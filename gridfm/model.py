"""Edge-state recurrent GridFM.

Unlike the prior node-centric model, every component-terminal incidence has a learned state.
Those states exchange messages with both the component and bus, propose complex terminal dV,
and are supervised on local complex voltage drops. A graph-global recurrent state supplies a
short path for long-range load aggregation without encoding feeder identity.
"""
from __future__ import annotations

import torch
from torch import nn

from .legacy import PE_DIM_EXT, SPECS, i_offset, n_slots, store_width, y_width
from .kcl_feedback import nodal_current_residual


def _pool(x: torch.Tensor, batch: torch.Tensor, n_graph: int, reduce: str) -> torch.Tensor:
    out = x.new_zeros(n_graph, x.shape[-1])
    out.index_add_(0, batch, x)
    if reduce == "sum":
        return out
    count = x.new_zeros(n_graph, 1)
    count.index_add_(0, batch, x.new_ones(x.shape[0], 1))
    return out / count.clamp_min(1)


class MLP(nn.Sequential):
    def __init__(self, din: int, dout: int, hidden: int, dropout: float = 0.0,
                 zero_last: bool = False):
        super().__init__(
            nn.Linear(din, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, dout),
        )
        if zero_last:
            nn.init.zeros_(self[-1].weight)
            nn.init.zeros_(self[-1].bias)


class FeatureStats(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("std", torch.ones(width))


class EdgeStateGridFM(nn.Module):
    """Recurrent heterogeneous component-terminal operator learner."""

    def __init__(self, hidden: int = 256, steps: int = 12, dropout: float = 0.0,
                 condition_on_scale: bool = True, normalize_features: bool = False,
                 aggregation: str = "mean", use_electrical_pe: bool = True,
                 directional_sweeps: bool = False, role_residual_heads: bool = False,
                 task_conditioning: bool = False, kcl_feedback: bool = False):
        super().__init__()
        if aggregation not in {"mean", "local_sum", "sum"}:
            raise ValueError(
                f"aggregation must be mean, local_sum, or sum, got {aggregation!r}"
            )
        self.hidden = hidden
        self.steps = steps
        self.condition_on_scale = condition_on_scale
        self.normalize_features = normalize_features
        self.aggregation = aggregation
        self.use_electrical_pe = use_electrical_pe
        self.directional_sweeps = directional_sweeps
        self.role_residual_heads_enabled = role_residual_heads
        self.task_conditioning = task_conditioning
        self.kcl_feedback_enabled = kcl_feedback
        # pu scale that normalizes the fed-back KCL residual; set from the corpus
        # current scaler by train.py/evaluate.py. asinh keeps it O(1).
        self.register_buffer("s_kcl", torch.tensor(1.0), persistent=False)
        self.feature_stats = nn.ModuleDict({s: FeatureStats(store_width(s)) for s in SPECS})
        node_in = 2 + 2 + 1 + 1 + 1 + PE_DIM_EXT
        self.node_encoder = MLP(node_in, hidden, hidden, dropout)
        self.comp_encoder = nn.ModuleDict({
            s: MLP((4 if condition_on_scale else 3) * store_width(s), hidden, hidden, dropout)
            for s in SPECS
        })
        self.slot_embedding = nn.ModuleDict({
            s: nn.Embedding(n_slots(s), hidden // 4) for s in SPECS
        })
        edge_in = 2 * hidden + hidden // 4
        self.edge_update = nn.ModuleDict({
            s: MLP(edge_in, hidden, hidden, dropout) for s in SPECS
        })
        # Zero initialization makes old checkpoints exactly equivalent until a
        # directional fine-tune learns leaf-to-root and root-to-leaf messages.
        self.directional_update = MLP(2 * hidden, hidden, hidden, dropout, zero_last=True)
        self.node_gru = nn.GRUCell(2 * hidden, hidden)
        # KCL-residual feedback: zero-initialized so a checkpoint without it is
        # reproduced exactly until the feedback path is trained (safe fine-tune).
        self.kcl_feedback_mlp = MLP(2, hidden, hidden, dropout, zero_last=True)
        self.comp_gru = nn.ModuleDict({s: nn.GRUCell(2 * hidden, hidden) for s in SPECS})
        self.global_gru = nn.GRUCell(hidden, hidden)
        self.task_encoder = MLP(4, hidden, hidden, dropout, zero_last=True)
        self.node_norm = nn.LayerNorm(hidden)
        self.comp_norm = nn.ModuleDict({s: nn.LayerNorm(hidden) for s in SPECS})

        # Exact zero means V=V_init and I/Y feature predictions decode to zero.
        # This is the safe physical baseline; random asinh-feature outputs can
        # decode through sinh to catastrophic currents before the first update.
        self.node_head = MLP(hidden, 2, hidden, dropout, zero_last=True)
        self.edge_dv_head = nn.ModuleDict({
            s: MLP(hidden, 2, hidden, dropout, zero_last=True) for s in SPECS
        })
        self.node_edge_gate = MLP(2 * hidden, 1, hidden, dropout)
        self.field_head = nn.ModuleDict({
            s: MLP(hidden, store_width(s), hidden, dropout, zero_last=True) for s in SPECS
        })
        self.role_residual_heads = nn.ModuleDict()
        for store in SPECS:
            ny, ni, width = y_width(store), i_offset(store), store_width(store)
            heads = {"y": MLP(hidden, ny, hidden, dropout, zero_last=True)}
            if ni > ny:
                heads["icomp"] = MLP(hidden, ni - ny, hidden, dropout, zero_last=True)
            heads["ibus"] = MLP(hidden, width - ni, hidden, dropout, zero_last=True)
            self.role_residual_heads[store] = nn.ModuleDict(heads)

    def set_feature_stats(self, stats: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
        for store, (mean, std) in stats.items():
            target = self.feature_stats[store]
            target.mean.copy_(mean.to(device=target.mean.device, dtype=target.mean.dtype))
            target.std.copy_(std.to(device=target.std.device, dtype=target.std.dtype))

    def _encode(self, batch):
        nd = batch["node"]
        dtype = next(self.node_encoder.parameters()).dtype
        vis = nd.vis_v.unsqueeze(1)
        pe = nd.pe[:, :PE_DIM_EXT].to(dtype)
        if not self.use_electrical_pe:
            # The final PE coordinate is computed from true variant-0 Y.  It is
            # valid for PF where Y is observed, but leaks targets in masked-Y
            # foundation tasks.  Retain structural RWSE and unweighted depth.
            pe = pe.clone()
            pe[:, -1] = 0
        node_x = torch.cat([
            nd.v_init.to(dtype),
            nd.dv.to(dtype) * vis,
            vis.to(dtype),
            nd.ground.to(dtype).unsqueeze(1),
            nd.slack.to(dtype).unsqueeze(1),
            pe,
        ], dim=1)
        hn = self.node_encoder(node_x)
        hc = {}
        for store in SPECS:
            st = batch[store]
            visible = st.vis.to(dtype)
            # Target features are asinh(x_pu / scale). The family-specific
            # scale (notably Line vs TriplexLine) is therefore part of the
            # coordinate system, not metadata the model may infer or memorize.
            # Omitting it made unseen components' current feature targets
            # ambiguous even when their physical parameters were visible.
            log_scale = torch.log10(st.scale.to(dtype).clamp_min(1e-12)).clamp(-12, 12) / 12
            values = st.x_true.to(dtype)
            if self.normalize_features:
                stats = self.feature_stats[store]
                values = (values - stats.mean.to(dtype)) / stats.std.to(dtype)
            parts = [values * visible, visible, st.act.to(dtype)]
            if self.condition_on_scale:
                parts.append(log_scale)
            x = torch.cat(parts, dim=1)
            hc[store] = self.comp_encoder[store](x)
        return hn, hc

    def _field_pred(self, store: str, hc_store: torch.Tensor) -> torch.Tensor:
        """Full store prediction vector from its hidden state (heads + std)."""
        head = self.field_head[store](hc_store)
        if self.role_residual_heads_enabled:
            roles = self.role_residual_heads[store]
            pieces = [roles["y"](hc_store)]
            if "icomp" in roles:
                pieces.append(roles["icomp"](hc_store))
            pieces.append(roles["ibus"](hc_store))
            head = head + torch.cat(pieces, dim=1)
        if self.normalize_features:
            head = head * self.feature_stats[store].std.to(head.dtype)
        return head

    def forward(self, batch, return_aux: bool = False):
        hn, hc = self._encode(batch)
        nd = batch["node"]
        node_batch = getattr(nd, "batch", None)
        if node_batch is None:
            node_batch = torch.zeros(nd.num_nodes, dtype=torch.long, device=hn.device)
        n_graph = int(node_batch.max().item()) + 1 if node_batch.numel() else 0
        global_reduce = "sum" if self.aggregation == "sum" else "mean"
        hg = _pool(hn, node_batch, n_graph, global_reduce)
        if self.task_conditioning:
            # Global missingness rates identify the requested operation without
            # exposing target values: V, Y, Icomp, and Ibus masked/active ratios.
            num = hn.new_zeros(n_graph, 4)
            den = hn.new_zeros(n_graph, 4)
            num[:, 0].index_add_(
                0, node_batch, nd.msk_v.to(hn.dtype)
            )
            den[:, 0].index_add_(
                0, node_batch, (~nd.ground).to(hn.dtype)
            )
            for store in SPECS:
                st = batch[store]
                if st.num_nodes == 0:
                    continue
                cb = getattr(st, "batch", None)
                if cb is None:
                    cb = torch.zeros(st.num_nodes, dtype=torch.long, device=hn.device)
                ny, ni = y_width(store), i_offset(store)
                for col, cols in enumerate((slice(0, ny), slice(ny, ni), slice(ni, None)), 1):
                    num[:, col].index_add_(
                        0, cb, st.msk[:, cols].sum(1).to(hn.dtype)
                    )
                    den[:, col].index_add_(
                        0, cb, st.act[:, cols].sum(1).to(hn.dtype)
                    )
            hg = hg + self.task_encoder(num / den.clamp_min(1))
        edge_state = {}

        for _ in range(self.steps):
            node_msg = hn.new_zeros(hn.shape)
            node_degree = hn.new_zeros(hn.shape[0], 1)
            up_msg = hn.new_zeros(hn.shape)
            down_msg = hn.new_zeros(hn.shape)
            comp_msg = {}
            for store in SPECS:
                es = batch[(store, "conn", "node")]
                comp, node, slot = es.edge_index[0], es.edge_index[1], es.slot
                if comp.numel() == 0:
                    comp_msg[store] = hc[store].new_zeros(hc[store].shape)
                    edge_state[store] = hc[store].new_zeros((0, self.hidden))
                    continue
                edge = self.edge_update[store](torch.cat([
                    hc[store][comp], hn[node], self.slot_embedding[store](slot)
                ], dim=1))
                # Autocast may return BF16 from the edge MLP while recurrent
                # states and index_add accumulators remain FP32. Accumulate in
                # the state dtype; this is also more stable for high-degree buses.
                edge = edge.to(dtype=hn.dtype)
                edge_state[store] = edge
                node_msg.index_add_(0, node, edge)
                node_degree.index_add_(0, node, edge.new_ones(edge.shape[0], 1))
                cm = hc[store].new_zeros(hc[store].shape)
                cd = hc[store].new_zeros(hc[store].shape[0], 1)
                cm.index_add_(0, comp, edge)
                cd.index_add_(0, comp, edge.new_ones(edge.shape[0], 1))
                comp_msg[store] = cm if self.aggregation == "sum" else cm / cd.clamp_min(1)

                if self.directional_sweeps and SPECS[store].terms > 1:
                    # For each multi-terminal component, aggregate latent
                    # evidence on the deeper side toward the source and source-
                    # side context toward descendants.  Exact (unclipped) BFS
                    # depth is structural only and never uses solved targets.
                    depth = getattr(nd, "depth_raw", nd.depth).to(edge.dtype)
                    edge_depth = depth[node]
                    n_comp = hc[store].shape[0]
                    min_depth = edge_depth.new_full((n_comp,), float("inf"))
                    min_depth.scatter_reduce_(
                        0, comp, edge_depth, reduce="amin", include_self=True
                    )
                    shallow = edge_depth <= min_depth[comp] + 1e-6
                    deep = ~shallow
                    shallow_sum = edge.new_zeros(n_comp, self.hidden)
                    shallow_count = edge.new_zeros(n_comp, 1)
                    deep_sum = edge.new_zeros(n_comp, self.hidden)
                    deep_count = edge.new_zeros(n_comp, 1)
                    if shallow.any():
                        shallow_sum.index_add_(0, comp[shallow], edge[shallow])
                        shallow_count.index_add_(
                            0, comp[shallow], edge.new_ones(int(shallow.sum()), 1)
                        )
                    if deep.any():
                        deep_sum.index_add_(0, comp[deep], edge[deep])
                        deep_count.index_add_(
                            0, comp[deep], edge.new_ones(int(deep.sum()), 1)
                        )
                    shallow_mean = shallow_sum / shallow_count.clamp_min(1)
                    deep_mean = deep_sum / deep_count.clamp_min(1)
                    if shallow.any():
                        up_msg.index_add_(0, node[shallow], deep_mean[comp[shallow]])
                    if deep.any():
                        down_msg.index_add_(0, node[deep], shallow_mean[comp[deep]])

            if self.aggregation == "mean":
                node_msg = node_msg / node_degree.clamp_min(1)
            if self.directional_sweeps:
                node_msg = node_msg + self.directional_update(
                    torch.cat([up_msg, down_msg], dim=1)
                )
            hg = self.global_gru(_pool(hn, node_batch, n_graph, global_reduce), hg)
            hn = self.node_norm(self.node_gru(torch.cat([node_msg, hg[node_batch]], 1), hn))
            if self.kcl_feedback_enabled:
                # Learned iterative solver (foundation, task-agnostic): compute
                # the nodal KCL residual of the COMPLETED terminal-current
                # estimates (observed-where-visible, predicted-where-masked) and
                # feed it back so the network refines its hidden state toward a
                # physically consistent solution — whatever variable is masked.
                # Σ Ibus is O(1)/well-conditioned AND differentiable (dr/dIbus=1),
                # so no detach; physics stays fp32. Re-normalize to bound compounding.
                ibus_feat = {s: self._field_pred(s, hc[s])[:, i_offset(s):].float()
                             for s in SPECS if batch[s].num_nodes}
                res = nodal_current_residual(batch, ibus_feat)
                res = torch.asinh(res / (self.s_kcl.float() + 1e-12))
                hn = self.node_norm(hn + self.kcl_feedback_mlp(res.to(hn.dtype)))
            for store in SPECS:
                st = batch[store]
                if st.num_nodes == 0:
                    continue
                cb = getattr(st, "batch", None)
                if cb is None:
                    cb = torch.zeros(st.num_nodes, dtype=torch.long, device=hn.device)
                hc[store] = self.comp_norm[store](self.comp_gru[store](
                    torch.cat([comp_msg[store], hg[cb]], 1), hc[store]
                ))

        node_base = self.node_head(hn)
        edge_sum = node_base.new_zeros(node_base.shape)
        edge_count = node_base.new_zeros(node_base.shape[0], 1)
        edge_dv = {}
        for store in SPECS:
            es = batch[(store, "conn", "node")]
            node = es.edge_index[1]
            proposal = self.edge_dv_head[store](edge_state[store])
            proposal = proposal.to(dtype=node_base.dtype)
            edge_dv[store] = proposal
            if node.numel():
                edge_sum.index_add_(0, node, proposal)
                edge_count.index_add_(0, node, proposal.new_ones(proposal.shape[0], 1))
        edge_mean = edge_sum / edge_count.clamp_min(1)
        gate = torch.sigmoid(self.node_edge_gate(torch.cat([hn, node_msg], 1)))
        node_dv = node_base + gate * (edge_mean - node_base)

        preds = {"node": node_dv}
        field_std = {}
        for store in SPECS:
            preds[store] = self._field_pred(store, hc[store])
            field_std[store] = self.feature_stats[store].std.to(preds[store].dtype)
        if return_aux:
            return preds, {
                "edge_dv": edge_dv, "node_base": node_base, "gate": gate,
                "field_std": field_std,
            }
        return preds


def load_compatible_state(model: nn.Module, state: dict) -> None:
    """Load pre-normalization checkpoints while rejecting unrelated mismatch."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = (
        "feature_stats.", "directional_update.", "role_residual_heads.",
        "task_encoder.", "kcl_feedback_mlp.", "s_kcl",
    )
    bad_missing = [key for key in missing if not key.startswith(allowed_missing)]
    allowed_obsolete = ("icomp_current_skip.",)
    bad_unexpected = [key for key in unexpected if not key.startswith(allowed_obsolete)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={bad_missing}, unexpected={bad_unexpected}"
        )

"""Edge-state recurrent GridFM.

Unlike the prior node-centric model, every component-terminal incidence has a learned state.
Those states exchange messages with both the component and bus, propose complex terminal dV,
and are supervised on local complex voltage drops. A graph-global recurrent state supplies a
short path for long-range load aggregation without encoding feeder identity.
"""
from __future__ import annotations

import torch
from torch import nn

from .legacy import PE_DIM_EXT, SPECS, n_slots, store_width


def _mean_pool(x: torch.Tensor, batch: torch.Tensor, n_graph: int) -> torch.Tensor:
    out = x.new_zeros(n_graph, x.shape[-1])
    out.index_add_(0, batch, x)
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


class EdgeStateGridFM(nn.Module):
    """Recurrent heterogeneous component-terminal operator learner."""

    def __init__(self, hidden: int = 256, steps: int = 12, dropout: float = 0.0):
        super().__init__()
        self.hidden = hidden
        self.steps = steps
        node_in = 2 + 2 + 1 + 1 + 1 + PE_DIM_EXT
        self.node_encoder = MLP(node_in, hidden, hidden, dropout)
        self.comp_encoder = nn.ModuleDict({
            s: MLP(3 * store_width(s), hidden, hidden, dropout) for s in SPECS
        })
        self.slot_embedding = nn.ModuleDict({
            s: nn.Embedding(n_slots(s), hidden // 4) for s in SPECS
        })
        edge_in = 2 * hidden + hidden // 4
        self.edge_update = nn.ModuleDict({
            s: MLP(edge_in, hidden, hidden, dropout) for s in SPECS
        })
        self.node_gru = nn.GRUCell(2 * hidden, hidden)
        self.comp_gru = nn.ModuleDict({s: nn.GRUCell(2 * hidden, hidden) for s in SPECS})
        self.global_gru = nn.GRUCell(hidden, hidden)
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

    def _encode(self, batch):
        nd = batch["node"]
        dtype = next(self.node_encoder.parameters()).dtype
        vis = nd.vis_v.unsqueeze(1)
        node_x = torch.cat([
            nd.v_init.to(dtype),
            nd.dv.to(dtype) * vis,
            vis.to(dtype),
            nd.ground.to(dtype).unsqueeze(1),
            nd.slack.to(dtype).unsqueeze(1),
            nd.pe[:, :PE_DIM_EXT].to(dtype),
        ], dim=1)
        hn = self.node_encoder(node_x)
        hc = {}
        for store in SPECS:
            st = batch[store]
            visible = st.vis.to(dtype)
            x = torch.cat([st.x_true.to(dtype) * visible, visible, st.act.to(dtype)], dim=1)
            hc[store] = self.comp_encoder[store](x)
        return hn, hc

    def forward(self, batch, return_aux: bool = False):
        hn, hc = self._encode(batch)
        nd = batch["node"]
        node_batch = getattr(nd, "batch", None)
        if node_batch is None:
            node_batch = torch.zeros(nd.num_nodes, dtype=torch.long, device=hn.device)
        n_graph = int(node_batch.max().item()) + 1 if node_batch.numel() else 0
        hg = _mean_pool(hn, node_batch, n_graph)
        edge_state = {}

        for _ in range(self.steps):
            node_msg = hn.new_zeros(hn.shape)
            node_degree = hn.new_zeros(hn.shape[0], 1)
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
                comp_msg[store] = cm / cd.clamp_min(1)

            node_msg = node_msg / node_degree.clamp_min(1)
            hg = self.global_gru(_mean_pool(hn, node_batch, n_graph), hg)
            hn = self.node_norm(self.node_gru(torch.cat([node_msg, hg[node_batch]], 1), hn))
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
        preds.update({s: self.field_head[s](hc[s]) for s in SPECS})
        if return_aux:
            return preds, {"edge_dv": edge_dv, "node_base": node_base, "gate": gate}
        return preds

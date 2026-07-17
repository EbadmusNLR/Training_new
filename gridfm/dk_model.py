#!/usr/bin/env python3
"""Iterative-solver GridFM for the datakit full-matrix format.

A weight-tied recurrent hetero message-passing network that maintains a latent
state for nodes and components, and — the key ingredient — each step feeds back
the exact nodal KCL residual of its own COMPLETED terminal-current estimates
(observed where visible, predicted where masked; well-conditioned O(1), no stiff
Y*V). It drives that residual to zero, i.e. a learned iterative power-system
solver, task-agnostic over whatever is masked.

Inputs are the datakit HeteroData (full-matrix Y_pu, per-terminal currents,
Icomp, V). Currents/Y are consumed via an asinh feature view; physics (KCL) runs
on decoded pu. See dk_physics for the full-matrix physics.
"""
from __future__ import annotations

import torch
from torch import nn

from .dk_physics import FC, STORES, terminal_slot, node_count
from .dk_data import PE_DIM, I_SCALE, Y_SCALE, feat, inv_feat
from .dk_tree import (SERIES_STORES, plan_to, reconstruct_vectorized,
                      reconstruct_full, recon_ctx_to)

# Genuinely SHUNT/local families whose terminal current is a well-conditioned
# function of bus V (I=Y@V-Icomp), so we decode from the predicted V instead of a
# free head. SERIES elements are excluded because I~Y*(V1-V2) amplifies V error:
# lines (~2e7x, unrecoverable) and transformers (~24x measured, 110% WAPE) both go
# to the structural tree-current path; the vsource slack keeps a free head too.
# Physics-decoded from V (I = Y@V - Icomp): every 1-terminal shunt AND both AMBIGUOUS
# 2-terminal stores. `reactor` was missing while SERIES_STORES contains it, so the model
# zeroed every SHUNT reactor (WAPE 1.0 on that family) -- the same silently-zero bug found
# in test_all.py. capacitor was already here; reactor is its exact counterpart.
PHYS_DECODE = {"load", "capacitor", "reactor", "pvsystem", "storage", "generator"}


class MLP(nn.Sequential):
    def __init__(self, din, dout, hidden, zero_last=False):
        super().__init__(nn.Linear(din, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Linear(hidden, dout))
        if zero_last:
            nn.init.zeros_(self[-1].weight); nn.init.zeros_(self[-1].bias)


class DKSolver(nn.Module):
    def __init__(self, hidden=256, steps=12, kcl_feedback=True, use_feat=True, scales=None,
                 exact_decoder=True):
        super().__init__()
        self.hidden = hidden
        self.steps = steps
        self.kcl_feedback = kcl_feedback
        self.use_feat = use_feat
        # reconstruct_full (exact) vs reconstruct_vectorized (the old model path).
        # MEASURED on SMART-DS variants, decoding from TRUTH V: full = 4.6e-08..1.8e-07,
        # vectorized = 3.0e-01..9.0e-01. The vectorized path is only accurate on tiny
        # synthetic cases (minimal_component 4.3e-03), i.e. on 2.8% of the corpus by node
        # count -- it was never right on the real feeders that carry 95% of the gradient.
        self.exact_decoder = exact_decoder
        self.stores = list(STORES)
        self.node_enc = MLP(2 + 2 + 1 + PE_DIM, hidden, hidden)  # v_init, dv*vis, vis, pe
        self.comp_enc = nn.ModuleDict()
        self.slot_emb = nn.ModuleDict()
        self.edge_mlp = nn.ModuleDict()
        self.comp_gru = nn.ModuleDict()
        self.cur_head = nn.ModuleDict()
        self.ic_head = nn.ModuleDict()
        for s in self.stores:
            _, nterm, _ = STORES[s]
            dim = nterm * FC
            # feat(Yr),feat(Yi) + feat(Icomp_r),feat(Icomp_i) (zeroed where hidden) + vis flag.
            # Icomp must be an ENCODER input, not only a physics constant: injection
            # estimation hides it, and a model that never sees it cannot miss it.
            self.comp_enc[s] = MLP(2 * dim * dim + 2 * FC + 1, hidden, hidden)
            # Icomp estimate head (feat space). Only consulted where vis_ic is False;
            # visible entries stay pinned to the data, like _decode_dv does for V.
            self.ic_head[s] = MLP(hidden, 2 * FC, hidden, zero_last=True)
            self.slot_emb[s] = nn.Embedding(dim, hidden // 4)
            self.edge_mlp[s] = MLP(2 * hidden + hidden // 4, hidden, hidden)
            self.comp_gru[s] = nn.GRUCell(hidden, hidden)
            self.cur_head[s] = MLP(hidden, 2 * dim, hidden, zero_last=True)  # I_feat r,i per slot
            # fitted global per-family scales (buffers -> saved & moved with model)
            isc = max(float(scales["I"][s]) if scales else I_SCALE, 1e-9)
            self.register_buffer(f"iscale_{s}", torch.tensor(isc))
            ys = torch.ones(dim, dim, 2)
            if scales:
                yd = scales["Y"][s]; eye = torch.eye(dim, dtype=torch.bool)
                ys[..., 0] = torch.where(eye, torch.tensor(float(yd["r_diag"])), torch.tensor(float(yd["r_off"])))
                ys[..., 1] = torch.where(eye, torch.tensor(float(yd["i_diag"])), torch.tensor(float(yd["i_off"])))
            else:
                ys = ys * Y_SCALE
            self.register_buffer(f"yscale_{s}", ys.clamp(min=1e-9))
        self.node_gru = nn.GRUCell(hidden, hidden)
        # Standardized-residual gauge: the head predicts z and dv = dv_std * z, so its
        # output (and its gradients) live at O(1) instead of O(|dv|) ~ 1e-2. This is the
        # mechanism behind the reference PINN's 7.5e-08 run (train_physics_informed_NN,
        # "standardized residual voltage-head"); without it the head must learn outputs
        # two orders below its init scale.
        self.node_head = MLP(hidden, 2, hidden, zero_last=True)         # z; dv = dv_std*z
        dv_std = scales.get("dv_std", [1.0, 1.0]) if scales else [1.0, 1.0]
        self.register_buffer("dv_std", torch.tensor(dv_std, dtype=torch.float32))
        self.kcl_mlp = MLP(2, hidden, hidden, zero_last=True)
        self.register_buffer("s_kcl", torch.tensor(float(scales["kcl"]) if scales else I_SCALE))

    def _iscale(self, s):
        return getattr(self, f"iscale_{s}")

    def _yscale(self, s):
        return getattr(self, f"yscale_{s}")

    def _decode_I(self, s, fr, fi):
        sc = self._iscale(s)
        return inv_feat(fr, sc, self.use_feat), inv_feat(fi, sc, self.use_feat)

    def _edges(self, batch):
        out = {}
        for s in self.stores:
            if s not in batch.node_types:
                continue
            _, nterm, _ = STORES[s]
            terms = []
            for t in range(1, nterm + 1):
                rel = (s, f"bus{t}", "node")
                if rel in batch.edge_types and batch[rel].edge_index.numel():
                    ei = batch[rel].edge_index
                    comp, node = ei[0], ei[1]
                    col = (t - 1) * FC + terminal_slot(comp)
                    terms.append((comp, node, col))
                else:
                    terms.append(None)
            out[s] = terms
        return out

    def _phys_current(self, s, batch, terms, v, icomp=None):
        """Physics-decoded terminal currents I = Y@V - Icomp from the current V
        estimate, for well-conditioned families (loads/shunts/transformers). V
        error maps ~linearly to current error here (unlike stiff series lines).
        `icomp` overrides the stored compensation: for injection estimation the
        model's Icomp ESTIMATE must drive the decode, so the current loss and KCL
        pull the estimate toward truth -- the iterated-unknown pattern."""
        st = batch[s]
        n, dim, _ = st.yr.shape
        Vlr = v.new_zeros(n, dim); Vli = v.new_zeros(n, dim)
        for t in terms:
            if t is None:
                continue
            comp, node, col = t
            Vlr[comp, col] = v[node, 0]
            Vli[comp, col] = v[node, 1]
        Ir = torch.bmm(st.yr, Vlr.unsqueeze(-1)).squeeze(-1) - torch.bmm(st.yi, Vli.unsqueeze(-1)).squeeze(-1)
        Ii = torch.bmm(st.yr, Vli.unsqueeze(-1)).squeeze(-1) + torch.bmm(st.yi, Vlr.unsqueeze(-1)).squeeze(-1)
        _, _, nic = STORES[s]
        if nic:
            icr, ici = (st.icr, st.ici) if icomp is None else icomp
            w = min(nic, dim, icr.shape[1])
            Ir = torch.cat([Ir[:, :w] - icr[:, :w], Ir[:, w:]], 1)
            Ii = torch.cat([Ii[:, :w] - ici[:, :w], Ii[:, w:]], 1)
        return Ir, Ii

    def _completed_currents(self, batch, edges, hc, v):
        """All terminal currents (pu): SHUNT families physics-decoded from V
        (well-conditioned), SERIES families (line/transformer/vsource) filled by
        the differentiable KCL subtree reconstruction over those shunts. No Y*V
        for stiff series, no free head. See dk_tree.reconstruct_full."""
        cur = {}
        aux = {"ic_est": {}, "ic_msk": {}}
        for s, terms in edges.items():
            if s in PHYS_DECODE:
                icomp = None
                st = batch[s]
                if hc is not None and hasattr(st, "vis_ic") and not bool(st.vis_ic.all()):
                    # Clamp the estimate's feature z to +-8 before inv_feat: inv_feat is a
                    # sinh, so an untrained head emitting z~20 decodes to 2.4e8x scale --
                    # measured ic_wape 16713% in the DDP smoke -- and that explosion drives
                    # the physics decode on injection samples, wedging whole runs (the T10
                    # s1 stall pattern). sinh(8)~1490x scale still covers any real Icomp.
                    z = self.ic_head[s](hc[s]).clamp(-8.0, 8.0)
                    er = inv_feat(z[:, :FC], self._iscale(s), self.use_feat)
                    ei = inv_feat(z[:, FC:], self._iscale(s), self.use_feat)
                    m = st.vis_ic.unsqueeze(1)
                    w = st.icr.shape[1]
                    icomp = (torch.where(m, st.icr, er[:, :w]),
                             torch.where(m, st.ici, ei[:, :w]))
                    aux["ic_est"][s] = (er[:, :w], ei[:, :w])
                    aux["ic_msk"][s] = ~st.vis_ic
                cur[s] = self._phys_current(s, batch, terms, v, icomp=icomp)
        # Zero ONLY the always-series stores. `reactor` is AMBIGUOUS: a grounded one is a
        # SHUNT and is physics-decoded exactly, while a both-ends-live one is series. Zeroing
        # the whole store (as before) wiped the decoded shunt reactors and left them at
        # exactly 0 -- the silently-zero signature. The tree sweep below overwrites only the
        # conductors that ARE tree edges, so series reactors still get their through-flow
        # while shunt reactors keep their decode. Same rule reconstruct_full uses.
        for s in ("line", "transformer", "vsource"):
            if s in edges:
                st = batch[s]; n, dim, _ = st.yr.shape
                z = v.new_zeros(n, dim)
                cur[s] = (z, z.clone())           # placeholder; reconstruction fills it
        self._last_aux = aux
        ctx = getattr(batch, "recon_ctx", None)
        if self.exact_decoder and ctx is not None:
            # The exact path: LV lines -> xfmr secondary KCL -> xfmr primary null-space
            # map -> all lines(+xfmr inj) -> parallel-line division -> vsource KCL, plus
            # the well-conditioned Yh line-charging common-mode from V. reconstruct_full
            # NEEDS V; reconstruct_vectorized never took it, which is why it could not be
            # right on feeders with line charging or tapped transformers.
            return reconstruct_full(batch, cur, v[:, 0], v[:, 1],
                                    ctx=recon_ctx_to(ctx, v.device, v.dtype))
        plan = plan_to(batch.tree_plan, v.device)
        return reconstruct_vectorized(plan, cur)

    def _kcl_residual(self, batch, edges, cur_preds):
        n_node = node_count(batch)
        dev = batch["node"].V_r_init_pu.device
        rr = torch.zeros(n_node, device=dev)
        ri = torch.zeros(n_node, device=dev)
        for s, terms in edges.items():
            ir, ii = cur_preds[s]
            for t in terms:
                if t is None:
                    continue
                comp, node, col = t
                rr.index_add_(0, node, ir[comp, col])
                ri.index_add_(0, node, ii[comp, col])
        res = torch.stack([rr, ri], 1)
        res[0] = 0.0  # ground
        return res

    def forward(self, batch):
        nd = batch["node"]
        dev = nd.V_r_init_pu.device
        edges = self._edges(batch)
        # node encode
        vis = nd.vis_v.unsqueeze(1).float()
        node_in = torch.cat([nd.v_init, nd.dv * vis, vis, nd.pe], 1)
        hn = self.node_enc(node_in)
        # component encode from asinh(Y)
        hc = {}
        for s in edges:
            st = batch[s]
            n = st.yr.shape[0]
            yst = torch.stack([st.yr, st.yi], -1)              # [n,dim,dim,2]
            yf = feat(yst, self._yscale(s), self.use_feat).reshape(n, -1)
            vis_ic = st.vis_ic if hasattr(st, "vis_ic") else torch.ones(n, dtype=torch.bool, device=yf.device)
            icr = yf.new_zeros(n, FC); ici = yf.new_zeros(n, FC)
            w = min(FC, st.icr.shape[1])
            icr[:, :w] = st.icr[:, :w]; ici[:, :w] = st.ici[:, :w]
            gate = vis_ic.unsqueeze(1).float()
            icf = torch.cat([feat(icr, self._iscale(s), self.use_feat) * gate,
                             feat(ici, self._iscale(s), self.use_feat) * gate, gate], 1)
            hc[s] = self.comp_enc[s](torch.cat([yf, icf], 1))
        for step in range(self.steps):
            node_msg = torch.zeros_like(hn)
            node_deg = torch.zeros(hn.shape[0], 1, device=dev)
            comp_msg = {s: torch.zeros_like(hc[s]) for s in edges}
            comp_deg = {s: torch.zeros(hc[s].shape[0], 1, device=dev) for s in edges}
            for s, terms in edges.items():
                for ti, t in enumerate(terms):
                    if t is None:
                        continue
                    comp, node, col = t
                    e = self.edge_mlp[s](torch.cat([hc[s][comp], hn[node], self.slot_emb[s](col)], 1))
                    node_msg.index_add_(0, node, e)
                    node_deg.index_add_(0, node, torch.ones(e.shape[0], 1, device=dev))
                    comp_msg[s].index_add_(0, comp, e)
                    comp_deg[s].index_add_(0, comp, torch.ones(e.shape[0], 1, device=dev))
            hn = self.node_gru(node_msg / node_deg.clamp(min=1), hn)
            for s in edges:
                hc[s] = self.comp_gru[s](comp_msg[s] / comp_deg[s].clamp(min=1), hc[s])
        # currents are exact functions of V now (physics-decode shunts + KCL tree
        # reconstruction, which enforces nodal balance structurally), so there is
        # no residual to feed back: the model is a pure V-predictor and the MP
        # depth is the iterative refinement. Reconstruct once at the end.
        dvp = self.node_head(hn) * self.dv_std
        v = nd.v_init + self._decode_dv(nd, hn, dvp)
        cur = self._completed_currents(batch, edges, hc, v)
        return dvp, cur, self._last_aux

    def _decode_dv(self, nd, hn, dvp=None):
        """V estimate delta: predicted where masked, pinned to truth where the
        voltage is observed (slack/ground in pf) so physics decode is anchored."""
        if dvp is None:
            dvp = self.node_head(hn) * self.dv_std
        return torch.where(nd.vis_v.unsqueeze(1), nd.dv, dvp)

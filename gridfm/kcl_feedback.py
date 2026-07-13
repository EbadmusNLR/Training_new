"""Differentiable nodal KCL residual of the model's current voltage estimate.

The unseen current error is voltage-generalization gated by stiff reactor/line
currents (experiments.md D-attrib): Ibus ≈ 2.7·V and <1% current needs V<0.37%,
below the recurrent model's own seen ceiling (0.7%). A learned iterative solver
(Donon 2020 Graph Neural Solver; PowerFlowNet) breaks such ceilings by enforcing
physics at inference: each recurrent step computes the exact nodal KCL residual
r_n = Σ_e (Y·V − Icomp)_{e→n} of the CURRENT voltage estimate and feeds it back,
so the network refines V toward a physically consistent state instead of
regressing it one-shot. This uses observed Y/Icomp (visible in PF/SE) and the
predicted V only — no linear/PF solve, no cheating.
"""
from __future__ import annotations

import torch

from .legacy import FC, SPECS, i_offset, physics, y_width

EPS = physics.EPS
CLAMP = 20.0


def nodal_current_residual(batch, ibus_feat: dict) -> torch.Tensor:
    """Nodal KCL residual [N,2] from COMPLETED terminal-current estimates.

    ibus_feat[store] is the model's predicted current-feature block (columns
    i_offset(store):). It is completed with the observed truth where visible,
    decoded to pu, and summed at each node. r_n = Σ_e Ibus_{e→n}, zeroed at
    ground. Unlike Y·V this is well-conditioned (O(1)) AND differentiable in the
    predictions (dr/dIbus = 1), so it needs no detach — the network learns to
    drive it to zero. Task-agnostic: works whether V, Y, Icomp, or Ibus is the
    masked unknown (it always checks current conservation of the completed
    estimate).
    """
    nd = batch["node"]
    n_node = nd.num_nodes
    dev = ibus_feat[next(iter(ibus_feat))].device
    rr = torch.zeros(n_node, device=dev, dtype=torch.float32)
    ri = torch.zeros_like(rr)
    for store, spec in SPECS.items():
        st = batch[store]
        if st.num_nodes == 0:
            continue
        ni = i_offset(store)
        vis_i, msk_i = st.vis[:, ni:], st.msk[:, ni:]
        bar = st.x_true[:, ni:].to(dev) * vis_i + ibus_feat[store] * msk_i
        ibus = torch.sinh(bar.clamp(-CLAMP, CLAMP)) * (st.scale[:, ni:].to(dev) + EPS)
        ibus = ibus * st.act[:, ni:].to(ibus.dtype)
        es = batch[(store, "conn", "node")]
        comp, node, slot = es.edge_index[0], es.edge_index[1], es.slot
        col_r = (slot // FC) * 2 * FC + slot % FC
        rr.index_add_(0, node, ibus[comp, col_r].to(rr.dtype))
        ri.index_add_(0, node, ibus[comp, col_r + FC].to(ri.dtype))
    res = torch.stack([rr, ri], dim=1)
    return res * (~nd.ground).unsqueeze(1).to(res.dtype)


def nodal_kcl_residual(batch, vr: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
    """Return [N,2] nodal residual Σ Ibus(Y_vis·V − Icomp_vis), zeroed at ground.

    Ibus per element uses the observed admittance/Icomp (masked entries drop to
    zero, which is the best available guidance for inverse tasks) and the given
    voltage `vr,vi`. Fully differentiable in `vr,vi`.
    """
    nd = batch["node"]
    n_node = nd.num_nodes
    rr = vr.new_zeros(n_node)
    ri = vr.new_zeros(n_node)
    for store, spec in SPECS.items():
        st = batch[store]
        if st.num_nodes == 0:
            continue
        ny, ni = y_width(store), i_offset(store)
        y_pu = physics.decode_truth(
            st.x_true[:, :ny] * st.vis[:, :ny], st.scale[:, :ny]
        ).to(vr.dtype)
        Vr, Vi = physics._slot_voltages(batch, store, vr, vi)
        Ir, Ii = physics._element_currents(store, y_pu, Vr, Vi)
        if spec.icomp:
            ic = physics.decode_truth(
                st.x_true[:, ny:ni] * st.vis[:, ny:ni], st.scale[:, ny:ni]
            ).to(vr.dtype)
            Ir = Ir.clone()
            Ii = Ii.clone()
            Ir[:, : spec.icomp] = Ir[:, : spec.icomp] - ic[:, : spec.icomp]
            Ii[:, : spec.icomp] = Ii[:, : spec.icomp] - ic[:, spec.icomp:]
        es = batch[(store, "conn", "node")]
        comp, node, slot = es.edge_index[0], es.edge_index[1], es.slot
        rr.index_add_(0, node, Ir[comp, slot].to(rr.dtype))
        ri.index_add_(0, node, Ii[comp, slot].to(ri.dtype))
    res = torch.stack([rr, ri], dim=1)
    res = res * (~nd.ground).unsqueeze(1).to(res.dtype)
    return res

"""Local, unrolled PF voltage refinement from the validated element contract.

This module never assembles or solves a global bus matrix.  It evaluates each
component's local ``Y V - Icomp`` current, scatters the residual to buses, and
applies a damped complex-Jacobi update.  The caller must provide an exact-PF
graph mask: hidden parameters or injections make this update ill-posed.
"""
from __future__ import annotations

import torch

from .legacy import FC, SPECS, i_offset, n_slots, physics, y_width


def _component_operator(store: str, y_pu: torch.Tensor) -> torch.Tensor:
    """Return the local complex terminal admittance matrix for one store."""
    spec = SPECS[store]
    tri = physics.tri_size(spec.ydim)
    if store == "line":
        ys = torch.complex(
            physics._tri_to_full(y_pu[:, :tri], FC),
            physics._tri_to_full(y_pu[:, tri : 2 * tri], FC),
        )
        a = ys + 1j * physics._tri_to_full(y_pu[:, 2 * tri :], FC)
        return torch.cat(
            [torch.cat([a, -ys], 2), torch.cat([-ys, a], 2)], 1
        )
    return torch.complex(
        physics._tri_to_full(y_pu[:, :tri], spec.ydim),
        physics._tri_to_full(y_pu[:, tri:], spec.ydim),
    )


def _residual_and_diagonal(
    batch, x_bar: dict[str, torch.Tensor], voltage: torch.Tensor, clamp: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate nodal ``sum(YV-Icomp)`` and its coalesced Jacobi diagonal."""
    residual = voltage.new_zeros(voltage.shape)
    diagonal = voltage.new_zeros(voltage.shape)
    for store, spec in SPECS.items():
        st = batch[store]
        if st.num_nodes == 0:
            continue
        ny, ni = y_width(store), i_offset(store)
        y_pu = physics.decode_completed(
            x_bar[store][:, :ny], st.scale[:, :ny], st.msk[:, :ny], clamp
        )
        operator = _component_operator(store, y_pu)
        es = batch[(store, "conn", "node")]
        comp, node, slot = es.edge_index[0], es.edge_index[1], es.slot
        slots = n_slots(store)
        slot_node = torch.full(
            (st.num_nodes, slots), -1, dtype=torch.long, device=node.device
        )
        slot_node[comp, slot] = node
        slot_voltage = voltage.new_zeros(st.num_nodes, slots)
        slot_voltage[comp, slot] = voltage[node]
        terminal = torch.einsum("kij,kj->ki", operator, slot_voltage)
        if spec.icomp:
            icomp = physics.decode_completed(
                x_bar[store][:, ny:ni], st.scale[:, ny:ni],
                st.msk[:, ny:ni], clamp,
            )
            terminal = terminal - torch.complex(
                icomp[:, : spec.icomp], icomp[:, spec.icomp :]
            )
        residual.index_add_(0, node, terminal[comp, slot])

        # If multiple local slots coalesce onto one global node, every matrix
        # entry between those slots contributes to the global diagonal.
        same_node = (
            (slot_node.unsqueeze(2) == slot_node.unsqueeze(1))
            & (slot_node.unsqueeze(2) >= 0)
        )
        local_diagonal = (operator * same_node).sum(2)
        diagonal.index_add_(0, node, local_diagonal[comp, slot])
    return residual, diagonal


def refine_pf_voltages(
    batch,
    preds: dict[str, torch.Tensor],
    clamp: float,
    graph_mask: torch.Tensor,
    *,
    steps: int,
    damping: float,
    eps: float = 1e-10,
    max_step_pu: float = 0.02,
    return_metrics: bool = False,
):
    """Apply guarded damped-Jacobi updates to exact-PF graphs only.

    Visible voltages, slack, ground, and every non-PF graph remain unchanged.
    ``preds`` is not mutated.  Metrics are mean complex KCL residual magnitudes
    over updated nodes, expressed in pu.
    """
    if steps <= 0:
        return (preds, {}) if return_metrics else preds
    if not 0 < damping <= 1:
        raise ValueError("damping must be in (0, 1]")
    if eps <= 0 or max_step_pu <= 0:
        raise ValueError("eps and max_step_pu must be positive")

    x_bar, vr, vi = physics.completed(batch, preds)
    voltage = torch.complex(vr, vi)
    nd = batch["node"]
    node_batch = getattr(nd, "batch", None)
    if node_batch is None:
        node_batch = torch.zeros(
            nd.num_nodes, dtype=torch.long, device=nd.msk_v.device
        )
    update = (
        graph_mask[node_batch]
        & nd.msk_v
        & ~nd.slack
        & ~nd.ground
    )
    if not bool(update.any()):
        return (preds, {}) if return_metrics else preds

    before = None
    for step_index in range(steps):
        residual, diagonal = _residual_and_diagonal(batch, x_bar, voltage, clamp)
        if step_index == 0:
            before = residual[update].abs().mean()
        denom = diagonal.abs().square() + eps * eps
        delta = -damping * diagonal.conj() * residual / denom
        magnitude = delta.abs().clamp_min(torch.finfo(delta.real.dtype).tiny)
        delta = delta * (max_step_pu / magnitude).clamp_max(1)
        safe = update & (diagonal.abs() > eps) & torch.isfinite(delta)
        voltage = torch.where(safe, voltage + delta, voltage)

    final_residual, _ = _residual_and_diagonal(batch, x_bar, voltage, clamp)
    dv = torch.stack([voltage.real, voltage.imag], 1) - nd.v_init
    # Keep the refined voltage in the persisted float64 coordinate.  Casting a
    # solved voltage back to float32 before YV loses enough low bits that stiff
    # line admittances can amplify roundoff into enormous terminal currents.
    node_pred = preds["node"].to(dv.dtype).clone()
    node_pred[update] = dv[update].to(node_pred.dtype)
    out = {**preds, "node": node_pred}
    if not return_metrics:
        return out
    metrics = {
        "refine_kcl_before_pu": float(before.detach()),
        "refine_kcl_after_pu": float(final_residual[update].abs().mean().detach()),
    }
    return out, metrics

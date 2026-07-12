"""Local current-only KCL projection; no voltage or power-flow solve."""
from __future__ import annotations

import torch

from .legacy import FC, SPECS, i_offset, physics


def project_kcl(batch, preds, clamp: float, mode: str = "series"):
    """Project terminal currents onto nodal KCL in physical pu.

    `series` preserves device/shunt heads and assigns their imbalance to
    line/transformer/source terminals. `equal` uses the minimum-norm equal
    correction over all incident terminals.
    """
    if mode not in {"equal", "series"}:
        raise ValueError(mode)
    xbar, _, _ = physics.completed(batch, preds)
    dev = preds["node"].device
    n_node = batch["node"].num_nodes
    sum_r = torch.zeros(n_node, dtype=torch.float64, device=dev)
    sum_i = torch.zeros_like(sum_r)
    decoded = {}
    incidence = {}
    series_count = torch.zeros(n_node, dtype=torch.float64, device=dev)
    all_count = torch.zeros_like(series_count)
    source_count = torch.zeros_like(series_count)
    for store in SPECS:
        st = batch[store]
        if st.num_nodes == 0:
            continue
        ni = i_offset(store)
        cur = physics.decode_completed(
            xbar[store][:, ni:].double(), st.scale[:, ni:].double(), st.msk[:, ni:], clamp
        )
        es = batch[(store, "conn", "node")]
        comp, node, slot = es.edge_index[0], es.edge_index[1], es.slot
        col_r = (slot // FC) * 2 * FC + slot % FC
        ir, ii = cur[comp, col_r], cur[comp, col_r + FC]
        sum_r.index_add_(0, node, ir)
        sum_i.index_add_(0, node, ii)
        ones = torch.ones_like(ir)
        all_count.index_add_(0, node, ones)
        if store in {"line", "transformer", "vsource"}:
            series_count.index_add_(0, node, ones)
        if store == "vsource":
            source_count.index_add_(0, node, ones)
        decoded[store] = cur
        incidence[store] = (comp, node, col_r)

    slack = batch["node"].slack
    out = dict(preds)
    for store, cur in decoded.items():
        comp, node, col_r = incidence[store]
        if mode == "equal":
            weight = 1.0 / all_count[node].clamp_min(1)
        else:
            is_series = store in {"line", "transformer", "vsource"}
            use_series = series_count[node] > 0
            weight = torch.where(
                use_series,
                torch.full_like(series_count[node], float(is_series)) /
                series_count[node].clamp_min(1),
                1.0 / all_count[node].clamp_min(1),
            )
            # The source terminal is the unique slack-current degree of freedom.
            if store == "vsource":
                weight = torch.where(slack[node], 1.0 / source_count[node].clamp_min(1), weight)
            else:
                weight = torch.where(slack[node] & (source_count[node] > 0), 0.0, weight)
        corrected = cur.clone()
        corrected[comp, col_r] -= weight * sum_r[node]
        corrected[comp, col_r + FC] -= weight * sum_i[node]
        st = batch[store]
        ni = i_offset(store)
        encoded = torch.asinh(corrected / (st.scale[:, ni:].double() + 1e-12))
        p = out[store].clone()
        take = st.msk[:, ni:]
        p[:, ni:] = torch.where(take, encoded.to(p.dtype), p[:, ni:])
        out[store] = p
    return out


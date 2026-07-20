"""Reconstruct a store's terminal currents as the nodal-KCL slack of the rest.

Attribution (scripts/attrib_current.py) showed the unseen line-series error is
driven by inaccurate reactor currents, and reactors are neutral-connection
elements whose terminal current is fixed by KCL at their (few-neighbour) nodes:
at reactor node n, I_reactor(n) = -(sum of every other completed terminal
current at n). Truth nodal KCL residual is ~1e-6 at every node including the
"ground"-flagged neutral nodes, so this is exact when the other currents are
exact and inherits their (small, shunt-level) error otherwise. This is a local
structural decoder over the incidence graph, not a power-flow solve: it never
uses Y or V, only already-decoded currents.

Apply AFTER the shunt (hybrid), line (tree) and source decodes so the currents
being summed are their most accurate values.
"""
from __future__ import annotations

import torch

from .legacy import FC, SPECS, i_offset, physics

EPS = physics.EPS


def kcl_decode_store(batch, preds, clamp: float, target: str):
    """Set masked terminal currents of `target` to the nodal-KCL slack.

    For each node, sum the completed terminal currents of every store except the
    target. Each target terminal at that node then takes an equal share of the
    negative of that sum (equal share only matters when several identical target
    terminals meet at one node, which is rare). Both real and imaginary parts.
    """
    st_t = batch[target]
    if st_t.num_nodes == 0:
        return preds
    dev = preds["node"].device
    x_bar, _, _ = physics.completed(batch, preds)
    n_node = batch["node"].num_nodes
    other_r = torch.zeros(n_node, dtype=torch.float64, device=dev)
    other_i = torch.zeros_like(other_r)

    for store in SPECS:
        if store == target or batch[store].num_nodes == 0:
            continue
        st = batch[store]
        ni = i_offset(store)
        ibus = physics.decoded_physical_currents(batch, x_bar, store, clamp)
        es = batch[(store, "conn", "node")]
        comp, node, slot = es.edge_index[0], es.edge_index[1], es.slot
        col_r = (slot // FC) * 2 * FC + slot % FC
        other_r.index_add_(0, node, ibus[comp, col_r])
        other_i.index_add_(0, node, ibus[comp, col_r + FC])

    es = batch[(target, "conn", "node")]
    comp, node, slot = es.edge_index[0], es.edge_index[1], es.slot
    target_count = torch.zeros(n_node, dtype=torch.float64, device=dev)
    target_count.index_add_(0, node, torch.ones_like(node, dtype=torch.float64))
    col_r = (slot // FC) * 2 * FC + slot % FC

    ni = i_offset(target)
    scale = st_t.scale[:, ni:].double()
    p = preds[target].clone()
    share = 1.0 / target_count[node].clamp_min(1)
    val_r = -other_r[node] * share
    val_i = -other_i[node] * share
    physical = physics.decoded_physical_currents(batch, x_bar, target, clamp)
    physical[comp, col_r] = val_r
    physical[comp, col_r + FC] = val_i
    terminal_feature = physics.physical_to_terminal_feature(
        batch, x_bar, target, physical, clamp
    )
    enc_r = torch.asinh(terminal_feature[comp, col_r] / (scale[comp, col_r] + EPS))
    enc_i = torch.asinh(
        terminal_feature[comp, col_r + FC] / (scale[comp, col_r + FC] + EPS)
    )
    take_r = st_t.msk[comp, ni + col_r]
    take_i = st_t.msk[comp, ni + col_r + FC]
    p[comp, ni + col_r] = torch.where(take_r, enc_r.to(p.dtype), p[comp, ni + col_r])
    p[comp, ni + col_r + FC] = torch.where(
        take_i, enc_i.to(p.dtype), p[comp, ni + col_r + FC])
    return {**preds, target: p}

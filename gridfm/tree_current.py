"""Radial series-current reconstruction from KCL; no voltage/PF solve.

Attribution (scripts/attrib_current.py, unseen PF) showed the line-series WAPE is
dominated not by voltage or shunt error but by the stiff REACTOR direct-head
current, which the original line-only sweep treated as a fixed nodal injection.
Reactors are paired series elements (truth |I1+I2|/|I| = 0.14%, scripts/
series_structure.py), so they belong inside the same KCL forest as lines.
Folding them in drops unseen line WAPE 7.7% -> 0.2% at truth-reactor and removes
the pollution at model reactor accuracy as well.

`decode_tree_series_currents` reconstructs the paired series current on every
edge of the combined line+reactor forest by subtree KCL. Every non-series
terminal current (loads, transformer, vsource, and each series element's own
shunt/common mode) is a known injection. Transformers stay injections: their
winding connection makes conductor currents non-paired (36.9% residual), so they
are handled elsewhere. One root residual per component remains for its
source/transformer boundary. All current operations are differentiable.
"""
from __future__ import annotations

from collections import defaultdict

import torch

from .legacy import FC, SPECS, i_offset, physics, y_width

PAIRED_SERIES = ("line", "reactor")


def decode_tree_series_currents(
    batch, preds, clamp: float, *, series_stores=PAIRED_SERIES,
    physics_shunt: bool = False,
):
    """Reconstruct paired series currents over a combined line+reactor forest."""
    series_stores = tuple(s for s in series_stores if batch[s].num_nodes > 0)
    if not series_stores:
        return preds
    xbar, vr, vi = physics.completed(batch, preds)
    dev = preds["node"].device
    n_node = batch["node"].num_nodes
    q = torch.zeros((n_node, 2), dtype=torch.float64, device=dev)
    boundary: set[int] = set()

    # Everything except the paired-series families is a known injection.
    for store in SPECS:
        if store in series_stores or batch[store].num_nodes == 0:
            continue
        st = batch[store]
        ni = i_offset(store)
        cur = physics.decoded_physical_currents(batch, xbar, store, clamp)
        es = batch[(store, "conn", "node")]
        comp, node, slot = es.edge_index[0], es.edge_index[1], es.slot
        col_r = (slot // FC) * 2 * FC + slot % FC
        vals = torch.stack((cur[comp, col_r], cur[comp, col_r + FC]), dim=1)
        q = q.index_add(0, node, vals)
        if store in {"transformer", "vsource"}:
            boundary.update(int(v) for v in node.detach().cpu().tolist())

    # Decode each series store's currents and lay out one graph edge per active
    # conductor. Terminal-1 slot s pairs with terminal-2 slot FC+s.
    store_cur = {s: physics.decode_completed(
        xbar[s][:, i_offset(s):].double(), batch[s].scale[:, i_offset(s):].double(),
        batch[s].msk[:, i_offset(s):], clamp) for s in series_stores}

    edges: list[tuple[int, int, int, int, int]] = []  # n1, n2, sidx, row, phase
    adjacency: dict[int, list[int]] = defaultdict(list)
    for sidx, store in enumerate(series_stores):
        st = batch[store]
        es = batch[(store, "conn", "node")]
        comp_cpu, node_cpu, slot_cpu = (
            v.detach().cpu() for v in (es.edge_index[0], es.edge_index[1], es.slot)
        )
        slot_node = torch.full((st.num_nodes, 2 * FC), -1, dtype=torch.long)
        slot_node[comp_cpu, slot_cpu] = node_cpu
        for row in range(st.num_nodes):
            for phase in range(FC):
                n1, n2 = int(slot_node[row, phase]), int(slot_node[row, FC + phase])
                if n1 < 0 or n2 < 0:
                    continue
                idx = len(edges)
                edges.append((n1, n2, sidx, row, phase))
                adjacency[n1].append(idx)
                adjacency[n2].append(idx)
    if not edges:
        return preds

    edge_n1 = torch.tensor([e[0] for e in edges], device=dev)
    edge_n2 = torch.tensor([e[1] for e in edges], device=dev)
    edge_sidx = torch.tensor([e[2] for e in edges], device=dev)
    edge_row = torch.tensor([e[3] for e in edges], device=dev)
    edge_phase = torch.tensor([e[4] for e in edges], device=dev)

    # Per-edge terminal currents gathered from the owning store.
    i1 = torch.zeros(len(edges), 2, dtype=torch.float64, device=dev)
    i2 = torch.zeros(len(edges), 2, dtype=torch.float64, device=dev)
    for sidx, store in enumerate(series_stores):
        m = edge_sidx == sidx
        if not m.any():
            continue
        cur = store_cur[store]
        row, phase = edge_row[m], edge_phase[m]
        c1, c2 = phase, 2 * FC + phase
        i1[m] = torch.stack((cur[row, c1], cur[row, c1 + FC]), dim=1)
        i2[m] = torch.stack((cur[row, c2], cur[row, c2 + FC]), dim=1)
    shunt = 0.5 * (i1 + i2)

    if physics_shunt and "line" in series_stores:
        # The paired common mode cancels the stiff series term exactly:
        # 0.5*(I1+I2) = 0.5*jYh*(V1+V2); absolute voltage, well conditioned.
        m = (edge_sidx == series_stores.index("line"))
        st = batch["line"]
        ny = y_width("line")
        y_pu = physics.decode_completed(
            xbar["line"][:, :ny].double(), st.scale[:, :ny].double(),
            st.msk[:, :ny], clamp,
        )
        tri = physics.tri_size(FC)
        yh = physics._tri_to_full(y_pu[:, 2 * tri:], FC)
        slot_vr, slot_vi = physics._slot_voltages(batch, "line", vr, vi)
        vsum = torch.complex(slot_vr[:, :FC] + slot_vr[:, FC:],
                             slot_vi[:, :FC] + slot_vi[:, FC:])
        shunt_all = 0.5j * torch.einsum("kij,kj->ki", yh.to(vsum.dtype), vsum)
        lrow, lphase = edge_row[m], edge_phase[m]
        shunt[m] = torch.stack(
            (shunt_all[lrow, lphase].real, shunt_all[lrow, lphase].imag), dim=1)

    q = q.index_add(0, edge_n1, shunt)
    q = q.index_add(0, edge_n2, shunt)

    # Forest discovery over the combined series graph.
    unseen_nodes = set(adjacency)
    slack = batch["node"].slack.detach().cpu()
    tree_children: list[int] = []
    tree_parents: list[int] = []
    tree_edge_ids: list[int] = []
    tree_depths: list[int] = []
    cycle_edge_ids: list[int] = []
    while unseen_nodes:
        component_seed = next(iter(unseen_nodes))
        stack, component_nodes, component_edges = [component_seed], set(), set()
        while stack:
            u = stack.pop()
            if u in component_nodes:
                continue
            component_nodes.add(u)
            for ei in adjacency[u]:
                component_edges.add(ei)
                n1, n2, *_ = edges[ei]
                stack.append(n2 if u == n1 else n1)
        unseen_nodes.difference_update(component_nodes)
        roots = [n for n in component_nodes if bool(slack[n])]
        if not roots:
            roots = [n for n in component_nodes if n in boundary]
        root = min(roots or component_nodes)

        parent = {root: -1}
        depth = {root: 0}
        parent_edge: dict[int, int] = {}
        order, stack, tree_edges = [], [root], set()
        while stack:
            u = stack.pop()
            order.append(u)
            for ei in adjacency[u]:
                n1, n2, *_ = edges[ei]
                v = n2 if u == n1 else n1
                if v in parent:
                    continue
                parent[v] = u
                depth[v] = depth[u] + 1
                parent_edge[v] = ei
                tree_edges.add(ei)
                stack.append(v)
        cycle_edge_ids.extend(component_edges - tree_edges)
        for child in order[1:]:
            tree_children.append(child)
            tree_parents.append(parent[child])
            tree_edge_ids.append(parent_edge[child])
            tree_depths.append(depth[child])

    # Preserve learned series flow on rare cycle chords.
    if cycle_edge_ids:
        cyc = torch.tensor(cycle_edge_ids, device=dev)
        flow = 0.5 * (i1[cyc] - i2[cyc])
        q = q.index_add(0, edge_n1[cyc], flow)
        q = q.index_add(0, edge_n2[cyc], -flow)

    out_cur = {s: store_cur[s].clone() for s in series_stores}
    if tree_children:
        children = torch.tensor(tree_children, device=dev)
        parents = torch.tensor(tree_parents, device=dev)
        tree_ei = torch.tensor(tree_edge_ids, device=dev)
        depths = torch.tensor(tree_depths, device=dev)
        subtree = q
        for level in range(int(depths.max().item()), 0, -1):
            take = depths == level
            child, ei = children[take], tree_ei[take]
            parent = parents[take]
            sign = torch.where(child == edge_n1[ei], 1.0, -1.0).unsqueeze(1)
            flow = -subtree[child] / sign
            edge_shunt = shunt[ei]
            sidx, row, phase = edge_sidx[ei], edge_row[ei], edge_phase[ei]
            col1, col2 = phase, 2 * FC + phase
            for si, store in enumerate(series_stores):
                sm = sidx == si
                if not sm.any():
                    continue
                r, p = row[sm], phase[sm]
                c1, c2 = p, 2 * FC + p
                f, sh = flow[sm], edge_shunt[sm]
                oc = out_cur[store]
                oc[r, c1] = f[:, 0] + sh[:, 0]
                oc[r, c1 + FC] = f[:, 1] + sh[:, 1]
                oc[r, c2] = -f[:, 0] + sh[:, 0]
                oc[r, c2 + FC] = -f[:, 1] + sh[:, 1]
            delta = torch.zeros_like(subtree)
            delta.index_add_(0, parent, subtree[child])
            subtree = subtree + delta

    out = dict(preds)
    for store in series_stores:
        st = batch[store]
        ni = i_offset(store)
        encoded = torch.asinh(out_cur[store] / (st.scale[:, ni:].double() + 1e-12))
        p = out[store].clone()
        take = st.msk[:, ni:]
        p[:, ni:] = torch.where(take, encoded.to(p.dtype), p[:, ni:])
        out[store] = p
    return out


def decode_tree_line_currents(batch, preds, clamp: float, *, physics_shunt: bool = False):
    """Backward-compatible line-only sweep (see decode_tree_series_currents)."""
    return decode_tree_series_currents(
        batch, preds, clamp, series_stores=("line",), physics_shunt=physics_shunt)

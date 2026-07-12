"""Radial line-series current reconstruction from KCL; no voltage/PF solve."""
from __future__ import annotations

from collections import defaultdict

import torch

from .legacy import FC, SPECS, i_offset, physics


def decode_tree_line_currents(batch, preds, clamp: float):
    """Reconstruct line series currents by subtree current accumulation.

    Learned non-line terminal currents and each line's learned shunt/common-mode
    current are preserved. For every conductor-wise line forest, KCL determines
    the remaining paired series current on each tree edge. One root residual per
    connected component remains for its source/transformer boundary.
    """
    if batch["line"].num_nodes == 0:
        return preds
    xbar, _, _ = physics.completed(batch, preds)
    dev = preds["node"].device
    n_node = batch["node"].num_nodes
    q = torch.zeros(n_node, dtype=torch.complex128)
    boundary: set[int] = set()

    # Everything except lines is a known injection into each line forest.
    for store in SPECS:
        if store == "line" or batch[store].num_nodes == 0:
            continue
        st = batch[store]
        ni = i_offset(store)
        cur = physics.decode_completed(
            xbar[store][:, ni:].double(), st.scale[:, ni:].double(),
            st.msk[:, ni:], clamp,
        ).cpu()
        es = batch[(store, "conn", "node")]
        comp, node, slot = (v.cpu() for v in (es.edge_index[0], es.edge_index[1], es.slot))
        col_r = (slot // FC) * 2 * FC + slot % FC
        vals = torch.complex(cur[comp, col_r], cur[comp, col_r + FC])
        q.index_add_(0, node, vals)
        if store in {"transformer", "vsource"}:
            boundary.update(int(v) for v in node.tolist())

    st = batch["line"]
    ni = i_offset("line")
    cur = physics.decode_completed(
        xbar["line"][:, ni:].double(), st.scale[:, ni:].double(),
        st.msk[:, ni:], clamp,
    ).cpu()
    es = batch[("line", "conn", "node")]
    comp, node, slot = (v.cpu() for v in (es.edge_index[0], es.edge_index[1], es.slot))
    slot_node = torch.full((st.num_nodes, 2 * FC), -1, dtype=torch.long)
    slot_node[comp, slot] = node

    # One graph edge per active conductor. Preserve common mode as a shunt
    # estimate; the paired differential mode is the series current to solve.
    edges: list[tuple[int, int, int, int, complex]] = []
    adjacency: dict[int, list[int]] = defaultdict(list)
    for row in range(st.num_nodes):
        for phase in range(FC):
            n1, n2 = int(slot_node[row, phase]), int(slot_node[row, FC + phase])
            if n1 < 0 or n2 < 0:
                continue
            c1, c2 = phase, 2 * FC + phase
            i1 = complex(float(cur[row, c1]), float(cur[row, c1 + FC]))
            i2 = complex(float(cur[row, c2]), float(cur[row, c2 + FC]))
            shunt = 0.5 * (i1 + i2)
            idx = len(edges)
            edges.append((n1, n2, row, phase, shunt))
            adjacency[n1].append(idx)
            adjacency[n2].append(idx)
            q[n1] += shunt
            q[n2] += shunt

    unseen_nodes = set(adjacency)
    slack = batch["node"].slack.cpu()
    while unseen_nodes:
        component_seed = next(iter(unseen_nodes))
        # Discover the connected line component before choosing its upstream root.
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
                parent_edge[v] = ei
                tree_edges.add(ei)
                stack.append(v)

        # Preserve learned series flow on rare cycle chords, then solve the
        # spanning tree so every non-root node satisfies KCL exactly.
        for ei in component_edges - tree_edges:
            n1, n2, row, phase, _ = edges[ei]
            c1, c2 = phase, 2 * FC + phase
            i1 = complex(float(cur[row, c1]), float(cur[row, c1 + FC]))
            i2 = complex(float(cur[row, c2]), float(cur[row, c2 + FC]))
            flow = 0.5 * (i1 - i2)
            q[n1] += flow
            q[n2] -= flow

        subtree = {n: complex(q[n]) for n in component_nodes}
        for child in reversed(order[1:]):
            ei = parent_edge[child]
            n1, n2, row, phase, shunt = edges[ei]
            sign_child = 1.0 if child == n1 else -1.0
            flow = -subtree[child] / sign_child
            c1, c2 = phase, 2 * FC + phase
            cur[row, c1], cur[row, c1 + FC] = flow.real + shunt.real, flow.imag + shunt.imag
            cur[row, c2], cur[row, c2 + FC] = -flow.real + shunt.real, -flow.imag + shunt.imag
            subtree[parent[child]] += subtree[child]

    encoded = torch.asinh(cur.to(dev) / (st.scale[:, ni:].double() + 1e-12))
    out = dict(preds)
    p = out["line"].clone()
    take = st.msk[:, ni:]
    p[:, ni:] = torch.where(take, encoded.to(p.dtype), p[:, ni:])
    out["line"] = p
    return out


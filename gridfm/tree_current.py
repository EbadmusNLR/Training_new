"""Radial line-series current reconstruction from KCL; no voltage/PF solve."""
from __future__ import annotations

from collections import defaultdict

import torch

from .legacy import FC, SPECS, i_offset, physics


def decode_tree_line_currents(
    batch, preds, clamp: float, *, line_current_unclamped: bool = False
):
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
    q = torch.zeros((n_node, 2), dtype=torch.float64, device=dev)
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
        )
        es = batch[(store, "conn", "node")]
        comp, node, slot = es.edge_index[0], es.edge_index[1], es.slot
        col_r = (slot // FC) * 2 * FC + slot % FC
        vals = torch.stack((cur[comp, col_r], cur[comp, col_r + FC]), dim=1)
        q = q.index_add(0, node, vals)
        if store in {"transformer", "vsource"}:
            boundary.update(int(v) for v in node.detach().cpu().tolist())

    st = batch["line"]
    ni = i_offset("line")
    if line_current_unclamped:
        cur = physics.decode_truth(
            xbar["line"][:, ni:].double(), st.scale[:, ni:].double()
        )
    else:
        cur = physics.decode_completed(
            xbar["line"][:, ni:].double(), st.scale[:, ni:].double(),
            st.msk[:, ni:], clamp,
        )
    es = batch[("line", "conn", "node")]
    comp_cpu, node_cpu, slot_cpu = (
        v.detach().cpu() for v in (es.edge_index[0], es.edge_index[1], es.slot)
    )
    slot_node = torch.full((st.num_nodes, 2 * FC), -1, dtype=torch.long)
    slot_node[comp_cpu, slot_cpu] = node_cpu

    # One graph edge per active conductor. Preserve common mode as a shunt
    # estimate; the paired differential mode is the series current to solve.
    edges: list[tuple[int, int, int, int]] = []
    adjacency: dict[int, list[int]] = defaultdict(list)
    for row in range(st.num_nodes):
        for phase in range(FC):
            n1, n2 = int(slot_node[row, phase]), int(slot_node[row, FC + phase])
            if n1 < 0 or n2 < 0:
                continue
            idx = len(edges)
            edges.append((n1, n2, row, phase))
            adjacency[n1].append(idx)
            adjacency[n2].append(idx)

    if not edges:
        return preds
    edge_n1 = torch.tensor([e[0] for e in edges], device=dev)
    edge_n2 = torch.tensor([e[1] for e in edges], device=dev)
    edge_row = torch.tensor([e[2] for e in edges], device=dev)
    edge_phase = torch.tensor([e[3] for e in edges], device=dev)
    c1 = edge_phase
    c2 = 2 * FC + edge_phase
    i1 = torch.stack((cur[edge_row, c1], cur[edge_row, c1 + FC]), dim=1)
    i2 = torch.stack((cur[edge_row, c2], cur[edge_row, c2 + FC]), dim=1)
    shunt = 0.5 * (i1 + i2)
    q = q.index_add(0, edge_n1, shunt)
    q = q.index_add(0, edge_n2, shunt)

    unseen_nodes = set(adjacency)
    slack = batch["node"].slack.detach().cpu()
    tree_children: list[int] = []
    tree_parents: list[int] = []
    tree_edge_ids: list[int] = []
    tree_depths: list[int] = []
    cycle_edge_ids: list[int] = []
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

    # Preserve learned series flow on rare cycle chords. All value operations
    # remain tensors, so gradients reach the current heads.
    if cycle_edge_ids:
        cyc = torch.tensor(cycle_edge_ids, device=dev)
        flow = 0.5 * (i1[cyc] - i2[cyc])
        q = q.index_add(0, edge_n1[cyc], flow)
        q = q.index_add(0, edge_n2[cyc], -flow)

    # A depth-vectorized reverse tree sweep accumulates each subtree. Topology
    # discovery is discrete, but every current operation is differentiable.
    out_cur = cur.clone()
    if tree_children:
        children = torch.tensor(tree_children, device=dev)
        parents = torch.tensor(tree_parents, device=dev)
        tree_ei = torch.tensor(tree_edge_ids, device=dev)
        depths = torch.tensor(tree_depths, device=dev)
        subtree = q
        for level in range(int(depths.max().item()), 0, -1):
            take = depths == level
            child, parent, ei = children[take], parents[take], tree_ei[take]
            sign = torch.where(child == edge_n1[ei], 1.0, -1.0).unsqueeze(1)
            flow = -subtree[child] / sign
            row, phase = edge_row[ei], edge_phase[ei]
            col1, col2 = phase, 2 * FC + phase
            edge_shunt = shunt[ei]
            out_cur[row, col1] = flow[:, 0] + edge_shunt[:, 0]
            out_cur[row, col1 + FC] = flow[:, 1] + edge_shunt[:, 1]
            out_cur[row, col2] = -flow[:, 0] + edge_shunt[:, 0]
            out_cur[row, col2 + FC] = -flow[:, 1] + edge_shunt[:, 1]
            delta = torch.zeros_like(subtree)
            delta.index_add_(0, parent, subtree[child])
            subtree = subtree + delta

    encoded = torch.asinh(out_cur / (st.scale[:, ni:].double() + 1e-12))
    out = dict(preds)
    p = out["line"].clone()
    take = st.msk[:, ni:]
    p[:, ni:] = torch.where(take, encoded.to(p.dtype), p[:, ni:])
    out["line"] = p
    return out

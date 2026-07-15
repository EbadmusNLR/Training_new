#!/usr/bin/env python3
"""Radial series-current reconstruction by subtree KCL (datakit full-matrix).

Line series current is `Ys*(V1-V2)` with V1~V2 to ~7 digits and Ys up to ~1e6, so
it is UNRECOVERABLE from bus voltages (fp32 or even the model's ~4% predicted V —
see test_physics.py / [[dgfm-line-current-unrecoverable-from-V]]). It IS, however,
exactly determined by KCL: on a radial feeder the series flow through any branch
equals the sum of all shunt injections in the subtree below it. That is well
conditioned and solver-free.

For each line conductor k we split the two terminal currents into
    shunt  s = 0.5 (I_bus1[k] + I_bus2[k])   (charging/common-mode, a known nodal injection)
    series f = 0.5 (I_bus1[k] - I_bus2[k])   (through-flow, reconstructed by subtree KCL)
and rebuild I_bus1[k] = f + s, I_bus2[k] = -f + s. Every non-line terminal current
(loads, caps, pv, storage, transformer, vsource) is a known nodal injection.

`reconstruct_series` takes a per-store current dict (stored truth for testing, or
model-predicted shunts for inference) and returns the reconstructed line currents.
All ops are differentiable (index_add / gather), so it trains end-to-end.
"""
from __future__ import annotations

import os
from collections import defaultdict, deque

import torch

from .dk_physics import FC, STORES, node_count, store_size, terminal_slot

SERIES = "line"
# series (through-flow) elements vs shunt (nodal-injection) elements
SERIES_STORES = ("line", "transformer", "vsource", "reactor")
SHUNT_STORES = ("load", "generator", "pvsystem", "storage", "capacitor")
# SHUNT vs SERIES is a property of each ELEMENT's connectivity, not of its store.
# MEASURED (minimal_component, 24k reactors): reactors are BOTH -- ~21/24 have
# bus2=ground (shunt) and ~3/24 have both terminals live (series). Capacitors are
# the same. So route per element:
#   a terminal grounded  -> SHUNT : physics-decode I=Y@V-Icomp (well-conditioned,
#                                   verified 4.2e-16 for reactors, 1.5e-16 for gens)
#   both terminals live  -> SERIES: through-flow from the subtree/mesh sweep, plus
#                                   the well-conditioned common-mode s=0.5(I1+I2)
# (transformer -> YPrim null-space map; vsource -> nodal KCL at the slack.)
AMBIG_STORES = ("capacitor", "reactor")          # per-element shunt-or-series
TREE_STORES = ("line", "reactor", "capacitor")   # ground-touching elems auto-excluded


def classify_series(data, store):
    """comps of a 2-terminal store that are SERIES (both terminals on live nodes).
    Everything else in the store is shunt-connected (a terminal is ground)."""
    if store not in data.node_types or store_size(data, store) == 0:
        return set()
    m1 = _slot_node_map(data, store, 1)
    m2 = _slot_node_map(data, store, 2)
    ser = set()
    for (c, sl), n1 in m1.items():
        n2 = m2.get((c, sl))
        if n2 is not None and n1 != 0 and n2 != 0:
            ser.add(c)
    return ser
# store -> int id for the tree's per-edge `sid`. Must cover EVERY store that can be
# a tree edge: the always-series ones AND the ambiguous ones (capacitor/reactor),
# whose both-terminals-live elements are series.
_SID_STORES = tuple(dict.fromkeys(tuple(SERIES_STORES) + tuple(AMBIG_STORES)))
_SID = {s: i for i, s in enumerate(_SID_STORES)}
_SID_INV = {i: s for s, i in _SID.items()}


# ---------------------------------------------------------------------------
# Vectorized, precomputable form: build the tree STRUCTURE once per feeder
# (topology only), then a differentiable level-by-level subtree sum reconstructs
# the series through-flows from the nodal injections in the model forward.
# ---------------------------------------------------------------------------

def _series_edges(data, stores):
    """All same-slot paired series conductors: (store, comp, n1, n2, cola, colb).
    n1/cola = reference terminal side; n2/colb = the other active terminal."""
    E = []
    for s in stores:
        if s not in data.node_types or s not in STORES:
            continue
        _, nterm, _ = STORES[s]
        term_maps = {t: _slot_node_map(data, s, t) for t in range(1, nterm + 1)}
        comps = {c for t in term_maps for (c, _s) in term_maps[t]}
        for c in comps:
            acts = [t for t in range(1, nterm + 1) if any(cc == c for (cc, _s) in term_maps[t])]
            if len(acts) != 2:
                continue
            ta, tb = acts
            for (cc, sl), n1 in term_maps[ta].items():
                if cc != c:
                    continue
                n2 = term_maps[tb].get((c, sl))
                if n2 is None:
                    continue
                E.append((s, c, n1, n2, (ta - 1) * FC + sl, (tb - 1) * FC + sl))
    return E


def _tree_from_edges(E, slack_set):
    """BFS forest (ground node 0 excluded). Returns per tree-edge arrays that
    drive the vectorized sweep: child/parent node, sign, level, and the target
    (store id, comp, cola, colb) to write the reconstructed current into."""
    # GROUND (node 0) is a root, and ground-touching edges are kept. A 4-wire line's
    # NEUTRAL is grounded at one end and carries real current; excluding every
    # ground-touching edge left it silently 0. Callers must pass only SERIES-element
    # conductors here -- a shunt capacitor/reactor leg also touches ground, and it is
    # physics-decoded exactly, so it must NOT become a tree edge.
    adj = defaultdict(list)
    for i, (s, c, n1, n2, ca, cb) in enumerate(E):
        adj[n1].append(i); adj[n2].append(i)
    seen = set(); parent_edge = {}; depth = {}; comp_of = {}
    # PRE-MARK every root. A root is an injection point (ground / slack / transformer
    # secondary) whose KCL is needed to determine what feeds it; giving it a parent
    # edge makes that KCL carry TWO unknowns and determines neither. Marking roots
    # lazily let BFS from one root adopt another as a CHILD, handing the connecting
    # edge the whole subtree's current (37Bus: the open-delta regulator's jumper
    # took the feeder current of the line next to it).
    roots = sorted(set(slack_set) | {0})
    for r in roots:
        seen.add(r); depth[r] = 0; comp_of[r] = r

    def _bfs(root):
        dq = deque([root])
        while dq:
            u = dq.popleft()
            for ei in adj[u]:
                s, c, n1, n2, ca, cb = E[ei]
                v = n2 if u == n1 else n1
                if v in seen:
                    continue
                seen.add(v); parent_edge[v] = ei; depth[v] = depth[u] + 1
                comp_of[v] = comp_of[u]; dq.append(v)

    for r in roots:
        _bfs(r)
    for r in sorted(adj.keys()):        # islands with no root of their own
        if r in seen:
            continue
        seen.add(r); depth[r] = 0; comp_of[r] = r; _bfs(r)
    child, parent, sign, level, sid, comp, cola, colb = ([] for _ in range(8))
    for v, ei in parent_edge.items():
        s, c, n1, n2, ca, cb = E[ei]
        child.append(v); parent.append(n1 if v == n2 else n2)
        sign.append(1.0 if v == n1 else -1.0); level.append(depth[v])
        sid.append(_SID[s]); comp.append(c); cola.append(ca); colb.append(cb)
    L = lambda x, dt: torch.tensor(x, dtype=dt)
    # Live edges that are not tree edges split into two DIFFERENT problems:
    #   CHORD  - both ends in the SAME rooted tree => a real independent loop. The
    #            current split is set by loop impedance (KVL) -> mesh_correct.
    #   BRIDGE - ends in DIFFERENT rooted trees => NOT a loop. It carries current
    #            BETWEEN two injection points, so no subtree sum can see it and KCL
    #            at either root has two unknowns. Determined by the transformers'
    #            own constitutive rows -> handed to the joint transformer system.
    # Conflating them is a bug either way: a bridge has no tree path, so mesh
    # analysis cannot even build its loop.
    tree_eids = set(parent_edge.values())
    chords, bridges = [], []
    for i, (s, c, n1, n2, ca, cb) in enumerate(E):
        if i in tree_eids or (n1 == 0 and n2 == 0):
            continue
        (chords if comp_of.get(n1) == comp_of.get(n2) else bridges).append(i)

    # ---- MESH tree: a spanning forest of the WHOLE graph, ROOTS IGNORED --------
    # The rooted forest above exists for the KCL SWEEP: its roots are injection
    # points whose KCL we need, so they must not get a parent edge. That makes it a
    # BAD basis for the CYCLE SPACE -- an edge between two rooted trees (a BRIDGE)
    # has no tree path, so its loop cannot be built at all. The two jobs want
    # different trees, and nothing says they must be the same one.
    #
    # Cycle space needs one spanning tree per CONNECTED component. Any non-tree edge
    # of THIS tree is then a chord with a real path, so every independent loop --
    # including ones that close through several bridges leaving one root (IEEE 30
    # Bus: 4 lines from node 49 into component 25 = 3 loops per phase = the 9
    # undetermined DOF) -- becomes visible to mesh_correct.
    #
    # Loop currents are invisible to EVERY KCL-derived equation (a circulating +d/-d
    # changes no nodal sum and no cut-set), so KVL+Z is the only thing that can fix
    # them. Measured: 18 cut-set rows added 0 rank.
    mseen, mparent_edge, mdepth, mparent_node = set(), {}, {}, {}
    for r in sorted(adj.keys()):
        if r in mseen:
            continue
        mseen.add(r); mdepth[r] = 0
        dq = deque([r])
        while dq:
            u = dq.popleft()
            for ei in adj[u]:
                _s, _c, n1, n2, _ca, _cb = E[ei]
                v = n2 if u == n1 else n1
                if v in mseen:
                    continue
                mseen.add(v); mparent_edge[v] = ei; mparent_node[v] = u
                mdepth[v] = mdepth[u] + 1
                dq.append(v)
    mtree_eids = set(mparent_edge.values())
    mchords = [i for i, (s, c, n1, n2, ca, cb) in enumerate(E)
               if i not in mtree_eids and not (n1 == 0 and n2 == 0)]
    parent_node = {v: (E[ei][3] if v == E[ei][2] else E[ei][2]) for v, ei in parent_edge.items()}
    return {
        "child": L(child, torch.long), "parent": L(parent, torch.long),
        "sign": L(sign, torch.float64), "level": L(level, torch.long),
        "sid": L(sid, torch.long), "comp": L(comp, torch.long),
        "cola": L(cola, torch.long), "colb": L(colb, torch.long),
        # loop/mesh support (python-side; topology-only)
        "chords": chords, "bridges": bridges, "comp_of": comp_of,
        "parent_edge": parent_edge, "parent_node": parent_node,
        "depth": depth,
        # cycle-space basis (separate tree; see above)
        "mchords": mchords, "mparent_edge": mparent_edge,
        "mparent_node": mparent_node, "mdepth": mdepth,
    }


def plan_to(plan, device):
    def mv(t):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
    return {"unified": mv(plan["unified"]), "line": mv(plan["line"]),
            "inj": {s: tuple(x.to(device) for x in plan["inj"][s]) for s in plan["inj"]},
            "n_node": plan["n_node"]}


def batch_plans(plans, node_counts, store_counts):
    """Merge per-feeder plans into one batched plan with global node/comp offsets
    (matching PyG's concatenation order), so the vectorized reconstruction runs
    once over the whole batch. store_counts[i] = {series_store: comp_count}."""
    all_stores = tuple(SHUNT_STORES) + tuple(SERIES_STORES)
    node_off = [0]
    for nc in node_counts:
        node_off.append(node_off[-1] + int(nc))
    soff = {s: [0] for s in all_stores}
    for sc in store_counts:
        for s in all_stores:
            soff[s].append(soff[s][-1] + int(sc.get(s, 0)))

    def merge(key):
        acc = {k: [] for k in ("child", "parent", "sign", "level", "sid", "comp", "cola", "colb")}
        for i, p in enumerate(plans):
            t = p[key]
            if t["child"].numel() == 0:
                continue
            acc["child"].append(t["child"] + node_off[i])
            acc["parent"].append(t["parent"] + node_off[i])
            acc["sign"].append(t["sign"]); acc["level"].append(t["level"])
            acc["sid"].append(t["sid"])
            off_vec = torch.tensor([soff[_SID_INV[k]][i] for k in range(len(_SID))], dtype=torch.long)
            acc["comp"].append(t["comp"] + off_vec[t["sid"]])
            acc["cola"].append(t["cola"]); acc["colb"].append(t["colb"])
        dt = {"sign": torch.float64}
        return {k: (torch.cat(v) if v else torch.zeros(0, dtype=dt.get(k, torch.long)))
                for k, v in acc.items()}

    inj = {}
    for s in all_stores:
        comps, cols, nodes = [], [], []
        for i, p in enumerate(plans):
            if s in p["inj"]:
                comp, col, node = p["inj"][s]
                comps.append(comp + soff[s][i]); cols.append(col); nodes.append(node + node_off[i])
        if comps:
            inj[s] = (torch.cat(comps), torch.cat(cols), torch.cat(nodes))
    return {"unified": merge("unified"), "line": merge("line"), "inj": inj, "n_node": node_off[-1]}


def _inj_index(data, store):
    """(comp, col, node) for every active terminal slot of a store, so a current
    tensor I[n,dim] can be scattered to nodal injections q[node] += I[comp,col]."""
    _, nterm, _ = STORES[store]
    comps, cols, nodes = [], [], []
    for t in range(1, nterm + 1):
        rel = (store, f"bus{t}", "node")
        if rel not in data.edge_types or data[rel].edge_index.numel() == 0:
            continue
        ei = data[rel].edge_index
        comp, node = ei[0], ei[1]
        col = (t - 1) * FC + terminal_slot(comp)
        comps.append(comp); cols.append(col); nodes.append(node)
    if comps:
        return torch.cat(comps), torch.cat(cols), torch.cat(nodes)
    z = torch.zeros(0, dtype=torch.long)
    return z, z, z


def build_tree_plan(data):
    """Precompute (topology-only) everything the vectorized reconstruction needs
    for one feeder: the unified tree (line+transformer+vsource -> exact xfmr/
    vsource) and the line-only tree (-> exact lines given the reconstructed
    xfmr/vsource injections), plus per-store injection scatter indices."""
    slack = data["node"].slack.tolist() if hasattr(data["node"], "slack") else []
    slack_set = {i for i, v in enumerate(slack) if v}
    inj = {s: _inj_index(data, s) for s in list(SHUNT_STORES) + list(SERIES_STORES)
           if s in data.node_types and store_size(data, s) > 0}
    uni = _tree_from_edges(_series_edges(data, SERIES_STORES), slack_set)
    line = _tree_from_edges(_series_edges(data, (SERIES,)), slack_set)
    n = node_count(data)
    # OPEN series conductors: terminal-slots NOT written by either tree sweep
    # (vsource ground edges, transformer/line cycle-chords). Resolved afterward by
    # nodal-KCL closure. Precompute their (comp,col,node) and per-node open count.
    written = set()
    for tr in (uni, line):
        for sid, c, ca, cb in zip(tr["sid"].tolist(), tr["comp"].tolist(),
                                  tr["cola"].tolist(), tr["colb"].tolist()):
            written.add((sid, c, ca)); written.add((sid, c, cb))
    opens = {}
    node_open = torch.zeros(n, dtype=torch.long)
    for s in SERIES_STORES:
        if s not in inj:
            continue
        comp, col, node = inj[s]
        sid = _SID[s]
        keep = torch.tensor([(sid, int(c), int(cl)) not in written and int(nd) != 0
                             for c, cl, nd in zip(comp.tolist(), col.tolist(), node.tolist())],
                            dtype=torch.bool)
        if keep.any():
            oc, ocol, onode = comp[keep], col[keep], node[keep]
            opens[s] = (oc, ocol, onode)
            node_open.index_add_(0, onode, torch.ones_like(onode))
    return {"unified": uni, "line": line, "inj": inj, "n_node": n,
            "open": opens, "node_open": node_open}


def _subtree_sum(q, tree):
    """Differentiable level-by-level subtree accumulation. q:[N,2]; returns the
    through-flow per tree-edge (= net injection of the child's subtree)."""
    subtree = q
    if tree["level"].numel():
        for Lvl in range(int(tree["level"].max()), 0, -1):
            m = tree["level"] == Lvl
            if m.any():
                subtree = subtree.index_add(0, tree["parent"][m], subtree[tree["child"][m]])
    if tree["child"].numel() == 0:
        return q.new_zeros(0, 2)
    return subtree[tree["child"]] * tree["sign"].unsqueeze(1).to(subtree.dtype)


def _assign(flow, tree, out, only):
    """Write reconstructed through-flow into the series current tensors (common
    mode s=0: I_a = -f, I_b = +f). Vectorized per store."""
    for s in only:
        m = tree["sid"] == _SID[s]
        if not m.any() or s not in out:
            continue
        comp, ca, cb, f = tree["comp"][m], tree["cola"][m], tree["colb"][m], flow[m]
        outr, outi = out[s]
        outr[comp, ca] = -f[:, 0]; outi[comp, ca] = -f[:, 1]
        outr[comp, cb] = f[:, 0];  outi[comp, cb] = f[:, 1]


def reconstruct_vectorized(plan, cur):
    """Differentiable series-current reconstruction using a precomputed plan.
    cur: {store:(Ir,Ii)} with accurate SHUNT currents (physics-decoded from V)
    and zero placeholders for the series stores; returns the same dict with
    line/transformer/vsource filled by their KCL reconstruction. Two vectorized
    subtree sweeps (unified -> xfmr/vsource, line -> lines), batched-friendly &
    differentiable. Runs in cur's dtype/device (fp32 GPU in the model)."""
    ref = cur[next(iter(cur))][0]
    dtype, dev = ref.dtype, ref.device
    n = plan["n_node"]
    out = {s: (cur[s][0].clone(), cur[s][1].clone()) for s in cur}

    def build_q(stores, src):
        q = torch.zeros(n, 2, dtype=dtype, device=dev)
        for s in stores:
            if s not in plan["inj"] or s not in src:
                continue
            comp, col, node = plan["inj"][s]
            Ir, Ii = src[s]
            q = q.index_add(0, node, torch.stack([Ir[comp, col], Ii[comp, col]], 1))
        return q

    q_shunt = build_q(SHUNT_STORES, cur)
    _assign(_subtree_sum(q_shunt, plan["unified"]), plan["unified"], out,
            only=("transformer", "vsource", "reactor"))
    q_line = q_shunt + build_q(("transformer", "vsource", "reactor"), out)
    _assign(_subtree_sum(q_line, plan["line"]), plan["line"], out, only=(SERIES,))
    return _kcl_close(out, plan, dtype, dev)


def _kcl_close(out, plan, dtype, dev, iters=8):
    """Fill the series conductors the tree sweeps left open (vsource root,
    transformer/line cycle-chords) by nodal KCL: each open conductor is (usually)
    the lone unknown at its terminal node, so I = -(residual of everything else
    there). Jacobi iteration handles the few coupled cases. Differentiable."""
    opens = plan.get("open") or {}
    if not opens:
        return out
    n = plan["n_node"]
    cnt = plan["node_open"].to(dev).clamp(min=1).to(dtype).unsqueeze(1)
    gmask = torch.ones(n, 1, dtype=dtype, device=dev); gmask[0] = 0.0   # ground has no KCL
    for _ in range(iters):
        r = torch.zeros(n, 2, dtype=dtype, device=dev)
        for s, (comp, col, node) in plan["inj"].items():
            if s not in out:
                continue
            Ir, Ii = out[s]
            r = r.index_add(0, node, torch.stack([Ir[comp, col], Ii[comp, col]], 1))
        r = r * gmask
        for s, (comp, col, node) in opens.items():
            if s not in out or comp.numel() == 0:
                continue
            share = r[node] / cnt[node]
            Ir, Ii = out[s]
            Ir = Ir.index_put((comp, col), Ir[comp, col] - share[:, 0])
            Ii = Ii.index_put((comp, col), Ii[comp, col] - share[:, 1])
            out[s] = (Ir, Ii)
    return out


def _slot_node_map(data, store, terminal):
    """(comp_row, slot) -> node for one terminal, from its edges."""
    rel = (store, f"bus{terminal}", "node")
    if rel not in data.edge_types or data[rel].edge_index.numel() == 0:
        return {}
    ei = data[rel].edge_index
    comp, node = ei[0], ei[1]
    slot = terminal_slot(comp)
    return {(int(c), int(s)): int(n) for c, s, n in
            zip(comp.tolist(), slot.tolist(), node.tolist())}


def line_conductor_edges(data):
    """One graph edge per line conductor present on BOTH terminals:
    (comp_row, slot, node1, node2)."""
    m1 = _slot_node_map(data, SERIES, 1)
    m2 = _slot_node_map(data, SERIES, 2)
    out = []
    for (c, s), n1 in m1.items():
        n2 = m2.get((c, s))
        if n2 is not None:
            out.append((c, s, n1, n2))
    return out


def _nodal_injection(data, cur, exclude):
    """q[node] (complex, [N,2]) = sum of every terminal current into the node,
    skipping the `exclude` store (its series part is reconstructed, not injected)."""
    n = node_count(data)
    q = torch.zeros(n, 2, dtype=torch.float64)
    for store, (Ir, Ii) in cur.items():
        if store == exclude:
            continue
        _, nterm, _ = STORES[store]
        for t in range(1, nterm + 1):
            rel = (store, f"bus{t}", "node")
            if rel not in data.edge_types or data[rel].edge_index.numel() == 0:
                continue
            ei = data[rel].edge_index
            comp, node = ei[0], ei[1]
            col = (t - 1) * FC + terminal_slot(comp)
            q[:, 0].index_add_(0, node, Ir[comp, col].double())
            q[:, 1].index_add_(0, node, Ii[comp, col].double())
    return q


def reconstruct_series(data, cur):
    """Return reconstructed line currents (Ir, Ii) [n_line, 2*FC] via subtree KCL."""
    st = data[SERIES]
    n_line = st["Ys_r_pu"].shape[0] if "Ys_r_pu" in st else st.ir.shape[0]
    Ir_s = cur[SERIES][0].double(); Ii_s = cur[SERIES][1].double()
    out_r = Ir_s.clone(); out_i = Ii_s.clone()

    edges = line_conductor_edges(data)
    if not edges:
        return out_r, out_i

    q = _nodal_injection(data, cur, exclude=SERIES)

    # per-edge shunt (known injection) and series flow (unknown, to solve)
    adj = defaultdict(list)          # node -> [edge_idx]
    e_n1 = torch.empty(len(edges), dtype=torch.long)
    e_n2 = torch.empty(len(edges), dtype=torch.long)
    e_comp = torch.empty(len(edges), dtype=torch.long)
    e_slot = torch.empty(len(edges), dtype=torch.long)
    shunt = torch.zeros(len(edges), 2, dtype=torch.float64)
    for i, (c, s, n1, n2) in enumerate(edges):
        col1, col2 = s, FC + s
        i1 = torch.stack([Ir_s[c, col1], Ii_s[c, col1]])
        i2 = torch.stack([Ir_s[c, col2], Ii_s[c, col2]])
        shunt[i] = 0.5 * (i1 + i2)
        e_n1[i] = n1; e_n2[i] = n2; e_comp[i] = c; e_slot[i] = s
        adj[n1].append(i); adj[n2].append(i)
    # shunt injects at both ends
    q[:, 0].index_add_(0, e_n1, shunt[:, 0]); q[:, 1].index_add_(0, e_n1, shunt[:, 1])
    q[:, 0].index_add_(0, e_n2, shunt[:, 0]); q[:, 1].index_add_(0, e_n2, shunt[:, 1])

    # forest from slack; each tree edge's parent->child gives one branch flow
    slack = data["node"].slack.tolist() if hasattr(data["node"], "slack") else []
    slack_set = {i for i, v in enumerate(slack) if v}
    seen = set()
    parent_edge = {}
    order = []
    roots = sorted(slack_set) or [min(adj)] if adj else []
    # BFS forest over all edge-touched nodes
    for root in list(roots) + sorted(adj.keys()):
        if root in seen:
            continue
        seen.add(root); dq = deque([root])
        while dq:
            u = dq.popleft(); order.append(u)
            for ei in adj[u]:
                v = int(e_n2[ei]) if u == int(e_n1[ei]) else int(e_n1[ei])
                if v in seen:
                    continue
                seen.add(v); parent_edge[v] = ei; dq.append(v)

    # leaf-to-root: subtree[node] accumulates injections; branch flow = -subtree[child]
    subtree = q.clone()
    flow = torch.zeros(len(edges), 2, dtype=torch.float64)
    for u in reversed(order):
        ei = parent_edge.get(u)
        if ei is None:
            continue
        f = subtree[u].clone()                     # net injection of u's subtree
        # sign: current from parent into child node u
        child_is_n1 = (u == int(e_n1[ei]))
        flow[ei] = f if child_is_n1 else -f
        p = int(e_n1[ei]) if not child_is_n1 else int(e_n2[ei])
        subtree[p] = subtree[p] + subtree[u]

    # I_bus1[k] = -flow + shunt ; I_bus2[k] = flow + shunt  (into-element sign)
    fr = -flow                                     # series contribution at bus1
    for i in range(len(edges)):
        c, s = int(e_comp[i]), int(e_slot[i])
        out_r[c, s] = fr[i, 0] + shunt[i, 0]
        out_i[c, s] = fr[i, 1] + shunt[i, 1]
        out_r[c, FC + s] = -fr[i, 0] + shunt[i, 0]
        out_i[c, FC + s] = -fr[i, 1] + shunt[i, 1]
    return out_r, out_i


def reconstruct_all(data, cur):
    """Full series reconstruction from shunt injections, using only the two
    proven-exact sweeps:
      1. unified tree over line+transformer+vsource -> take TRANSFORMER & VSOURCE
         (both exact: they are subtree sums of the shunt loads through the tree);
      2. line-only tree -> LINES, with the reconstructed transformer/vsource fed
         back in as known injections (reconstruct_series is exact given them).
    The unified sweep's own line output is discarded (it undercounts on meshed
    LV/transformer junctions); lines always come from the clean line-only tree.
    Returns {store: (Ir, Ii)} for every series store present."""
    uni = reconstruct_unified(data, cur)
    cur2 = dict(cur)
    for s in ("transformer", "vsource", "reactor"):
        if s in uni:
            cur2[s] = uni[s]
    out = {s: uni[s] for s in uni}
    if SERIES in data.node_types and SERIES in cur:
        out[SERIES] = reconstruct_series(data, cur2)
    return out


def transformer_null_map(Yr, Yi, thr=1e-4):
    """Per-transformer map M with I_primary = M @ I_secondary, from the YPrim
    null space (the amp-turn constraints). Type-agnostic (wye/delta/center-tap):
    Y encodes it. Returns (prim_slots, sec_slots, Mr, Mi) or None. numpy in/out."""
    import numpy as np
    Y = Yr.astype(np.complex128) + 1j * Yi
    diag = np.abs(np.diag(Y))
    if diag.max() <= 0:
        return None
    act = np.where(diag > 1e-9 * diag.max())[0]
    prim = np.array([i for i in act if i < FC], dtype=int)          # bus1 = primary
    sec = np.array([i for i in act if i >= FC], dtype=int)          # bus2/bus3 = secondary
    if prim.size == 0 or sec.size == 0:
        return None
    Ya = Y[np.ix_(act, act)]
    _, S, Vh = np.linalg.svd(Ya)
    null_mask = (S / S.max()) < thr
    N = Vh[null_mask].conj().T                                      # Ya @ N ~ 0  -> nᵀI=0
    if N.shape[1] == 0:
        return None
    pl = np.array([k for k, i in enumerate(act) if i < FC])
    sl = np.array([k for k, i in enumerate(act) if i >= FC])
    # N[prim]ᵀ I_prim = -N[sec]ᵀ I_sec  ->  I_prim = M @ I_sec
    M = -np.linalg.pinv(N[pl].T) @ N[sl].T
    return prim, sec, M.real.copy(), M.imag.copy()


def build_xfmr_system(data, thr=1e-6, unsolved=None, bridges=(), comp_of=None,
                      loop_dof=0, kvl=None):
    """JOINT transformer solve, per group of transformers that share a node.

    Supersedes the per-transformer K/U map, which asked "is this conductor the lone
    unknown at its node?" and could not represent the answer "no". Both settings of
    that test fail on real networks:
      * per-transformer census -> an OPEN-WYE/OPEN-DELTA bank (two single-phase
        transformers sharing a secondary node) has each one believe it owns the
        node, so each takes HALF the current;
      * global census -> a TRANSMISSION bus with several transformers on it
        disqualifies all of them, K comes back empty and NOTHING is solved.
    Both are the same fact: KCL at a shared node gives only the SUM of the unknowns
    there. That is one equation about several conductors, so it belongs in a system,
    not in a per-element map.

    Per group, unknowns x = every active conductor of its transformers, and rows are
      * one KCL row per shared secondary node:  sum(x at node) = -(other injections)
      * per transformer, directions n with   nᵀI = (Yn)ᵀV   (transformers carry no
        Icomp: they are linear, I = Y@V exactly).
    Rows are added least-stiff-first (‖Yn‖ = the singular value multiplies the V
    error) until the system determines x. An isolated transformer at unique nodes
    reduces EXACTLY to the old map (3 KCL rows + 4 null rows = 7 unknowns), so this
    generalises without special-casing anything.

    BRIDGE conductors (a line between two rooted trees) join the same system: no
    subtree sum can see them, and KCL at either root then has two unknowns. They add
    their own well-conditioned row  I1 + I2 = Yh(V1+V2)  (the charging; the stiff
    series part cancels in the SUM), and the transformers' constitutive rows supply
    the rest. 37Bus: 14 transformer conductors + 2 bridge conductors = 16 unknowns
    against 6 KCL + 1 bridge + 9 null rows = 16.

    Returns per-group precompute; x = P @ b is assembled by _apply_xfmr_system.
    """
    import numpy as np
    if "transformer" not in data.node_types or store_size(data, "transformer") == 0:
        return []
    st = data["transformer"]
    Yr = st["Yxfmr_r_pu"].reshape(-1, 3 * FC, 3 * FC).double().numpy()
    Yi = st["Yxfmr_i_pu"].reshape(-1, 3 * FC, 3 * FC).double().numpy()
    slot_node = {}
    for t in (1, 2, 3):
        rel = ("transformer", f"bus{t}", "node")
        if rel not in data.edge_types or not data[rel].edge_index.numel():
            continue
        ei = data[rel].edge_index
        k = terminal_slot(ei[0])
        for c, kk, nd in zip(ei[0].tolist(), k.tolist(), ei[1].tolist()):
            slot_node[(int(c), (t - 1) * FC + int(kk))] = int(nd)
    act_of = {}
    for row in range(Yr.shape[0]):
        diag = np.abs(np.diag(Yr[row] + 1j * Yi[row]))
        if diag.max() > 0:
            act_of[row] = [int(i) for i in np.where(diag > 1e-9 * diag.max())[0]]

    # Unknown conductors, keyed (store, comp, slot): every active transformer
    # conductor + both terminals of every bridge conductor.
    node_conds = defaultdict(list)
    for r, a in act_of.items():
        for s in a:
            nd = slot_node.get((r, s), 0)
            if nd != 0:
                node_conds[nd].append(("transformer", r, s))
    bpairs = []
    for (bs, bc, bn1, bn2, bca, bcb) in bridges:
        ka, kb = (bs, bc, bca), (bs, bc, bcb)
        node_conds[bn1].append(ka); node_conds[bn2].append(kb)
        bpairs.append((ka, kb))
    # KCL is usable only at a ROOT: everything else at it (children subtree sums,
    # shunts) is known. Transformer-secondary roots qualify; the slack does not
    # (its vsource is itself unknown until the end).
    slack_s, xsec = _slack_xfmrsec_roots(data)
    knodes = sorted(nd for nd in node_conds if nd in xsec)
    # CUT-SET rows. A BRIDGE has 2 unknown terminals but only ONE row of its own
    # (I1+I2 = Yh(V1+V2)); the second must come from KCL, which exists only at
    # transformer-secondary roots -- so a bridge landing on ordinary nodes stays
    # underdetermined (IEEE 30 Bus: rank 50 < 62). Summing KCL over a WHOLE rooted
    # component closes it: every internal 2-terminal element contributes only
    # I1+I2 = its charging (known from V), so
    #     sum(boundary conductors of T) = -(sum of everything else in T)
    # which is one well-conditioned equation per component -- currents only, no
    # impedance, no V1-V2. Components containing the SLACK are skipped: their
    # vsource is itself unknown until the end.
    cutsets = []
    if comp_of:
        by_root = defaultdict(list)
        for nd, root in comp_of.items():
            if nd != 0:
                by_root[root].append(nd)
        for root, nodes in sorted(by_root.items()):
            ns = set(nodes)
            if ns & slack_s:
                continue
            keys = sorted({k for nd in nodes for k in node_conds.get(nd, ())})
            if keys:
                cutsets.append((sorted(ns), keys))

    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for r, a in act_of.items():                 # a transformer's slots are coupled by its Y
        ks = [("transformer", r, s) for s in a]
        for k in ks[1:]:
            union(ks[0], k)
    for ka, kb in bpairs:                       # a bridge's two terminals are coupled
        union(ka, kb)
    for nd in knodes:                           # unknowns meeting at a KCL node interact
        ks = node_conds[nd]
        for k in ks[1:]:
            union(ks[0], k)
    for _ns, ks in cutsets:                     # a cut-set couples everything on it
        for k in ks[1:]:
            union(ks[0], k)

    allkeys = [("transformer", r, s) for r, a in act_of.items() for s in a] \
        + [k for p in bpairs for k in p]
    groups = defaultdict(list)
    for k in allkeys:
        groups[find(k)].append(k)
    kn_of = defaultdict(list)
    for nd in knodes:
        kn_of[find(node_conds[nd][0])].append(nd)
    cs_of = defaultdict(list)
    for ns, ks in cutsets:
        cs_of[find(ks[0])].append((ns, ks))

    out = []
    for g, keys in sorted(groups.items(), key=lambda kv: str(kv[0])):
        xi = {k: i for i, k in enumerate(keys)}
        Nx = len(xi)
        gk = sorted(kn_of.get(g, []))
        R = np.zeros((len(gk), Nx), dtype=np.complex128)
        for i, nd in enumerate(gk):
            for k in node_conds[nd]:
                R[i, xi[k]] = 1.0
        rank = np.linalg.matrix_rank(R, tol=thr) if R.shape[0] else 0
        cur = R
        # CUT-SET rows: sum of this component's boundary/unknown conductors.
        # Added ONLY if the row raises the rank. A redundant row is NOT free: the rhs
        # is assembled from a half-converged Jacobi sweep, so a row that duplicates a
        # KCL row carries a DIFFERENT rhs mid-iteration. pinv then least-squares the
        # disagreement and smears it across unknowns that were already exact --
        # measured, trans_3w_center_tap went 6.5e-11 -> 6.6e-01 (transformer 8.3e-01,
        # vsource silently ZERO) purely from adding redundant cut-sets. Same failure
        # mode as the rank-deficient IEEE 30 Bus case: only ever hand pinv a system
        # whose rows are independent.
        gcs, cut_keep = cs_of.get(g, []), []
        for i, (_ns, ks) in enumerate(gcs):
            row = np.zeros(Nx, dtype=np.complex128)
            for k in ks:
                if k in xi:
                    row[xi[k]] = 1.0
            if not np.any(row):
                continue
            test = np.vstack([cur, row[None, :]]) if cur.shape[0] else row[None, :]
            rr = np.linalg.matrix_rank(test, tol=thr)
            if rr > rank:
                cur, rank = test, rr
                cut_keep.append(i)
        gcs = [gcs[i] for i in cut_keep]
        # BRIDGE rows: I1 + I2 = Yh(V1+V2). Well-conditioned -- the stiff series part
        # cancels in the sum, exactly as for the line charging common-mode.
        gpairs = [(ka, kb) for (ka, kb) in bpairs if ka in xi]
        for ka, kb in gpairs:
            row = np.zeros(Nx, dtype=np.complex128)
            row[xi[ka]] = 1.0; row[xi[kb]] = 1.0
            cur = np.vstack([cur, row[None, :]])
            rank = np.linalg.matrix_rank(cur, tol=thr)
        # KVL rows (see build_kvl_rows). Added BEFORE the transformer's constitutive
        # directions because they are well-conditioned (Z and currents only), and the
        # whole point of the greedy order is least-stiff-first.
        kvl_sel, kvl_W = [], []
        if kvl is not None:
            klive, KW, _kch, KE = kvl
            for ci in range(KW.shape[0]):
                if rank >= Nx:
                    break
                row = np.zeros(Nx, dtype=np.complex128)
                wrest = np.zeros(len(klive), dtype=np.complex128)
                for k, ei in enumerate(klive):
                    w = complex(KW[ci, k])
                    if w == 0:
                        continue
                    s2, c2, _n1, _n2, ca2, cb2 = KE[ei]
                    ka, kb = (s2, c2, ca2), (s2, c2, cb2)
                    if ka in xi and kb in xi:      # a BRIDGE unknown of THIS group
                        row[xi[ka]] -= 0.5 * w
                        row[xi[kb]] += 0.5 * w
                    else:                          # known branch -> rhs, from f
                        wrest[k] = w
                if not np.any(row):                # loop touches none of our unknowns
                    continue
                test = np.vstack([cur, row[None, :]])
                rr = np.linalg.matrix_rank(test, tol=thr)
                if rr > rank:
                    cur, rank = test, rr
                    kvl_sel.append(ci); kvl_W.append(wrest)
        rows_ = sorted({r for (st, r, _s) in keys if st == "transformer"})
        # candidate constraint rows, least-stiff first
        cands = []
        for r in rows_:
            act = act_of[r]
            Y = Yr[r].astype(np.complex128) + 1j * Yi[r]
            Ya = Y[np.ix_(act, act)]
            _, S, Vh = np.linalg.svd(Ya)
            smax = max(S.max(), 1e-300)
            for j in range(len(S)):
                nv = Vh[j].conj()                       # direction in act-space
                row = np.zeros(Nx, dtype=np.complex128)
                for k2, s in enumerate(act):
                    row[xi[("transformer", r, s)]] = nv[k2]
                cands.append((S[j] / smax, row, r, act, Ya @ nv))
        cands.sort(key=lambda t: t[0])
        dirs = []; svs = []
        for _sv, row, r, act, Yn in cands:
            if rank >= Nx:
                break
            test = np.vstack([cur, row[None, :]])
            rr = np.linalg.matrix_rank(test, tol=thr)
            if rr > rank:
                cur, rank = test, rr
                dirs.append((r, act, Yn)); svs.append(float(_sv))
        # A deficient group is still RETURNED (with its null basis), so a caller can
        # close it with KVL. Whether an unclosed deficiency is fatal is the CALLER's
        # decision (_xsys_or_raise) -- skipping the group here would leave its currents
        # at exactly ZERO, the silent-wrong failure every bug in this decoder has worn.
        #
        # The leftover DOF are CIRCULATING (loop) currents -- they satisfy every row
        # here BY CONSTRUCTION (a +d/-d around a loop changes no nodal sum), so no
        # extra KCL-derived row can ever pin them: measured, 18 cut-set rows added
        # +0 rank. Only KVL+Z can.
        NB = None
        if rank < Nx:
            if unsolved is not None:
                unsolved.append((rows_, f"underdetermined: rank {rank} < {Nx} unknowns"))
            _u, _s, _vh = np.linalg.svd(cur)
            NB = _vh[rank:].conj().T                      # [Nx, Nx-rank]
        P = np.linalg.pinv(cur)
        pos = {st: [i for i, (s_, _c, _s2) in enumerate(keys) if s_ == st]
               for st in {k[0] for k in keys}}
        out.append({
            "comps": rows_, "nkcl": len(gk), "nbridge": len(gpairs),
            "ncut": len(gcs), "keys": keys,
            "null": NB, "ndef": int(Nx - rank),
            "nkvl": len(kvl_sel),
            "kvl_W": (torch.from_numpy(np.stack(kvl_W)) if kvl_W else None),
            "cutnodes": [torch.tensor(ns, dtype=torch.long) for ns, _ks in gcs],
            "svs": svs, "cond": float(np.linalg.cond(cur)),
            "knodes": torch.tensor(gk, dtype=torch.long),
            "Pr": torch.from_numpy(P.real.copy()), "Pi": torch.from_numpy(P.imag.copy()),
            # scatter targets per store: x[pos] -> out[store][comp, slot]
            "scatter": {st: (torch.tensor(ps, dtype=torch.long),
                             torch.tensor([keys[i][1] for i in ps], dtype=torch.long),
                             torch.tensor([keys[i][2] for i in ps], dtype=torch.long))
                        for st, ps in pos.items()},
            # bridge rhs lookup: (comp, slot%FC) into the charging table
            "bridge_cs": [(ka[1], ka[2] % FC) for (ka, _kb) in gpairs],
            "dirs": [(torch.tensor([slot_node.get((r, s), 0) for s in act], dtype=torch.long),
                      torch.from_numpy(Yn.real.copy()), torch.from_numpy(Yn.imag.copy()))
                     for (r, act, Yn) in dirs],
        })
    return out


def _apply_xfmr_system(data, out, groups, vr, vi, n, cs=None, kvl=None):
    """Solve every group: x = P @ [ -(other injections at KCL nodes) ;
    (Yn)ᵀV per constraint row ; Yh(V1+V2) per bridge row ]. Every unknown conductor
    (transformer AND bridge) is zeroed first so the residual sees ONLY the other
    elements' injections. `cs` = (csr, csi) line charging, for the bridge rows."""
    if "transformer" not in out or not groups:
        return
    for g in groups:                       # zero every unknown before the residual
        for st, (pos, ci, si) in g["scatter"].items():
            if st in out:
                Ar, Ai = out[st]
                out[st] = (Ar.index_put((ci, si), torch.zeros(len(ci), dtype=Ar.dtype)),
                           Ai.index_put((ci, si), torch.zeros(len(ci), dtype=Ai.dtype)))
    r0 = _full_residual(data, out, n)
    if os.environ.get("XDBG"):
        for g in groups:
            kn = g["knodes"].tolist()
            print(f"    [xdbg] comps={g['comps']} r0@knodes="
                  f"{[complex(round(float(r0[k,0]),8), round(float(r0[k,1]),8)) for k in kn]}",
                  flush=True)
    vrd, vid = vr.double(), vi.double()
    # KVL rhs needs the through-flow of every branch the loop closes through. Those
    # are KNOWN this iteration (the sweep just wrote them); only the group's own
    # bridges are unknown, and those sit on the lhs.
    fbr = None
    if kvl is not None and any(g.get("nkvl") for g in groups):
        klive, _KW, _kch, KE = kvl
        fbr = _branch_f(out, KE, klive)
    for g in groups:
        nk, nb, nc = g["nkcl"], g["nbridge"], g.get("ncut", 0)
        nv = g.get("nkvl", 0)
        br = torch.zeros(nk + nc + nb + nv + len(g["dirs"]), dtype=torch.float64)
        bi = torch.zeros_like(br)
        if nk:
            br[:nk] = -r0[g["knodes"], 0]; bi[:nk] = -r0[g["knodes"], 1]
        # cut-set: sum(unknowns on the component) = -(everything else summed over it)
        for j, ns in enumerate(g.get("cutnodes", [])):
            br[nk + j] = -r0[ns, 0].sum(); bi[nk + j] = -r0[ns, 1].sum()
        for j, (c_, sl_) in enumerate(g["bridge_cs"]):
            if cs is not None:               # I1 + I2 = Yh(V1+V2) = 2 * (half-charging)
                br[nk + nc + j] = 2.0 * cs[0][c_, sl_]; bi[nk + nc + j] = 2.0 * cs[1][c_, sl_]
        # KVL: (mᵀZ)f = 0 -> (lhs on our bridges) = -(mᵀZ) f over the known branches
        if nv:
            rhs = -(g["kvl_W"].to(torch.complex128) @ fbr)
            br[nk + nc + nb:nk + nc + nb + nv] = rhs.real
            bi[nk + nc + nb:nk + nc + nb + nv] = rhs.imag
        for j, (nd, Ynr, Yni) in enumerate(g["dirs"]):
            Va_r, Va_i = vrd[nd], vid[nd]           # V[0] == 0, so ground slots vanish
            br[nk + nc + nb + nv + j] = Ynr @ Va_r - Yni @ Va_i
            bi[nk + nc + nb + nv + j] = Ynr @ Va_i + Yni @ Va_r
        Pr, Pi = g["Pr"], g["Pi"]
        xr = Pr @ br - Pi @ bi
        xi_ = Pr @ bi + Pi @ br
        for st, (pos, ci, si) in g["scatter"].items():
            if st not in out:
                continue
            Ar, Ai = out[st]
            out[st] = (Ar.index_put((ci, si), xr[pos]), Ai.index_put((ci, si), xi_[pos]))


def build_xfmr_maps(data, thr=1e-6, unsolved=None):
    """Per-transformer YPrim map I_U = A @ V_act + B @ I_K, where K = the conductors
    nodal KCL can determine (load-facing SECONDARY conductors at unique non-ground
    nodes) and U = everything else (primary + grounded/shared-neutral secondary).
    Handles center-tap (shared neutral) and any connection, straight from Y.

    `unsolved` (optional list) collects comps for which no determined map exists --
    the caller MUST treat those as errors, never as zero current."""
    import numpy as np
    from collections import Counter
    maps = []
    if "transformer" not in data.node_types or store_size(data, "transformer") == 0:
        return maps
    st = data["transformer"]
    Yr = st["Yxfmr_r_pu"].reshape(-1, 12, 12).numpy()
    Yi = st["Yxfmr_i_pu"].reshape(-1, 12, 12).numpy()
    # slot -> node per transformer row, from the terminal edges
    slot_node = {}
    for t in (1, 2, 3):
        rel = ("transformer", f"bus{t}", "node")
        if rel not in data.edge_types or not data[rel].edge_index.numel():
            continue
        ei = data[rel].edge_index
        comp, node = ei[0], ei[1]
        k = terminal_slot(comp)
        for c, kk, nd in zip(comp.tolist(), k.tolist(), node.tolist()):
            slot_node[(int(c), (t - 1) * FC + int(kk))] = int(nd)
    # ACTIVE slots per transformer, then a GLOBAL node census over them. A conductor
    # is determinable by nodal KCL only if it is the ONLY unknown transformer
    # conductor at its node -- and that must be counted across ALL transformers, not
    # within one: an open-wye/open-delta bank is two single-phase transformers whose
    # secondaries SHARE a node, so KCL there yields only their SUM. Counting per
    # transformer made each one believe it owned the node and take half the current.
    act_of = {}
    for row in range(Yr.shape[0]):
        diag = np.abs(np.diag(Yr[row] + 1j * Yi[row]))
        if diag.max() <= 0:
            continue
        act_of[row] = [int(i) for i in np.where(diag > 1e-9 * diag.max())[0]]
    gcnt = Counter(slot_node[(r, s)] for r, a in act_of.items() for s in a
                   if slot_node.get((r, s), 0) != 0)
    for row, act in act_of.items():
        Y = Yr[row].astype(np.complex128) + 1j * Yi[row]
        nodes = {s: slot_node.get((row, s), 0) for s in act}
        K = [s for s in act if s >= FC and nodes[s] != 0 and gcnt[nodes[s]] == 1]
        U = [s for s in act if s not in K]
        if not K or not U:
            if unsolved is not None and U:
                unsolved.append((int(row), "no K conductors (no load-facing secondary)"))
            continue
        Ya = Y[np.ix_(act, act)]
        _, S, Vh = np.linalg.svd(Ya)
        Kl = [act.index(s) for s in K]; Ul = [act.index(s) for s in U]
        # EXACT constraint (not the ideal amp-turn approximation): since I = Y@V and Y
        # is symmetric, nᵀI = nᵀYV = (Yn)ᵀV for EVERY n -- null-ness is not required for
        # correctness, only for CONDITIONING (‖Yn‖ = the singular value is the price:
        # it multiplies the V error). So take the |U| least-stiff INDEPENDENT directions
        # that actually determine I_U, smallest singular value first.
        #   N_UᵀI_U = (YN)ᵀV - N_KᵀI_K   ->   I_U = A @ V_act + B @ I_K
        # Null vectors are free and usually suffice (they are the amp-turn constraints,
        # and the (Yn)ᵀV term is exactly the magnetizing current that nᵀI=0 drops). But
        # they are NOT always enough: a grounded-wye primary feeding a floating-wye
        # secondary has a null vector supported only on K (the secondary common mode),
        # leaving the primary ZERO-SEQUENCE unconstrained -- that current flows through
        # the magnetizing branch alone, so it is a function of V and is genuinely absent
        # from I_K. Thresholding the null space silently min-normed it to 0.
        sel, rank = [], 0
        for j in np.argsort(S):
            cand = sel + [int(j)]
            r = np.linalg.matrix_rank(Vh[cand].conj().T[Ul].T, tol=thr)
            if r > rank:
                sel, rank = cand, r
            if rank == len(U):
                break
        if rank < len(U):
            if unsolved is not None:
                unsolved.append((int(row), f"underdetermined: rank {rank} < |U| {len(U)}"))
            continue
        N = Vh[sel].conj().T
        P = np.linalg.pinv(N[Ul].T)                       # square & invertible by construction
        A = P @ (Ya @ N).T                                # [nU, nact]  (V_act -> I_U)
        B = -P @ N[Kl].T                                  # [nU, nK]    (I_K   -> I_U)
        maps.append({"comp": int(row), "K": torch.tensor(K, dtype=torch.long),
                     "U": torch.tensor(U, dtype=torch.long),
                     "act": torch.tensor(act, dtype=torch.long),
                     "anode": torch.tensor([nodes[s] for s in act], dtype=torch.long),
                     "Ar": torch.tensor(A.real.copy()), "Ai": torch.tensor(A.imag.copy()),
                     "Br": torch.tensor(B.real.copy()), "Bi": torch.tensor(B.imag.copy())})
    return maps


def _slack_xfmrsec_roots(data):
    slack, xsec = set(), set()
    rel = ("vsource", "bus1", "node")
    if rel in data.edge_types and data[rel].edge_index.numel():
        slack = {int(x) for x in data[rel].edge_index[1].tolist() if int(x) != 0}
    for t in (2, 3):
        rel = ("transformer", f"bus{t}", "node")
        if rel in data.edge_types and data[rel].edge_index.numel():
            xsec.update(int(x) for x in data[rel].edge_index[1].tolist() if int(x) != 0)
    return slack, xsec


def _full_residual(data, out, n):
    r = torch.zeros(n, 2, dtype=torch.float64)
    for s, (Ir, Ii) in out.items():
        comp, col, node = _inj_index(data, s)
        if comp.numel():
            r = r.index_add(0, node, torch.stack([Ir[comp, col], Ii[comp, col]], 1))
    r[0] = 0.0
    return r


def _set_terminals_by_kcl(data, out, store, terminals, n):
    """Set a store's terminal-conductor currents so nodal KCL closes at their
    nodes (each is the lone unknown there): I -= residual(node)."""
    r = _full_residual(data, out, n)
    Ir, Ii = out[store]
    for t in terminals:
        rel = (store, f"bus{t}", "node")
        if rel not in data.edge_types or not data[rel].edge_index.numel():
            continue
        ei = data[rel].edge_index
        comp, node = ei[0], ei[1]
        col = (t - 1) * FC + terminal_slot(comp)
        Ir = Ir.index_put((comp, col), Ir[comp, col] - r[node, 0])
        Ii = Ii.index_put((comp, col), Ii[comp, col] - r[node, 1])
    out[store] = (Ir, Ii)


def _apply_xfmr_maps(out, xmaps, vr=None, vi=None):
    """Close the unknown transformer conductors U from the KCL-known K:
        I_U = A @ V_act + B @ I_K
    The A@V term is the EXACT (Yn)ᵀV magnetizing part -- well-conditioned because
    Yn is tiny for near-null n. Drop it (vr=None) and you get the ideal amp-turn
    approximation, which floors the transformer at ~0.5-1%."""
    if "transformer" not in out or not xmaps:
        return
    Ir, Ii = out["transformer"]
    for m in xmaps:
        c, K, U = m["comp"], m["K"], m["U"]
        Br, Bi = m["Br"].double(), m["Bi"].double()
        Ikr, Iki = Ir[c, K], Ii[c, K]
        pr = Br @ Ikr - Bi @ Iki
        pi = Br @ Iki + Bi @ Ikr
        if vr is not None:
            Ar, Ai = m["Ar"].double(), m["Ai"].double()
            nd = m["anode"]
            Var, Vai = vr[nd].double(), vi[nd].double()
            pr = pr + (Ar @ Var - Ai @ Vai)
            pi = pi + (Ar @ Vai + Ai @ Var)
        Ir = Ir.index_put((torch.full_like(U, c), U), pr)
        Ii = Ii.index_put((torch.full_like(U, c), U), pi)
    out["transformer"] = (Ir, Ii)


def _line_charging(data, vr, vi):
    """Well-conditioned line common-mode (charging) current per conductor:
    0.5*Yh*(V1+V2), with Yh = A+B recovered from the fused 8x8 YPrim block
    [[A,B],[B^T,A]] (line.tex: Ys=-B, Yh=A+B). Returns (sr,si) [n_line,4]."""
    st = data["line"]
    # Yh now comes DIRECTLY from the corpus (Yh_i_pu, purely susceptive) instead of
    # the old Yh = A+B cancellation off the fused 8x8 -- exact by construction.
    Yh_i = st["Yh_i_pu"].reshape(-1, 4, 4).double()
    Yh_r = torch.zeros_like(Yh_i)
    nl = Yh_i.shape[0]
    V1r = torch.zeros(nl, 4, dtype=torch.float64); V1i = torch.zeros(nl, 4, dtype=torch.float64)
    V2r = torch.zeros(nl, 4, dtype=torch.float64); V2i = torch.zeros(nl, 4, dtype=torch.float64)
    for t, (Vr_, Vi_) in ((1, (V1r, V1i)), (2, (V2r, V2i))):
        rel = ("line", f"bus{t}", "node")
        if rel not in data.edge_types or not data[rel].edge_index.numel():
            continue
        ei = data[rel].edge_index
        comp, node = ei[0], ei[1]
        slot = terminal_slot(comp)
        Vr_[comp, slot] = vr[node].double(); Vi_[comp, slot] = vi[node].double()
    Vsr = (V1r + V2r).unsqueeze(-1); Vsi = (V1i + V2i).unsqueeze(-1)
    sr = 0.5 * (torch.bmm(Yh_r, Vsr) - torch.bmm(Yh_i, Vsi)).squeeze(-1)
    si = 0.5 * (torch.bmm(Yh_r, Vsi) + torch.bmm(Yh_i, Vsr)).squeeze(-1)
    return sr, si


class UnsupportedNetwork(Exception):
    """The decoder cannot handle this network structure. Raised rather than
    silently returning wrong currents -- a foundation model must know when it is
    out of distribution."""


def check_assumptions(data, raise_on_fail=True):
    """Guard the decoder's structural assumptions. SMART-DS satisfies all of these,
    but 'never seen here' != 'safe' -- we only found the mesh problem because ONE
    feeder happened to have loops. Anything unhandled must FAIL LOUDLY, because the
    silent failure mode is a current of exactly ZERO (that is how the vsource
    grounded terminal and the transformer neutrals hid for so long).

    Returns a list of violations; raises UnsupportedNetwork if raise_on_fail.
    """
    bad = []
    # A2/A3: vsource must be a single slack whose bus2 is ground (bus2 = -bus1).
    if "vsource" in data.node_types and store_size(data, "vsource") > 0:
        if store_size(data, "vsource") > 1:
            bad.append(f"multiple vsources ({store_size(data,'vsource')}): the "
                       f"bus2=-bus1 slack mirror assumes a single slack")
        m2 = _slot_node_map(data, "vsource", 2)
        live2 = [n for n in m2.values() if n != 0]
        if live2:
            bad.append(f"vsource bus2 is NOT ground (nodes {live2[:4]}): it is a real "
                       f"2-terminal source, so I_bus2 = -I_bus1 is invalid")
    # A6: capacitors are treated as shunt injections.
    if "capacitor" in data.node_types and store_size(data, "capacitor") > 0:
        m1 = _slot_node_map(data, "capacitor", 1); m2 = _slot_node_map(data, "capacitor", 2)
        ser = [(c, sl) for (c, sl), n1 in m1.items()
               if n1 != 0 and m2.get((c, sl), 0) != 0]
        if ser:
            bad.append(f"{len(ser)} SERIES capacitor conductors (both terminals live): "
                       f"capacitors are reconstructed as shunt injections")
    # A_generic: any 2-terminal SERIES store not routed anywhere is silently 0.
    for s in SERIES_STORES:
        if s in (SERIES, "transformer", "vsource") or s in TREE_STORES:
            continue        # line/reactor/cap = tree(+per-element shunt split),
                            # transformer = null-space map, vsource = slack KCL
        if s in data.node_types and store_size(data, s) > 0:
            bad.append(f"{store_size(data, s)} '{s}' elements: series store is not in "
                       f"the reconstruction tree -> its current would be silently 0")
    if bad and raise_on_fail:
        raise UnsupportedNetwork("; ".join(bad))
    return bad


def _xsys_or_raise(data, bridges, comp_of=None, loop_dof=0, kvl=None):
    """An underdetermined group is SKIPPED by build_xfmr_system, which leaves its
    currents at exactly ZERO -- the silent-wrong failure every bug in this decoder
    has presented as. Refuse the feeder instead."""
    uns = []
    xs = build_xfmr_system(data, bridges=bridges, unsolved=uns, comp_of=comp_of,
                           loop_dof=loop_dof, kvl=kvl)
    if uns:
        raise UnsupportedNetwork(
            f"transformer group(s) underdetermined: {uns}. The leftover DOF are not "
            "loop currents (those are handled: pinv gives the KCL particular solution "
            "and mesh_correct adds the KVL/Z loop part), so something else is "
            "unconstrained. Refusing rather than returning silently-zero currents.")
    return xs


def _bridge_inj(bridges):
    """Scatter index (store -> comp, col, node) for both terminals of every bridge
    conductor, so their solved current can be injected into the nodal balance."""
    by = defaultdict(lambda: ([], [], []))
    for (s, c, n1, n2, ca, cb) in bridges:
        for col, nd in ((ca, n1), (cb, n2)):
            by[s][0].append(c); by[s][1].append(col); by[s][2].append(nd)
    return {s: (torch.tensor(a, dtype=torch.long), torch.tensor(b, dtype=torch.long),
                torch.tensor(c_, dtype=torch.long)) for s, (a, b, c_) in by.items()}


def _y_fingerprint(data):
    """Reference to the Y a ctx's transformer maps were built from, so a ctx reused
    on a variant with different Y is caught LOUDLY instead of quietly decoding wrong."""
    if "transformer" not in data.node_types or store_size(data, "transformer") == 0:
        return None
    st = data["transformer"]
    return (st["Yxfmr_r_pu"].detach().clone(), st["Yxfmr_i_pu"].detach().clone())


def build_recon_ctx(data, topo=None):
    """Per-variant precompute. Two very different lifetimes are mixed in here:

      TOPOLOGY-only (tree, injection indices, series classification) -- driven by
        edge_index, which IS static across a feeder's variants -> cacheable.
      Y-DEPENDENT (the transformer null-space maps) -- driven by Yxfmr, which is
        NOT static: variants change transformer TAPS, so A/B change with them.

    Pass `topo` (a ctx from a previous variant of the SAME feeder) to reuse the
    topology precompute; the Y-dependent maps are always rebuilt. Caching the
    whole ctx across variants silently decoded taps at the wrong ratio (6.5e-1
    on a variant whose variant-0 sibling read 1.4e-8) -- reconstruct_full now
    rejects a stale ctx rather than trusting it."""
    if topo is not None:
        ctx = dict(topo)
        ctx["xmaps"] = _xsys_or_raise(data, ctx["bridges"], None,
                                      len(ctx["ltree"].get("mchords", [])),
                                      kvl=ctx.get("kvl"))
        ctx["yref"] = _y_fingerprint(data)
        return ctx
    slack, xsec = _slack_xfmrsec_roots(data)
    # TREE_STORES: every 2-terminal series store whose through-flow the subtree/mesh
    # sweep must reconstruct. Reactors belong here too -- they are series elements,
    # and leaving them out means their current is silently ZERO (A1 in the audit).
    ser = {s: classify_series(data, s) for s in AMBIG_STORES}
    # Only SERIES-element conductors may enter the tree. The tree now keeps
    # ground-touching edges (a line's grounded neutral is a real branch), so the
    # shunt capacitors/reactors -- whose grounded leg is physics-decoded EXACTLY --
    # must be filtered out here instead, or the sweep would overwrite them.
    Eline = [e for e in _series_edges(data, TREE_STORES)
             if e[0] not in AMBIG_STORES or e[1] in ser.get(e[0], set())]
    ltree = _tree_from_edges(Eline, slack | xsec)
    bridges = [Eline[i] for i in ltree["bridges"]]
    # KVL closure for the bridge loops. Topology + LINE impedance only, both static
    # across a feeder's variants (only transformer TAPS move), so it lives in the
    # reusable topology half of the ctx.
    kr = build_kvl_rows(data, Eline, ltree)
    kvl = (kr[0], kr[1], kr[2], Eline) if kr else None
    return {
        "Eline": Eline,
        "ltree": ltree,
        "bridges": bridges,
        "kvl": kvl,
        # bridge conductors are injections the subtree sweep cannot see: their
        # current must enter q like a transformer's, or every sum above them is short
        "binj": _bridge_inj(bridges),
        # comp_of=None DISABLES the cut-set rows. They are RETRACTED: measured, they
        # take trans_3w_center_tap from 6.5e-11 to 6.6e-01 (transformer 8.3e-01,
        # vsource silently ZERO) -- and not merely by being redundant, since the row
        # is rank-INCREASING and still wrong, so the cut-set equation itself does not
        # hold on that network. They also bought nothing: IEEE 30 Bus is refused with
        # or without them. Do not re-enable without a test that the row holds at TRUTH
        # currents on a feeder with grounded/center-tap windings.
        "xmaps": _xsys_or_raise(data, bridges, None,
                                len(ltree.get("mchords", [])), kvl=kvl),
        "inj": {s: _inj_index(data, s) for s in tuple(SHUNT_STORES) + tuple(SERIES_STORES)
                if s in data.node_types and store_size(data, s) > 0},
        # per-ELEMENT shunt/series split for the ambiguous 2-terminal stores
        "ser": ser,
        "n": node_count(data),
        "yref": _y_fingerprint(data),
    }


def reconstruct_full(data, cur, vr=None, vi=None, ctx=None):
    """Corrected per-feeder reconstruction (validation reference). Order:
    LV lines(shunts, rooted at slack+xfmr-sec) -> xfmr secondary(nodal KCL) ->
    xfmr primary(YPrim null-space map) -> all lines(+xfmr inj) -> parallel-line
    current division -> vsource(KCL). If (vr,vi) given, adds the well-conditioned
    Yh line charging common-mode. Pass `ctx` (build_recon_ctx) to reuse the
    topology precompute across a feeder's variants."""
    if ctx is None:
        ctx = build_recon_ctx(data)
    else:
        # A ctx carries transformer maps built from a SPECIFIC Y. Variants retap the
        # transformers, so a ctx reused blind decodes at the wrong turns ratio -- and
        # it does so silently, which is how this went unnoticed across 100 variants.
        yr = ctx.get("yref"); yn = _y_fingerprint(data)
        stale = (yr is None) != (yn is None) or (
            yr is not None and not (torch.equal(yr[0], yn[0]) and torch.equal(yr[1], yn[1])))
        if stale:
            raise UnsupportedNetwork(
                "stale ctx: transformer Y differs from the one its null-space maps were "
                "built from (variants change taps). Rebuild per variant with "
                "build_recon_ctx(data, topo=ctx) -- topology is reused, maps are not.")
    n = ctx["n"]
    out = {s: (cur[s][0].double().clone(), cur[s][1].double().clone()) for s in cur}
    for s in ("line", "transformer", "vsource"):  # always-series: zero placeholders
        if s in out:
            z = torch.zeros_like(out[s][0]); out[s] = (z, z.clone())
    # AMBIGUOUS stores (capacitor/reactor): shunt-connected elements keep their exact
    # physics-decode (I=Y@V-Icomp, verified 1e-16); SERIES-connected ones (both
    # terminals live) must instead get s + tree-flow, so zero only those rows and
    # seed them with their well-conditioned common-mode s = 0.5*(I1+I2), whose stiff
    # series part cancels algebraically.
    ser_cm = {}
    for s in AMBIG_STORES:
        sers = ctx["ser"].get(s, set())
        if s not in out or not sers:
            continue
        rows = torch.tensor(sorted(sers), dtype=torch.long)
        Or, Oi = out[s]
        cm_r = 0.5 * (Or[rows, :FC] + Or[rows, FC:2 * FC])
        cm_i = 0.5 * (Oi[rows, :FC] + Oi[rows, FC:2 * FC])
        Or[rows] = 0.0; Oi[rows] = 0.0
        Or[rows[:, None], torch.arange(FC)] = cm_r
        Or[rows[:, None], torch.arange(FC, 2 * FC)] = cm_r
        Oi[rows[:, None], torch.arange(FC)] = cm_i
        Oi[rows[:, None], torch.arange(FC, 2 * FC)] = cm_i
        out[s] = (Or, Oi)
        ser_cm[s] = (rows, cm_r, cm_i)
    ltree = ctx["ltree"]
    xmaps = ctx["xmaps"]
    # line charging (common-mode) from Yh -- added to both terminals so it does
    # not disturb the KCL/through-flow (it cancels in I1-I2, appears in I1+I2).
    if vr is not None and "line" in out:
        csr, csi = _line_charging(data, vr, vi)
        lr, li = out["line"]
        lr[:, :4] += csr; lr[:, 4:] += csr; li[:, :4] += csi; li[:, 4:] += csi
        out["line"] = (lr, li)

    # The line's OWN charging is a KNOWN nodal injection at both of its terminals
    # (it cancels in I1-I2 but not in I1+I2). It must enter q, or every subtree sum
    # is off by the charging accumulated below it.
    q_charge = torch.zeros(n, 2, dtype=torch.float64)
    if vr is not None and "line" in out:
        for t in (1, 2):
            rel = ("line", f"bus{t}", "node")
            if rel not in data.edge_types or not data[rel].edge_index.numel():
                continue
            ei = data[rel].edge_index
            comp, node = ei[0], ei[1]
            slot = terminal_slot(comp)
            q_charge = q_charge.index_add(0, node, torch.stack([csr[comp, slot], csi[comp, slot]], 1))

    def build_q(stores):
        q = q_charge.clone()
        for s in stores:
            src = out if (s in SERIES_STORES or s in AMBIG_STORES) else cur
            if s not in src:
                continue
            if s not in ctx["inj"]:
                continue
            comp, col, node = ctx["inj"][s]
            if comp.numel():
                Ir, Ii = src[s]
                q = q.index_add(0, node, torch.stack([Ir[comp, col].double(), Ii[comp, col].double()], 1))
        return q

    # Fixed point: LV lines need the transformer current, the transformer
    # secondary needs the LV lines (KCL). Jacobi-iterate lines <-> transformer.
    line_stores = list(SHUNT_STORES) + ["transformer", "vsource", "reactor"]
    binj = ctx.get("binj") or {}
    bkeep = {s: (out[s][0][c, col].clone(), out[s][1][c, col].clone())
             for s, (c, col, _nd) in binj.items() if s in out}
    for _ in range(int(os.environ.get("JACOBI", "6"))):
        # lines from shunts + current transformer/vsource estimate
        if "line" in out:
            z = torch.zeros_like(out["line"][0]); lr, li = z, z.clone()
            if vr is not None:
                lr[:, :4] += csr; lr[:, 4:] += csr; li[:, :4] += csi; li[:, 4:] += csi
            out["line"] = (lr, li)
        # BRIDGE conductors are not tree edges, so the sweep never writes them and
        # the reset above would wipe last iteration's solved value. Restore it, and
        # inject it into q -- the subtree sums above a bridge are short without it.
        for s, (c, col, _nd) in binj.items():
            if s in out and s in bkeep:
                Ar, Ai = out[s]
                out[s] = (Ar.index_put((c, col), bkeep[s][0]),
                          Ai.index_put((c, col), bkeep[s][1]))
        q = build_q(line_stores)
        for s, (c, col, nd) in binj.items():
            if s in out:
                Ar, Ai = out[s]
                q = q.index_add(0, nd, torch.stack([Ar[c, col].double(), Ai[c, col].double()], 1))
        _add_series_flow(_subtree_sum(q, ltree), ltree, out, only=TREE_STORES)
        # transformers + bridges: ONE joint solve per group (shared-node KCL rows,
        # bridge I1+I2=Yh(V1+V2) rows, per-Y constraint rows). Replaces "set
        # secondaries by KCL, then map U from K", which assumed each secondary was
        # the lone unknown at its node.
        if "transformer" in out:
            _apply_xfmr_system(data, out, xmaps, vr, vi, n,
                               cs=(csr, csi) if vr is not None else None,
                               kvl=ctx.get("kvl"))
            bkeep = {s: (out[s][0][c, col].clone(), out[s][1][c, col].clone())
                     for s, (c, col, _nd) in binj.items() if s in out}
    # NON-RADIAL networks: subtree-KCL is L equations short (L = independent loops)
    # because it cannot know how current splits between parallel paths. Close it
    # with KVL around each fundamental loop (mesh analysis). Handles parallel lines
    # (L=1) AND genuinely meshed feeders, using impedances + the KCL tree currents
    # -- never V1-V2 -- so no Y@V stiffness. Required for a FOUNDATION model: the
    # radial assumption must not be baked into the physics.
    if "line" in out and ctx.get("Eline") is not None:
        mesh_correct(data, out, ltree, ctx["Eline"])
    # vsource <- nodal KCL at slack (last, once lines have converged). Its bus2 is
    # always ground in practice (vsource.py _is_ground_like_bus2), so it behaves as
    # a SHUNT at the slack: a series source branch with I_bus2 = -I_bus1
    # (vsource.tex: I1 = y(V1-V2) - yE, Icomp1 ~ yE, Icomp2 ~ -yE). The grounded
    # terminal carries no edge, so KCL can't set it -- mirror it from bus1.
    if "vsource" in out:
        _set_terminals_by_kcl(data, out, "vsource", (1,), n)
        vr_, vi_ = out["vsource"]
        vr_[:, FC:2 * FC] = -vr_[:, 0:FC]
        vi_[:, FC:2 * FC] = -vi_[:, 0:FC]
        out["vsource"] = (vr_, vi_)
    return out


def _tree_path(u, v, tree):
    """Tree path u->v as [(edge_id, +1 if traversed n1->n2 else -1)]."""
    pe, pn, dp = tree["parent_edge"], tree["parent_node"], tree["depth"]
    up, dn = [], []
    a, b = u, v
    while dp.get(a, 0) > dp.get(b, 0):
        up.append(a); a = pn[a]
    while dp.get(b, 0) > dp.get(a, 0):
        dn.append(b); b = pn[b]
    while a != b:
        up.append(a); a = pn[a]
        dn.append(b); b = pn[b]
    return up, list(reversed(dn))


def _series_yeff(data, store, cache):
    """Effective SERIES admittance [ncomp, FC, FC] of a 2-terminal store, from its
    own Y alone -- no element-type knowledge.

    For the standard pi primitive YPrim = [[A, B], [Bᵀ, D]] the through-branch sees
    the series admittance plus half the shunt at each end:
        A = Ys + Yh,  B = -Ys   ->   (A - B)/2 = Ys + Yh/2
    which is exactly the line's Z^-1, and for a pure series element (Yh = 0) reduces
    to Ys. One formula covers line / reactor / capacitor, so a new series component
    type needs no new code here.
    """
    if store in cache:
        return cache[store]
    st = data[store]
    if store == "line":                       # split blocks are stored: use them directly
        ys_r = st["Ys_r_pu"].reshape(-1, FC, FC).double()
        ys_i = st["Ys_i_pu"].reshape(-1, FC, FC).double()
        yh_i = st["Yh_i_pu"].reshape(-1, FC, FC).double()
        Y = ys_r + 1j * (ys_i + 0.5 * yh_i)
    else:
        prefix, _, _ = STORES[store]
        Yf = (st[f"{prefix}_r_pu"].reshape(-1, 2 * FC, 2 * FC).double()
              + 1j * st[f"{prefix}_i_pu"].reshape(-1, 2 * FC, 2 * FC).double())
        A = Yf[:, :FC, :FC]; B = Yf[:, :FC, FC:]
        Y = 0.5 * (A - B)
    cache[store] = Y
    return Y


def _branch_Z(data, E, live, bidx=None):
    """Branch impedance over `live`, BLOCK-diagonal per ELEMENT so an element's own
    conductors stay coupled (phase mutual impedance is included, not dropped)."""
    if bidx is None:
        bidx = {e: k for k, e in enumerate(live)}
    nb = len(live)
    Z = torch.zeros(nb, nb, dtype=torch.complex128)
    by_elem = defaultdict(list)
    for ei in live:
        by_elem[(E[ei][0], E[ei][1])].append(ei)
    yeff = {}
    for (store, cmp_), eids in by_elem.items():
        Y = _series_yeff(data, store, yeff)[cmp_]
        slots = [E[e][4] % FC for e in eids]
        sub = Y[slots][:, slots]
        try:
            Zsub = torch.linalg.inv(sub)
        except Exception:
            Zsub = torch.linalg.pinv(sub)
        for a, ea in enumerate(eids):
            for b, eb in enumerate(eids):
                Z[bidx[ea], bidx[eb]] = Zsub[a, b]
    return Z


def _branch_f(out, E, live):
    """Per-branch through-flow f = 0.5*(I_colb - I_cola). The charging common-mode
    cancels in the difference, so f is the series part only."""
    f = torch.zeros(len(live), dtype=torch.complex128)
    for k, ei in enumerate(live):
        s, cmp_, _n1, _n2, ca, cb = E[ei]
        Or, Oi = out[s]
        f[k] = complex(0.5 * (Or[cmp_, cb] - Or[cmp_, ca]),
                       0.5 * (Oi[cmp_, cb] - Oi[cmp_, ca]))
    return f


def _loop_vec(ei, E, mt, bidx, nb):
    """Fundamental loop of chord `ei` w.r.t. the mesh forest, as a signed incidence
    vector over branches: chord oriented n1->n2, closed by the tree path n2->n1."""
    m = torch.zeros(nb, dtype=torch.complex128)
    _s, _c, n1, n2, _ca, _cb = E[ei]
    m[bidx[ei]] = 1.0
    up, dn = _tree_path(n2, n1, mt)
    for node in up:
        e2 = mt["parent_edge"][node]
        if e2 in bidx:
            m[bidx[e2]] += 1.0 if node == E[e2][2] else -1.0
    for node in dn:
        e2 = mt["parent_edge"][node]
        if e2 in bidx:
            m[bidx[e2]] += -1.0 if node == E[e2][2] else 1.0
    return m


def build_kvl_rows(data, E, ltree):
    """KVL rows for the loops the joint system CANNOT see, ready to go INTO it.

    The undetermined DOF of the transformer/bridge system are circulating currents.
    MEASURED (IEEE 30 Bus): the 9 null modes carry 100% of their weight on LINE
    conductors and 0.00% on any transformer winding -- they are pure line loops that
    close through BRIDGES, and the transformer is merely the ROOT of the component a
    bridge lands in. So this needs no transformer loop model, no turns-ratio in the
    KVL, and no impedance form for a winding (which does not exist -- YPrim is
    singular).

    Bolting a post-hoc mesh_correct on afterwards does NOT work: the group system is
    rank deficient, so pinv least-squares an inconsistent mid-Jacobi rhs and smears
    the error into the DETERMINED unknowns too (transformers went to WAPE 1.07).
    Feeding these rows INTO the system instead makes it full rank, so pinv is a true
    inverse and nothing smears.

    One row per BRIDGE that is a chord of the mesh forest (a bridge that is a mesh
    TREE edge closes no loop). Row: (mᵀZ) f = 0 around the loop -- currents and
    impedances only, never V1-V2, so no Y@V stiffness is reintroduced.
    Returns (live, W [L,nb] complex, chords) or None.
    """
    live = [i for i, e in enumerate(E) if e[2] != 0 and e[3] != 0]
    if not live:
        return None
    bidx = {e: k for k, e in enumerate(live)}
    mt = {"parent_edge": ltree.get("mparent_edge", ltree["parent_edge"]),
          "parent_node": ltree.get("mparent_node", ltree["parent_node"]),
          "depth": ltree.get("mdepth", ltree["depth"])}
    mtree_eids = set(mt["parent_edge"].values())
    chords = [i for i in ltree.get("bridges", ()) if i not in mtree_eids and i in bidx]
    if not chords:
        return None
    Z = _branch_Z(data, E, live, bidx)
    W = torch.stack([_loop_vec(ei, E, mt, bidx, len(live)) @ Z for ei in chords])
    return live, W, chords


def mesh_correct(data, out, tree, E):
    """GENERAL loop (mesh) correction -- makes the decoder valid on NON-RADIAL
    networks, not just radial ones.

    Subtree-KCL determines a branch current only on a tree: with L independent
    loops it is L equations short, and the missing information is how current
    SPLITS between parallel paths -- set by the loop impedances, which KCL never
    sees. Each chord defines one fundamental loop (chord + tree path); KVL gives
        sum_k  Z_k . I_k  = 0   around that loop.
    With I = I_tree + M @ J (M = fundamental loop matrix, J = loop currents):
        (Mᵀ Z M) J = -(Mᵀ Z I_tree)
    an L x L solve. Uses Z = (Ys + Yh/2)^-1 and the KCL tree currents -- never
    V1-V2 -- so it does NOT reintroduce the Y@V stiffness. The parallel-line
    current divider is the L=1 special case of this.
    """
    chords = list(tree["chords"])
    if not chords:
        return
    mt = tree
    # branch set = every live series conductor-edge (tree + chords), ANY store.
    # NOT just lines: a reactor/capacitor can close a loop too, and a chord that is
    # filtered out here never gets a current at all -- it stays silently ZERO.
    live = [i for i, e in enumerate(E) if e[2] != 0 and e[3] != 0]
    bidx = {e: k for k, e in enumerate(live)}
    nb, nl = len(live), len(chords)
    # fundamental loop matrix M [nb, nl]
    M = torch.zeros(nb, nl, dtype=torch.float64)
    for c, ei in enumerate(chords):
        s, comp, n1, n2, ca, cb = E[ei]
        M[bidx[ei], c] = 1.0                       # chord, oriented n1->n2
        up, dn = _tree_path(n2, n1, mt)            # return path n2 -> n1 (cycle tree)
        for node in up:
            e2 = mt["parent_edge"][node]
            if e2 in bidx:
                M[bidx[e2], c] += 1.0 if node == E[e2][2] else -1.0
        for node in dn:
            e2 = mt["parent_edge"][node]
            if e2 in bidx:
                M[bidx[e2], c] += -1.0 if node == E[e2][2] else 1.0
    Z = _branch_Z(data, E, live, bidx)
    f = _branch_f(out, E, live)
    Mc = M.to(torch.complex128)
    Zl = Mc.T @ Z @ Mc                                   # [L,L] loop impedance
    b = Mc.T @ (Z @ f)                                   # [L]
    try:
        J = torch.linalg.solve(Zl, -b)
    except Exception:
        J = torch.linalg.lstsq(Zl, (-b).unsqueeze(1)).solution.squeeze(1)
    fnew = f + Mc @ J
    # rebuild terminal currents, preserving each branch's own charging s
    for ei in live:
        s, cmp_, n1, n2, ca, cb = E[ei]
        k = bidx[ei]
        Or, Oi = out[s]
        cs_r = 0.5 * (Or[cmp_, ca] + Or[cmp_, cb]); cs_i = 0.5 * (Oi[cmp_, ca] + Oi[cmp_, cb])
        Or[cmp_, ca] = cs_r - fnew[k].real; Or[cmp_, cb] = cs_r + fnew[k].real
        Oi[cmp_, ca] = cs_i - fnew[k].imag; Oi[cmp_, cb] = cs_i + fnew[k].imag


def _split_parallel_lines(data, out):
    """Redistribute through-flow across PARALLEL line conductors (same node pair).

    A radial spanning tree cannot split current between parallel branches: it makes
    one a tree edge (which then carries the whole combined flow) and the other a
    chord (zero). The physical split is current division by admittance,
        I_k = I_total * Ys_k / sum_j Ys_j
    which is exact. Only the group's total is KCL-determined; the split needs Ys.
    """
    E = _series_edges(data, (SERIES,))
    groups = defaultdict(list)
    for i, (s, c, n1, n2, ca, cb) in enumerate(E):
        if n1 == 0 or n2 == 0:
            continue
        groups[(min(n1, n2), max(n1, n2))].append(i)
    par = {k: v for k, v in groups.items() if len(v) > 1}
    if not par:
        return
    ys_r = data["line"]["Ys_r_pu"].reshape(-1, FC, FC).double()
    ys_i = data["line"]["Ys_i_pu"].reshape(-1, FC, FC).double()
    Or, Oi = out["line"]
    for _, members in par.items():
        # each member's series part f = 0.5*(I_colb - I_cola); only the tree edge
        # holds the (combined) flow, chords hold 0 -> their sum is the true total.
        tr = ti = 0.0
        wr, wi = [], []
        for i in members:
            s_, c, n1, n2, ca, cb = E[i]
            tr = tr + 0.5 * (Or[c, cb] - Or[c, ca])
            ti = ti + 0.5 * (Oi[c, cb] - Oi[c, ca])
            k = ca % FC                                  # conductor slot
            wr.append(ys_r[c, k, k]); wi.append(ys_i[c, k, k])
        Wr = sum(wr); Wi = sum(wi)
        den = Wr * Wr + Wi * Wi
        if float(den.abs() if torch.is_tensor(den) else abs(den)) < 1e-30:
            continue
        for n, i in enumerate(members):
            s_, c, n1, n2, ca, cb = E[i]
            # share = w_k / W  (complex);  f_k = total * share
            sr = (wr[n] * Wr + wi[n] * Wi) / den
            si = (wi[n] * Wr - wr[n] * Wi) / den
            fr = tr * sr - ti * si
            fi = tr * si + ti * sr
            # keep each member's own charging s = 0.5*(I_cola + I_colb)
            cs_r = 0.5 * (Or[c, ca] + Or[c, cb]); cs_i = 0.5 * (Oi[c, ca] + Oi[c, cb])
            Or[c, ca] = cs_r - fr; Or[c, cb] = cs_r + fr
            Oi[c, ca] = cs_i - fi; Oi[c, cb] = cs_i + fi
    out["line"] = (Or, Oi)


def _add_series_flow(flow, tree, out, only):
    """Like _assign but ADDS the through-flow to existing series currents (so a
    charging common-mode seeded beforehand is preserved): I_a += -f, I_b += +f."""
    for s in only:
        m = tree["sid"] == _SID[s]
        if not m.any() or s not in out:
            continue
        comp, ca, cb, f = tree["comp"][m], tree["cola"][m], tree["colb"][m], flow[m]
        outr, outi = out[s]
        outr = outr.index_put((comp, ca), outr[comp, ca] - f[:, 0])
        outi = outi.index_put((comp, ca), outi[comp, ca] - f[:, 1])
        outr = outr.index_put((comp, cb), outr[comp, cb] + f[:, 0])
        outi = outi.index_put((comp, cb), outi[comp, cb] + f[:, 1])
        out[s] = (outr, outi)


def _active_terminals(data, store, comp_row, nterm):
    """Which terminals of a given component row carry any edge."""
    out = []
    for t in range(1, nterm + 1):
        m = _slot_node_map(data, store, t)
        if any(c == comp_row for (c, _s) in m):
            out.append(t)
    return out


def reconstruct_unified(data, cur, series_stores=SERIES_STORES):
    """Reconstruct ALL 2-terminal series currents (line/vsource/2-winding
    transformer/reactor) by ONE subtree-KCL sweep over a unified tree, from the
    SHUNT injections (loads/caps/pv/storage) plus each series element's own
    common-mode. Same-slot terminal pairing (exact for wye-wye; the delta-wye
    phase map is what this test measures). Returns {store: (Ir,Ii)}."""
    present = [s for s in series_stores if s in data.node_types and s in cur]
    out = {s: (cur[s][0].double().clone(), cur[s][1].double().clone()) for s in present}

    # nodal injections from the shunt stores (loads/caps/pv/storage)
    q = _nodal_injection(data, {s: cur[s] for s in SHUNT_STORES if s in cur}, exclude=None)

    # collect one paired-conductor edge per (series store, comp, slot)
    E = []  # (store, comp, slot, n1, n2, col1, col2)
    for s in present:
        _, nterm, _ = STORES[s]
        m1 = _slot_node_map(data, s, 1)
        # cache per-terminal slot->node maps
        term_maps = {t: _slot_node_map(data, s, t) for t in range(1, nterm + 1)}
        # per comp, pick the 2 active terminals (ref = lowest, other = next)
        comps = {c for (c, _s) in m1}
        Ir_s, Ii_s = cur[s][0].double(), cur[s][1].double()
        for c in comps:
            acts = [t for t in range(1, nterm + 1) if any(cc == c for (cc, _s) in term_maps[t])]
            if len(acts) != 2:
                continue
            ta, tb = acts
            for (cc, sl), n1 in term_maps[ta].items():
                if cc != c:
                    continue
                n2 = term_maps[tb].get((c, sl))
                if n2 is None:
                    continue
                col1 = (ta - 1) * FC + sl
                col2 = (tb - 1) * FC + sl
                E.append((s, c, sl, n1, n2, col1, col2, ta, tb))

    if not E:
        return out

    adj = defaultdict(list)
    e_n1 = torch.tensor([e[3] for e in E]); e_n2 = torch.tensor([e[4] for e in E])
    shunt = torch.zeros(len(E), 2, dtype=torch.float64)
    for i, (s, c, sl, n1, n2, col1, col2, ta, tb) in enumerate(E):
        Ir_s, Ii_s = cur[s][0].double(), cur[s][1].double()
        i1 = torch.stack([Ir_s[c, col1], Ii_s[c, col1]])
        i2 = torch.stack([Ir_s[c, col2], Ii_s[c, col2]])
        shunt[i] = 0.5 * (i1 + i2)
        # node 0 is ground: it connects every neutral/return, so routing series
        # through-flow through it corrupts the tree. Ground-touching series
        # terminals (e.g. vsource->ground) are a boundary, not a tree edge: they
        # keep their stored/predicted value and inject nothing here.
        if n1 == 0 or n2 == 0:
            continue
        adj[n1].append(i); adj[n2].append(i)
        q[:, 0].index_add_(0, e_n1[i:i+1], shunt[i:i+1, 0]); q[:, 1].index_add_(0, e_n1[i:i+1], shunt[i:i+1, 1])
        q[:, 0].index_add_(0, e_n2[i:i+1], shunt[i:i+1, 0]); q[:, 1].index_add_(0, e_n2[i:i+1], shunt[i:i+1, 1])

    slack = data["node"].slack.tolist() if hasattr(data["node"], "slack") else []
    roots = sorted(i for i, v in enumerate(slack) if v)
    seen = {0}; parent_edge = {}; order = []       # never traverse ground (node 0)
    for root in list(roots) + sorted(adj.keys()):
        if root in seen:
            continue
        seen.add(root); dq = deque([root])
        while dq:
            u = dq.popleft(); order.append(u)
            for ei in adj[u]:
                v = int(e_n2[ei]) if u == int(e_n1[ei]) else int(e_n1[ei])
                if v in seen:
                    continue
                seen.add(v); parent_edge[v] = ei; dq.append(v)

    subtree = q.clone()
    flow = torch.zeros(len(E), 2, dtype=torch.float64)
    tree_edges = set()
    for u in reversed(order):
        ei = parent_edge.get(u)
        if ei is None:
            continue
        tree_edges.add(ei)
        child_is_n1 = (u == int(e_n1[ei]))
        flow[ei] = subtree[u] if child_is_n1 else -subtree[u]
        p = int(e_n2[ei]) if child_is_n1 else int(e_n1[ei])
        subtree[p] = subtree[p] + subtree[u]

    # overwrite ONLY the tree-reconstructed series conductors; chords / ground
    # boundaries keep their stored/predicted value (out was cloned from cur).
    for i in tree_edges:
        s, c, sl, n1, n2, col1, col2, ta, tb = E[i]
        outr, outi = out[s]
        outr[c, col1] = -flow[i, 0] + shunt[i, 0]; outi[c, col1] = -flow[i, 1] + shunt[i, 1]
        outr[c, col2] = flow[i, 0] + shunt[i, 0];  outi[c, col2] = flow[i, 1] + shunt[i, 1]
    return out

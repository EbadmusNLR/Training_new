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

from collections import defaultdict, deque

import torch

from .dk_physics import FC, STORES, node_count, store_size, terminal_slot

SERIES = "line"
# series (through-flow) elements vs shunt (nodal-injection) elements
SERIES_STORES = ("line", "transformer", "vsource", "reactor")
SHUNT_STORES = ("load", "generator", "pvsystem", "storage", "capacitor")
_SID = {s: i for i, s in enumerate(SERIES_STORES)}       # series store -> int id
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
    adj = defaultdict(list)
    for i, (s, c, n1, n2, ca, cb) in enumerate(E):
        if n1 == 0 or n2 == 0:
            continue
        adj[n1].append(i); adj[n2].append(i)
    seen = {0}; parent_edge = {}; depth = {}
    for root in sorted(slack_set) + sorted(adj.keys()):
        if root in seen:
            continue
        seen.add(root); depth[root] = 0; dq = deque([root])
        while dq:
            u = dq.popleft()
            for ei in adj[u]:
                s, c, n1, n2, ca, cb = E[ei]
                v = n2 if u == n1 else n1
                if v in seen:
                    continue
                seen.add(v); parent_edge[v] = ei; depth[v] = depth[u] + 1; dq.append(v)
    child, parent, sign, level, sid, comp, cola, colb = ([] for _ in range(8))
    for v, ei in parent_edge.items():
        s, c, n1, n2, ca, cb = E[ei]
        child.append(v); parent.append(n1 if v == n2 else n2)
        sign.append(1.0 if v == n1 else -1.0); level.append(depth[v])
        sid.append(_SID[s]); comp.append(c); cola.append(ca); colb.append(cb)
    L = lambda x, dt: torch.tensor(x, dtype=dt)
    return {
        "child": L(child, torch.long), "parent": L(parent, torch.long),
        "sign": L(sign, torch.float64), "level": L(level, torch.long),
        "sid": L(sid, torch.long), "comp": L(comp, torch.long),
        "cola": L(cola, torch.long), "colb": L(colb, torch.long),
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
    n_line = st["Yline_r_pu"].shape[0] if "Yline_r_pu" in st else st.ir.shape[0]
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


def build_xfmr_maps(data, thr=1e-4):
    """Per-transformer YPrim null-space map I_U = M @ I_K, where K = the conductors
    nodal KCL can determine (load-facing SECONDARY conductors at unique non-ground
    nodes) and U = everything else (primary + grounded/shared-neutral secondary).
    The null space (amp-turn constraints) closes U from K -- handles center-tap
    (shared neutral) and any connection, straight from Y."""
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
    for row in range(Yr.shape[0]):
        Y = Yr[row].astype(np.complex128) + 1j * Yi[row]
        diag = np.abs(np.diag(Y))
        if diag.max() <= 0:
            continue
        act = [int(i) for i in np.where(diag > 1e-9 * diag.max())[0]]
        nodes = {s: slot_node.get((row, s), 0) for s in act}
        cnt = Counter(nodes.values())
        K = [s for s in act if s >= FC and nodes[s] != 0 and cnt[nodes[s]] == 1]
        U = [s for s in act if s not in K]
        if not K or not U:
            continue
        Ya = Y[np.ix_(act, act)]
        _, S, Vh = np.linalg.svd(Ya)
        N = Vh[(S / S.max()) < thr].conj().T
        if N.shape[1] == 0:
            continue
        Kl = [act.index(s) for s in K]; Ul = [act.index(s) for s in U]
        # EXACT constraint (not the ideal amp-turn approximation): since I = Y@V and
        # Y is symmetric, nᵀI = nᵀYV = (Yn)ᵀV for every null vector n. For near-null
        # n, Yn is tiny -> this term is well-conditioned (no cancellation) and it is
        # exactly the magnetizing/no-load current the nᵀI=0 idealization drops.
        #   N_UᵀI_U = (YN)ᵀV - N_KᵀI_K   ->   I_U = A @ V_act + B @ I_K
        P = np.linalg.pinv(N[Ul].T)
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
    Yr = st["Yline_r_pu"].reshape(-1, 8, 8).double()
    Yi = st["Yline_i_pu"].reshape(-1, 8, 8).double()
    Yh_r = Yr[:, :4, :4] + Yr[:, :4, 4:]
    Yh_i = Yi[:, :4, :4] + Yi[:, :4, 4:]
    nl = Yr.shape[0]
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


def reconstruct_full(data, cur, vr=None, vi=None):
    """Corrected per-feeder reconstruction (validation reference). Order:
    LV lines(shunts, rooted at slack+xfmr-sec) -> xfmr secondary(nodal KCL) ->
    xfmr primary(YPrim null-space map) -> all lines(+xfmr inj) -> vsource(KCL).
    If (vr,vi) given, adds the well-conditioned Yh line charging common-mode."""
    n = node_count(data)
    out = {s: (cur[s][0].double().clone(), cur[s][1].double().clone()) for s in cur}
    for s in SERIES_STORES:                       # zero the series placeholders
        if s in out:
            z = torch.zeros_like(out[s][0]); out[s] = (z, z.clone())
    slack, xsec = _slack_xfmrsec_roots(data)
    ltree = _tree_from_edges(_series_edges(data, (SERIES,)), slack | xsec)
    xmaps = build_xfmr_maps(data)
    # line charging (common-mode) from Yh -- added to both terminals so it does
    # not disturb the KCL/through-flow (it cancels in I1-I2, appears in I1+I2).
    if vr is not None and "line" in out:
        csr, csi = _line_charging(data, vr, vi)
        lr, li = out["line"]
        lr[:, :4] += csr; lr[:, 4:] += csr; li[:, :4] += csi; li[:, 4:] += csi
        out["line"] = (lr, li)

    def build_q(stores):
        q = torch.zeros(n, 2, dtype=torch.float64)
        for s in stores:
            src = out if s in SERIES_STORES else cur
            if s not in src:
                continue
            comp, col, node = _inj_index(data, s)
            if comp.numel():
                Ir, Ii = src[s]
                q = q.index_add(0, node, torch.stack([Ir[comp, col].double(), Ii[comp, col].double()], 1))
        return q

    # Fixed point: LV lines need the transformer current, the transformer
    # secondary needs the LV lines (KCL). Jacobi-iterate lines <-> transformer.
    line_stores = list(SHUNT_STORES) + ["transformer", "vsource", "reactor"]
    for _ in range(6):
        # lines from shunts + current transformer/vsource estimate
        if "line" in out:
            z = torch.zeros_like(out["line"][0]); lr, li = z, z.clone()
            if vr is not None:
                lr[:, :4] += csr; lr[:, 4:] += csr; li[:, :4] += csi; li[:, 4:] += csi
            out["line"] = (lr, li)
        _add_series_flow(_subtree_sum(build_q(line_stores), ltree), ltree, out, only=(SERIES,))
        # transformer secondary (nodal KCL) -> primary (YPrim null-space map)
        if "transformer" in out:
            _set_terminals_by_kcl(data, out, "transformer", (2, 3), n)
            _apply_xfmr_maps(out, xmaps, vr, vi)
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

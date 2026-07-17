#!/usr/bin/env python3
"""Datakit training dataset for the iterative-solver GridFM.

Wraps datakit's FeederScenarios (which reconstructs a per-variant HeteroData with
full-matrix Y_pu, per-terminal currents, Icomp, and V) and adds the training-time
machinery the model needs, all computed ON DEMAND (never baked into the corpus):

  * feeder-disjoint train/seen/unseen/test splits
  * cheap per-feeder structural positional encoding (cached once per topology)
  * per-terminal edge->conductor-slot columns (for full-matrix physics)
  * masking per task (pf / se / param / injection) into vis/msk flags
  * optional asinh feature view of Y and currents (ablatable), pu kept for physics

The physics (dk_physics) consumes the pu tensors; the model consumes the feature
view (or pu directly when feat is off).
"""
from __future__ import annotations

import glob
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
from core.scenario_store import FeederScenarios  # noqa: E402

from .dk_physics import FC, STORES, store_size, node_count, terminal_slot, line_yprim  # noqa: E402


def store_yfull(st, store, dim):
    """Full [n,dim,dim] Y for a store. The line stores the physical blocks
    (Ys 4x4 + Yh 4x4 imag-only) rather than a fused 8x8, so rebuild its YPrim;
    every other store keeps its fused matrix. Keeps the model's single-Y feature."""
    if store == "line":
        return line_yprim(st, dtype=st["Ys_r_pu"].dtype)
    prefix = STORES[store][0]
    n = st[f"{prefix}_r_pu"].shape[0]
    return (st[f"{prefix}_r_pu"].reshape(n, dim, dim),
            st[f"{prefix}_i_pu"].reshape(n, dim, dim))

PE_DIM = 6  # [log_deg, hop/10, is_slack, is_ground, deg_ratio, log1p_nbr_deg]


# ----------------------------------------------------------------------------- splits
def discover_feeders(root: str) -> list[str]:
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "static.pt")))


def split_feeders(feeders: list[str], train_frac=0.8, val_frac=0.1, seed=42):
    """Deterministic feeder-disjoint split by name hash (stable across runs)."""
    def h(name):
        return int(hashlib.md5((str(seed) + os.path.basename(name)).encode()).hexdigest(), 16)
    ordered = sorted(feeders, key=h)
    n = len(ordered)
    ntr = int(round(train_frac * n))
    nval = int(round(val_frac * n))
    return {"train": ordered[:ntr], "unseen": ordered[ntr:ntr + nval], "test": ordered[ntr + nval:]}


# ----------------------------------------------------------------------------- per-feeder
class DKFeeder:
    """Cached per-topology structure for one feeder + its variant reader."""

    def __init__(self, feeder_dir: str):
        self.dir = feeder_dir
        self.scen = FeederScenarios(feeder_dir)
        self.name = os.path.basename(feeder_dir)
        base = self.scen[0]
        self.n_node = node_count(base)
        self.stores = [s for s in STORES if s in base.node_types and store_size(base, s) > 0]
        # edge (comp, node, slot-column) per store/terminal, precomputed once
        self.edges: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        deg = torch.zeros(self.n_node)
        adj_r, adj_c = [], []
        for s in self.stores:
            _, nterm, _ = STORES[s]
            self.edges[s] = []
            for t in range(1, nterm + 1):
                rel = (s, f"bus{t}", "node")
                if rel not in base.edge_types:
                    self.edges[s].append((torch.zeros(0, dtype=torch.long),) * 3)
                    continue
                ei = base[rel].edge_index
                comp, node = ei[0], ei[1]
                col = (t - 1) * FC + terminal_slot(comp)
                self.edges[s].append((comp, node, col))
                deg.index_add_(0, node, torch.ones_like(node, dtype=torch.float))
                # component acts as a hyperedge: connect its nodes (cheap: to first node)
                adj_r.append(node)
                adj_c.append(node)
        self.slack = self._slack_mask(base)
        self.ground = torch.zeros(self.n_node, dtype=torch.bool)
        self.ground[0] = True
        self.pe = self._pe(base, deg)
        self._pe_cached = self.pe
        # precompute the KCL tree-current plan once per topology (like the PE)
        base["node"].slack = self.slack
        from .dk_tree import build_tree_plan, check_assumptions, build_recon_ctx
        # Fail loudly on structures the current decoder cannot reconstruct, rather
        # than training on silently-zero currents (every past bug looked like that).
        check_assumptions(base)
        self.plan = build_tree_plan(base)
        # Topology half of the EXACT decoder's precompute (tree, KVL rows, injection
        # indices, series classification) -- all driven by edge_index, which is static
        # across this feeder's variants. The Y-dependent transformer null-space maps are
        # NOT static (variants move taps) and are rebuilt per variant in the collate.
        # Caching this matters: rebuilding the whole ctx costs 2.09s on a 9710-node /
        # 517-transformer feeder vs 0.142s reusing topology (14.7x), and the reused ctx
        # reconstructs to the identical WAPE.
        self.recon_topo = build_recon_ctx(base)

    def _slack_mask(self, base) -> torch.Tensor:
        m = torch.zeros(self.n_node, dtype=torch.bool)
        rel = ("vsource", "bus1", "node")
        if "vsource" in base.node_types and rel in base.edge_types:
            m[base[rel].edge_index[1]] = True
        m[0] = False  # ground never slack
        return m

    def _pe(self, base, deg) -> torch.Tensor:
        """Cheap structural PE: node degree + BFS hop depth from the slack, over a
        node graph where each multi-terminal component links its bus1 node to the
        matching-slot node on every other terminal (the electrical series path)."""
        import collections
        nbr = collections.defaultdict(set)
        for s in self.stores:
            _, nterm, _ = STORES[s]
            if nterm < 2:
                continue
            c0, n0, col0 = self.edges[s][0]
            first = {(int(c), int(k) % FC): int(n) for c, n, k in zip(c0.tolist(), n0.tolist(), col0.tolist())}
            for t in range(1, nterm):
                ct, nt_, colt = self.edges[s][t]
                for c, n, k in zip(ct.tolist(), nt_.tolist(), colt.tolist()):
                    a = first.get((int(c), int(k) % FC))
                    if a is not None:
                        nbr[a].add(int(n)); nbr[int(n)].add(a)
        hop = torch.full((self.n_node,), 30.0)
        srcs = torch.where(self.slack)[0].tolist()
        seen = set(srcs)
        q = collections.deque((s, 0) for s in srcs)
        for s in srcs:
            hop[s] = 0.0
        while q:
            u, d = q.popleft()
            for v in nbr[u]:
                if v not in seen:
                    seen.add(v); hop[v] = float(d + 1); q.append((v, d + 1))
        maxdeg = float(deg.clamp(min=1).max())
        return torch.stack([
            torch.log1p(deg),
            hop.clamp(max=30) / 10.0,
            self.slack.float(),
            self.ground.float(),
            deg / maxdeg,
            torch.log1p(deg) / (torch.log1p(torch.tensor(maxdeg)) + 1e-6),
        ], dim=1)

    def sample(self, variant: int):
        return self.scen[variant]


# ----------------------------------------------------------------------------- features
CLAMP = 20.0
# fallback per-family scales if none are fitted (rough pu magnitudes)
Y_SCALE = 10.0
I_SCALE = 0.05


def asinh_feat(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Invertible compressed view (ablatable). pu stays the source of truth."""
    return torch.asinh(x / scale)


def feat(x: torch.Tensor, scale, use_asinh: bool = True) -> torch.Tensor:
    """Normalized (optionally asinh-compressed) feature view of pu tensor x.
    `scale` is the fitted global per-family scale (scalar or broadcastable)."""
    z = x / scale
    return torch.asinh(z) if use_asinh else z


def inv_feat(z: torch.Tensor, scale, use_asinh: bool = True) -> torch.Tensor:
    """Inverse of `feat`: decode a feature back to pu (invertible, physics-exact)."""
    z = torch.sinh(z.clamp(-CLAMP, CLAMP)) if use_asinh else z
    return z * scale


def _p95(chunks, floor: float = 1e-9, cap: int = 400_000) -> float:
    if not chunks:
        return floor
    x = torch.cat([c.flatten() for c in chunks])
    x = x[x > 0]
    if x.numel() == 0:
        return floor
    if x.numel() > cap:
        x = x[torch.randint(0, x.numel(), (cap,))]
    return max(float(torch.quantile(x, 0.95)), floor)


def fit_scales(feeders, variants, max_feeders: int = 60, max_variants: int = 4,
               cap_per_chunk: int = 20_000):
    """Fit ONE global, per-family scale set over a sample of the TRAIN split.

        s_I[store]           = P95(|I_bus|)  per component family
        s_Y[store,part,band] = P95(|Y_pu|)   per (family, real/imag, diag/off-diag)

    This is the cross-feeder normalization the old JSON pipeline proved out
    (featurizing.py), recomputed on-demand for the datakit full-matrix stores.
    Per-feeder scaling would be a foundation-model bug: the same physical
    quantity must encode to the same feature in every feeder.
    """
    fsub = feeders[:max_feeders]
    vsub = (variants[:max_variants] if variants else [0])
    Ib: dict[str, list] = {s: [] for s in STORES}
    Yb: dict[str, dict] = {s: {"r_diag": [], "r_off": [], "i_diag": [], "i_off": []} for s in STORES}

    def _keep(t):
        t = t.flatten()
        if t.numel() > cap_per_chunk:
            t = t[torch.randint(0, t.numel(), (cap_per_chunk,))]
        return t

    dvs: list[torch.Tensor] = []
    for f in fsub:
        for v in vsub:
            data = f.sample(v)
            nd = data["node"]
            dv = torch.stack([nd.V_r_pu - nd.V_r_init_pu,
                              nd.V_i_pu - nd.V_i_init_pu], 1)
            if dv.shape[0] > cap_per_chunk:
                dv = dv[torch.randint(0, dv.shape[0], (cap_per_chunk,))]
            dvs.append(dv)
            for s in f.stores:
                prefix, nterm, _ = STORES[s]
                st = data[s]
                dim = nterm * FC
                for t in range(1, nterm + 1):
                    if f"I_r_bus{t}_pu" in st:
                        Ib[s].append(_keep(torch.hypot(st[f"I_r_bus{t}_pu"], st[f"I_i_bus{t}_pu"])))
                yr, yi = store_yfull(st, s, dim)
                eye = torch.eye(dim, dtype=torch.bool)
                Yb[s]["r_diag"].append(_keep(yr[:, eye].abs()))
                Yb[s]["r_off"].append(_keep(yr[:, ~eye].abs()))
                Yb[s]["i_diag"].append(_keep(yi[:, eye].abs()))
                Yb[s]["i_off"].append(_keep(yi[:, ~eye].abs()))

    Iscale = {s: _p95(Ib[s]) for s in STORES}
    Yscale = {s: {k: _p95(Yb[s][k]) for k in Yb[s]} for s in STORES}
    med = [Iscale[s] for s in STORES if Ib[s]]
    kcl = float(np.median(med)) if med else I_SCALE
    # Train-set std of the voltage residual per (re, im): the head's output gauge.
    # The reference PINN's route to 7.5e-08 ("standardized residual voltage-head")
    # predicts z with dv = std*z, so the head works at O(1) whatever the corpus's
    # deviation from nominal (theirs: voltage_residual_std; see grid_state_pinn.py).
    dv_all = torch.cat(dvs) if dvs else torch.zeros(1, 2)
    dv_std = [max(float(dv_all[:, 0].std()), 1e-6), max(float(dv_all[:, 1].std()), 1e-6)]
    return {"I": Iscale, "Y": Yscale, "kcl": max(kcl, 1e-9), "dv_std": dv_std}


def _ydim(store):
    _, nterm, _ = STORES[store]
    return nterm * FC


# ----------------------------------------------------------------------------- masking
# Every task here is IDENTIFIABLE: the visible fields physically determine the targets
# (checked by scripts/check_pf_determinacy.py -- Ybus V = sum(Icomp) with the visible
# voltages pinned is nonsingular on 100% of this corpus). Blind random masking is
# deliberately absent: jointly hiding Y and Icomp at one operating point leaves the
# targets underdetermined, and training on arbitrary targets manufactures an
# architecture mystery out of a data problem (see the 2026-07-12 foundation contract).


def _set_comp_masks(data):
    for s in data.node_types:
        if s == "node" or s not in STORES:
            continue
        st = data[s]
        n = st.yr.shape[0]
        st.vis_y = torch.ones(n, dtype=torch.bool)
        st.vis_ic = torch.ones(n, dtype=torch.bool)
        st.msk_i = torch.ones(n, dtype=torch.bool)   # currents are always targets


def mask_pf(data, rng=None):
    """Power flow: hide non-slack/non-ground voltages and all terminal currents;
    Y, Icomp, slack V, and every V_init are observed."""
    nd = data["node"]
    nd.vis_v = nd.slack | nd.ground            # observed voltages
    nd.msk_v = ~nd.vis_v                        # targets
    _set_comp_masks(data)
    return data


def mask_se(data, rng):
    """Known-injection state estimation: Y and Icomp visible (the injections are known),
    plus a random subset of voltage 'measurements'; recover the unmeasured state.

    Strictly easier than pf per sample (more V visible), but a different conditional --
    the model must USE arbitrary interior measurements, not just the slack boundary.
    The measured fraction is drawn per sample so one checkpoint spans sparse SCADA
    (~10%) to dense PMU (~60%) coverage."""
    nd = data["node"]
    frac = float(rng.uniform(0.1, 0.6))
    meas = torch.from_numpy(rng.random(nd.num_nodes) < frac)
    nd.vis_v = nd.slack | nd.ground | meas
    nd.msk_v = ~nd.vis_v
    _set_comp_masks(data)
    return data


PC_STORES = ("load", "generator", "pvsystem", "storage")


def mask_injection(data, rng):
    """Injection estimation: full state visible (all V), Y visible; hide Icomp on a
    random subset of power-conversion components and recover it.

    Identifiability needs AT MOST ONE hidden-Icomp component per node: KCL at a node
    determines the sum of its terminal currents, so one unknown Icomp per node is pinned
    (I_term = -(sum of others), Icomp = Y@V - I_term) while two hidden on one node leave
    only their sum determined. Masking respects that constraint by construction."""
    nd = data["node"]
    nd.vis_v = torch.ones(nd.num_nodes, dtype=torch.bool)
    nd.msk_v = ~nd.vis_v                       # no voltage targets
    _set_comp_masks(data)
    taken = np.zeros(int(nd.num_nodes), dtype=bool)
    for s in PC_STORES:
        if s not in data.node_types or s not in STORES:
            continue
        st = data[s]
        n = st.yr.shape[0]
        rel = (s, "bus1", "node")
        if rel not in data.edge_types or not data[rel].edge_index.numel():
            continue
        ei = data[rel].edge_index
        comp_nodes = [[] for _ in range(n)]
        for c, nd_i in zip(ei[0].tolist(), ei[1].tolist()):
            comp_nodes[c].append(nd_i)
        vis = torch.ones(n, dtype=torch.bool)
        order = rng.permutation(n)
        for c in order:
            if rng.random() > 0.35:
                continue
            nodes = comp_nodes[int(c)]
            if not nodes or any(taken[x] for x in nodes):
                continue
            vis[int(c)] = False
            for x in nodes:
                taken[x] = True
        st.vis_ic = vis
    return data


def mask_random(data, rng):
    """THE pretraining objective: one random conditional per sample over ALL fields.

    No task presets. Independent per-sample rates decide what is visible:
      V     : slack+ground always, plus a Bernoulli(p_v) subset, p_v ~ U(0, 0.9)
      Icomp : Bernoulli(p_ic) subset per PC component,        p_ic ~ U(0.4, 1.0)
      Y     : visible (no Y head yet -- the stated capability boundary)
    Everything hidden is a target; currents are always targets (they are decoded from
    the V estimate + Icomp estimate, so they supervise both). The model therefore
    learns p(hidden | visible) across the whole family of conditionals -- pf
    (p_v=0, p_ic=1), se (p_v>0, p_ic=1), injection est. (p_v=1, p_ic<1) and every
    mixture in between are POINTS in this distribution, recovered at inference by
    choosing the mask, not separate skills.

    Some corners are underdetermined (two hidden Icomps on one node leave only their
    sum). That is a noise floor on the pretraining loss, not a defect: the model
    learns the conditional expectation there. What the 2026-07-12 contract actually
    forbids is CLAIMING identifiability from such corners -- so capability CLAIMS are
    evaluated on the determinate lenses (pf/se/injection presets below), while
    training samples the full distribution."""
    nd = data["node"]
    p_v = float(rng.uniform(0.0, 0.9))
    meas = torch.from_numpy(rng.random(int(nd.num_nodes)) < p_v)
    nd.vis_v = nd.slack | nd.ground | meas
    nd.msk_v = ~nd.vis_v
    _set_comp_masks(data)
    p_ic = float(rng.uniform(0.4, 1.0))
    for s in PC_STORES:
        if s not in data.node_types or s not in STORES:
            continue
        st = data[s]
        n = st.yr.shape[0]
        st.vis_ic = torch.from_numpy(rng.random(n) < p_ic)
    return data


def mask_random_safe(data, rng):
    """Foundation objective: ONE identifiable task per sample, chosen at random --
    the model learns every conditional (all the interactions), never an
    underdetermined one. Extend the pool as new heads land (one-entry Y completion
    still needs a Y head)."""
    r = rng.random()
    if r < 1 / 3:
        return mask_pf(data, rng)
    if r < 2 / 3:
        return mask_se(data, rng)
    return mask_injection(data, rng)


TASKS = {"pf": mask_pf, "se": mask_se, "injection": mask_injection,
         "random_safe": mask_random_safe, "random": mask_random}


# ----------------------------------------------------------------------------- dataset
class DKDataset(torch.utils.data.Dataset):
    def __init__(self, feeders: list[DKFeeder], variants: list[int], task: str = "pf",
                 use_feat: bool = True, seed: int = 0):
        self.feeders = feeders
        self.items = [(fi, v) for fi in range(len(feeders)) for v in variants]
        self.task = task
        self.use_feat = use_feat
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return len(self.items)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _item_rng(self, idx):
        """Per-item generator derived from (seed, epoch, idx). A shared self.rng is
        wrong under num_workers>0: fork gives every worker a COPY, so all workers
        draw the SAME mask sequence and 'random' masks repeat across the batch."""
        return np.random.default_rng((self.seed * 1_000_003 + self.epoch) * 4_294_967_291 + idx)

    def __getitem__(self, idx):
        fi, variant = self.items[idx]
        feeder = self.feeders[fi]
        data = feeder.sample(variant)
        data.feeder_id = torch.tensor([fi])          # for the collate to fetch the plan
        nd = data["node"]
        # node fields
        vinit = torch.stack([nd.V_r_init_pu, nd.V_i_init_pu], 1)
        vsol = torch.stack([nd.V_r_pu, nd.V_i_pu], 1)
        nd.v_init = vinit
        nd.dv = vsol - vinit
        nd.pe = feeder.pe
        nd.slack = feeder.slack
        nd.ground = feeder.ground
        nd.num_nodes = feeder.n_node               # explicit: PyG cannot infer it
        # component fields: reshape Y to [n,dim,dim], stack currents/icomp
        for s in feeder.stores:
            st = data[s]
            prefix, nterm, nic = STORES[s]
            dim = nterm * FC
            n = st[f"{prefix}_r_pu"].shape[0]
            st.yr, st.yi = store_yfull(st, s, dim)
            ir = torch.zeros(n, dim); ii = torch.zeros(n, dim)
            for t in range(1, nterm + 1):
                if f"I_r_bus{t}_pu" in st:
                    ir[:, (t - 1) * FC:t * FC] = st[f"I_r_bus{t}_pu"]
                    ii[:, (t - 1) * FC:t * FC] = st[f"I_i_bus{t}_pu"]
            st.ir = ir; st.ii = ii
            st.num_nodes = n                       # explicit component count
            if nic:
                st.icr = st.Icomp_r_pu; st.ici = st.Icomp_i_pu
            else:
                st.icr = torch.zeros(n, FC); st.ici = torch.zeros(n, FC)
        TASKS[self.task](data, self._item_rng(idx))
        # The corpus is fp64 BY DESIGN (generating quality data), but the model
        # trains in fp32. Cast at this boundary -- not in the corpus -- so the
        # reference decoder keeps its fp64 inputs.
        for store in data.node_types:
            st = data[store]
            for k, v in list(st.items()):
                if torch.is_tensor(v) and v.dtype == torch.float64:
                    st[k] = v.float()
        return data


def make_dk_collate(feeders):
    """Collate that batches the graph, the KCL tree plan, AND the exact decoder's
    reconstruction context (both offset to match PyG's node/comp concatenation)."""
    from torch_geometric.data import Batch
    from .dk_physics import ensure_batch_schema
    from .dk_tree import (batch_plans, batch_recon_ctx, build_recon_ctx,
                          SHUNT_STORES, SERIES_STORES)

    def collate(samples):
        fids = [int(s.feeder_id) for s in samples]
        plans = [feeders[f].plan for f in fids]
        # Per-VARIANT, reusing this feeder's cached topology: only the transformer
        # null-space maps are rebuilt, because variants move taps. Built on the pristine
        # sample BEFORE ensure_batch_schema/from_data_list -- the order the batched-recon
        # verification used to reach max|batched - per_feeder| = 1.084e-19.
        ctxs = [build_recon_ctx(s, topo=feeders[f].recon_topo)
                for s, f in zip(samples, fids)]
        # EVERY sample must share ONE schema before Batch.from_data_list. PyG accumulates
        # edge_index offsets only over the samples that HAVE a relation, so a feeder without
        # pvsystem/storage is skipped in that cumsum and every LATER feeder's pvsystem edges
        # point INTO an earlier feeder's node range. MEASURED on 6 SMART-DS feeders: 494
        # pvsystem + 182 storage node indices wrong; load/line/transformer/vsource (present in
        # every feeder) were fine. That silently corrupted training: _edges reads
        # batch[rel].edge_index, so _phys_current gathered the wrong v[node] and the KCL
        # residual scattered PV/storage current onto another feeder's nodes.
        ensure_batch_schema(samples)
        node_counts = [int(s["node"].num_nodes) for s in samples]
        keys = tuple(SHUNT_STORES) + tuple(SERIES_STORES)
        store_counts = [{st: int(s[st].ir.shape[0]) for st in keys
                         if st in s.node_types and hasattr(s[st], "ir")} for s in samples]
        batch = Batch.from_data_list(samples)
        return (batch,
                batch_plans(plans, node_counts, store_counts),
                batch_recon_ctx(ctxs, node_counts, store_counts))

    return collate

#!/usr/bin/env python3
"""Scenario-store dataset for GridFM training.

Loads training_data/<corpus>/<feeder>/{static.pt,dynamic.npy} directly and turns
each variant into a HeteroData sample carrying, per component store:

    x_true [n, W]   all *_feat fields concatenated (Y triangles, then optional
                    Icomp current features, then bus-terminal currents) in the
                    canonical SPECS order
    act    [n, W]   connectivity-derived active mask (structural zeros are 0)
    scale  [n, W]   per-entry asinh scale so pu = sinh(x) * (scale + eps)

and per node: v_init / dv (= V_pu - V_init_pu), ground flag, and a KCL mask for
non-ground nodes. KCL sums decoded physical terminal currents.

Edges are re-keyed at slot level: (store, "t{T}s{S}", node) per active conductor
slot, so both message passing and the physics losses know which node each padded
tensor slot refers to. Slot identity is recovered from the stored bus{T} edge
order, which json_to_hetero emits sorted by (component row, slot) with
leading-slot active masks.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import os

import numpy as np
import torch
from torch_geometric.data import HeteroData

FC = 4          # conductor slots per terminal (3 phases + neutral)
EPS = 1e-12     # featurizing.py asinh epsilon
PE_K = 8        # random-walk steps for the node structural encoding
PE_DIM = PE_K + 1   # RWSE-K + log degree (legacy checkpoints use exactly this)
PE_DENSE_MAX = 512  # feeders above this use the sparse PE (dense is O(n^2/n^3))
PE_CACHE_VERSION = 1
# Extended PE appends grid-electrical columns AFTER the legacy block, so
# pe[:, :PE_DIM] is unchanged for old checkpoints:
#   [PE_DIM]   hop depth from the source nodes (BFS), /10
#   [PE_DIM+1] log1p path impedance to source (Dijkstra, w = 1/max|Y_diag|)
PE_DIM_EXT = PE_DIM + 2


@dataclass(frozen=True)
class StoreSpec:
    terms: int                       # number of bus terminals
    ydim: int                        # order of the packed Y matrix
    yfields: tuple[tuple[str, str], ...]  # (field base name, "r"|"i")
    family: str                      # scaler family (line resolved per row)
    icomp: int = 0                   # Icomp feature width per real/imag part


SPECS: dict[str, StoreSpec] = {
    "line": StoreSpec(2, 4, (("Ys_r_tri", "r"), ("Ys_i_tri", "i"), ("Yh_i_tri", "i")), "Line"),
    "capacitor": StoreSpec(2, 8, (("Ycap_r_tri", "r"), ("Ycap_i_tri", "i")), "Capacitor"),
    "reactor": StoreSpec(2, 8, (("Yreactor_r_tri", "r"), ("Yreactor_i_tri", "i")), "Reactor"),
    "transformer": StoreSpec(3, 12, (("Yxfmr_r_tri", "r"), ("Yxfmr_i_tri", "i")), "Transformer"),
    "vsource": StoreSpec(2, 8, (("Ysource_r_tri", "r"), ("Ysource_i_tri", "i")), "Vsource", 8),
    "load": StoreSpec(1, 4, (("Yload_r_tri", "r"), ("Yload_i_tri", "i")), "Load", 4),
    "generator": StoreSpec(1, 4, (("Ygen_r_tri", "r"), ("Ygen_i_tri", "i")), "Generator", 4),
    "pvsystem": StoreSpec(1, 4, (("Ypv_r_tri", "r"), ("Ypv_i_tri", "i")), "PVSystem", 4),
    "storage": StoreSpec(1, 4, (("Ystorage_r_tri", "r"), ("Ystorage_i_tri", "i")), "Storage", 4),
}
NODE_FIELDS = ("V_r_init_pu", "V_i_init_pu", "V_r_pu", "V_i_pu")

# Target-independent device definitions needed by deterministic passive-Y
# decoders.  These are carried alongside x_true, never concatenated into the
# learned electrical feature vector, and deliberately exclude every stored Y
# answer field.
PASSIVE_DEFINITION_FIELDS: dict[str, tuple[str, ...]] = {
    "line": (
        "physics_params", "physics_mask", "physics_supported",
        "terminal_kv_base", "system_base_mva",
    ),
    "transformer": (
        "physics_params", "physics_supported", "terminal_kv_base",
        "system_base_mva", "physics_extra_params", "physics_extra_mask",
        "physics_v2_supported",
    ),
    "generator": (
        "physics_params", "physics_mask", "physics_supported",
        "terminal_kv_base", "system_base_mva",
    ),
    "capacitor": (
        "physics_params", "physics_mask", "physics_supported",
        "terminal_kv_base", "system_base_mva",
    ),
    "reactor": (
        "physics_params", "physics_mask", "physics_supported",
        "terminal_kv_base", "system_base_mva",
    ),
    "load": (
        "physics_params", "physics_mask", "physics_supported",
        "terminal_kv_base", "system_base_mva",
    ),
    "pvsystem": (
        "physics_params", "physics_mask", "physics_supported",
        "terminal_kv_base", "system_base_mva",
    ),
    "vsource": (
        "physics_params", "physics_mask", "physics_supported",
        "terminal_kv_base", "system_base_mva",
    ),
    "storage": (
        "physics_params", "physics_mask", "physics_supported",
        "terminal_kv_base", "system_base_mva",
    ),
}


def _passive_definitions(skel, by_field: dict, store: str, n: int) -> dict:
    """Resolve target-independent definition tensors without touching Y fields."""
    definitions = {}
    for name in PASSIVE_DEFINITION_FIELDS.get(store, ()):
        entry = by_field.get((store, name))
        if entry is None:
            continue
        shape = tuple(int(v) for v in entry["shape"])
        if not shape or shape[0] != n:
            raise ValueError(f"{store}.{name} shape {shape}")
        if entry["static"]:
            value = getattr(skel[store], name).reshape(shape).clone()
            definitions[name] = (value, None, shape)
        else:
            definitions[name] = (
                None, (int(entry["offset"]), int(entry["numel"])), shape
            )
    return definitions


def tri_size(dim: int) -> int:
    return dim * (dim + 1) // 2


def tri_rc(dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Row/col index of each packed lower-triangular entry (row-major)."""
    rows = [r for r in range(dim) for _ in range(r + 1)]
    cols = [c for r in range(dim) for c in range(r + 1)]
    return torch.tensor(rows), torch.tensor(cols)


def field_layout(store: str) -> list[tuple[str, int]]:
    """Canonical (tensor name, width) order backing the x_true columns."""
    spec = SPECS[store]
    out = [(f"{name}_feat", tri_size(spec.ydim)) for name, _ in spec.yfields]
    if spec.icomp:
        out += [("Icomp_r_feat", spec.icomp), ("Icomp_i_feat", spec.icomp)]
    for t in range(1, spec.terms + 1):
        out += [(f"I_r_bus{t}_feat", FC), (f"I_i_bus{t}_feat", FC)]
    return out


def store_width(store: str) -> int:
    return sum(w for _, w in field_layout(store))


def y_width(store: str) -> int:
    return len(SPECS[store].yfields) * tri_size(SPECS[store].ydim)


def icomp_width(store: str) -> int:
    return 2 * SPECS[store].icomp


def i_offset(store: str) -> int:
    return y_width(store) + icomp_width(store)


def n_slots(store: str) -> int:
    return SPECS[store].terms * FC


def _slot_maps(edge_index: torch.Tensor, n_comp: int) -> torch.Tensor:
    """[n_comp, FC] node row per slot (-1 inactive) from a bus{t} edge_index."""
    out = torch.full((n_comp, FC), -1, dtype=torch.long)
    comp, node = edge_index[0], edge_index[1]
    counts = torch.bincount(comp, minlength=n_comp)
    if int(counts.max().item() if n_comp else 0) > FC:
        raise ValueError("terminal with more than FC active slots")
    starts = torch.cumsum(counts, 0) - counts
    slots = torch.arange(comp.numel()) - starts[comp]
    out[comp, slots] = node
    return out


def _line_scale_names(triplex: torch.Tensor) -> list[str]:
    return ["TriplexLine" if bool(f) else "Line" for f in triplex]


class FeederCache:
    """Static skeleton, layouts, masks, scales and edges for one feeder store."""

    def __init__(self, feeder_dir: Path, scaler: dict, line_triplex: torch.Tensor | None,
                 dtype: torch.dtype | None = None):
        meta = torch.load(Path(feeder_dir) / "static.pt", map_location="cpu", weights_only=False)
        if meta.get("schema") != 1:
            raise ValueError(f"{feeder_dir}: unsupported scenario schema {meta.get('schema')}")
        if meta.get("basis", "feat") != "feat":
            raise ValueError(
                f"{feeder_dir}: scenario store basis={meta.get('basis')} is not training-ready; "
                "run featurize_scenario_stores.py before PE/training"
            )
        self.name = Path(feeder_dir).name
        self._dir = Path(feeder_dir)
        # Loaded fully into RAM (corpus ~4G): keeping 2000 memmaps open exceeds
        # ulimit -n, and Python 3.14's forkserver cannot pass that many fds to
        # DataLoader workers. Workers fork and share these pages copy-on-write.
        # dtype=None keeps the corpus dtype (float64: machine-precision physics
        # gate); training passes float32 — model and losses run fp32 anyway.
        raw_dyn = np.load(Path(feeder_dir) / "dynamic.npy", allow_pickle=False)
        corpus_dtype = torch.float64 if raw_dyn.dtype == np.float64 else torch.float32
        self.dtype = dtype or corpus_dtype
        self.n_variants = len(meta["variants"])
        skel = meta["skeleton"]
        by_field = {(e["store"], e["field"]): e for e in meta["layout"]}
        # Keep only the small definition slices in fp64 before casting the
        # electrical training row to fp32. Exact passive-Y reconstruction is
        # sensitive to exact device parameters below fp32 spacing.
        definition_dynamic_values: dict[str, dict[str, np.ndarray]] = {}
        if self.dtype == torch.float32 and corpus_dtype == torch.float64:
            for store, names in PASSIVE_DEFINITION_FIELDS.items():
                for name in names:
                    entry = by_field.get((store, name))
                    if entry is None or entry["static"]:
                        continue
                    offset, numel = int(entry["offset"]), int(entry["numel"])
                    definition_dynamic_values.setdefault(store, {})[name] = np.array(
                        raw_dyn[:, offset:offset + numel], dtype=np.float64, copy=True
                    )
            self.dyn = raw_dyn.astype(np.float32)  # halves bulk RAM/collation bytes
        else:
            self.dyn = raw_dyn

        def resolve(store: str, field: str, n: int, w: int):
            """Return (static tensor | None, dynamic (offset, numel) | None).

            A field absent from this feeder's layout is treated as an all-zero
            static fill. Sparser corpora (e.g. a SMART-DS vsource with no Norton
            compensation omits its Icomp block, which is physically zero) then
            load against the full fixed schema; complete corpora are unaffected.
            """
            entry = by_field.get((store, field))
            if entry is None:
                return torch.zeros(n, w, dtype=self.dtype), None
            if entry["shape"][0] != n or int(np.prod(entry["shape"])) != n * w:
                raise ValueError(f"{self.name}: {store}.{field} shape {entry['shape']}")
            if entry["static"]:
                return getattr(skel[store], field).reshape(n, w), None
            return None, (int(entry["offset"]), int(entry["numel"]))

        # node voltages: template + dynamic fill slices
        self.n_node = int(by_field[("node", "V_r_init_pu")]["shape"][0])
        self.v_tmpl = torch.zeros(self.n_node, 4, dtype=self.dtype)  # columns follow NODE_FIELDS
        self.v_dyn: list[tuple[int, tuple[int, int]]] = []
        for col, f in enumerate(NODE_FIELDS):
            stat, dyn = resolve("node", f, self.n_node, 1)
            if stat is not None:
                self.v_tmpl[:, col] = stat.reshape(-1)
            else:
                self.v_dyn.append((col, dyn))

        # component stores: x template + dynamic fills + masks/scales/edges
        self.stores: dict[str, dict] = {}
        for store, spec in SPECS.items():
            layout = field_layout(store)
            if (store, layout[0][0]) not in by_field:
                continue
            n = int(by_field[(store, layout[0][0])]["shape"][0])
            tmpl = torch.zeros(n, store_width(store), dtype=self.dtype)
            dyn_fill: list[tuple[int, int, tuple[int, int]]] = []
            col = 0
            for fname, w in layout:
                stat, dyn = resolve(store, fname, n, w)
                if stat is not None:
                    tmpl[:, col:col + w] = stat
                else:
                    dyn_fill.append((col, w, dyn))
                col += w

            # slot maps -> ONE packed edge store per component store:
            # edge_index [2, E] (comp -> node) + slot id [E] in 0..terms*FC-1.
            slot = torch.stack(
                [_slot_maps(skel[(store, f"bus{t}", "node")].edge_index, n)
                 for t in range(1, spec.terms + 1)], dim=1)          # [n, T, FC]
            cond = slot >= 0                                          # [n, T, FC]
            comp, term, sl = cond.nonzero(as_tuple=True)
            edge_index = torch.stack([comp, slot[comp, term, sl]])
            edge_slot = term * FC + sl
            # active masks: Y entries live where both stacked conductors are active
            rows, cols_ = tri_rc(spec.ydim)
            stacked = cond.any(1) if store == "line" else cond.reshape(n, -1)
            act_tri = stacked[:, rows] & stacked[:, cols_]            # [n, tri]
            # Icomp and I parts use (all real slots | all imag slots) and
            # (I_r, I_i) per terminal respectively.
            act_icomp = (
                cond.reshape(n, -1).repeat(1, 2)[:, : 2 * spec.icomp]
                if spec.icomp else torch.zeros(n, 0, dtype=torch.bool)
            )
            act_i = torch.cat([cond[:, t].repeat(1, 2) for t in range(spec.terms)], dim=1)
            act = torch.cat([act_tri.repeat(1, len(spec.yfields)), act_icomp, act_i], dim=1)

            # per-entry asinh scales (line family varies per row)
            fams = _line_scale_names(line_triplex) if store == "line" and line_triplex is not None \
                else [spec.family] * n
            diag = rows == cols_
            y_cols = []
            for _, part in spec.yfields:
                y_cols.append(torch.where(
                    diag,
                    torch.tensor([scaler["admittance"][f][f"Y_{part}_diag_scale"] for f in fams], dtype=self.dtype).unsqueeze(1),
                    torch.tensor([scaler["admittance"][f][f"Y_{part}_offdiag_scale"] for f in fams], dtype=self.dtype).unsqueeze(1),
                ))
            i_scale = torch.tensor([scaler["current"][f]["I_scale"] for f in fams], dtype=self.dtype)
            icomp_cols = i_scale.unsqueeze(1).expand(n, 2 * spec.icomp) if spec.icomp else torch.zeros(n, 0, dtype=self.dtype)
            i_cols = i_scale.unsqueeze(1).expand(n, 2 * FC * spec.terms)
            scale = torch.cat(y_cols + [icomp_cols, i_cols], dim=1)

            try:
                definitions = _passive_definitions(skel, by_field, store, n)
            except ValueError as exc:
                raise ValueError(f"{self.name}: {exc}") from exc

            self.stores[store] = dict(
                n=n, tmpl=tmpl, dyn=dyn_fill, act=act, scale=scale,
                edge_index=edge_index, edge_slot=edge_slot,
                definitions=definitions,
                definition_values=definition_dynamic_values.get(store, {}),
            )

        v0 = self._voltages(0)
        self.ground = (v0[:, :2].abs() < 1e-9).all(1)
        self.kcl_mask = ~self.ground

        # Node structural PE: RWSE (return probabilities of a k-step random walk
        # on the node co-incidence graph) + log degree. Sign/basis-free (unlike
        # LapPE) and static per feeder.
        # --- source / slack nodes (always; PF boundary condition) ---
        # The source terminal is the PF boundary condition.  Its three solved
        # phase voltages are measurements, not prediction targets, in every
        # downstream task.  Keep this explicit instead of inferring it again in
        # masking.py; filtering ground also excludes a grounded source neutral.
        src_nodes = torch.zeros(self.n_node, dtype=torch.bool)
        if "vsource" in self.stores:
            ei = self.stores["vsource"]["edge_index"]
            sl = self.stores["vsource"]["edge_slot"]
            src_nodes[ei[1][sl < FC]] = True          # bus1 slots only
        self.slack = src_nodes & ~self.ground
        if int(self.slack.sum()) != 3:
            raise ValueError(
                f"{self.name}: expected exactly three non-ground slack phase nodes "
                f"at vsource bus1, found {int(self.slack.sum())}"
            )

        # --- positional encoding: cached per topology; the dense O(n^2/n^3)
        # formulation below is exact but only tractable for the ~100-node
        # minimal_component feeders. Large SMART-DS feeders (thousands of nodes)
        # use an equivalent sparse computation (_positional_encoding_sparse).
        cached = self._load_pe_cache()
        if cached is not None:
            self.pe, self.depth, self.depth_raw = cached
        elif self.n_node <= PE_DENSE_MAX:
            adj = torch.zeros(self.n_node, self.n_node, dtype=torch.float64)
            for info in self.stores.values():
                inc = torch.zeros(info["n"], self.n_node, dtype=torch.float64)
                inc[info["edge_index"][0], info["edge_index"][1]] = 1.0
                adj += inc.T @ inc
            adj.fill_diagonal_(0.0)
            adj = (adj > 0).double()
            deg = adj.sum(1)
            p_step = adj / deg.clamp(min=1).unsqueeze(1)
            diags, p_k = [], p_step
            for _ in range(PE_K):
                diags.append(torch.diagonal(p_k))
                p_k = p_k @ p_step
            pe_legacy = torch.cat([torch.stack(diags, 1), torch.log1p(deg).unsqueeze(1)], 1)
            wmat = torch.full((self.n_node, self.n_node), float("inf"), dtype=torch.float64)
            row0 = torch.from_numpy(np.array(self.dyn[0], dtype=np.float64))
            for store, info in self.stores.items():
                ny = y_width(store)
                x0 = info["tmpl"].double().clone()        # variant-0 values: template
                for col, w, (off, numel) in info["dyn"]:  # + dynamic fills
                    x0[:, col:col + w] = row0[off:off + numel].reshape(info["n"], w)
                y_pu = torch.sinh(x0[:, :ny]) * (info["scale"][:, :ny].double() + EPS)
                rows, cols_ = tri_rc(SPECS[store].ydim)
                nf = len(SPECS[store].yfields)
                tri = tri_size(SPECS[store].ydim)
                diag_vals = y_pu.reshape(-1, nf, tri)[:, :, rows == cols_]
                zc = 1.0 / diag_vals.abs().flatten(1).max(dim=1).values.clamp(min=1e-9)
                ei = info["edge_index"]
                for c in range(info["n"]):
                    nodes = ei[1][ei[0] == c].unique()
                    a, b = torch.meshgrid(nodes, nodes, indexing="ij")
                    cur = wmat[a, b]
                    wmat[a, b] = torch.minimum(cur, torch.full_like(cur, float(zc[c])))
            wmat.fill_diagonal_(float("inf"))
            hops = torch.full((self.n_node,), 99.0)
            zdist = torch.full((self.n_node,), float("inf"), dtype=torch.float64)
            hops[src_nodes] = 0.0
            zdist[src_nodes] = 0.0
            finite_adj = ~torch.isinf(wmat)
            for _ in range(self.n_node):                  # Bellman-Ford (tiny graphs)
                nh = torch.where(finite_adj, hops.unsqueeze(1) + 1, torch.full_like(wmat, torch.inf).float()).min(0).values
                nz = torch.where(finite_adj, zdist.unsqueeze(1) + wmat, torch.inf).min(0).values
                hops = torch.minimum(hops, nh)
                zdist = torch.minimum(zdist, nz)
            self.depth_raw = hops
            self.depth = hops.clamp(max=20)
            zdist = torch.where(torch.isinf(zdist), torch.full_like(zdist, 1e6), zdist)
            self.pe = torch.cat([pe_legacy, (self.depth / 10).unsqueeze(1),
                                 torch.log1p(zdist).float().unsqueeze(1)], 1).to(self.dtype)
            self._save_pe_cache()
        else:
            self._positional_encoding_sparse(src_nodes)
            self._save_pe_cache()

    def _load_pe_cache(self):
        f = self._dir / f"pe_cache_v{PE_CACHE_VERSION}.pt"
        if not f.is_file():
            return None
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            return None
        if int(d.get("n_node", -1)) != self.n_node:
            return None
        return d["pe"].to(self.dtype), d["depth"], d["depth_raw"]

    def _save_pe_cache(self):
        f = self._dir / f"pe_cache_v{PE_CACHE_VERSION}.pt"
        tmp = f.with_suffix(f".tmp.{os.getpid()}")
        try:
            torch.save({"n_node": self.n_node, "pe": self.pe,
                        "depth": self.depth, "depth_raw": self.depth_raw}, tmp)
            os.replace(tmp, f)
        except Exception:
            Path(tmp).unlink(missing_ok=True)

    def _positional_encoding_sparse(self, src_nodes: torch.Tensor) -> None:
        """Sparse equivalent of the dense PE, for large (SMART-DS) feeders.

        Same quantities as the dense path — RWSE diag(P^k) + log degree, plus BFS
        hop depth and Dijkstra path-impedance from the source — but built on the
        sparse co-incidence graph so it is O(n+e) shortest paths instead of the
        dense O(n^2) matrices / O(n) Bellman-Ford iterations.
        """
        import scipy.sparse as sp
        from scipy.sparse.csgraph import dijkstra

        n = self.n_node
        row0 = np.array(self.dyn[0], dtype=np.float64)
        ar, ac = [], []            # binary co-incidence edges
        wr, wc, wv = [], [], []    # weighted edges (weight = component zc)
        for store, info in self.stores.items():
            ny = y_width(store)
            x0 = info["tmpl"].double().clone().numpy()
            for col, w, (off, numel) in info["dyn"]:
                x0[:, col:col + w] = row0[off:off + numel].reshape(info["n"], w)
            y_pu = np.sinh(x0[:, :ny]) * (info["scale"][:, :ny].double().numpy() + float(EPS))
            rows, cols_ = tri_rc(SPECS[store].ydim)
            nf = len(SPECS[store].yfields)
            tri = tri_size(SPECS[store].ydim)
            dmask = (rows == cols_).numpy()
            diag_vals = y_pu.reshape(-1, nf, tri)[:, :, dmask]
            zc = 1.0 / np.clip(np.abs(diag_vals).reshape(info["n"], -1).max(1), 1e-9, None)
            ei = info["edge_index"].numpy()
            comp, node = ei[0], ei[1]
            order = np.argsort(comp, kind="stable")
            comp_s, node_s = comp[order], node[order]
            bounds = np.searchsorted(comp_s, np.arange(info["n"] + 1))
            for c in range(info["n"]):
                nodes = np.unique(node_s[bounds[c]:bounds[c + 1]])
                if len(nodes) < 2:
                    continue
                a = np.repeat(nodes, len(nodes))
                b = np.tile(nodes, len(nodes))
                m = a != b
                ar.append(a[m]); ac.append(b[m])
                wr.append(a[m]); wc.append(b[m]); wv.append(np.full(int(m.sum()), zc[c]))
        ar, ac = np.concatenate(ar), np.concatenate(ac)
        A = sp.coo_matrix((np.ones(len(ar)), (ar, ac)), shape=(n, n)).tocsr()
        A.data[:] = 1.0
        deg = np.asarray(A.sum(1)).ravel()
        # RWSE diag(P^k), P row-normalized co-incidence
        Pn = (sp.diags(1.0 / np.clip(deg, 1, None)) @ A).tocsr()
        Pk = Pn.copy()
        diags = []
        for k in range(PE_K):
            diags.append(Pk.diagonal())
            if k < PE_K - 1:
                Pk = (Pk @ Pn).tocsr()
        pe_legacy = np.concatenate([np.stack(diags, 1), np.log1p(deg)[:, None]], 1)
        # weighted graph: min zc over shared components (min-reduce duplicates)
        wr, wc, wv = np.concatenate(wr), np.concatenate(wc), np.concatenate(wv)
        key = wr.astype(np.int64) * n + wc
        o = np.lexsort((wv, key))
        key_s, wv_s, wr_s, wc_s = key[o], wv[o], wr[o], wc[o]
        first = np.concatenate([[True], key_s[1:] != key_s[:-1]])
        W = sp.coo_matrix((wv_s[first], (wr_s[first], wc_s[first])), shape=(n, n)).tocsr()
        src_idx = np.where(src_nodes.numpy())[0]
        hops = dijkstra(A, directed=False, indices=src_idx, unweighted=True, min_only=True)
        zdist = dijkstra(W, directed=False, indices=src_idx, min_only=True)
        hops = np.where(np.isinf(hops), 99.0, hops)
        zdist = np.where(np.isinf(zdist), 1e6, zdist)
        self.depth_raw = torch.from_numpy(hops).float()
        self.depth = self.depth_raw.clamp(max=20)
        pe = np.concatenate([pe_legacy, (self.depth.numpy() / 10)[:, None],
                             np.log1p(zdist)[:, None]], 1)
        self.pe = torch.from_numpy(pe).to(self.dtype)

    def _voltages(self, variant: int) -> torch.Tensor:
        row = self.dyn[variant]
        v = self.v_tmpl.clone()
        for col, (off, numel) in self.v_dyn:
            v[:, col] = torch.from_numpy(row[off:off + numel])
        return v

    def sample(self, variant: int) -> HeteroData:
        row = self.dyn[variant]  # read-only source; every use copies into new tensors
        data = HeteroData()

        v = self.v_tmpl.clone()
        for col, (off, numel) in self.v_dyn:
            v[:, col] = torch.from_numpy(row[off:off + numel])
        nd = data["node"]
        nd.num_nodes = self.n_node
        nd.v_init = v[:, :2].contiguous()
        nd.dv = (v[:, 2:] - v[:, :2]).contiguous()
        nd.ground = self.ground
        nd.slack = self.slack
        nd.kcl_mask = self.kcl_mask
        nd.pe = self.pe
        nd.depth = self.depth
        nd.depth_raw = self.depth_raw

        for store in SPECS:
            st = data[store]
            info = self.stores.get(store)
            if info is None:
                st.num_nodes = 0
                w = store_width(store)
                st.x_true = torch.zeros(0, w, dtype=self.dtype)
                st.act = torch.zeros(0, w, dtype=torch.bool)
                st.scale = torch.zeros(0, w, dtype=self.dtype)
                # Adapters can register zero-row attributes that must exist on
                # every store for heterogeneous PyG collation, including
                # feeders where this component family is absent.
                for name, value in getattr(
                    self, "empty_derived_definitions", {}
                ).get(store, {}).items():
                    setattr(st, name, value)
            else:
                x = info["tmpl"].clone()
                for col, w, (off, numel) in info["dyn"]:
                    x[:, col:col + w] = torch.from_numpy(row[off:off + numel]).reshape(info["n"], w)
                st.num_nodes = info["n"]
                st.x_true = x
                st.act = info["act"]
                st.scale = info["scale"]
                for name, (static, dynamic, shape) in info["definitions"].items():
                    if static is not None:
                        value = static.clone()
                    elif name in info.get("definition_values", {}):
                        value = torch.from_numpy(
                            info["definition_values"][name][variant]
                        ).reshape(shape)
                    else:
                        offset, numel = dynamic
                        value = torch.from_numpy(row[offset:offset + numel]).reshape(shape)
                    setattr(st, name, value)
                # Downstream adapters may derive target-independent tensors
                # from device definitions after the cache is constructed. A
                # derived value is either topology-static or indexed by
                # scenario variant; it is kept separate from x_true and from
                # the raw scenario-store offsets.
                for name, (static, per_variant) in info.get(
                    "derived_definitions", {}
                ).items():
                    value = static if static is not None else per_variant[variant]
                    setattr(st, name, value)
            es = data[(store, "conn", "node")]
            if info is None:
                es.edge_index = torch.zeros(2, 0, dtype=torch.long)
                es.slot = torch.zeros(0, dtype=torch.long)
            else:
                es.edge_index = info["edge_index"]
                es.slot = info["edge_slot"]
        return data


# ── corpus discovery, splits, line-family flags ─────────────────────────────

def discover_feeders(root: Path) -> list[Path]:
    # recursive: nested corpora (SMART-DS region/substation/feeder) keep stores
    # below the first level; flat corpora are unaffected.
    return sorted(p.parent for p in Path(root).rglob("static.pt")
                  if (p.parent / "dynamic.npy").is_file())


def split_of(name: str, train_frac: float, val_frac: float) -> str:
    h = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < train_frac:
        return "train"
    if h < train_frac + val_frac:
        return "val_unseen"
    return "test"


def split_of_group(
    group: str, train_frac: float, val_frac: float, seed: int = 42,
) -> str:
    """Audited split rule for content-equivalent topology groups."""
    h = int(hashlib.md5(f"{seed}{group}".encode()).hexdigest(), 16) / float(1 << 128)
    if h < train_frac:
        return "train"
    if h < train_frac + val_frac:
        return "val_unseen"
    return "test"


def topology_group(feeder: Path, root: Path, groups: dict[str, str]) -> str:
    """Resolve a feeder to its cross-corpus structural fingerprint."""
    relative = feeder.relative_to(root)
    key = relative.as_posix()
    # The validated v4 synthetic export retains the v3 topology/name set while
    # changing scenarios and metadata. Reuse the audited structural IDs.
    if key in groups:
        return groups[key]
    if key.startswith("minimal_component_v4/"):
        legacy = "minimal_component/" + key.split("/", 1)[1]
        if legacy in groups:
            return groups[legacy]
    return "name:" + feeder.name


def line_triplex_flags(root: Path, feeders: list[Path], cache_path: Path) -> dict[str, torch.Tensor]:
    """Per-feeder is_triplex_line row flags, read once from the baseline JSONs.

    The flags select the Line vs TriplexLine scaler family per row; they are not
    stored in the HeteroData. Consolidated into ONE cache file (HPC: no file sprawl).
    """
    if cache_path.is_file():
        flags = torch.load(cache_path, weights_only=True)
        if all(f.name in flags for f in feeders):
            return flags
    flags = {}
    for f in feeders:
        # nested corpora keep json/<source-relative-path>/master.json
        jp = Path(root) / "json" / f.relative_to(root) / "master.json"
        if not jp.is_file():
            jp = Path(root) / "json" / f.name / "master.json"
        payload = json.loads(jp.read_text())
        lines = payload.get("Line") or []
        flags[f.name] = torch.tensor([bool(e.get("is_triplex_line", False)) for e in lines])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(flags, cache_path)
    return flags


class ScenarioDataset(torch.utils.data.Dataset):
    """Flat (feeder, variant) index over a list of FeederCache objects.

    Masking is applied online in __getitem__ (see masking.py); the RNG is seeded
    per (seed, epoch, item) so eval sets are reproducible with a fixed epoch.
    """

    def __init__(self, caches: list[FeederCache], items: list[tuple[int, int]],
                 mask_cfg: dict, seed: int, train: bool):
        self.caches, self.items, self.mask_cfg = caches, items, mask_cfg
        self.seed, self.train, self.epoch = seed, train, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch if self.train else 0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> HeteroData:
        from .masking import apply_masks

        fi, variant = self.items[idx]
        data = self.caches[fi].sample(variant)
        rng = np.random.default_rng((self.seed, self.epoch, idx))
        apply_masks(data, self.mask_cfg, rng)
        return data


def build_datasets(cfg: dict, mask_cfg: dict, seed: int, limit: int | None = None):
    """Returns (train, val_seen, val_unseen, test) ScenarioDatasets."""
    from .masking import validate_mask_cfg

    validate_mask_cfg(mask_cfg)
    root = Path(cfg["root"])
    feeders = discover_feeders(root)
    # Fractional diagnostics need reproducible component-family slices, not
    # whichever feeder happens to sort first. Offset is deliberately applied
    # before the CLI limit and defaults to zero for all existing configs.
    offset = int(cfg.get("feeder_offset", 0))
    if offset < 0 or offset >= len(feeders):
        raise ValueError(f"data.feeder_offset={offset} outside [0, {len(feeders)})")
    feeders = feeders[offset:]
    if limit is not None:
        feeders = feeders[:limit]
    scaler = json.loads((root / "feature_scaler.json").read_text())
    topology_manifest = cfg.get("topology_manifest")
    topology_groups = {}
    if topology_manifest:
        payload = json.loads(Path(topology_manifest).read_text())
        topology_groups = payload.get("feeders", {})
        if not topology_groups:
            raise ValueError(f"empty topology manifest: {topology_manifest}")
    # New corpora may deliberately use one physical Line coordinate for every
    # line row. In that case no baseline-JSON triplex lookup is needed (and the
    # feature corpus can remain self-contained).
    if "TriplexLine" in scaler.get("admittance", {}):
        flags = line_triplex_flags(
            root, feeders, Path(cfg["cache_dir"]) / "line_triplex.pt"
        )
    else:
        flags = {}

    dtype = torch.float32 if cfg.get("cast_float32", True) else None
    caches, splits = [], []
    for f in feeders:
        cache = FeederCache(f, scaler, flags.get(f.name), dtype=dtype)
        if topology_groups:
            group = topology_group(f, root, topology_groups)
            if cfg.get("require_topology_manifest_coverage", False) and group.startswith(
                "name:"
            ):
                raise ValueError(f"topology manifest has no entry for {f.relative_to(root)}")
            cache.split_group = group
            split = split_of_group(
                group, cfg["train_frac"], cfg["val_frac"],
                int(cfg.get("split_seed", 42)),
            )
        else:
            cache.split_group = "name:" + f.name
            split = split_of(f.name, cfg["train_frac"], cfg["val_frac"])
        caches.append(cache)
        splits.append(split)

    n_train_var = int(cfg["train_variants"])
    items = {"train": [], "seen": [], "val_unseen": [], "test": []}
    for i, (c, s) in enumerate(zip(caches, splits)):
        if s == "train":
            items["train"] += [(i, v) for v in range(min(n_train_var, c.n_variants))]
            items["seen"] += [(i, v) for v in range(n_train_var, min(n_train_var + cfg["eval_variants"], c.n_variants))]
        else:
            items[s] += [(i, v) for v in range(min(cfg["eval_variants"], c.n_variants))]

    mk = lambda key, train: ScenarioDataset(caches, items[key], mask_cfg, seed, train)
    return mk("train", True), mk("seen", False), mk("val_unseen", False), mk("test", False)

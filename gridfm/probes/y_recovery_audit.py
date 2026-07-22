#!/usr/bin/env python3
"""Audit the constant-Y assumption behind T22 and repeat recovery honestly.

For every selected feeder and target element this probe:

* checks bitwise equality and relative drift of the target primitive Y;
* separately checks passive-network Y and all exported component Y;
* reports the singular spectrum/effective rank of the stacked voltage matrix;
* fits one common target Y for K snapshots and reports its equation residual;
* evaluates held-out voltage after actually replacing the held-out target Y.

The legacy T22 held-out number is retained for diagnosis.  When target Y varies,
that path adds ``Yrec - Y_variant0`` to a Y-bus containing ``Y_holdout`` and
therefore does not replace the held-out target.  ``v_skill_correct`` uses the
proper ``Yrec - Y_holdout`` patch.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# This audit worktree lives beside, rather than beneath, ``Training_new``.
# Seed the canonical legacy contract path because ``gridfm.legacy`` derives its
# root from the worktree location and imports those modules by their old names.
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")

from Datakit.core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, FC, node_count
from gridfm.probes.y_recovery_probe import elem_edges, terminal_currents_all
from gridfm.tests.test_ladder import build_ybus, _y_full


DEFAULT_ROOT = "/kfs2/projects/gogpt/Ebadmus/training_data"
DEFAULT_CORPORA = ["SMART-DS_1000", "new_dss_data", "dss_data", "minimal_component"]
PASSIVE_STORES = ("line", "transformer", "capacitor", "reactor")


def _complex_y(d, store: str) -> np.ndarray | None:
    if store not in d.node_types:
        return None
    prefix, nterm, _ = STORES[store]
    field = f"{prefix}_r_pu"
    if field not in d[store] or d[store][field].shape[0] == 0:
        return None
    yr, yi = _y_full(d[store], prefix, nterm * FC, torch.float64, store=store)
    return yr.numpy() + 1j * yi.numpy()


def _y_state(d, stores) -> dict[str, np.ndarray]:
    return {store: y for store in stores if (y := _complex_y(d, store)) is not None}


def _state_equal(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    return a.keys() == b.keys() and all(
        a[key].shape == b[key].shape and np.array_equal(a[key], b[key]) for key in a
    )


def _state_rel_drift(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    if a.keys() != b.keys() or any(a[key].shape != b[key].shape for key in a):
        return float("inf")
    num = sum(float(np.abs(a[key] - b[key]).sum()) for key in a)
    den = sum(float(np.abs(a[key]).sum()) for key in a) + 1e-300
    return num / den


def _state_digest(state: dict[str, np.ndarray]) -> str:
    h = hashlib.blake2b(digest_size=10)
    for key in sorted(state):
        a = np.ascontiguousarray(state[key])
        h.update(key.encode())
        h.update(str(a.shape).encode())
        h.update(a.view(np.uint8))
    return h.hexdigest()


def _effective_ranks(s: np.ndarray, shape: tuple[int, int]) -> dict[str, float | int]:
    if not len(s) or s[0] == 0:
        return {"rank_lapack": 0, "rank_rel1e8": 0, "rank_rel1e10": 0, "stable_rank": 0.0}
    tol = np.finfo(float).eps * max(shape) * s[0]
    return {
        "rank_lapack": int(np.sum(s > tol)),
        "rank_rel1e8": int(np.sum(s / s[0] > 1e-8)),
        "rank_rel1e10": int(np.sum(s / s[0] > 1e-10)),
        "stable_rank": float(np.sum(s * s) / (s[0] * s[0])),
    }


def _pivot_order(vectors: np.ndarray) -> list[int]:
    """Greedy D-optimal ordering using only candidate terminal voltages.

    At each step it maximizes ``logdet(G + vᴴv + λI)``. A tiny
    scale-relative ridge makes this a stable rank-revealing design before the
    Gram matrix becomes full rank.
    No current or target-Y value enters the selection.
    """
    vectors = np.asarray(vectors, dtype=np.complex128)
    if vectors.ndim != 2 or not len(vectors):
        return []
    dim = vectors.shape[1]
    scale = max(float(np.max(np.sum(np.abs(vectors) ** 2, axis=1))), 1e-300)
    ridge = scale * 1e-14
    gram = np.zeros((dim, dim), dtype=np.complex128)
    remaining = list(range(len(vectors)))
    order: list[int] = []
    eye = np.eye(dim, dtype=np.complex128)
    while remaining:
        scores = []
        for idx in remaining:
            v = vectors[idx]
            candidate = gram + np.outer(v.conj(), v) + ridge * eye
            sign, logdet = np.linalg.slogdet(candidate)
            score = float(logdet) if np.real(sign) > 0 else float("-inf")
            scores.append((score, -idx, idx))
        _, _, chosen = max(scores)
        v = vectors[chosen]
        gram += np.outer(v.conj(), v)
        order.append(chosen)
        remaining.remove(chosen)
    return order


def _structured_recovery(v: np.ndarray, b: np.ndarray, kind: str) -> tuple[np.ndarray, float]:
    """Fit a complex-symmetric or zero-row-sum symmetric primitive Y.

    ``v`` and ``b`` are K-by-N terminal voltage/current matrices satisfying
    ``b[k] = Y @ v[k]``. The Laplacian basis enforces reciprocity and KCL/gauge
    structure by representing Y as sums of ``(e_i-e_j)(e_i-e_j)^T``.
    """
    k_count, dim = v.shape
    bases: list[np.ndarray] = []
    if kind in {"symmetric", "laplacian"}:
        params = ([(p, q) for p in range(dim) for q in range(p, dim)]
                  if kind == "symmetric" else
                  [(p, q) for p in range(dim) for q in range(p + 1, dim)])
        for p, q in params:
            basis = np.zeros((dim, dim), dtype=np.complex128)
            if kind == "symmetric":
                basis[p, q] = 1.0
                basis[q, p] = 1.0
            else:
                basis[p, p] = basis[q, q] = 1.0
                basis[p, q] = basis[q, p] = -1.0
            bases.append(basis)
    elif kind == "two_terminal":
        if dim % 2:
            raise ValueError("two-terminal basis needs equal terminal widths")
        width = dim // 2
        # Reciprocal pi block: Y=[[A,B],[B,A]], with A and B symmetric.
        for block in ("diag", "cross"):
            for p in range(width):
                for q in range(p, width):
                    basis = np.zeros((dim, dim), dtype=np.complex128)
                    pairs = ((p, q), (p + width, q + width)) if block == "diag" else (
                        (p, q + width), (p + width, q)
                    )
                    for i, j in pairs:
                        basis[i, j] = 1.0
                        basis[j, i] = 1.0
                    bases.append(basis)
    else:
        raise ValueError(kind)
    design = np.zeros((k_count * dim, len(bases)), dtype=np.complex128)
    target = b.reshape(-1)
    for k in range(k_count):
        for col, basis in enumerate(bases):
            design[k * dim:(k + 1) * dim, col] = basis @ v[k]
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    y = sum((value * basis for value, basis in zip(coef, bases)),
            start=np.zeros((dim, dim), dtype=np.complex128))
    fit = float(np.abs((v @ y.T) - b).sum() / (np.abs(b).sum() + 1e-300))
    return y, fit


def _voltage_skill(dh, s_target: str, c: int, cols: list[int], edges, yrec, y_reference) -> float:
    """Held-out direct solve after adding yrec-y_reference to its existing Ybus."""
    n = node_count(dh)
    ybus, rhs = build_ybus(dh, n)
    dy = np.zeros_like(yrec)
    dy[np.ix_(cols, cols)] = yrec[np.ix_(cols, cols)] - y_reference[np.ix_(cols, cols)]
    nodes = {a: nd_i for a, nd_i in edges}
    for a in cols:
        for b in cols:
            ybus[nodes[a], nodes[b]] += dy[a, b]

    vt = dh["node"].V_r_pu.double().numpy() + 1j * dh["node"].V_i_pu.double().numpy()
    vi = dh["node"].V_r_init_pu.double().numpy() + 1j * dh["node"].V_i_init_pu.double().numpy()
    visible = np.zeros(n, dtype=bool)
    visible[0] = True
    rel = ("vsource", "bus1", "node")
    if rel in dh.edge_types and dh[rel].edge_index.numel():
        visible[dh[rel].edge_index[1].numpy()] = True
    free = np.where(~visible)[0]
    fixed = np.where(visible)[0]
    bfree = rhs[free] - ybus[np.ix_(free, fixed)] @ vt[fixed]
    try:
        solved = np.linalg.solve(ybus[np.ix_(free, free)], bfree)
    except np.linalg.LinAlgError:
        # Some synthetic connection families deliberately expose a floating
        # common/zero-sequence mode. Their local excitation rank and Y fit are
        # still auditable even though a unique full-network voltage solve is
        # undefined.
        return float("nan")
    return float(np.abs(solved - vt[free]).sum() / (np.abs(vt[free] - vi[free]).sum() + 1e-300))


def audit_target(
    fdir: str,
    s_target: str,
    k_list: tuple[int, ...],
    holdout: int,
    selection: str = "prefix",
    candidates: int | None = None,
    component_index: int = 0,
):
    scenarios = FeederScenarios(fdir)
    candidates = int(candidates or max(k_list))
    if candidates < max(k_list):
        raise ValueError(f"candidates={candidates} must be >= max K={max(k_list)}")
    if holdout < candidates:
        raise ValueError("holdout must be outside the candidate-selection pool")
    need = max(candidates, holdout + 1)
    if len(scenarios) < need:
        raise ValueError(f"need {need} variants, found {len(scenarios)}")

    d0 = scenarios[0]
    y0_all = _complex_y(d0, s_target)
    if y0_all is None:
        return None
    c = component_index if component_index >= 0 else y0_all.shape[0] + component_index
    if c < 0 or c >= y0_all.shape[0]:
        raise IndexError(
            f"component_index={component_index} outside {s_target} count={y0_all.shape[0]}"
        )
    edges = elem_edges(d0, s_target, c)
    if not edges:
        return None
    cols = [a for a, _ in edges]
    dim = y0_all.shape[-1]
    y0 = y0_all[c]
    y0_block = y0[np.ix_(cols, cols)]
    target_y_sv = np.linalg.svd(y0_block, compute_uv=False)
    target_y_ranks = _effective_ranks(target_y_sv, y0_block.shape)
    identity = np.eye(len(cols), dtype=np.complex128)
    _, symmetric_structure_rel = _structured_recovery(
        identity, y0_block.T, "symmetric"
    )
    _, laplacian_structure_rel = _structured_recovery(
        identity, y0_block.T, "laplacian"
    )
    if s_target == "line" and len(cols) % 2 == 0:
        _, two_terminal_structure_rel = _structured_recovery(
            identity, y0_block.T, "two_terminal"
        )
    else:
        two_terminal_structure_rel = None

    passive0 = _y_state(d0, PASSIVE_STORES)
    all0 = _y_state(d0, STORES)
    passive_exact = True
    all_exact = True
    passive_drift_max = 0.0
    all_drift_max = 0.0
    target_exact = True
    target_drift = []
    target_hashes = []
    rows_a = {a: [] for a in cols}
    rows_b = {a: [] for a in cols}
    oracle_num_by_snapshot = []
    oracle_den_by_snapshot = []

    for k in range(candidates):
        d = scenarios[k]
        yk_all = _complex_y(d, s_target)
        if yk_all is None or yk_all.shape != y0_all.shape:
            raise ValueError(f"variant {k}: target Y shape changed")
        yk = yk_all[c]
        target_exact &= np.array_equal(yk, y0)
        target_drift.append(float(np.abs(yk[np.ix_(cols, cols)] - y0[np.ix_(cols, cols)]).sum()
                                  / (np.abs(y0[np.ix_(cols, cols)]).sum() + 1e-300)))
        target_hashes.append(_state_digest({s_target: yk}))

        passive = _y_state(d, PASSIVE_STORES)
        all_y = _y_state(d, STORES)
        passive_exact &= _state_equal(passive0, passive)
        all_exact &= _state_equal(all0, all_y)
        passive_drift_max = max(passive_drift_max, _state_rel_drift(passive0, passive))
        all_drift_max = max(all_drift_max, _state_rel_drift(all0, all_y))

        per, total, _ = terminal_currents_all(d, node_count(d0))
        i_target, _, vloc, _, _ = per[s_target]
        vk = vloc[c, cols]
        oracle_num_k = 0.0
        oracle_den_k = 0.0
        for a, nd_i in edges:
            ielem = -(total[nd_i] - i_target[c, a])
            rows_a[a].append(vk)
            rows_b[a].append(ielem)
            oracle = yk[a, cols] @ vk
            oracle_num_k += abs(oracle - ielem)
            oracle_den_k += abs(ielem)
        oracle_num_by_snapshot.append(oracle_num_k)
        oracle_den_by_snapshot.append(oracle_den_k)

    dh = scenarios[holdout]
    yh_all = _complex_y(dh, s_target)
    if yh_all is None or yh_all.shape != y0_all.shape:
        raise ValueError("held-out target Y shape changed")
    yh = yh_all[c]
    holdout_drift = float(np.abs(yh[np.ix_(cols, cols)] - y0[np.ix_(cols, cols)]).sum()
                          / (np.abs(y0[np.ix_(cols, cols)]).sum() + 1e-300))

    excitation_all = np.asarray(rows_a[cols[0]])
    order = list(range(candidates)) if selection == "prefix" else _pivot_order(excitation_all)
    records = []
    for k in k_list:
        chosen = order[:k]
        yrec = np.zeros((dim, dim), dtype=np.complex128)
        fit_num = 0.0
        fit_den = 0.0
        # Every terminal row sees the same stacked local-voltage matrix.
        excitation = excitation_all[chosen]
        singular_values = np.linalg.svd(excitation, compute_uv=False)
        ranks = _effective_ranks(singular_values, excitation.shape)
        for a in cols:
            amat = np.asarray(rows_a[a])[chosen]
            bvec = np.asarray(rows_b[a])[chosen]
            x, *_ = np.linalg.lstsq(amat, bvec, rcond=None)
            yrec[a, cols] = x
            fit_num += float(np.abs(amat @ x - bvec).sum())
            fit_den += float(np.abs(bvec).sum())

        bmat = np.column_stack([np.asarray(rows_b[a])[chosen] for a in cols])
        ys, ys_fit = _structured_recovery(excitation, bmat, "symmetric")
        yl, yl_fit = _structured_recovery(excitation, bmat, "laplacian")
        if s_target == "line" and len(cols) % 2 == 0:
            yt, yt_fit = _structured_recovery(excitation, bmat, "two_terminal")
        else:
            yt = None
            yt_fit = None

        block = np.ix_(cols, cols)
        yerr0 = float(np.abs(yrec[block] - y0[block]).sum() / (np.abs(y0[block]).sum() + 1e-300))
        record = {
            "K": k,
            "selection": selection,
            "selected_indices": chosen,
            "selected_variant_ids": [int(scenarios.variant_ids[j]) for j in chosen],
            "connected_slots": len(cols),
            "singular_values": [float(x) for x in singular_values],
            "condition_nonzero": float(singular_values[0] / singular_values[-1])
            if len(singular_values) and singular_values[-1] > 0 else float("inf"),
            **ranks,
            "target_y_exact": bool(all(np.array_equal(_complex_y(scenarios[j], s_target)[c], y0)
                                        for j in chosen)),
            "target_y_drift_max": max(target_drift[j] for j in chosen),
            "common_fit_relres": fit_num / (fit_den + 1e-300),
            "oracle_dynamic_y_relres": sum(oracle_num_by_snapshot[j] for j in chosen)
            / (sum(oracle_den_by_snapshot[j] for j in chosen) + 1e-300),
            "y_relerr_vs_variant0": yerr0,
            "symmetric_fit_relres": ys_fit,
            "symmetric_y_relerr": float(
                np.abs(ys - y0_block).sum() / (np.abs(y0_block).sum() + 1e-300)
            ),
            "laplacian_fit_relres": yl_fit,
            "laplacian_y_relerr": float(
                np.abs(yl - y0_block).sum() / (np.abs(y0_block).sum() + 1e-300)
            ),
            "two_terminal_fit_relres": yt_fit,
            "two_terminal_y_relerr": (float(
                np.abs(yt - y0_block).sum() / (np.abs(y0_block).sum() + 1e-300)
            ) if yt is not None else None),
            "v_skill_legacy_patch": _voltage_skill(dh, s_target, c, cols, edges, yrec, y0),
            "v_skill_correct_patch": _voltage_skill(dh, s_target, c, cols, edges, yrec, yh),
        }
        records.append(record)

    meta = {
        "feeder": fdir,
        "target": s_target,
        "component_index": c,
        "connected_columns": cols,
        "variant_ids": [int(scenarios.variant_ids[k]) for k in range(candidates)],
        "candidate_variant_ids": [int(scenarios.variant_ids[k]) for k in range(candidates)],
        "selection": selection,
        "target_y_exact_all": target_exact,
        "target_y_drift_max": max(target_drift),
        "target_y_unique_hashes": len(set(target_hashes)),
        "passive_y_exact_all": passive_exact,
        "passive_y_drift_max": passive_drift_max,
        "all_component_y_exact_all": all_exact,
        "all_component_y_drift_max": all_drift_max,
        "holdout": holdout,
        "holdout_target_y_drift": holdout_drift,
        "target_symmetry_rel": float(
            np.abs(y0_block - y0_block.T).sum() / (np.abs(y0_block).sum() + 1e-300)
        ),
        "target_rowsum_rel": float(
            np.abs(y0_block.sum(axis=1)).sum() / (np.abs(y0_block).sum() + 1e-300)
        ),
        "target_y_singular_values": [float(value) for value in target_y_sv],
        "target_y_rank_lapack": target_y_ranks["rank_lapack"],
        "target_y_rank_rel1e8": target_y_ranks["rank_rel1e8"],
        "symmetric_structure_rel": symmetric_structure_rel,
        "laplacian_structure_rel": laplacian_structure_rel,
        "two_terminal_structure_rel": two_terminal_structure_rel,
        "variant0_passive_y_hash": _state_digest(passive0),
        "variant0_all_y_hash": _state_digest(all0),
    }
    return {"meta": meta, "records": records}


def _selected_feeders(
    root: str,
    corpora: list[str],
    per_corpus: int,
    feeder_pattern: str | None = None,
):
    for corpus in corpora:
        stores = sorted(glob.glob(os.path.join(root, corpus, "*", "static.pt")))
        if feeder_pattern:
            stores = [
                path for path in stores
                if fnmatch.fnmatch(os.path.basename(os.path.dirname(path)), feeder_pattern)
            ]
        step = max(1, len(stores) // per_corpus)
        picked = 0
        for path in stores[::step]:
            if picked >= per_corpus:
                break
            yield corpus, os.path.dirname(path)
            picked += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--corpora", nargs="+", default=DEFAULT_CORPORA)
    parser.add_argument("--per-corpus", type=int, default=3)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--holdout", type=int, default=90)
    parser.add_argument("--selection", choices=("prefix", "pivot"), default="prefix")
    parser.add_argument("--candidates", type=int, default=None,
                        help="candidate snapshots available to the selector; defaults to max K")
    parser.add_argument("--component-index", type=int, default=0,
                        help="component row to audit; use -1 for the last rendered target")
    parser.add_argument("--feeder-pattern", default=None,
                        help="optional shell-style pattern matched against feeder directory names")
    parser.add_argument("--targets", nargs="+", choices=tuple(PASSIVE_STORES),
                        default=["line", "transformer"],
                        help="component families to audit")
    parser.add_argument("--output")
    args = parser.parse_args()
    k_list = tuple(sorted(set(args.k)))

    payload = {"root": args.root, "k": list(k_list), "holdout": args.holdout,
               "selection": args.selection, "candidates": args.candidates,
               "feeder_pattern": args.feeder_pattern, "targets": args.targets,
               "audits": [], "failures": []}
    header = ("corpus/feeder", "target", "K", "Ysame", "rank", "smin/smax", "Ydrift", "fit_res", "Yerr0", "Vcorrect")
    print(f"{header[0]:44s} {header[1]:11s} {header[2]:>3s} {header[3]:>5s} {header[4]:>5s} "
          f"{header[5]:>10s} {header[6]:>10s} {header[7]:>10s} {header[8]:>10s} {header[9]:>10s}")
    for corpus, fdir in _selected_feeders(
        args.root, args.corpora, args.per_corpus, args.feeder_pattern
    ):
        name = f"{corpus}/{os.path.basename(fdir)}"
        for target in args.targets:
            try:
                audit = audit_target(fdir, target, k_list, args.holdout,
                                     selection=args.selection, candidates=args.candidates,
                                     component_index=args.component_index)
            except Exception as exc:
                failure = {"feeder": fdir, "target": target, "error": f"{type(exc).__name__}: {exc}"}
                payload["failures"].append(failure)
                print(f"{name[:44]:44s} {target:11s} FAIL {failure['error']}", flush=True)
                continue
            if audit is None:
                continue
            audit["meta"]["corpus"] = corpus
            payload["audits"].append(audit)
            for row in audit["records"]:
                sv = row["singular_values"]
                ratio = sv[-1] / sv[0] if sv and sv[0] else 0.0
                print(f"{name[:44]:44s} {target:11s} {row['K']:3d} "
                      f"{str(row['target_y_exact']):>5s} {row['rank_rel1e8']:5d} {ratio:10.2e} "
                      f"{row['target_y_drift_max']:10.2e} {row['common_fit_relres']:10.2e} "
                      f"{row['y_relerr_vs_variant0']:10.2e} {row['v_skill_correct_patch']:10.2e}", flush=True)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    changed = sum(not a["meta"]["target_y_exact_all"] for a in payload["audits"])
    print(f"SUMMARY targets={len(payload['audits'])} target_Y_changed={changed} failures={len(payload['failures'])}")
    if not payload["audits"]:
        print("ERROR: selection produced no auditable target components")
    return 1 if payload["failures"] or not payload["audits"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

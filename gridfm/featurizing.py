#!/usr/bin/env python3
"""Deterministic ML feature transforms for GridFM artifacts.

Physical `*_pu` tensors are the source of truth and are used for all physics.
`*_feat` tensors are a deterministic, invertible VIEW of the pu tensors,
conditioned for training and made comparable ACROSS FEEDERS by a single
global scaler.

Design (per-feeder scaling would be a foundational-model bug: the same
physical quantity must encode to the same feature in every feeder):

    Current    I_*_bus*_feat = asinh((I_bus_pu + Icomp_pu) / (s_I[family] + eps))
    where Icomp_pu uses the terminal slice for active devices. The feature keeps the
        existing I_*_bus*_feat tensor name because that is the PT schema.
    s_I[family] = P95(|I_bus_pu + Icomp_pu|) over the whole training corpus,
         per component family.

  Admittance Y_feat = asinh(Y_pu / (s_Y[family,part,band] + eps))
         s_Y pooled globally per (family, real/imag, diag/offdiag).

The pu tensors themselves are made cross-feeder comparable upstream by using a
single fixed system MVA base for every feeder (see GLOBAL_BASE_MVA in
build_master_json.py). Without that, no scaler could make currents comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parent
sys.path[:0] = [str(MODULE_ROOT), str(PROJECT_ROOT)]

# eps guards divide-by-zero only; nonzero is required for robust asinh scaling.
FEAT_EPS = 1e-12
# Floor for fitted scales: below any real pu current/admittance magnitude, so it
# never rescales real data — it only prevents a divide-by-zero for a degenerate
# family that has no nonzero samples at all.
SCALE_FLOOR = 1e-12
DEFAULT_SCALER_PATH = Path("data") / "feature_scaler.json"

# Canonical component spec, shared by fit / apply / validation.
#   json_key, n_term (bus{t} terminals), y_fields, y_dim (Y matrix order),
#   icomp (Icomp slot count; 0 = passive)
SPECS: dict[str, dict[str, Any]] = {
    "line": {"json_key": "Line", "n_term": 2, "y_fields": ("Yline_r_tri", "Yline_i_tri"), "y_dim": 8, "icomp": 0},
    "capacitor": {"json_key": "Capacitor", "n_term": 2, "y_fields": ("Ycap_r_tri", "Ycap_i_tri"), "y_dim": 8, "icomp": 0},
    "reactor": {"json_key": "Reactor", "n_term": 2, "y_fields": ("Yreactor_r_tri", "Yreactor_i_tri"), "y_dim": 8, "icomp": 0},
    "transformer": {"json_key": "Transformer", "n_term": 3, "y_fields": ("Yxfmr_r_tri", "Yxfmr_i_tri"), "y_dim": 12, "icomp": 0},
    "vsource": {"json_key": "Vsource", "n_term": 2, "y_fields": ("Ysource_r_tri", "Ysource_i_tri"), "y_dim": 8, "icomp": 8},
    "load": {"json_key": "Load", "n_term": 1, "y_fields": ("Yload_r_tri", "Yload_i_tri"), "y_dim": 4, "icomp": 4},
    "generator": {"json_key": "Generator", "n_term": 1, "y_fields": ("Ygen_r_tri", "Ygen_i_tri"), "y_dim": 4, "icomp": 4},
    "pvsystem": {"json_key": "PVSystem", "n_term": 1, "y_fields": ("Ypv_r_tri", "Ypv_i_tri"), "y_dim": 4, "icomp": 4},
    "storage": {"json_key": "Storage", "n_term": 1, "y_fields": ("Ystorage_r_tri", "Ystorage_i_tri"), "y_dim": 4, "icomp": 4},
}

# Definition-only metadata is not a learned electrical target. Preserve it
# through pu -> feature conversion so exact passive-device decoders remain
# available without reading any stored Y answer.
PASSIVE_DEFINITION_FIELDS: dict[str, tuple[str, ...]] = {
    "line": (
        "physics_params", "physics_mask", "physics_supported",
        "physics_family_code", "physics_reason_code", "physics_schema_version",
        "terminal_kv_base", "system_base_mva",
    ),
    "transformer": (
        "physics_params", "physics_mask", "physics_supported",
        "physics_schema_version", "terminal_kv_base", "system_base_mva",
        "physics_extra_params", "physics_extra_mask", "physics_v2_supported",
    ),
    "generator": (
        "physics_params", "physics_mask", "physics_supported",
        "physics_schema_version", "terminal_kv_base", "system_base_mva",
    ),
    "capacitor": (
        "physics_params", "physics_mask", "physics_supported",
        "physics_schema_version", "terminal_kv_base", "system_base_mva",
    ),
    "reactor": (
        "physics_params", "physics_mask", "physics_supported",
        "physics_schema_version", "terminal_kv_base", "system_base_mva",
    ),
    "load": (
        "physics_params", "physics_mask", "physics_supported",
        "physics_schema_version", "terminal_kv_base", "system_base_mva",
    ),
    "pvsystem": (
        "physics_params", "physics_mask", "physics_supported",
        "physics_schema_version", "terminal_kv_base", "system_base_mva",
    ),
    "vsource": (
        "physics_params", "physics_mask", "physics_supported",
        "physics_schema_version", "terminal_kv_base", "system_base_mva",
    ),
}

# Current datakit pu tensors are full matrices; DG_FM_Training consumes packed
# lower triangles in these exact field names and dimensions.
SCENARIO_Y_FIELDS: dict[str, tuple[int, tuple[tuple[str, str, str], ...]]] = {
    "line": (4, (
        ("Ys_r_pu", "Ys_r_tri_feat", "r"),
        ("Ys_i_pu", "Ys_i_tri_feat", "i"),
        ("Yh_i_pu", "Yh_i_tri_feat", "i"),
    )),
    "capacitor": (8, (
        ("Ycap_r_pu", "Ycap_r_tri_feat", "r"),
        ("Ycap_i_pu", "Ycap_i_tri_feat", "i"),
    )),
    "reactor": (8, (
        ("Yreactor_r_pu", "Yreactor_r_tri_feat", "r"),
        ("Yreactor_i_pu", "Yreactor_i_tri_feat", "i"),
    )),
    "transformer": (12, (
        ("Yxfmr_r_pu", "Yxfmr_r_tri_feat", "r"),
        ("Yxfmr_i_pu", "Yxfmr_i_tri_feat", "i"),
    )),
    "vsource": (8, (
        ("Ysource_r_pu", "Ysource_r_tri_feat", "r"),
        ("Ysource_i_pu", "Ysource_i_tri_feat", "i"),
    )),
    "load": (4, (
        ("Yload_r_pu", "Yload_r_tri_feat", "r"),
        ("Yload_i_pu", "Yload_i_tri_feat", "i"),
    )),
    "generator": (4, (
        ("Ygen_r_pu", "Ygen_r_tri_feat", "r"),
        ("Ygen_i_pu", "Ygen_i_tri_feat", "i"),
    )),
    "pvsystem": (4, (
        ("Ypv_r_pu", "Ypv_r_tri_feat", "r"),
        ("Ypv_i_pu", "Ypv_i_tri_feat", "i"),
    )),
    "storage": (4, (
        ("Ystorage_r_pu", "Ystorage_r_tri_feat", "r"),
        ("Ystorage_i_pu", "Ystorage_i_tri_feat", "i"),
    )),
}


def _as_float_list(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for value in values:
        try:
            out.append(float(value))
        except Exception:
            out.append(0.0)
    return out


def _slice_with_zero_pad(values: list[float], start: int, width: int) -> list[float]:
    if width <= 0:
        return []
    slice_vals = values[start : start + width]
    if len(slice_vals) < width:
        slice_vals = slice_vals + [0.0] * (width - len(slice_vals))
    return slice_vals


def _terminal_current_with_icomp_pu(entry: dict[str, Any], spec: dict[str, Any], term: int, part: str) -> list[float]:
    i_vals = _as_float_list(entry.get(f"I_{part}_bus{term}_pu"))
    if not i_vals:
        return []

    if int(spec.get("icomp", 0)) <= 0:
        return i_vals

    icomp = _as_float_list(entry.get(f"Icomp_{part}_pu"))
    start = (term - 1) * len(i_vals)
    ic_slice = _slice_with_zero_pad(icomp, start, len(i_vals))
    return [i_vals[idx] + ic_slice[idx] for idx in range(len(i_vals))]


def quantile_95(values: list[float]) -> float:
    """P95 of |values|, floored only to guard divide-by-zero."""
    if not values:
        return SCALE_FLOOR
    seq = sorted(float(v) for v in values)
    idx = int(math.ceil(0.95 * (len(seq) - 1)))
    return max(float(seq[idx]), SCALE_FLOOR)


def bus_name_from_ref(ref: Any) -> str:
    if not isinstance(ref, str):
        return ""
    return ref.split(".", 1)[0].strip().lower()


def bus_triplex_map(payload: dict[str, Any]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for bus in payload.get("Bus", []) if isinstance(payload.get("Bus"), list) else []:
        if not isinstance(bus, dict):
            continue
        name = str(bus.get("Name", "")).strip().lower()
        if name:
            out[name] = bool(bus.get("is_triplex_bus", False))
    return out


def component_family(comp: str, entry: dict[str, Any], bus_triplex: dict[str, bool]) -> str:
    if comp == "line":
        return "TriplexLine" if bool(entry.get("is_triplex_line", False)) else "Line"
    if comp == "load":
        return "TriplexLoad" if bus_triplex.get(bus_name_from_ref(entry.get("Bus1")), False) else "Load"
    return {
        "transformer": "Transformer", "vsource": "Vsource", "generator": "Generator",
        "pvsystem": "PVSystem", "storage": "Storage", "capacitor": "Capacitor", "reactor": "Reactor",
    }.get(comp, comp)


def tri_diag_indices(dim: int) -> set[int]:
    out: set[int] = set()
    idx = 0
    for r in range(dim):
        for c in range(r + 1):
            if r == c:
                out.add(idx)
            idx += 1
    return out


# ── scaler fit ──────────────────────────────────────────────────────────────

def new_scaler_metadata(epsilon: float = FEAT_EPS) -> dict[str, Any]:
    return {"epsilon": float(epsilon), "current": {}, "admittance": {}}


def _collect_scaler_samples(payload, specs):
    """Per-payload |I_pu + Icomp_pu| and |Y_pu| samples."""
    triplex = bus_triplex_map(payload)
    current_samples: dict[str, list[float]] = {}
    y_samples: dict[tuple[str, str, str], list[float]] = {}
    for comp, spec in specs.items():
        entries = payload.get(spec["json_key"], [])
        if not isinstance(entries, list):
            continue
        diag_idx = tri_diag_indices(int(spec["y_dim"]))
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            family = component_family(comp, entry, triplex)
            cur = current_samples.setdefault(family, [])
            for t in range(1, int(spec["n_term"]) + 1):
                ir = _terminal_current_with_icomp_pu(entry, spec, t, "r")
                ii = _terminal_current_with_icomp_pu(entry, spec, t, "i")
                cur.extend(math.hypot(float(a), float(b)) for a, b in zip(ir, ii))
            for y_field in spec["y_fields"]:
                vals = entry.get(f"{y_field}_pu", [])
                if not isinstance(vals, list):
                    continue
                part = "r" if "_r_" in y_field else "i"
                for k, v in enumerate(vals):
                    band = "diag" if k in diag_idx else "offdiag"
                    y_samples.setdefault((family, part, band), []).append(abs(float(v)))
    return current_samples, y_samples


def _finalize_scaler_meta(current_samples, y_samples, epsilon, training_split, fit_source):
    """Percentile scales from pooled samples (fast; shared by serial+parallel)."""
    meta = new_scaler_metadata(epsilon=epsilon)
    for family, vals in current_samples.items():
        nonzero_vals = [float(v) for v in vals if float(v) > 0.0]
        has_nonzero = bool(nonzero_vals)
        scale = quantile_95(nonzero_vals if has_nonzero else vals)
        if has_nonzero and scale <= SCALE_FLOOR:
            raise ValueError(
                f"Current scale collapse for active family {family}: "
                f"P95={scale:.3g} with nonzero data (max |I_pu|={max(vals):.3g})."
            )
        meta["current"][family] = {"I_scale": scale, "transform": "asinh"}
    for (family, part, band), vals in y_samples.items():
        nonzero_vals = [float(v) for v in vals if float(v) > 0.0]
        has_nonzero = bool(nonzero_vals)
        scale = quantile_95(nonzero_vals if has_nonzero else vals)
        if has_nonzero and scale <= SCALE_FLOOR:
            raise ValueError(
                f"Admittance scale collapse for active bucket {family}/{part}/{band}: "
                f"P95={scale:.3g} with nonzero data (max |Y_pu|={max(vals):.3g})."
            )
        fam = meta["admittance"].setdefault(family, {"transform": "asinh"})
        fam[f"Y_{part}_{band}_scale"] = scale
    if training_split is not None:
        meta["training_split"] = training_split
    if fit_source is not None:
        meta["fit_source"] = fit_source
    return meta


def fit_scaler_from_payloads(
    payloads: list[dict[str, Any]],
    specs: dict[str, dict[str, Any]] | None = None,
    epsilon: float = FEAT_EPS,
    training_split: str | None = None,
    fit_source: list[str] | None = None,
) -> dict[str, Any]:
    """Fit ONE global scaler by pooling |I_pu + Icomp_pu| and |Y_pu| across ALL payloads.

    Pass the whole training corpus here (never a single feeder) — that is the
    entire point of the artifact.
    """
    specs = specs or SPECS
    current_samples: dict[str, list[float]] = {}
    y_samples: dict[tuple[str, str, str], list[float]] = {}
    for payload in payloads:
        cur, ys = _collect_scaler_samples(payload, specs)
        for fam, vals in cur.items():
            current_samples.setdefault(fam, []).extend(vals)
        for key, vals in ys.items():
            y_samples.setdefault(key, []).extend(vals)
    return _finalize_scaler_meta(
        current_samples, y_samples, epsilon, training_split, fit_source
    )


def _scaler_worker(path_str):
    payload = json.loads(Path(path_str).read_text())
    return _collect_scaler_samples(payload, SPECS)


def fit_scaler_from_json_paths(
    json_paths: list[Path],
    specs: dict[str, dict[str, Any]] | None = None,
    epsilon: float = FEAT_EPS,
    training_split: str = "train",
    workers: int = 1,
) -> dict[str, Any]:
    """Fit the scaler from JSON paths. workers>1 parallelizes the per-feeder Y·V
    sample collection (the bottleneck on large SMART-DS feeders)."""
    specs = specs or SPECS
    if workers and workers > 1 and len(json_paths) > 1 and specs is SPECS:
        # Parallelize the per-feeder Y·V sample collection (the bottleneck on
        # large SMART-DS feeders). fork avoids re-importing the caller's __main__.
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor
        try:
            _ctx = _mp.get_context("fork")
        except ValueError:
            _ctx = None
        current_samples: dict[str, list[float]] = {}
        y_samples: dict[tuple[str, str, str], list[float]] = {}
        with ProcessPoolExecutor(max_workers=workers, mp_context=_ctx) as pool:
            for cur, ys in pool.map(_scaler_worker, [str(p) for p in json_paths],
                                    chunksize=4):
                for fam, vals in cur.items():
                    current_samples.setdefault(fam, []).extend(vals)
                for key, vals in ys.items():
                    y_samples.setdefault(key, []).extend(vals)
        return _finalize_scaler_meta(
            current_samples, y_samples, epsilon, training_split,
            [str(jp) for jp in json_paths],
        )
    payloads = [json.loads(Path(jp).read_text()) for jp in json_paths]
    return fit_scaler_from_payloads(
        payloads, specs, epsilon=epsilon,
        training_split=training_split, fit_source=[str(jp) for jp in json_paths],
    )


def current_scale(meta: dict[str, Any], family: str) -> float:
    cur = meta.get("current", {})
    if family not in cur:
        raise KeyError(f"Missing current scaler for family {family}")
    if "I_scale" not in cur[family]:
        raise KeyError(f"Missing I_scale for current family {family}")
    return float(cur[family]["I_scale"])


def admittance_scale(meta: dict[str, Any], family: str, part: str, band: str) -> float:
    adm = meta.get("admittance", {})
    if family not in adm:
        raise KeyError(f"Missing admittance scaler for family {family}")
    key = f"Y_{part}_{band}_scale"
    if key not in adm[family]:
        raise KeyError(f"Missing {key} for admittance family {family}")
    return float(adm[family][key])


# ── transforms (all invertible) ─────────────────────────────────────────────

def admittance_to_feat(y_pu: float, scale: float, epsilon: float) -> float:
    return math.asinh(float(y_pu) / (float(scale) + float(epsilon)))


def admittance_from_feat(y_feat: float, scale: float, epsilon: float) -> float:
    return math.sinh(float(y_feat)) * (float(scale) + float(epsilon))


def current_to_feat(i_pu: float, scale: float, epsilon: float) -> float:
    return math.asinh(float(i_pu) / (float(scale) + float(epsilon)))


def current_from_feat(i_feat: float, scale: float, epsilon: float) -> float:
    return math.sinh(float(i_feat)) * (float(scale) + float(epsilon))


def _require_list(entry: dict[str, Any], field: str, where: str) -> list[Any]:
    values = entry.get(field)
    if not isinstance(values, list):
        raise KeyError(f"{where}: missing required list field {field}")
    return values


def _strip_existing_feat_fields(payload: dict[str, Any], specs: dict[str, dict[str, Any]]) -> None:
    # Prevent stale/legacy features from surviving partial updates.
    nodes = payload.get("Node", [])
    if isinstance(nodes, list):
        for node in nodes:
            if isinstance(node, dict):
                for key in list(node.keys()):
                    if key.endswith("_feat"):
                        del node[key]

    for spec in specs.values():
        entries = payload.get(spec["json_key"], [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for key in list(entry.keys()):
                if key.endswith("_feat"):
                    del entry[key]


def validate_asinh_scaler_metadata(meta: dict[str, Any]) -> None:
    eps = float(meta.get("epsilon", 0.0))
    if eps <= 0.0:
        raise ValueError("FeatureScaler epsilon must be > 0 for robust asinh scaling")

    cur = meta.get("current", {})
    adm = meta.get("admittance", {})
    if not isinstance(cur, dict) or not isinstance(adm, dict):
        raise ValueError("FeatureScaler must contain current and admittance mappings")

    for family, cfg in cur.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"current scaler for {family} must be an object")
        if cfg.get("transform") != "asinh":
            raise ValueError(f"current scaler for {family} must declare transform='asinh'")
        if float(cfg.get("I_scale", 0.0)) <= 0.0:
            raise ValueError(f"current scaler for {family} has non-positive I_scale")

    for family, cfg in adm.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"admittance scaler for {family} must be an object")
        if cfg.get("transform") != "asinh":
            raise ValueError(f"admittance scaler for {family} must declare transform='asinh'")


# ── apply: stamp *_feat into a payload from a FIXED global scaler ────────────

def apply_features_to_payload(
    payload: dict[str, Any],
    scaler: dict[str, Any],
    specs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write deterministic `*_feat` fields into `payload` using a fixed scaler.

    Reads only the physical `*_pu` fields; never refits. The scaler must be the
    single global artifact fit over the training corpus.
    """
    specs = specs or SPECS
    validate_asinh_scaler_metadata(scaler)
    eps = float(scaler.get("epsilon", FEAT_EPS))

    _strip_existing_feat_fields(payload, specs)

    # Node voltage features are computed on demand by downstream consumers
    # (e.g., .pt builder) from V_*_pu and V_*_init_pu and are not persisted.

    triplex = bus_triplex_map(payload)
    for comp, spec in specs.items():
        entries = payload.get(spec["json_key"], [])
        if not isinstance(entries, list):
            continue
        diag_idx = tri_diag_indices(int(spec["y_dim"]))
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            where = f"{spec['json_key']} {entry.get('Name', '<unnamed>')}"
            family = component_family(comp, entry, triplex)
            i_scale = current_scale(scaler, family)

            for t in range(1, int(spec["n_term"]) + 1):
                for part in ("r", "i"):
                    i_with_icomp_vals = _terminal_current_with_icomp_pu(entry, spec, t, part)
                    if not i_with_icomp_vals:
                        raise KeyError(
                            f"{where}: missing required list field I_{part}_bus{t}_pu"
                        )
                    entry[f"I_{part}_bus{t}_feat"] = [
                        current_to_feat(float(v), i_scale, eps) for v in i_with_icomp_vals
                    ]

            for y_field in spec["y_fields"]:
                vals = _require_list(entry, f"{y_field}_pu", where)
                part = "r" if "_r_" in y_field else "i"
                out = []
                for idx, v in enumerate(vals):
                    band = "diag" if idx in diag_idx else "offdiag"
                    out.append(admittance_to_feat(float(v), admittance_scale(scaler, family, part, band), eps))
                entry[f"{y_field}_feat"] = out

    return payload


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    seq = sorted(float(v) for v in values)
    idx = int(math.ceil(float(q) * (len(seq) - 1)))
    return float(seq[idx])


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "max": max(values) if values else 0.0,
    }


def summarize_feature_magnitudes(payloads: list[dict[str, Any]], specs: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, float]]:
    specs = specs or SPECS
    i_abs: list[float] = []
    y_abs: list[float] = []

    for payload in payloads:
        for comp, spec in specs.items():
            entries = payload.get(spec["json_key"], [])
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                for t in range(1, int(spec["n_term"]) + 1):
                    ir = entry.get(f"I_r_bus{t}_feat", [])
                    ii = entry.get(f"I_i_bus{t}_feat", [])
                    if isinstance(ir, list) and isinstance(ii, list):
                        i_abs.extend(math.hypot(float(a), float(b)) for a, b in zip(ir, ii))
                for y_field in spec["y_fields"]:
                    y_vals = entry.get(f"{y_field}_feat", [])
                    if isinstance(y_vals, list):
                        y_abs.extend(abs(float(v)) for v in y_vals)

    return {"I_feat": _summary(i_abs), "Y_feat": _summary(y_abs)}


def summarize_feature_magnitudes_from_json_paths(json_paths: list[Path]) -> dict[str, dict[str, float]]:
    payloads = [json.loads(Path(jp).read_text()) for jp in json_paths]
    return summarize_feature_magnitudes(payloads)


# ── scaler artifact IO ──────────────────────────────────────────────────────

def load_scaler_metadata(path: Path, epsilon_default: float = FEAT_EPS) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Scaler metadata must be an object: {path}")
    meta = new_scaler_metadata(epsilon=float(raw.get("epsilon", epsilon_default)))
    if isinstance(raw.get("current"), dict):
        meta["current"] = raw["current"]
    if isinstance(raw.get("admittance"), dict):
        meta["admittance"] = raw["admittance"]
    for key in ("training_split", "fit_source"):
        if key in raw:
            meta[key] = raw[key]
    return meta


def save_scaler_metadata(meta: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))


# ── CLI: fit/apply global scaler in-place on JSON corpus ───────────────────

def _discover_jsons(search_root: Path) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for p in sorted(search_root.glob("**/json/master*.json")):
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def _atomic_write_json(path: Path, payload: dict[str, Any], indent: int) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=None if indent == 0 else indent) + "\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _fit_scaler(train_paths: list[Path], split_label: str) -> dict[str, Any]:
    print(f"Fitting global scaler over {len(train_paths)} feeder(s) [{split_label}] ...")
    scaler = fit_scaler_from_json_paths(train_paths, training_split=split_label)
    n_cur = len(scaler.get("current", {}))
    n_adm = len(scaler.get("admittance", {}))
    print(f"  fit {n_cur} current families, {n_adm} admittance families")
    return scaler


def _apply_scaler(all_paths: list[Path], scaler: dict[str, Any], indent: int) -> tuple[list[Path], list[tuple[Path, str]]]:
    ok_paths: list[Path] = []
    bad: list[tuple[Path, str]] = []
    for p in all_paths:
        try:
            payload = json.loads(p.read_text())
            apply_features_to_payload(payload, scaler)
            payload["FeatureScaler"] = scaler
            _atomic_write_json(p, payload, indent)
            ok_paths.append(p)
        except Exception as exc:  # noqa: BLE001
            bad.append((p, str(exc)))
    return ok_paths, bad


def _print_feature_distribution_stats(json_paths: list[Path]) -> None:
    stats = summarize_feature_magnitudes_from_json_paths(json_paths)
    i_stats = stats.get("I_feat", {})
    y_stats = stats.get("Y_feat", {})
    print("Feature magnitude summary (absolute values):")
    print(
        "  |I_feat|: "
        f"count={int(i_stats.get('count', 0))} "
        f"p50={i_stats.get('p50', 0.0):.6g} "
        f"p95={i_stats.get('p95', 0.0):.6g} "
        f"p99={i_stats.get('p99', 0.0):.6g} "
        f"max={i_stats.get('max', 0.0):.6g}"
    )
    print(
        "  |Y_feat|: "
        f"count={int(y_stats.get('count', 0))} "
        f"p50={y_stats.get('p50', 0.0):.6g} "
        f"p95={y_stats.get('p95', 0.0):.6g} "
        f"p99={y_stats.get('p99', 0.0):.6g} "
        f"max={y_stats.get('max', 0.0):.6g}"
    )


def _top_feature_outliers_from_json_paths(
    json_paths: list[Path],
    scaler: dict[str, Any],
    top_k: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    i_rows: list[dict[str, Any]] = []
    y_rows: list[dict[str, Any]] = []

    for jp in json_paths:
        payload = json.loads(Path(jp).read_text())
        triplex = bus_triplex_map(payload)
        for comp, spec in SPECS.items():
            entries = payload.get(spec["json_key"], [])
            if not isinstance(entries, list):
                continue
            diag_idx = tri_diag_indices(int(spec["y_dim"]))
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                family = component_family(comp, entry, triplex)
                i_scale = current_scale(scaler, family)

                for t in range(1, int(spec["n_term"]) + 1):
                    for part in ("r", "i"):
                        feat_field = f"I_{part}_bus{t}_feat"
                        pu_vals = _terminal_current_with_icomp_pu(entry, spec, t, part)
                        feat_vals = entry.get(feat_field, [])
                        if not pu_vals or not isinstance(feat_vals, list):
                            continue
                        for idx, (pu_v, feat_v) in enumerate(zip(pu_vals, feat_vals)):
                            i_rows.append(
                                {
                                    "abs_feat": abs(float(feat_v)),
                                    "family": family,
                                    "field": f"{feat_field}[{idx}]",
                                    "pu": float(pu_v),
                                    "scale": float(i_scale),
                                    "file": str(jp),
                                }
                            )

                for y_field in spec["y_fields"]:
                    pu_field = f"{y_field}_pu"
                    feat_field = f"{y_field}_feat"
                    pu_vals = entry.get(pu_field, [])
                    feat_vals = entry.get(feat_field, [])
                    if not isinstance(pu_vals, list) or not isinstance(feat_vals, list):
                        continue
                    part = "r" if "_r_" in y_field else "i"
                    for idx, (pu_v, feat_v) in enumerate(zip(pu_vals, feat_vals)):
                        band = "diag" if idx in diag_idx else "offdiag"
                        y_scale = admittance_scale(scaler, family, part, band)
                        y_rows.append(
                            {
                                "abs_feat": abs(float(feat_v)),
                                "family": family,
                                "field": f"{feat_field}[{idx}]",
                                "pu": float(pu_v),
                                "scale": float(y_scale),
                                "file": str(jp),
                            }
                        )

    i_rows.sort(key=lambda row: row["abs_feat"], reverse=True)
    y_rows.sort(key=lambda row: row["abs_feat"], reverse=True)
    return i_rows[:top_k], y_rows[:top_k]


def _print_top_feature_outliers(json_paths: list[Path], scaler: dict[str, Any], top_k: int = 20) -> None:
    top_i, top_y = _top_feature_outliers_from_json_paths(json_paths, scaler, top_k=top_k)

    print(f"Top {top_k} largest |I_feat| entries:")
    if not top_i:
        print("  <none>")
    for idx, row in enumerate(top_i, 1):
        print(
            f"  {idx:2d}. |I_feat|={row['abs_feat']:.6g} "
            f"family={row['family']} field={row['field']} "
            f"pu={row['pu']:.6g} scale={row['scale']:.6g} file={row['file']}"
        )

    print(f"Top {top_k} largest |Y_feat| entries:")
    if not top_y:
        print("  <none>")
    for idx, row in enumerate(top_y, 1):
        print(
            f"  {idx:2d}. |Y_feat|={row['abs_feat']:.6g} "
            f"family={row['family']} field={row['field']} "
            f"pu={row['pu']:.6g} scale={row['scale']:.6g} file={row['file']}"
        )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fit global feature scaler and stamp *_feat into JSONs.")
    ap.add_argument("--search-root", type=Path, default=Path("data"),
                    help="Root to discover json/master*.json to apply features to.")
    ap.add_argument("--train-glob", type=str, default=None,
                    help="Glob (relative to CWD) selecting fit/train jsons. Default: all discovered.")
    ap.add_argument("--scaler", type=Path, default=DEFAULT_SCALER_PATH,
                    help="Path to write (or, with --apply-only, read) scaler artifact.")
    ap.add_argument("--apply-only", action="store_true",
                    help="Skip fitting; load existing scaler and only stamp features.")
    ap.add_argument("--fit-only", action="store_true", help="Fit and save scaler; do not stamp features.")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent for rewritten files (0 = compact).")
    return ap.parse_args()


def _scenario_baseline_json_path(
    root: Path, feeder_dir: Path, source_root: Path | None = None,
) -> Path:
    rel = feeder_dir.relative_to(source_root or root)
    nested = root / "json" / rel / "master.json"
    if nested.is_file():
        return nested
    flat = root / "json" / feeder_dir.name / "master.json"
    if flat.is_file():
        return flat
    raise FileNotFoundError(f"baseline master.json not found for {feeder_dir}")


def _scenario_families_by_store(payload: dict[str, Any]) -> dict[str, list[str]]:
    from datakit.core.hetero_graph import COMPONENTS

    triplex = bus_triplex_map(payload)
    out: dict[str, list[str]] = {}
    for store_name, spec in COMPONENTS.items():
        entries = payload.get(spec["json_key"], [])
        if not isinstance(entries, list):
            continue
        out[store_name] = [
            component_family(store_name, entry, triplex)
            for entry in entries
            if isinstance(entry, dict)
        ]
    return out


def _scenario_unified_families(raw_data) -> dict[str, list[str]]:
    """One physical coordinate per component family; no baseline JSON needed."""
    out = {}
    for store_name, (dim, y_fields) in SCENARIO_Y_FIELDS.items():
        if store_name not in raw_data.node_types:
            continue
        first = getattr(raw_data[store_name], y_fields[0][0], None)
        if not hasattr(first, "shape"):
            continue
        family = "Line" if store_name == "line" else SPECS[store_name]["json_key"]
        out[store_name] = [family] * int(first.shape[0])
    return out


def _scenario_copy_node_and_edges(src, dst) -> None:
    import torch

    for field in src["node"].keys():
        value = getattr(src["node"], field)
        setattr(dst["node"], field, value.clone() if torch.is_tensor(value) else value)
    for rel in src.edge_types:
        dst[rel].edge_index = src[rel].edge_index


def _scenario_build_feat_sample(raw_data, scaler: dict[str, Any], families: dict[str, list[str]]):
    import torch
    from torch_geometric.data import HeteroData

    from datakit.core.hetero_graph import COMPONENTS, FIXED_CONDUCTORS

    feat_data = HeteroData()
    _scenario_copy_node_and_edges(raw_data, feat_data)
    eps = float(scaler.get("epsilon", FEAT_EPS))

    for store_name, spec in COMPONENTS.items():
        if store_name not in raw_data.node_types:
            continue
        src = raw_data[store_name]
        if not src.keys():
            continue
        dst = feat_data[store_name]
        row_families = families.get(store_name, [])
        dim, y_fields = SCENARIO_Y_FIELDS[store_name]
        n_rows = int(getattr(src, y_fields[0][0]).shape[0])
        if row_families and len(row_families) != n_rows:
            raise ValueError(f"{store_name}: family rows {len(row_families)} != tensor rows {n_rows}")
        if not row_families and n_rows:
            raise ValueError(f"{store_name}: missing family labels for {n_rows} rows")
        rows, cols = torch.tril_indices(dim, dim)
        diag = rows == cols
        for pu_field, feat_field, part in y_fields:
            full = getattr(src, pu_field).reshape(n_rows, dim, dim)
            packed = full[:, rows, cols]
            out = torch.empty_like(packed)
            for row, family in enumerate(row_families):
                scales = torch.tensor(
                    [
                        admittance_scale(
                            scaler, family, part, "diag" if bool(is_diag) else "offdiag"
                        )
                        for is_diag in diag
                    ],
                    dtype=packed.dtype,
                )
                out[row] = torch.asinh(packed[row] / (scales + eps))
            setattr(dst, feat_field, out)

        icomp_slots = int(spec.get("icomp_slots", 0))
        if icomp_slots:
            for part in ("r", "i"):
                pu_tensor = getattr(src, f"Icomp_{part}_pu")
                out = torch.empty_like(pu_tensor)
                for row, family in enumerate(row_families):
                    scale = current_scale(scaler, family)
                    out[row] = torch.asinh(pu_tensor[row] / (scale + eps))
                setattr(dst, f"Icomp_{part}_feat", out)

        for term in range(1, int(spec["terminals"]) + 1):
            for part in ("r", "i"):
                pu_tensor = getattr(src, f"I_{part}_bus{term}_pu")
                out = torch.empty_like(pu_tensor)
                icomp_tensor = (
                    getattr(src, f"Icomp_{part}_pu") if icomp_slots else None
                )
                start = (term - 1) * FIXED_CONDUCTORS
                for row, family in enumerate(row_families):
                    values = pu_tensor[row]
                    if icomp_tensor is not None:
                        values = values + icomp_tensor[row, start:start + FIXED_CONDUCTORS]
                    scale = current_scale(scaler, family)
                    out[row] = torch.asinh(values / (scale + eps))
                setattr(dst, f"I_{part}_bus{term}_feat", out)

        for field in PASSIVE_DEFINITION_FIELDS.get(store_name, ()):
            value = getattr(src, field, None)
            if torch.is_tensor(value):
                setattr(dst, field, value.clone())

        dst.num_nodes = n_rows

    return feat_data


def _scenario_featurize_one_feeder(
    root_str: str, feeder_dir_str: str, out_root_str: str, json_root_str: str,
    scaler: dict[str, Any], overwrite: bool, unified_line_scale: bool,
) -> tuple[str, str]:
    import datakit.core.scenario_store as scenario_store

    root = Path(root_str)
    feeder_dir = Path(feeder_dir_str)
    target_dir = Path(out_root_str) / feeder_dir.relative_to(root)
    if (target_dir / "static.pt").is_file() and not overwrite:
        target = scenario_store.FeederScenarios(target_dir)
        if target.basis == "feat":
            return str(feeder_dir.relative_to(root)), "cached"
    ds = scenario_store.FeederScenarios(feeder_dir)
    if ds.basis != "pu":
        raise ValueError(f"{feeder_dir.name}: expected pu-basis store, got {ds.basis}")

    if unified_line_scale:
        families = None
    else:
        payload = json.loads(
            _scenario_baseline_json_path(
                Path(json_root_str), feeder_dir, source_root=root
            ).read_text()
        )
        families = _scenario_families_by_store(payload)
    writer = scenario_store.ScenarioWriter(basis="feat")
    for idx, variant_id in enumerate(ds.variant_ids):
        raw_data = ds[idx]
        row_families = _scenario_unified_families(raw_data) if families is None else families
        writer.add(
            int(variant_id), _scenario_build_feat_sample(raw_data, scaler, row_families)
        )
    writer.finalize(target_dir)
    return str(feeder_dir.relative_to(root)), "ok"


def _scenario_discover_feeders(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob("static.pt") if (p.parent / "dynamic.npy").is_file())


def scenario_stores_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Rewrite pu-basis scenario stores into feature-space scenario stores. "
            "This is the post-selection stage for corpora generated by make_training_pt.py "
            "with --store-basis pu."
        )
    )
    ap.add_argument("--root", type=Path, required=True, help="scenario-store root containing feature_scaler.json and feeder stores")
    ap.add_argument(
        "--scaler", type=Path,
        help="global scaler artifact; defaults to <root>/feature_scaler.json",
    )
    ap.add_argument(
        "--out-root", type=Path,
        help="write a separate feature corpus (recommended); default rewrites --root",
    )
    ap.add_argument(
        "--json-root", type=Path,
        help="root containing baseline json/<relative feeder>/master.json; defaults to --root",
    )
    ap.add_argument(
        "--unified-line-scale", action="store_true",
        help="use one Line scale for all line rows and avoid baseline-JSON family lookup",
    )
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--limit", type=int, default=None, help="only the first N feeders")
    ap.add_argument(
        "--limit-per-corpus", type=int,
        help="select this many hash-shuffled feeders from each top-level corpus",
    )
    ap.add_argument("--selection-seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="rewrite stores even if they already declare basis=feat")
    args = ap.parse_args(argv)

    root = args.root.resolve()
    scaler_path = (args.scaler or (root / "feature_scaler.json")).resolve()
    scaler = load_scaler_metadata(scaler_path)
    out_root = (args.out_root or root).resolve()
    json_root = (args.json_root or root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if out_root != root:
        shutil.copy2(scaler_path, out_root / "feature_scaler.json")
    feeders = _scenario_discover_feeders(root)
    if args.limit is not None and args.limit_per_corpus is not None:
        raise ValueError("use only one of --limit and --limit-per-corpus")
    if args.limit_per_corpus is not None:
        grouped: dict[str, list[Path]] = {}
        for feeder in feeders:
            corpus = feeder.relative_to(root).parts[0]
            grouped.setdefault(corpus, []).append(feeder)
        feeders = []
        for corpus in sorted(grouped):
            rows = sorted(
                grouped[corpus],
                key=lambda path: hashlib.sha256(
                    f"{args.selection_seed}|{path.relative_to(root)}".encode()
                ).digest(),
            )
            feeders.extend(rows[:args.limit_per_corpus])
    if args.limit is not None:
        feeders = feeders[: args.limit]
    if not feeders:
        raise SystemExit(f"no feeder stores under {root}")

    print(f"featurizing {len(feeders)} scenario stores under {root} on {args.workers} workers", flush=True)
    ok = cached = fail = 0
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        future_map = {
            pool.submit(
                _scenario_featurize_one_feeder,
                str(root), str(feeder), str(out_root), str(json_root), scaler,
                bool(args.overwrite), bool(args.unified_line_scale),
            ): feeder
            for feeder in feeders
        }
        for idx, fut in enumerate(as_completed(future_map), 1):
            feeder = future_map[fut]
            try:
                _, status = fut.result()
            except Exception as exc:  # noqa: BLE001
                fail += 1
                print(f"FAIL {feeder.name}: {type(exc).__name__}: {exc}", flush=True)
            else:
                if status == "cached":
                    cached += 1
                else:
                    ok += 1
            if idx == len(feeders) or idx % max(1, min(25, len(feeders))) == 0:
                print(
                    f"scenario-featurize: feeders={idx}/{len(feeders)} ok={ok} cached={cached} failed={fail}",
                    flush=True,
                )
    return 1 if fail else 0


def main() -> int:
    args = _parse_args()

    root = args.search_root.resolve()
    all_paths = _discover_jsons(root)
    if not all_paths:
        print(f"No json/master*.json under {root}", file=sys.stderr)
        return 2

    if args.apply_only:
        if not args.scaler.exists():
            print(f"--apply-only needs an existing scaler at {args.scaler}", file=sys.stderr)
            return 2
        scaler = load_scaler_metadata(args.scaler)
        validate_asinh_scaler_metadata(scaler)
        print(f"Loaded scaler from {args.scaler}")
    else:
        if args.train_glob:
            train_paths = sorted(Path.cwd().glob(args.train_glob))
            split_label = args.train_glob
        else:
            train_paths = all_paths
            split_label = "all-discovered"
        if not train_paths:
            print(f"No train jsons matched {args.train_glob!r}", file=sys.stderr)
            return 2
        scaler = _fit_scaler(train_paths, split_label)
        save_scaler_metadata(scaler, args.scaler)
        print(f"Saved scaler -> {args.scaler}")
        if args.fit_only:
            return 0

    ok_paths, bad = _apply_scaler(all_paths, scaler, args.indent)
    print(f"Stamped features into {len(ok_paths)}/{len(all_paths)} feeders.")
    if ok_paths:
        _print_feature_distribution_stats(ok_paths)
        _print_top_feature_outliers(ok_paths, scaler, top_k=20)
    for p, err in bad[:20]:
        print(f"FAILED {p}: {err}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"scenario-stores", "scenario-stores-featurize"}:
        raise SystemExit(scenario_stores_main(sys.argv[2:]))
    raise SystemExit(main())

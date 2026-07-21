#!/usr/bin/env python3
"""Re-encode a scenario-store corpus with floored feature scales (no OpenDSS).

Why: P95-per-family asinh scales degenerate when a family's values are ~0 in the
corpus (TriplexLine I_scale ~9e-10, Storage/Capacitor Y scales at the 1e-12
floor). Near-zero physical values then encode as +/-15 feats where tiny model
errors decode into enormous relative noise, and the family sits far outside
every other family's feat range. Flooring each scale at
    alpha * median(scales of the same kind across families)
keeps healthy families bit-identical in spirit (their scales don't move) while
degenerate families encode honestly as ~0.

The transform is exact per entry (featurizing.py maps are invertible):
    x' = asinh( sinh(x) * (s_old+eps) / (s_new+eps) )
so structural zeros stay exactly 0, the static/dynamic split stays valid, and
the physics gate must still pass at machine precision on the output.

Usage (sbatch a CPU node; ~2000 feeders):
    python scripts/generate_training_data/reencode_corpus.py \
        --src  ../training_data/minimal_component_ifields \
        --out  ../training_data/minimal_component_v2 \
        --flags /kfs2/projects/gogpt/Ebadmus/DG_FM_Training/cache/line_triplex.pt
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import torch

# Pinned schema: keep this experiment independent of external exporter edits.
FEAT_EPS = 1e-12
SPECS = {
    "line": {"json_key": "Line", "y_fields": ("Ys_r_tri", "Ys_i_tri", "Yh_i_tri"), "y_dim": 4, "icomp": 0},
    "capacitor": {"json_key": "Capacitor", "y_fields": ("Ycap_r_tri", "Ycap_i_tri"), "y_dim": 8, "icomp": 0},
    "reactor": {"json_key": "Reactor", "y_fields": ("Yreactor_r_tri", "Yreactor_i_tri"), "y_dim": 8, "icomp": 0},
    "transformer": {"json_key": "Transformer", "y_fields": ("Yxfmr_r_tri", "Yxfmr_i_tri"), "y_dim": 12, "icomp": 0},
    "vsource": {"json_key": "Vsource", "y_fields": ("Ysource_r_tri", "Ysource_i_tri"), "y_dim": 8, "icomp": 8},
    "load": {"json_key": "Load", "y_fields": ("Yload_r_tri", "Yload_i_tri"), "y_dim": 4, "icomp": 4},
    "generator": {"json_key": "Generator", "y_fields": ("Ygen_r_tri", "Ygen_i_tri"), "y_dim": 4, "icomp": 4},
    "pvsystem": {"json_key": "PVSystem", "y_fields": ("Ypv_r_tri", "Ypv_i_tri"), "y_dim": 4, "icomp": 4},
    "storage": {"json_key": "Storage", "y_fields": ("Ystorage_r_tri", "Ystorage_i_tri"), "y_dim": 4, "icomp": 4},
}


def tri_diag_indices(dim: int) -> set[int]:
    out, idx = set(), 0
    for row in range(dim):
        for col in range(row + 1):
            if row == col:
                out.add(idx)
            idx += 1
    return out

_WORKER: dict = {}


def floored_scaler(scaler: dict, alpha_i: float, alpha_y: float) -> dict:
    """New scaler with per-kind median floors; annotates what moved."""
    out = json.loads(json.dumps(scaler))  # deep copy
    i_scales = [v["I_scale"] for v in scaler["current"].values()]
    i_floor = alpha_i * statistics.median(i_scales)
    for fam, cfg in out["current"].items():
        cfg["I_scale"] = max(float(cfg["I_scale"]), i_floor)
    for key in ("Y_r_diag_scale", "Y_r_offdiag_scale", "Y_i_diag_scale", "Y_i_offdiag_scale"):
        vals = [float(v[key]) for v in scaler["admittance"].values() if key in v]
        floor = alpha_y * statistics.median(vals)
        for fam, cfg in out["admittance"].items():
            if key in cfg:
                cfg[key] = max(float(cfg[key]), floor)
    out["reencode"] = {"alpha_i": alpha_i, "alpha_y": alpha_y,
                       "i_floor": i_floor, "src_note": "floored re-encode; see reencode_corpus.py"}
    return out


def _entry_scales(store_key: str, spec: dict, field: str, fams: list[str], scaler: dict) -> np.ndarray | None:
    """[n_rows, width] scale array for one *_feat field, or None if not a feat field."""
    dim = int(spec["y_dim"])
    diag = tri_diag_indices(dim)
    tri = dim * (dim + 1) // 2
    if field.startswith("I_") or field.startswith("Icomp_"):
        s = np.array([scaler["current"][f]["I_scale"] for f in fams])[:, None]
        width = spec["icomp"] if field.startswith("Icomp_") else 4
        return np.repeat(s, width, axis=1)
    for y_field in spec["y_fields"]:
        if field == f"{y_field}_feat":
            part = "r" if "_r_" in y_field else "i"
            cols = np.array([
                [scaler["admittance"][f][f"Y_{part}_{'diag' if k in diag else 'offdiag'}_scale"]
                 for k in range(tri)] for f in fams])
            return cols
    return None


def reencode_feeder(feeder_name: str) -> str:
    src, out = _WORKER["src"] / feeder_name, _WORKER["out"] / feeder_name
    old, new = _WORKER["old"], _WORKER["new"]
    flags = _WORKER["flags"].get(feeder_name)
    eps = FEAT_EPS

    meta = torch.load(src / "static.pt", map_location="cpu", weights_only=False)
    if flags is None:
        json_path = _WORKER["src"] / "json" / feeder_name / "master.json"
        if not json_path.is_file():
            raise FileNotFoundError(
                f"missing Line/TriplexLine metadata for {feeder_name}: {json_path}"
            )
        payload = json.loads(json_path.read_text())
        flags = [bool(row.get("is_triplex_line", False)) for row in payload.get("Line", [])]
    source_dyn = np.load(src / "dynamic.npy", allow_pickle=False)
    source_dtype = source_dyn.dtype
    # Do the invertible coordinate change in float64, then preserve the corpus
    # storage dtype declared by static.pt/schema metadata.
    dyn = source_dyn.astype(np.float64)
    skel = meta["skeleton"]

    def fams_for(store_key: str, n: int) -> list[str]:
        if store_key == "line":
            if len(flags) != n:
                raise ValueError(
                    f"{feeder_name}: line-family flags {len(flags)} != line rows {n}"
                )
            return ["TriplexLine" if bool(f) else "Line" for f in flags]
        return [ {"line": "Line"}.get(store_key, SPECS[store_key]["json_key"]) ] * n

    def ratio(store_key, spec, field, n):
        s_old = _entry_scales(store_key, spec, field, fams_for(store_key, n), old)
        if s_old is None:
            return None
        s_new = _entry_scales(store_key, spec, field, fams_for(store_key, n), new)
        return (s_old + eps) / (s_new + eps)

    for entry in meta["layout"]:
        store_key, field = entry["store"], entry["field"]
        spec = SPECS.get(store_key)
        if spec is None or not field.endswith("_feat"):
            continue
        n = int(entry["shape"][0])
        r = ratio(store_key, spec, field, n)
        if r is None or (r == 1.0).all():
            continue    # scales unchanged -> keep the field bit-identical
        if entry["static"]:
            t = getattr(skel[store_key], field)
            v = t.double().reshape(n, -1).numpy()
            setattr(skel[store_key], field,
                    torch.from_numpy(np.arcsinh(np.sinh(v) * r)).reshape(t.shape).to(t.dtype))
        else:
            off, numel = int(entry["offset"]), int(entry["numel"])
            block = dyn[:, off:off + numel].reshape(dyn.shape[0], n, -1)
            dyn[:, off:off + numel] = np.arcsinh(np.sinh(block) * r[None]).reshape(dyn.shape[0], -1)

    out.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(
        out / "dynamic.npy", mode="w+", dtype=source_dtype, shape=dyn.shape
    )
    mm[:] = dyn
    mm.flush()
    del mm
    torch.save(meta, out / "static.pt")
    return feeder_name


def _init(src, out, old, new, flags):
    _WORKER.update(src=Path(src), out=Path(out), old=old, new=new, flags=flags)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--flags", type=Path, required=True,
                    help="consolidated line_triplex.pt (per-feeder per-row bool)")
    ap.add_argument("--alpha-i", type=float, default=1e-3)
    ap.add_argument("--alpha-y", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--limit", type=int, default=None, help="only the first N feeders (smoke)")
    args = ap.parse_args()

    src, out = args.src.resolve(), args.out.resolve()
    old = json.loads((src / "feature_scaler.json").read_text())
    new = floored_scaler(old, args.alpha_i, args.alpha_y)
    # plain bool lists: torch tensors in initargs exhaust fds under Python
    # 3.14's forkserver (every tensor ships through a file descriptor)
    flags = {k: [bool(x) for x in v] for k, v in torch.load(args.flags, weights_only=True).items()}

    feeders = sorted(p.name for p in src.iterdir()
                     if p.is_dir() and (p / "static.pt").is_file())
    if args.limit:
        feeders = feeders[:args.limit]
    print(f"re-encoding {len(feeders)} feeders {src.name} -> {out.name}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "feature_scaler.json").write_text(json.dumps(new, indent=2, sort_keys=True))
    if not (out / "json").exists():
        os.symlink(src / "json", out / "json")
    shutil.copy2(src / "manifest.json", out / "manifest.json")

    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork"),
                             initializer=_init,
                             initargs=(str(src), str(out), old, new, flags)) as pool:
        futs = [pool.submit(reencode_feeder, f) for f in feeders]
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 200 == 0 or done == len(feeders):
                print(f"{done}/{len(feeders)} ({done / (time.time() - t0):.1f}/s)", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

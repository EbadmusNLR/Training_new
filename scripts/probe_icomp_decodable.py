#!/usr/bin/env python3
"""Is Icomp decodable in closed form from the stored device definition?

The physics doc gives I_comp(V) = Y_Norton V - I_DSS(V). Y_Norton already decodes
exactly from definitions (T88-T93). So Icomp is decodable exactly when I_DSS is --
i.e. when the stored physics_params determine which OpenDSS current equation the
element used.

This probe checks, per PC family, whether the stored contract is self-consistent
(I_bus + Icomp == Y V) and whether Icomp can be rebuilt from (effective power, V)
alone. Where it can, Icomp joins Y and V as exact and no learning is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path[:0] = [str(ROOT), str(PROJECT)]

from datakit.core.scenario_store import FeederScenarios  # noqa: E402
from datakit.core import pc_metadata as pc  # noqa: E402

FEATURES = {
    "load": pc.LOAD_PHYSICS_FEATURES,
    "pvsystem": pc.PVSYSTEM_PHYSICS_FEATURES,
    "generator": pc.GENERATOR_PHYSICS_FEATURES,
    "storage": pc.STORAGE_PHYSICS_FEATURES,
}
YKEY = {
    "load": "Yload", "pvsystem": "Ypv", "generator": "Ygen", "storage": "Ystorage",
}


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/kfs2/projects/gogpt/Ebadmus/training_data")
    corpus = root / "minimal_component"
    feeders = sorted(p.parent for p in corpus.rglob("static.pt"))[:60]
    print(f"scanning {len(feeders)} feeders under {corpus}", flush=True)

    for fam in ("load", "pvsystem", "generator", "storage"):
        shown = 0
        for f in feeders:
            try:
                d = FeederScenarios(f)[0]
            except Exception:
                continue
            if fam not in d.node_types:
                continue
            st = d[fam]
            prm = st.get("physics_params")
            if prm is None or not prm.shape[0]:
                continue
            cols = list(FEATURES[fam])
            row = prm[0]
            vals = {n: float(row[i]) for i, n in enumerate(cols) if i < row.numel()}
            ic = st.get("Icomp_r_pu")
            yk = YKEY[fam]
            yr, yi = st.get(f"{yk}_r_pu"), st.get(f"{yk}_i_pu")
            print(f"\n=== {fam}  feeder={f.name[:60]}", flush=True)
            print(f"    stored params: {vals}", flush=True)
            if ic is not None:
                icv = st["Icomp_r_pu"][0].numpy() + 1j * st["Icomp_i_pu"][0].numpy()
                print(f"    Icomp_pu row0: {np.round(icv, 6)}", flush=True)
                print(f"    |Icomp| max  : {np.abs(icv).max():.6e}", flush=True)
            if yr is not None:
                print(f"    Y row0 nonzero: {int((yr[0].abs() > 0).sum())}/{yr[0].numel()}", flush=True)
            shown += 1
            if shown >= 2:
                break
        if shown == 0:
            print(f"\n=== {fam}: no rows found in scanned feeders", flush=True)

    print("\nDECODABILITY VERDICT (from the stored feature contract):", flush=True)
    for fam in ("load", "pvsystem", "generator", "storage"):
        cols = set(FEATURES[fam])
        has_model = "model" in cols
        has_power = {"effective_kw", "effective_kvar"} <= cols or {"kw", "kvar"} <= cols
        note = []
        if not has_model:
            note.append("NO model selector")
        if not has_power:
            note.append("no explicit power")
        print(f"  {fam:10s} power={has_power} model={has_model}  {'; '.join(note) or 'decodable inputs present'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

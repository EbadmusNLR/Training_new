"""Scan the dss_data corpus for DE-ENERGIZED / degenerate networks.

W and H solve "successfully" (converged=True) but sit at V=0 everywhere except the
slack: nothing downstream is energized. That is why W's currents are ~7e-08 (which I
mis-filed as a metric artifact) and why H's Y_series is exactly singular.

A converged solve is NOT evidence of a usable network. Measure the fraction of nodes
at V~0 straight from the BUILT corpus (that is what training would actually see), so
this catches every instance rather than the two I happened to trip over.
"""
import glob, os, sys
import numpy as np
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from core.scenario_store import FeederScenarios

TD = "/kfs2/projects/gogpt/Ebadmus/training_data"
corpus = sys.argv[1] if len(sys.argv) > 1 else "dss_data"
rows = []
for p in sorted(glob.glob(os.path.join(TD, corpus, "*", "static.pt"))):
    name = os.path.basename(os.path.dirname(p))
    try:
        fs = FeederScenarios(os.path.dirname(p))
        d = fs[0]
        nd = d["node"]
        v = (nd.V_r_pu.double() ** 2 + nd.V_i_pu.double() ** 2).sqrt().numpy()
        v = v[1:]                       # drop ground (node 0)
        dead = float((v < 1e-6).mean())
        rows.append((dead, name, len(v), float(np.median(v)), float(v.max())))
    except Exception as e:
        rows.append((-1.0, name, 0, 0.0, 0.0))
rows.sort(reverse=True)
print(f"=== {corpus}: fraction of nodes at V ~ 0 (de-energized) ===")
print(f"{'dead%':>7s} {'nodes':>7s} {'medV':>7s} {'maxV':>7s}  feeder")
for dead, name, n, med, mx in rows:
    if dead > 0.01 or dead < 0:
        print(f"{100*dead:7.1f} {n:7d} {med:7.4f} {mx:7.4f}  {name[:52]}")
n_bad = sum(1 for r in rows if r[0] > 0.5)
print(f"\nfeeders with >50% of nodes de-energized: {n_bad} / {len(rows)}")
print(f"feeders with ANY de-energized node:      {sum(1 for r in rows if r[0] > 0.01)} / {len(rows)}")

"""Is _flatten_dss FAITHFUL, and does it actually unblock the perturber?

Two independent things to prove before rebuilding any corpus:
  1. FAITHFUL: the flattened master must solve to the SAME voltages as the original.
     Inlining includes can break relative paths / ordering, which would silently
     change the network -- worse than the bug being fixed.
  2. EFFECTIVE: scale_dss_text on the flattened text must actually move the loads
     (the original text had no `New Load.` lines at all for these feeders).
"""
import random
import sys
from pathlib import Path

import numpy as np
import opendssdirect as dss

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
from datakit.pipelines.make_training_pt import _flatten_dss
from datakit.pipelines import param_sampler

DATA = Path("/kfs2/projects/gogpt/Ebadmus/data/dss_data")
FEEDERS = ["IEEE 30 Bus", "123Bus", "IEEE9500", "13Bus", "37Bus", "4Bus-OYOD-Bal"]


def solve_text(text: str, workdir: Path, tag: str):
    tmp = workdir / f".flatcheck_{tag}.dss"
    tmp.write_text(text)
    try:
        dss.Text.Command("Clear")
        dss.Text.Command(f'Redirect "{tmp}"')
        dss.Text.Command("Set ControlMode=OFF")
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            return None
        return np.array(dss.Circuit.AllBusMagPu())
    except Exception as e:
        print(f"      solve({tag}) EXCEPTION: {str(e)[:90]}")
        return None
    finally:
        tmp.unlink(missing_ok=True)


ranges = param_sampler.load_ranges(None)
print(f"{'feeder':16s} {'orig loads':>10s} {'flat loads':>10s} {'V match':>12s} {'perturb moves V':>16s}")
for name in FEEDERS:
    m = DATA / name / "network" / "Master.dss"
    if not m.is_file():
        print(f"{name:16s}  (no Master.dss)")
        continue
    orig = m.read_text()
    flat = _flatten_dss(m)
    n_orig = sum(1 for l in orig.splitlines() if l.strip().lower().startswith("new load."))
    n_flat = sum(1 for l in flat.splitlines() if l.strip().lower().startswith("new load."))

    v_o = solve_text(orig, m.parent, "orig")
    v_f = solve_text(flat, m.parent, "flat")
    if v_o is None or v_f is None or v_o.shape != v_f.shape:
        vm = "SOLVE FAIL" if (v_o is None or v_f is None) else f"shape {v_o.shape}!={v_f.shape}"
    else:
        vm = f"{np.abs(v_o - v_f).max():.2e}"

    # does perturbation now actually move the operating point?
    rng = random.Random(7)
    pert = param_sampler.scale_dss_text(flat, rng, ranges, impedance=False, source_pu=True)
    v_p = solve_text(pert, m.parent, "pert")
    if v_p is None or v_f is None or v_p.shape != v_f.shape:
        pm = "n/a"
    else:
        pm = f"{np.abs(v_p - v_f).max():.2e}"
    print(f"{name:16s} {n_orig:10d} {n_flat:10d} {vm:>12s} {pm:>16s}")
print("\nV match  = |V(flattened) - V(original)|max   -> must be ~0 (faithful)")
print("perturb  = |V(perturbed) - V(flattened)|max  -> must be > 0 (variants differ)")

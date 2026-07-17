"""VERIFY unused SMART-DS candidates by SOLVING them, then copy the healthy ones into
data/SMART-DS_1000 (preserving the v1.0/peak/... relative path so datakit finds them).

Health must be MEASURED, not assumed: the two feeders this replaces compiled and converged
fine yet had 95.7% / 99.8% of nodes at V=0. Criteria: converged, >=5 loads, <1% dead nodes,
sane Vmax. Feeder dirs are self-contained (all Redirects are local), so copying the dir works.
"""
import json, os, shutil, sys
import numpy as np
import opendssdirect as dss

CAND = sys.argv[1]; NEED = int(sys.argv[2]) if len(sys.argv) > 2 else 2
DEST = "/kfs2/projects/gogpt/Ebadmus/data/SMART-DS_1000/v1.0/peak"
cands = json.load(open(CAND))
print(f"{len(cands)} candidates to screen; need {NEED}")

def health(path):
    try:
        dss.Text.Command("Clear")
        dss.Text.Command(f'Redirect "{os.path.join(path, "Master.dss")}"')
        dss.Text.Command("Set ControlMode=OFF")
        dss.Solution.Solve()
        if not dss.Solution.Converged(): return None, "not converged"
        v = np.array(dss.Circuit.AllBusMagPu())
        v = v[np.isfinite(v)]
        if v.size == 0: return None, "no buses"
        dead = float((v < 1e-6).mean())
        nl = dss.Loads.Count()
        return {"dead": dead, "vmax": float(v.max()), "vmin": float(v.min()),
                "nload": nl, "nbus": int(v.size)}, None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:60]}"

picked = []
for nl_txt, rel, src in cands:
    if len(picked) >= NEED: break
    h, err = health(src)
    if err: print(f"  SKIP {os.path.basename(src):34s} {err}"); continue
    if h["nload"] < 5 or h["dead"] > 0.01 or not (0.5 < h["vmax"] < 1.5):
        print(f"  SKIP {os.path.basename(src):34s} dead={100*h['dead']:.1f}% "
              f"Vmax={h['vmax']:.3f} loads={h['nload']}"); continue
    print(f"  OK   {os.path.basename(src):34s} dead={100*h['dead']:.1f}% "
          f"Vmax={h['vmax']:.3f} Vmin={h['vmin']:.3f} loads={h['nload']} buses={h['nbus']}")
    picked.append((rel, src, h))

if len(picked) < NEED:
    print(f"ONLY {len(picked)}/{NEED} healthy candidates found"); sys.exit(1)
for rel, src, h in picked:
    dst = os.path.join(DEST, rel)
    if os.path.exists(dst): print("  exists, skip:", rel); continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copytree(src, dst)
    print(f"COPIED -> data/SMART-DS_1000/v1.0/peak/{rel}  ({h['nload']} loads)")
print("\nDone. Rebuild these two into training_data/SMART-DS_1000 with the datakit "
      "make_training_pt pipeline to restore the 1000 count.")

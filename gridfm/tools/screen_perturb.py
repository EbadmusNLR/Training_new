"""Screen SMART-DS feeder dirs against the REAL datakit perturbation gate.

The gate that matters is NOT "does it solve at base" -- it is "does a FULL-STRENGTH
perturbation draw still converge". Two feeders were excluded for failing exactly this
while solving fine unperturbed, so screening replacements on a base solve (as I did once
already) validates the wrong thing.

Replicates make_training_pt._build_variant: _flatten_dss -> param_sampler.scale_dss_text
(strength=1.0) -> export_circuit_to_json (compile+solve). Reports the convergence rate
over N independent full-strength draws.

  python gridfm/screen_perturb.py <ndraws> <feeder_dir> [<feeder_dir> ...]
"""
import os, random, sys, tempfile
from pathlib import Path
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
from datakit.pipelines.make_training_pt import (_flatten_dss, _is_nonconvergence_error)
from datakit.pipelines import param_sampler
from datakit.pipelines.build_master_json import export_circuit_to_json

def screen(feeder_dir: str, ndraws: int, ranges):
    master = Path(feeder_dir) / "Master.dss"
    if not master.is_file():
        return None, "no Master.dss"
    try:
        base_text = _flatten_dss(master)
    except Exception as e:
        return None, f"flatten failed: {str(e)[:50]}"
    ok = nonconv = other = 0
    for k in range(ndraws):
        rng = random.Random(1234 + k)
        text = param_sampler.scale_dss_text(base_text, rng, ranges, impedance=False,
                                            source_pu=True, strength=1.0)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=f".screen{k}.dss",
                                             prefix=".scr_", dir=master.parent, delete=False) as h:
                h.write(text); tmp = Path(h.name)
            export_circuit_to_json(tmp, None, indent=0, include_extras=False)
            ok += 1
        except Exception as e:
            if _is_nonconvergence_error(str(e)): nonconv += 1
            else: other += 1
        finally:
            if tmp is not None:
                try: tmp.unlink()
                except OSError: pass
    return {"ok": ok, "nonconv": nonconv, "other": other, "n": ndraws}, None

if __name__ == "__main__":
    n = int(sys.argv[1]); ranges = param_sampler.load_ranges(None)
    print(f"{'feeder':40s} {'converged':>12s} {'nonconv':>8s} {'other':>6s}")
    for fd in sys.argv[2:]:
        r, err = screen(fd, n, ranges)
        nm = os.path.basename(fd)[:38]
        if err: print(f"  {nm:40s} ERROR: {err}"); continue
        print(f"  {nm:40s} {r['ok']:5d}/{r['n']:<6d} {r['nonconv']:8d} {r['other']:6d}")

"""Are a feeder's VARIANTS actually different, and WHICH quantity is being perturbed?
Per feeder, across variants, measure the relative SPREAD of:
  V     - the solved voltage (the effect a perturbation should have)
  Icomp - load/gen compensation current (the usual perturbation TARGET)
  Yline - line series admittance (impedance perturbation, if any)
  Yxfmr - transformer admittance (tap perturbation, if any)
spread ~ 0  => that quantity is IDENTICAL across variants (not perturbed).
Reports how many feeders have DEAD variants (V identical => 100 duplicate samples).
"""
import argparse, glob, hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES

def spread(vals):
    if len(vals) < 2: return 0.0
    t = torch.stack(vals); m = t.mean(0)
    return float((t - m).abs().mean() / (m.abs().mean() + 1e-30))

def _fingerprint(t):
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def _electrical_arrays(d):
    """Return only the four reconstruction arrays, grouped by logical target."""
    def cat(xs):
        return torch.cat([x.detach().cpu().flatten() for x in xs]) if xs else torch.zeros(0)

    node = d["node"]
    out = {
        "V": cat([node["V_r_pu"], node["V_i_pu"]]),
    }
    ibus, icomp, y = [], [], []
    for store, (prefix, nterm, _) in STORES.items():
        if store not in d.node_types:
            continue
        st = d[store]
        for terminal in range(1, nterm + 1):
            for suffix in ("r", "i"):
                key = f"I_{suffix}_bus{terminal}_pu"
                if key in st:
                    ibus.append(st[key])
        for suffix in ("r", "i"):
            key = f"Icomp_{suffix}_pu"
            if key in st:
                icomp.append(st[key])
            key = f"{prefix}_{suffix}_pu"
            if key in st:
                y.append(st[key])
        if store == "line" and "Yh_i_pu" in st:
            y.append(st["Yh_i_pu"])
    out["Ibus"] = cat(ibus)
    out["Icomp"] = cat(icomp)
    out["Y"] = cat(y)
    out["electrical"] = cat([out["V"], out["Ibus"], out["Icomp"], out["Y"]])
    return out


def one(args):
    path, max_variants = args
    name = os.path.basename(os.path.dirname(path))
    try:
        fs = FeederScenarios(os.path.dirname(path)); nv = len(fs)
        if nv < 2: return (name, nv, None, {"err": "fewer than two variants"})
        K = min(nv, max_variants) if max_variants else nv
        V, IC, YL, YX = [], [], [], []
        hashes = {k: [] for k in ("V", "Ibus", "Icomp", "Y", "electrical")}
        for k in range(K):
            d = fs[k]
            arrays = _electrical_arrays(d)
            for key, value in arrays.items():
                hashes[key].append(_fingerprint(value))
            vv = torch.stack([d["node"].V_r_pu, d["node"].V_i_pu]).flatten()
            V.append(vv)
            ic = []
            for s in ("load","pvsystem","generator","storage"):
                if s in d.node_types and "Icomp_r_pu" in d[s]:
                    ic.append(d[s]["Icomp_r_pu"].flatten())
            IC.append(torch.cat(ic) if ic else torch.zeros(1))
            YL.append(d["line"]["Ys_r_pu"].flatten() if "line" in d.node_types and d["line"]["Ys_r_pu"].numel() else torch.zeros(1))
            YX.append(d["transformer"]["Yxfmr_r_pu"].flatten() if "transformer" in d.node_types and d["transformer"]["Yxfmr_r_pu"].numel() else torch.zeros(1))
        duplicates = {key: len(vals) - len(set(vals)) for key, vals in hashes.items()}
        return (name, nv, duplicates, {"V":spread(V), "Icomp":spread(IC), "Yline":spread(YL), "Yxfmr":spread(YX)})
    except Exception as e:
        return (name, 0, None, {"err": str(e)[:200]})

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--corpus", default="SMART-DS_1000")
    ap.add_argument("--root", default="", help="Scenario-store corpus root; overrides --corpus")
    ap.add_argument("--workers", type=int, default=64); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-variants", type=int, default=0, help="0 audits every stored variant")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--fail-on-errors", action="store_true")
    ap.add_argument("--require-no-electrical-duplicates", action="store_true")
    a = ap.parse_args()
    root = a.root or f"/kfs2/projects/gogpt/Ebadmus/training_data/{a.corpus}"
    fs = sorted(glob.glob(os.path.join(root, "*", "static.pt")))
    if not fs:
        raise SystemExit(f"no scenario stores found under {root}")
    if a.n: fs = fs[::max(1,len(fs)//a.n)][:a.n]
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("fork")) as ex:
        rows = list(ex.map(one, [(p, a.max_variants) for p in fs], chunksize=1))
    ok = [r for r in rows if r[2] is not None]
    dead = [r for r in ok if r[3].get("V",0) < 1e-9]                       # V identical across variants
    icdead = [r for r in ok if r[3].get("Icomp",0) < 1e-9]                 # loads not perturbed
    full_dupes = [r for r in ok if r[2]["electrical"]]
    label = a.root or a.corpus
    print(f"=== {label}: {len(ok)} feeders (>=2 variants)")
    print(f"  V IDENTICAL across variants (DUPLICATES): {len(dead)}/{len(ok)}")
    print(f"  Icomp identical (loads NOT perturbed):    {len(icdead)}/{len(ok)}")
    print(f"  feeders with repeated four-array snapshots: {len(full_dupes)}/{len(ok)}")
    for key in ("V", "Ibus", "Icomp", "Y", "electrical"):
        print(f"  exact repeated {key:10s} snapshots: {sum(r[2][key] for r in ok)}")
    import statistics as st
    for q in ("V","Icomp","Yline","Yxfmr"):
        sv = sorted(r[3].get(q,0) for r in ok)
        if sv:
            med = sv[len(sv)//2]; frac0 = sum(1 for x in sv if x<1e-9)/len(sv)
            print(f"  {q:6s}: median spread {med:.2e}   fraction ~0: {100*frac0:.1f}%")
    if dead[:5]:
        print("  sample DUPLICATE feeders:")
        for n,nv,dupes,sp in dead[:5]: print(f"    {n[-50:]:52s} Icomp spread {sp.get('Icomp',0):.1e}")
    if a.json_out:
        payload = {
            "root": root,
            "feeders": len(ok),
            "errors": len(rows) - len(ok),
            "feeders_with_repeated_electrical_snapshots": len(full_dupes),
            "exact_repeated_snapshots": {
                key: sum(r[2][key] for r in ok)
                for key in ("V", "Ibus", "Icomp", "Y", "electrical")
            },
            "rows": [
                {"name": n, "variants": nv, "duplicates": dupes, "spread": sp}
                for n, nv, dupes, sp in rows
            ],
        }
        with open(a.json_out, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"  wrote {a.json_out}")
    if a.fail_on_errors and len(rows) != len(ok):
        raise SystemExit(f"audit failed: {len(rows) - len(ok)} stores could not be read")
    if a.require_no_electrical_duplicates and full_dupes:
        raise SystemExit(
            f"audit failed: {len(full_dupes)} feeders contain repeated four-array snapshots"
        )

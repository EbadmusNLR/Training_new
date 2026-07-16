"""Verify the WAPE~1.0 outliers ON THE DATA, then move them (+ their DSS source) out.

WAPE 1.0 with |I|~0 is a DEAD or LOAD-FREE circuit, not a decoder failure -- decide on the
node voltages / stored |I| / load count, never on the score or a name match.
"""
import glob, json, os, re, shutil, sys, torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, store_size, stored_currents
ROOT = "/kfs2/projects/gogpt/Ebadmus"
EXC = f"{ROOT}/data/excluded"
cands = []
for tag, corpus in [("bx","new_dss_data"),("rc","SMART-DS_1000"),("bx","dss_data"),("rc","minimal_component")]:
    for f in glob.glob(f"runs/{tag}_{corpus}/shard_*.json"):
        for r in json.load(open(f)):
            if r["err"]: continue
            fn=sum(v[0] for v in r["acc"].values()); fd=sum(v[1] for v in r["acc"].values())
            if fn/(fd+1e-30) > 0.99 and fd < 1e-5: cands.append((corpus, r["name"]))
moved = {"de_energized": 0, "no_load": 0}
for corpus, name in sorted(set(cands)):
    store = f"{ROOT}/training_data/{corpus}/{name}"
    if not os.path.isdir(store): print("  gone:", name); continue
    d = FeederScenarios(store)[0]
    vm = (d["node"].V_r_pu.double()**2 + d["node"].V_i_pu.double()**2).sqrt()[1:]
    dead = float((vm < 1e-6).double().mean()); tot = 0.0; nload = 0
    for s in STORES:
        if s in d.node_types and store_size(d, s) > 0:
            a,b = stored_currents(d, s, dtype=torch.float64); tot += float(a.abs().sum()+b.abs().sum())
            if s == "load": nload = store_size(d, s)
    if tot > 1e-5: print(f"  KEEP (real |I|={tot:.1e}): {name}"); continue
    if dead > 0.5:
        kind="de_energized"; reason=(f"{100*dead:.1f}% of nodes at V=0 with converged=True, while the "
            f"circuit HAS {nload} loads. |I| = {tot:.2e}: no current flows. Same signature as the "
            f"excluded W network.")
    elif nload == 0:
        kind="no_load"; reason=(f"ZERO loads in the extracted circuit while the network is fully "
            f"energized ({100*dead:.1f}% dead nodes, Vmax {float(vm.max()):.3f}). |I| = {tot:.2e}: no "
            f"power flow to learn. NOTE: 0 extracted loads may be a datakit load-extraction / "
            f"redirect-inlining gap rather than a genuinely load-free network -- worth checking "
            f"alongside the variant-perturbation work.")
    else:
        print(f"  KEEP (not classifiable: dead={dead:.2f} nload={nload}): {name}"); continue
    dst = os.path.join(EXC, kind, name); os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst): shutil.rmtree(dst)
    shutil.move(store, dst)
    dss_note = "no matching source dir found"
    base = re.sub(r"__[0-9a-f]{12}$", "", name)
    dsrc = f"{ROOT}/data/{corpus}/{base}"
    if os.path.isdir(dsrc):
        shutil.move(dsrc, os.path.join(dst, "_dss_source")); dss_note = f"moved from data/{corpus}/{base} -> _dss_source/"
    elif corpus == "SMART-DS_1000":
        dss_note = ("NOT moved: SMART-DS sources live in a shared nested scenario tree "
                    "(v1.0/peak/<region>/.../opendss_no_loadshapes/<substation>/) whose substation "
                    "dir ALSO serves healthy sibling feeders -- moving it would break them.")
    open(os.path.join(dst,"EXCLUDED_REASON.md"),"w").write(
        f"# Excluded from training_data/{corpus}\n\n{reason}\n\nDSS source: {dss_note}\n\n"
        "Decided on the DATA (node voltages, stored |I|, load count), never on the WAPE score\n"
        "or a name match: a dead or load-free circuit scores WAPE 1.0 and merely LOOKS like a\n"
        "decoder failure.\n")
    moved[kind] += 1
print("moved:", moved)
for c in ("SMART-DS_1000","new_dss_data","dss_data","minimal_component"):
    n = len(glob.glob(f"{ROOT}/training_data/{c}/*/static.pt")); print(f"  training_data/{c}: {n} feeders")

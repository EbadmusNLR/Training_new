import sys, time, glob, os, json, torch, numpy as np
sys.path.insert(0,"/kfs2/projects/gogpt/Ebadmus/Training_new")
import data as D

# 1) SMALL feeder: sparse-vs-dense equivalence by forcing both paths
base_s="/kfs2/projects/gogpt/Ebadmus/training_data/minimal_component"
scaler_s=json.loads(open(base_s+"/feature_scaler.json").read())
sf=[os.path.dirname(p) for p in glob.glob(base_s+"/*/static.pt")][0]
for p in glob.glob(sf+"/pe_cache_v*.pt"): os.remove(p)
fc=D.FeederCache(sf, scaler_s, None)  # dense path (small)
pe_dense, depth_dense = fc.pe.clone(), fc.depth.clone()
# force sparse path on the same small feeder
D.PE_DENSE_MAX=0
for p in glob.glob(sf+"/pe_cache_v*.pt"): os.remove(p)
fc2=D.FeederCache(sf, scaler_s, None)
pe_sp, depth_sp = fc2.pe.clone(), fc2.depth.clone()
D.PE_DENSE_MAX=512
n=fc.n_node
print(f"SMALL feeder n={n}")
print(f"  depth match: max|Δ|={ (depth_dense-depth_sp).abs().max():.3e}")
print(f"  pe   match: max|Δ|={ (pe_dense-pe_sp).abs().max():.3e}  (RWSE cols 0:8, deg 8, hop 9, z 10)")
print(f"  per-col max|Δ|: {[round(float((pe_dense[:,i]-pe_sp[:,i]).abs().max()),4) for i in range(pe_dense.shape[1])]}")

# 2) BIG feeder: timing
base_b="/kfs2/projects/gogpt/Ebadmus/training_data/smartds1000_pilot"
scaler_b=json.loads(open(base_b+"/feature_scaler.json").read())
bf=[os.path.dirname(p) for p in glob.glob(base_b+"/*/static.pt")][0]
for p in glob.glob(bf+"/pe_cache_v*.pt"): os.remove(p)
t=time.time(); fcb=D.FeederCache(bf, scaler_b, None); dt=time.time()-t
print(f"BIG feeder n={fcb.n_node}: build {dt:.1f}s, pe shape {tuple(fcb.pe.shape)}")
t=time.time(); fcb2=D.FeederCache(bf, scaler_b, None); dt2=time.time()-t
print(f"  cached reload: {dt2:.2f}s")

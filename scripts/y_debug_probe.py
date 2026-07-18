import sys, torch
sys.path.insert(0, '/kfs2/projects/gogpt/Ebadmus/datakit')
sys.path.insert(0, '/kfs2/projects/gogpt/Ebadmus/Training_new')
sys.path.insert(0, '/kfs2/projects/gogpt/Ebadmus/Training_new/scripts')
from gridfm.dk_data import DKFeeder, DKDataset, make_dk_collate, fit_scales
from gridfm.dk_model import DKSolver
from dk_train import losses
torch.manual_seed(0)
fd = DKFeeder('/kfs2/projects/gogpt/Ebadmus/training_data/minimal_component/t1_1250_k46_transformer_transformer_1ph_2w_yd_load_delta_abc_m8_zipv_load_wye_3ph_n_m2_const_z_plus41')
sc = fit_scales([fd], [0, 1])
ds = DKDataset([fd], [0], task='param')
m = DKSolver(hidden=64, steps=3, scales=sc, four_mask=True); m.skip_current = True
batch, plan, rctx = make_dk_collate([fd], need_ctx=False)([ds[0]])
batch.tree_plan = plan; batch.recon_ctx = rctx
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for it in range(200):
    dv, cur, aux = m(batch)
    loss, met = losses(batch, dv, cur, sc, aux=aux, w_i=0.0, w_kcl=0.0, norm=True)
    opt.zero_grad(); loss.backward(); opt.step()
    if it in (0, 50, 199):
        print(f'step {it}: loss={float(loss):.4f} y_wape={met["y_wape"]:.1f}%')
        for s, (er, ei) in aux['y_est'].items():
            mm = aux['y_msk'][s]
            st = batch[s]
            es = torch.stack([er[mm], ei[mm]], -1).detach()
            tr = torch.stack([st.yr[mm], st.yi[mm]], -1)
            print(f'  {s}: nhid={int(mm.sum())} |est|max={float(es.abs().max()):.3e} '
                  f'|tr|max={float(tr.abs().max()):.3e} |est|sum={float(es.abs().sum()):.3e} '
                  f'|tr|sum={float(tr.abs().sum()):.3e} '
                  f'wape={100*float((es-tr).abs().sum()/(tr.abs().sum()+1e-30)):.1f}%')

#!/usr/bin/env python3
"""Physics losses and masked-reconstruction metrics on decoded pu quantities.

Everything here works on COMPLETED values  x̄ = vis·x_true + msk·x̂  (observed
entries kept, predictions substituted only where masked; structural zeros stay
zero because vis and msk are subsets of act).

Element identity (docs/physics.tex): the grid is described by I_bus and Icomp
(stored separately in the corpus as I_*_bus*_pu and Icomp_*_pu). The stored
terminal TARGET is their sum,
    (I_bus_pu + Icomp_pu) = Y_pu · V_pu,
so it must never be read as the physical bus current on its own. Physical KCL
uses I_bus_pu = (I_bus_pu + Icomp_pu) - Icomp_pu; lines use the
series/shunt block form I1 = (Ys+Yh)V1 - Ys·V2, I2 = -Ys·V1 + (Ys+Yh)V2.
The element loss is expressed in the asinh feature metric (well-conditioned):
    L_elem = MSE( asinh(Y̅V̄ / s_I),
                  asinh(Ibar_feat / s_I) )
             over active terminal-current entries.

KCL holds for physical terminal current I_bus, so the KCL loss covers every
non-ground node.

The training loop keeps these calculations outside autocast: sinh amplifies
rounding, and the persisted corpus is float64 so truth checks can stay strict.
"""
from __future__ import annotations

import torch

from .data import EPS, FC, SPECS, field_layout, i_offset, n_slots, tri_rc, tri_size, y_width

_TRI_RC = {d: tri_rc(d) for d in (4, 8, 12)}

# Families whose corpus-wide current scale is numerically zero (e.g. TriplexLine,
# I_scale ~1e-9: unloaded secondaries) carry no computable I=YV signal — float32
# cancellation noise in Y·V exceeds the target itself. Exclude them from the
# ELEMENT loss only; they remain masked-reconstruction targets and KCL terms.
ELEM_MIN_I_SCALE = 1e-6


def decode(x_feat: torch.Tensor, scale: torch.Tensor, clamp: float) -> torch.Tensor:
    """Decode a model prediction, clamped so early training cannot overflow."""
    return torch.sinh(x_feat.clamp(-clamp, clamp)) * (scale + EPS)


def decode_truth(x_feat: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode persisted/observed features exactly; truth must never be clamped."""
    return torch.sinh(x_feat) * (scale + EPS)


def decode_completed(x_feat: torch.Tensor, scale: torch.Tensor, pred_mask: torch.Tensor,
                     clamp: float) -> torch.Tensor:
    """Decode xbar while clamping only entries supplied by the model.

    xbar mixes observed truth and masked predictions. Clamping the whole tensor
    silently changes large observed Y/Icomp/Ibus values and breaks the exact
    element identity even when every physical input is ground truth.
    """
    safe_feat = torch.where(pred_mask, x_feat.clamp(-clamp, clamp), x_feat)
    return torch.sinh(safe_feat) * (scale + EPS)


def completed(batch, preds) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Per-store completed x̄ plus completed node voltage (v̄_r, v̄_i)."""
    x_bar = {s: batch[s].x_true * batch[s].vis + preds[s] * batch[s].msk for s in SPECS}
    nd = batch["node"]
    dv_bar = nd.dv * nd.vis_v.unsqueeze(1) + preds["node"] * nd.msk_v.unsqueeze(1)
    v_bar = nd.v_init + dv_bar
    return x_bar, v_bar[:, 0], v_bar[:, 1]


def clamp_structural_zeros(batch, preds) -> dict[str, torch.Tensor]:
    """Clamp inactive padded component slots to zero before losses/physics."""
    out = {"node": preds["node"]}
    for store in SPECS:
        out[store] = preds[store] * batch[store].act.to(dtype=preds[store].dtype)
    return out


def _tri_to_full(tri: torch.Tensor, dim: int) -> torch.Tensor:
    rows, cols = (t.to(tri.device) for t in _TRI_RC[dim])
    m = tri.new_zeros(tri.shape[0], dim, dim)
    m[:, rows, cols] = tri
    m[:, cols, rows] = tri
    return m


def _slot_voltages(batch, store: str, vr: torch.Tensor, vi: torch.Tensor):
    """Stacked terminal voltages [n, FC·terms]; zeros at inactive slots."""
    es = batch[(store, "conn", "node")]
    comp, node, slot = es.edge_index[0], es.edge_index[1], es.slot
    n = batch[store].num_nodes
    Vr = vr.new_zeros(n, FC * SPECS[store].terms)
    Vi = vi.new_zeros(n, FC * SPECS[store].terms)
    Vr[comp, slot] = vr[node]
    Vi[comp, slot] = vi[node]
    return Vr, Vi


def _element_currents(store: str, y_pu: torch.Tensor, Vr: torch.Tensor, Vi: torch.Tensor):
    """Y·V per element in pu, from decoded packed triangles."""
    # Inference solvers deliberately preserve voltage in float64/complex128.
    # Promote Y to that dtype so downstream current/physics evaluation keeps
    # the solved low-order voltage-drop bits instead of failing or demoting.
    y_pu = y_pu.to(dtype=Vr.dtype)
    spec = SPECS[store]
    tri = tri_size(spec.ydim)
    if store == "line":
        Ys_r = _tri_to_full(y_pu[:, :tri], 4)
        Ys_i = _tri_to_full(y_pu[:, tri:2 * tri], 4)
        A_r, A_i = Ys_r, Ys_i + _tri_to_full(y_pu[:, 2 * tri:], 4)   # Ys + jYh
        V1r, V1i, V2r, V2i = Vr[:, :4], Vi[:, :4], Vr[:, 4:], Vi[:, 4:]
        def cmv(Mr, Mi, xr, xi):
            return (torch.einsum("kij,kj->ki", Mr, xr) - torch.einsum("kij,kj->ki", Mi, xi),
                    torch.einsum("kij,kj->ki", Mr, xi) + torch.einsum("kij,kj->ki", Mi, xr))
        a1r, a1i = cmv(A_r, A_i, V1r, V1i)
        s2r, s2i = cmv(Ys_r, Ys_i, V2r, V2i)
        s1r, s1i = cmv(Ys_r, Ys_i, V1r, V1i)
        a2r, a2i = cmv(A_r, A_i, V2r, V2i)
        return torch.cat([a1r - s2r, a2r - s1r], 1), torch.cat([a1i - s2i, a2i - s1i], 1)
    Yr = _tri_to_full(y_pu[:, :tri], spec.ydim)
    Yi = _tri_to_full(y_pu[:, tri:], spec.ydim)
    Ir = torch.einsum("kij,kj->ki", Yr, Vr) - torch.einsum("kij,kj->ki", Yi, Vi)
    Ii = torch.einsum("kij,kj->ki", Yr, Vi) + torch.einsum("kij,kj->ki", Yi, Vr)
    return Ir, Ii


def _slot_to_col(store: str, dev) -> torch.Tensor:
    """Map slot-major index t*FC+s to the real-part I column t*2FC+s."""
    slots_idx = torch.arange(SPECS[store].terms * FC, device=dev)
    return (slots_idx // FC) * 2 * FC + slots_idx % FC


def decoded_icomp_in_terminal_layout(batch, x_bar, store: str, clamp: float,
                                      dtype=torch.float64) -> torch.Tensor:
    """Decode Icomp and align it with the packed terminal-current columns."""
    st, spec = batch[store], SPECS[store]
    ni = i_offset(store)
    out = torch.zeros(
        st.num_nodes, 2 * spec.terms * FC, device=st.scale.device, dtype=dtype
    )
    if not spec.icomp or st.num_nodes == 0:
        return out
    ny = y_width(store)
    ic = decode_completed(
        x_bar[store][:, ny:ni].to(dtype), st.scale[:, ny:ni].to(dtype),
        st.msk[:, ny:ni], clamp,
    )
    col_r = _slot_to_col(store, st.scale.device)
    out[:, col_r] = ic[:, :spec.icomp]
    out[:, col_r + FC] = ic[:, spec.icomp:]
    return out


def decoded_physical_currents(batch, x_bar, store: str, clamp: float,
                              dtype=torch.float64) -> torch.Tensor:
    """Return physical ``I_bus = (I_bus + Icomp) - Icomp`` in terminal layout."""
    st = batch[store]
    ni = i_offset(store)
    total = decode_completed(
        x_bar[store][:, ni:].to(dtype), st.scale[:, ni:].to(dtype),
        st.msk[:, ni:], clamp,
    )
    return total - decoded_icomp_in_terminal_layout(
        batch, x_bar, store, clamp, dtype=dtype
    )


def physical_to_terminal_feature(batch, x_bar, store: str,
                                 physical: torch.Tensor, clamp: float) -> torch.Tensor:
    """Return the stored target ``I_bus + Icomp`` (= Y V) in physical pu."""
    return physical + decoded_icomp_in_terminal_layout(
        batch, x_bar, store, clamp, dtype=physical.dtype
    )


def kcl_decode_icomp(batch, preds, clamp: float):
    """Recover one hidden PC Icomp per conductor node from physical KCL.

    The identity ``sum((I_bus + Icomp) - Icomp)=0`` gives a hidden injection exactly
    when every incident terminal feature is visible and there is precisely one
    hidden Icomp at that conductor node. All other rows keep the learned head.
    """
    x_bar, _, _ = completed(batch, preds)
    dev = preds["node"].device
    n_node = batch["node"].num_nodes
    sum_r = torch.zeros(n_node, dtype=torch.float64, device=dev)
    sum_i = torch.zeros_like(sum_r)
    valid_node = torch.ones(n_node, dtype=torch.bool, device=dev)
    hidden_count = torch.zeros(n_node, dtype=torch.long, device=dev)
    candidates = {}
    pc_stores = {"load", "generator", "pvsystem", "storage"}

    for store, spec in SPECS.items():
        st = batch[store]
        if st.num_nodes == 0:
            continue
        ny, ni = y_width(store), i_offset(store)
        total = decode_completed(
            x_bar[store][:, ni:].double(), st.scale[:, ni:].double(),
            st.msk[:, ni:], clamp,
        )
        es = batch[(store, "conn", "node")]
        comp, node, slot = es.edge_index[0], es.edge_index[1], es.slot
        col_r = (slot // FC) * 2 * FC + slot % FC
        terminal_visible = (
            st.vis[comp, ni + col_r] & st.vis[comp, ni + col_r + FC]
        )
        valid_node[node[~terminal_visible]] = False
        sum_r.index_add_(0, node, total[comp, col_r])
        sum_i.index_add_(0, node, total[comp, col_r + FC])

        if not spec.icomp:
            continue
        ic = decode_completed(
            x_bar[store][:, ny:ni].double(), st.scale[:, ny:ni].double(),
            st.msk[:, ny:ni], clamp,
        )
        ic_slot_node = torch.full(
            (st.num_nodes, spec.icomp), -1, dtype=torch.long, device=dev
        )
        ic_edge = slot < spec.icomp
        ic_slot_node[comp[ic_edge], slot[ic_edge]] = node[ic_edge]
        rows, slots = (ic_slot_node >= 0).nonzero(as_tuple=True)
        if not rows.numel():
            continue
        nodes = ic_slot_node[rows, slots]
        visible = (
            st.vis[rows, ny + slots]
            & st.vis[rows, ny + spec.icomp + slots]
        )
        hidden = (
            st.msk[rows, ny + slots]
            & st.msk[rows, ny + spec.icomp + slots]
        )
        sum_r.index_add_(0, nodes[visible], -ic[rows[visible], slots[visible]])
        sum_i.index_add_(
            0, nodes[visible], -ic[rows[visible], spec.icomp + slots[visible]]
        )
        eligible = hidden if store in pc_stores else torch.zeros_like(hidden)
        eligible_nodes = nodes[eligible]
        hidden_count.index_add_(
            0, eligible_nodes, torch.ones_like(eligible_nodes)
        )
        candidates[store] = (rows[eligible], slots[eligible], eligible_nodes)
        invalid = ~(visible | eligible)
        valid_node[nodes[invalid]] = False

    out = dict(preds)
    for store in pc_stores:
        if batch[store].num_nodes:
            out[store] = preds[store].clone()
    for store, (rows, slots, nodes) in candidates.items():
        good = valid_node[nodes] & (hidden_count[nodes] == 1)
        st, spec = batch[store], SPECS[store]
        ny = y_width(store)
        rows, slots, nodes = rows[good], slots[good], nodes[good]
        real_cols = ny + slots
        imag_cols = ny + spec.icomp + slots
        encoded_r = torch.asinh(
            sum_r[nodes] / (st.scale[rows, real_cols].double() + EPS)
        )
        encoded_i = torch.asinh(
            sum_i[nodes] / (st.scale[rows, imag_cols].double() + EPS)
        )
        out[store][rows, real_cols] = encoded_r.to(out[store].dtype)
        out[store][rows, imag_cols] = encoded_i.to(out[store].dtype)
    return out


def decode_currents(batch, preds, clamp: float):
    """Replace predicted terminal-current features with exact ``Y·V``.

    The feature corpus stores the sum ``I_bus + Icomp = Y·V``. Physical KCL
    current is recovered separately as ``I_bus = (I_bus + Icomp) - Icomp``. Do not
    subtract Icomp here or the reconstruction target changes meaning.
    """
    x_bar, vr, vi = completed(batch, preds)      # I part of x_bar unused here
    out = dict(preds)
    for store, spec in SPECS.items():
        st = batch[store]
        if st.num_nodes == 0:
            continue
        ny, ni = y_width(store), i_offset(store)
        if hasattr(st, "metadata_y_pu"):
            y_pu = st.metadata_y_pu.to(device=vr.device, dtype=torch.float64)
        else:
            y_pu = decode_completed(
                x_bar[store][:, :ny], st.scale[:, :ny], st.msk[:, :ny], clamp
            )
        Vr, Vi = _slot_voltages(batch, store, vr, vi)
        Ir, Ii = _element_currents(store, y_pu, Vr, Vi)
        col_r = _slot_to_col(store, vr.device)
        ibus = Ir.new_zeros(Ir.shape[0], 2 * spec.terms * FC)
        ibus[:, col_r] = Ir
        ibus[:, col_r + FC] = Ii
        ifeat = torch.asinh(ibus / (st.scale[:, ni:] + EPS)) * st.act[:, ni:].to(ibus.dtype)
        p = (
            out[store].double().clone()
            if hasattr(st, "metadata_y_pu") else out[store].clone()
        )
        p[:, ni:] = ifeat
        out[store] = p
    return out


def exact_pf_solve(batch, preds, clamp: float, return_info: bool = False):
    """Solve the bus system Ysys·V = ΣIcomp where element physics determines it.

    When every active Y and Icomp entry of a graph is observed (the pf task),
    KCL over I_bus = Y·V − Icomp assembles a linear complex system whose
    solution IS the state — no learning needed. Masked node voltages in such
    graphs are overwritten with the solve (complex128, per-graph dense; graphs
    are tiny). Graphs with any hidden Y/Icomp, or a singular system, keep the
    head predictions. Inference-only postprocess, the pf-task ceiling.
    """
    info = {
        "graphs": 0, "eligible": 0, "solved": 0, "unstable": 0,
        "solved_v_abs_error": 0.0, "solved_v_abs_truth": 0.0,
    }

    def finish(value):
        return (value, info) if return_info else value

    x_bar, _, _ = completed(batch, preds)
    nd = batch["node"]
    N = nd.num_nodes
    gid = getattr(nd, "batch", None)
    gid = gid.cpu() if gid is not None else torch.zeros(N, dtype=torch.long)
    n_graph = int(gid.max()) + 1 if N else 0
    info["graphs"] = n_graph
    ok = torch.ones(n_graph, dtype=torch.bool)
    rows, cols, vals = [], [], []
    rhs = torch.zeros(N, dtype=torch.complex128)
    for store, spec in SPECS.items():
        st = batch[store]
        n = st.act.shape[0]
        if n == 0:
            continue
        ny, ni = y_width(store), i_offset(store)
        es = batch[(store, "conn", "node")]
        comp, node, slot = (t.cpu() for t in (es.edge_index[0], es.edge_index[1], es.slot))
        ns = n_slots(store)
        sn = torch.full((n, ns), -1, dtype=torch.long)
        sn[comp, slot] = node
        cg = torch.zeros(n, dtype=torch.long)
        cg[comp] = gid[node]
        undet = (st.act[:, :ni] & ~st.vis[:, :ni]).any(dim=1).cpu()
        ok[cg[undet]] = False
        if hasattr(st, "metadata_y_pu"):
            y_pu = st.metadata_y_pu.double().cpu()
        else:
            y_pu = decode_completed(
                x_bar[store][:, :ny].double(), st.scale[:, :ny].double(),
                st.msk[:, :ny], clamp,
            ).cpu()
        tri = tri_size(spec.ydim)
        if store == "line":
            Ys = torch.complex(_tri_to_full(y_pu[:, :tri], 4), _tri_to_full(y_pu[:, tri:2 * tri], 4))
            A = Ys + 1j * _tri_to_full(y_pu[:, 2 * tri:], 4)
            Yf = torch.cat([torch.cat([A, -Ys], 2), torch.cat([-Ys, A], 2)], 1)
        else:
            Yf = torch.complex(_tri_to_full(y_pu[:, :tri], spec.ydim),
                               _tri_to_full(y_pu[:, tri:], spec.ydim))
        pr = sn.unsqueeze(2).expand(n, ns, ns)
        pc = sn.unsqueeze(1).expand(n, ns, ns)
        m = (pr >= 0) & (pc >= 0)
        rows.append(pr[m])
        cols.append(pc[m])
        vals.append(Yf[m])
        if spec.icomp:
            ic = decode_completed(x_bar[store][:, ny:ni].double(), st.scale[:, ny:ni].double(),
                                  st.msk[:, ny:ni], clamp).cpu()
            icc = torch.complex(ic[:, :spec.icomp], ic[:, spec.icomp:])
            v = sn >= 0
            rhs.index_add_(0, sn[v], icc[v])
    if not rows or not bool(ok.any()):
        return finish(preds)
    R, C, V = torch.cat(rows), torch.cat(cols), torch.cat(vals)
    eg = gid[R]
    order = torch.argsort(eg)
    R, C, V, eg = R[order], C[order], V[order], eg[order]
    esplit = torch.bincount(eg, minlength=n_graph).tolist()
    norder = torch.argsort(gid)
    nsplit = torch.bincount(gid, minlength=n_graph).tolist()
    local = torch.empty(N, dtype=torch.long)
    ground = nd.ground.cpu()
    visible_v = nd.vis_v.cpu()
    known_v = torch.complex(
        (nd.v_init[:, 0] + nd.dv[:, 0]).double().cpu(),
        (nd.v_init[:, 1] + nd.dv[:, 1]).double().cpu(),
    )
    msk_v = nd.msk_v.cpu()
    vinit = nd.v_init.double().cpu()
    # Preserve the complex128 solve through downstream YV evaluation.  A
    # float32 cast loses low voltage-drop bits that stiff line admittances
    # amplify into catastrophic current error, even when V WAPE is tiny.
    out_node = preds["node"].double().clone()
    info["eligible"] = int(ok.sum())
    e0 = n0 = 0
    for g in range(n_graph):
        ecnt, ncnt = esplit[g], nsplit[g]
        if not ok[g] or ncnt == 0 or ecnt == 0:
            e0, n0 = e0 + ecnt, n0 + ncnt
            continue
        nidx = norder[n0:n0 + ncnt]
        local[nidx] = torch.arange(ncnt)
        A = torch.zeros(ncnt, ncnt, dtype=torch.complex128)
        A.index_put_((local[R[e0:e0 + ecnt]], local[C[e0:e0 + ecnt]]),
                     V[e0:e0 + ecnt], accumulate=True)
        b = rhs[nidx].clone()
        # PF exposes ground and the three solved slack-phase voltages as
        # boundary conditions. Solving the Vsource internal equivalent for
        # those already-observed voltages is both unnecessary and numerically
        # fragile; anchor every visible boundary exactly.
        fixed_mask = visible_v[nidx] | ground[nidx]
        fixed = local[nidx[fixed_mask]]
        A[fixed, :] = 0
        A[fixed, fixed] = 1
        b[fixed] = known_v[nidx[fixed_mask]]
        try:
            LU, piv = torch.linalg.lu_factor(A)
            x = torch.linalg.lu_solve(LU, piv, b.unsqueeze(1)).squeeze(1)
            residual = (A @ x - b).abs().max()
            backward_scale = (
                A.abs().matmul(x.abs()) + b.abs()
            ).max().clamp_min(1.0)
            # A large response to an arbitrary RHS perturbation is expected for
            # physically stiff distribution systems and was incorrectly
            # rejecting valid feeders. Fail only on a non-finite or inaccurate
            # solve; LU itself raises for a genuinely singular system.
            if not bool(
                torch.isfinite(x.real).all()
                and torch.isfinite(x.imag).all()
                and residual <= 1e-10 * backward_scale
            ):
                raise RuntimeError
        except RuntimeError:
            info["unstable"] += 1
            e0, n0 = e0 + ecnt, n0 + ncnt
            continue
        info["solved"] += 1
        hidden = msk_v[nidx]
        if bool(hidden.any()):
            info["solved_v_abs_error"] += float(
                (x[hidden] - known_v[nidx][hidden]).abs().sum()
            )
            info["solved_v_abs_truth"] += float(
                known_v[nidx][hidden].abs().sum()
            )
        dv = torch.stack([x.real, x.imag], 1) - vinit[nidx]
        take = msk_v[nidx]
        out_node[nidx[take].to(out_node.device)] = dv[take].to(out_node.device, out_node.dtype)
        e0, n0 = e0 + ecnt, n0 + ncnt
    return finish({**preds, "node": out_node})


def selective_decode_currents(batch, preds, clamp: float):
    """Overwrite masked current entries that physics fully determines.

    A component row is "determined" when every active Y and Icomp entry is
    observed and every incident node voltage is observed — there the stored
    terminal feature Y·V is exact (machine precision on truth data), so the head is
    replaced. Elsewhere (masked Y or masked V) the head stays: decoding through
    predictions amplifies errors by stiff |Y| (E5/E5b evidence). Inference-time
    postprocess only — never used in training.
    """
    x_bar, vr, vi = completed(batch, preds)
    out = dict(preds)
    for store, spec in SPECS.items():
        st = batch[store]
        if st.num_nodes == 0:
            continue
        ny, ni = y_width(store), i_offset(store)
        params_ok = (st.vis[:, :ni] | ~st.act[:, :ni]).all(dim=1)
        es = batch[(store, "conn", "node")]
        comp, node = es.edge_index[0], es.edge_index[1]
        bad = torch.zeros(st.num_nodes, device=vr.device)
        bad.index_add_(0, comp, (~batch["node"].vis_v[node]).float())
        row_ok = params_ok & (bad == 0)
        if not bool(row_ok.any()):
            continue
        y_pu = decode_completed(x_bar[store][:, :ny], st.scale[:, :ny],
                                st.msk[:, :ny], clamp)
        Vr, Vi = _slot_voltages(batch, store, vr, vi)
        Ir, Ii = _element_currents(store, y_pu, Vr, Vi)
        col_r = _slot_to_col(store, vr.device)
        ibus = Ir.new_zeros(Ir.shape[0], 2 * spec.terms * FC)
        ibus[:, col_r] = Ir
        ibus[:, col_r + FC] = Ii
        ifeat = torch.asinh(ibus / (st.scale[:, ni:] + EPS)) * st.act[:, ni:].to(ibus.dtype)
        take = row_ok.unsqueeze(1) & st.msk[:, ni:]
        p = out[store].clone()
        p[:, ni:] = torch.where(take, ifeat, p[:, ni:])
        out[store] = p
    return out


def kcl_decode_vsource(batch, preds, clamp: float):
    """Determine masked vsource terminal current from KCL at the slack phases.

    Slack voltage is an observed boundary condition, but source balancing
    current is not. At a non-ground node incident to exactly one vsource slot,
    KCL uniquely gives that current as minus every other completed terminal
    current. The paired ground-terminal current is its negative (an exact
    corpus invariant). This is a local structural decoder, not a PF solve.
    """
    st_src = batch["vsource"]
    if st_src.num_nodes == 0:
        return preds
    dev = preds["node"].device
    x_bar, _, _ = completed(batch, preds)
    n_node = batch["node"].num_nodes
    other_r = torch.zeros(n_node, dtype=torch.float64, device=dev)
    other_i = torch.zeros_like(other_r)

    for store in SPECS:
        if store == "vsource" or batch[store].num_nodes == 0:
            continue
        st = batch[store]
        ni = i_offset(store)
        ibus = decoded_physical_currents(batch, x_bar, store, clamp)
        es = batch[(store, "conn", "node")]
        comp, node, slot = es.edge_index[0], es.edge_index[1], es.slot
        col_r = (slot // FC) * 2 * FC + slot % FC
        other_r.index_add_(0, node, ibus[comp, col_r])
        other_i.index_add_(0, node, ibus[comp, col_r + FC])

    es = batch[("vsource", "conn", "node")]
    comp, node, slot = es.edge_index[0], es.edge_index[1], es.slot
    source_count = torch.zeros(n_node, dtype=torch.long, device=dev)
    source_count.index_add_(0, node, torch.ones_like(node))
    slot_node = torch.full((st_src.num_nodes, 2 * FC), -1, dtype=torch.long, device=dev)
    slot_node[comp, slot] = node

    ni = i_offset("vsource")
    p = preds["vsource"].clone()
    for phase in range(FC):
        src_node = slot_node[:, phase]
        valid = src_node >= 0
        safe_node = src_node.clamp(min=0)
        valid &= ~batch["node"].ground[safe_node]
        valid &= source_count[safe_node] == 1
        if not valid.any():
            continue
        ir = -other_r[safe_node]
        ii = -other_i[safe_node]
        physical = torch.zeros(
            st_src.num_nodes, 2 * SPECS["vsource"].terms * FC,
            dtype=torch.float64, device=dev,
        )
        physical[:, phase] = ir
        physical[:, phase + FC] = ii
        physical[:, 2 * FC + phase] = -ir
        physical[:, 3 * FC + phase] = -ii
        terminal_feature = physical_to_terminal_feature(
            batch, x_bar, "vsource", physical, clamp
        )
        for col in (phase, phase + FC, 2 * FC + phase, 3 * FC + phase):
            take = valid & st_src.msk[:, ni + col]
            encoded = torch.asinh(
                terminal_feature[:, col] / (st_src.scale[:, ni + col].double() + EPS)
            )
            p[:, ni + col] = torch.where(take, encoded.to(p.dtype), p[:, ni + col])
    return {**preds, "vsource": p}


def vdrop_loss(batch, vr, vi, floor: float = 1e-6):
    """Relative supervision of completed line terminal-voltage differences.

    Stiff series currents are set by V1−V2 (~1e-3 pu), not by the ~1 pu bus
    voltages; this trains exactly that quantity (PINN recipe). Counted per
    conductor slot where both terminals are present and at least one is masked.
    """
    st = batch[("line", "conn", "node")]
    nd = batch["node"]
    if st.edge_index.numel() == 0:
        return torch.zeros((), device=vr.device)
    n = batch["line"].num_nodes
    Vr, Vi = _slot_voltages(batch, "line", vr, vi)                    # completed
    v_true = nd.v_init + nd.dv
    Tr, Ti = _slot_voltages(batch, "line", v_true[:, 0], v_true[:, 1])  # truth
    comp, node, slot = st.edge_index[0], st.edge_index[1], st.slot
    present = torch.zeros(n, 2 * FC, dtype=torch.bool, device=vr.device)
    hidden = torch.zeros_like(present)
    present[comp, slot] = True
    hidden[comp, slot] = nd.msk_v[node]
    pair = present[:, :FC] & present[:, FC:] & (hidden[:, :FC] | hidden[:, FC:])
    if not bool(pair.any()):
        return torch.zeros((), device=vr.device)
    pd_r = (Vr[:, :FC] - Vr[:, FC:]) - (Tr[:, :FC] - Tr[:, FC:])
    pd_i = (Vi[:, :FC] - Vi[:, FC:]) - (Ti[:, :FC] - Ti[:, FC:])
    true_sq = (Tr[:, :FC] - Tr[:, FC:]) ** 2 + (Ti[:, :FC] - Ti[:, FC:]) ** 2
    # Huber on the RELATIVE drop error r = |Δdrop|/|drop|: quadratic near zero,
    # linear (constant, nonzero gradient) for large r. Neither the raw ratio
    # (rel ~ 1e6 on an untrained model, dominates everything) nor log1p (its
    # gradient 1/(1+rel) vanishes exactly where the signal is needed — E6b's
    # V collapse) behaves at both ends.
    # Index BEFORE sqrt: unselected entries can be exactly 0 and sqrt'(0)=inf
    # turns zero upstream gradient into NaN (0·inf) — E7's step-1 NaN. The eps
    # guards exact zeros inside selected pairs.
    rel = (pd_r ** 2 + pd_i ** 2) / (true_sq + floor)
    r = (rel[pair] + 1e-12).sqrt()
    return torch.where(r < 1, 0.5 * r ** 2, r - 0.5).mean()


def physics_losses(batch, x_bar, vr, vi, clamp: float, s_kcl: float,
                   detach_currents: bool = False):
    """Returns (loss_elem, loss_kcl, metrics dict). All in float32/pu.

    detach_currents: stop-gradient on the current side of the element identity —
    physics then regularizes predicted V/Y toward consistency with the (mostly
    observed) currents, without dragging predicted currents toward an imperfect
    Y·V (E3 showed that pull hurts current reconstruction; experiments.md).
    """
    dev = vr.device
    elem_sq = torch.zeros((), device=dev)
    elem_abs = torch.zeros((), device=dev)
    n_elem = torch.zeros((), device=dev)
    kcl_r = vr.new_zeros(vr.shape[0])
    kcl_i = vr.new_zeros(vr.shape[0])

    for store, spec in SPECS.items():
        st = batch[store]
        if st.num_nodes == 0:
            continue
        ny = y_width(store)
        ni = i_offset(store)
        y_pu = decode_completed(x_bar[store][:, :ny], st.scale[:, :ny],
                                st.msk[:, :ny], clamp)
        Vr, Vi = _slot_voltages(batch, store, vr, vi)
        Ir, Ii = _element_currents(store, y_pu, Vr, Vi)

        # I columns per terminal are [r FC | i FC]: slot t*FC+s -> col t*2FC+s.
        # Icomp packing is slot-major too, so the same column map applies.
        col_r = _slot_to_col(store, dev)
        yv_pred = Ir.new_zeros(Ir.shape[0], 2 * spec.terms * FC)
        yv_pred[:, col_r] = Ir
        yv_pred[:, col_r + FC] = Ii
        terminal_total = decode_completed(
            x_bar[store][:, ni:], st.scale[:, ni:], st.msk[:, ni:], clamp
        ).to(Ir.dtype)
        elem_bar = terminal_total
        physical_ibus = terminal_total.clone()
        if spec.icomp:
            ic = decode_completed(x_bar[store][:, ny:ni], st.scale[:, ny:ni],
                                  st.msk[:, ny:ni], clamp).to(Ir.dtype)
            physical_ibus[:, col_r] -= ic[:, :spec.icomp]
            physical_ibus[:, col_r + FC] -= ic[:, spec.icomp:]
        i_scale = (st.scale[:, ni:] + EPS).to(Ir.dtype)
        act_i = (st.act[:, ni:] & (st.scale[:, ni:] > ELEM_MIN_I_SCALE)).to(Ir.dtype)
        elem_side = elem_bar.detach() if detach_currents else elem_bar
        diff = (torch.asinh(yv_pred / i_scale) - torch.asinh(elem_side / i_scale)) * act_i
        # Huber (delta=1): stiff elements (line/xfmr) amplify masked-V errors by
        # Y ~ 1e4-1e6 pu, giving asinh diffs of 5-10 with near-singular gradients
        # that fight the mask loss (E1 vs E2b in experiments.md). Linearizing the
        # tail keeps the power-flow coupling without letting it dominate.
        a = diff.abs()
        elem_sq = elem_sq + torch.where(a < 1, 0.5 * diff ** 2, a - 0.5).sum()
        n_elem = n_elem + act_i.sum()
        elem_abs = elem_abs + ((yv_pred - elem_bar).abs() * act_i).sum()

        es = batch[(store, "conn", "node")]
        comp, node, slot = es.edge_index[0], es.edge_index[1], es.slot
        # I columns per terminal are [r FC | i FC]: slot t*FC+s -> col t*2FC+s
        col_r = (slot // FC) * 2 * FC + slot % FC
        kcl_r.index_add_(0, node, physical_ibus[comp, col_r])
        kcl_i.index_add_(0, node, physical_ibus[comp, col_r + FC])

    kcl_mask = batch["node"].kcl_mask
    res_r, res_i = kcl_r[kcl_mask], kcl_i[kcl_mask]
    loss_kcl = (torch.asinh(res_r / s_kcl) ** 2 + torch.asinh(res_i / s_kcl) ** 2).mean() \
        if kcl_mask.any() else torch.zeros((), device=dev)
    loss_elem = elem_sq / n_elem.clamp(min=1)
    metrics = {
        "elem_residual_pu": (elem_abs / n_elem.clamp(min=1)).item(),
        "kcl_residual_pu": torch.hypot(res_r, res_i).mean().item() if kcl_mask.any() else 0.0,
    }
    return loss_elem, loss_kcl, metrics


def detailed_metrics(batch, preds, clamp: float) -> dict[str, float]:
    """Multi-lens eval report (eval-only; too heavy for the train loop).

    Relative errors answer "how many % off": V against |V_true| (≈1 pu, floored
    at 0.1), I/Y against |truth| floored at the family scale (so near-zero
    truths don't explode the ratio). Percentiles expose the tail the mean
    hides; the depth buckets show how error grows with electrical distance
    from the source — the power-flow propagation lens.
    """
    out: dict[str, float] = {}
    nd = batch["node"]
    mv = nd.msk_v
    if mv.any():
        err = (preds["node"].float() - nd.dv)[mv].norm(dim=1)
        vt = (nd.v_init + nd.dv)[mv].norm(dim=1).clamp(min=0.1)
        rel = err / vt
        q = torch.quantile(rel, torch.tensor([0.5, 0.95], device=rel.device, dtype=rel.dtype))
        out["V_rel_p50"], out["V_rel_p95"], out["V_rel_max"] = q[0].item(), q[1].item(), rel.max().item()
        d = nd.depth[mv]
        for name, lo, hi in (("d0_2", 0, 2), ("d3_5", 3, 5), ("d6p", 6, 98)):
            sel = (d >= lo) & (d <= hi)
            if sel.any():
                out[f"V_rel_{name}"] = rel[sel].mean().item()
        # Feasibility judgment: does the predicted state detect ANSI voltage
        # violations (|V| outside [0.95, 1.05] pu)? Recall is the operator's
        # metric — a missed violation costs more than a false alarm.
        vmag_hat = (nd.v_init + preds["node"].float())[mv].norm(dim=1)
        vmag_true = (nd.v_init + nd.dv)[mv].norm(dim=1)
        bad_hat = (vmag_hat < 0.95) | (vmag_hat > 1.05)
        bad_true = (vmag_true < 0.95) | (vmag_true > 1.05)
        out["feas_acc"] = (bad_hat == bad_true).float().mean().item()
        out["feas_viol_rate"] = bad_true.float().mean().item()
        if bad_true.any():
            out["feas_recall"] = (bad_hat & bad_true).float().sum().item() / bad_true.sum().item()
        if bad_hat.any():
            out["feas_precision"] = (bad_hat & bad_true).float().sum().item() / bad_hat.sum().item()

    # Current-error decomposition over masked I entries (all magnitude-weighted,
    # sum|err|/sum|I_true|, so tiny currents cannot dominate a ratio):
    #   I_head_magw        the head's direct prediction
    #   I_dec_truthV_magw  stored I_bus + Icomp = Y·V with TRUTH Y/V — isolates
    #                      decoder/data defects (≈0 on a determinate corpus)
    #   I_dec_predV_magw   same physics through PREDICTED V — isolates how V
    #                      error amplifies through stiff |Y|
    nd_ = batch["node"]
    v_hat = nd_.v_init + nd_.dv * nd_.vis_v.unsqueeze(1) + preds["node"].float() * nd_.msk_v.unsqueeze(1)
    v_tru = nd_.v_init + nd_.dv
    num_h = num_dt = num_dp = den_magw = 0.0
    i_rel_all, y_rel_all = [], []
    for s in SPECS:
        st = batch[s]
        if st.num_nodes == 0:
            continue
        spec = SPECS[s]
        ny, ni = y_width(s), i_offset(s)
        m = st.msk
        pu_p = decode(preds[s].float(), st.scale, clamp)
        pu_t = decode_truth(st.x_true, st.scale)
        rel = (pu_p - pu_t).abs() / (pu_t.abs() + st.scale)
        mi = m[:, ny:]
        if mi.any():
            r = rel[:, ny:][mi]
            out[f"I_rel_{s}"] = r.mean().item()
            i_rel_all.append(r)
            y_tru = pu_t[:, :ny]
            col_r = _slot_to_col(s, v_hat.device)
            ibus = {}
            for name, vr_, vi_ in (("t", v_tru[:, 0], v_tru[:, 1]), ("p", v_hat[:, 0], v_hat[:, 1])):
                Vr, Vi = _slot_voltages(batch, s, vr_, vi_)
                Ir, Ii = _element_currents(s, y_tru, Vr, Vi)
                b = Ir.new_zeros(Ir.shape[0], 2 * spec.terms * FC)
                b[:, col_r] = Ir
                b[:, col_r + FC] = Ii
                ibus[name] = b
            mI = m[:, ni:]          # terminal currents only (Icomp is an input,
            it = pu_t[:, ni:]       # not Y·V-decodable)
            den_magw += it.abs()[mI].sum().item()
            num_h += (pu_p[:, ni:] - it).abs()[mI].sum().item()
            num_dt += (ibus["t"] - it).abs()[mI].sum().item()
            num_dp += (ibus["p"] - it).abs()[mI].sum().item()
        my = m[:, :ny]
        if my.any():
            y_rel_all.append(rel[:, :ny][my])
    if den_magw > 0:
        out["I_head_magw"] = num_h / den_magw
        out["I_dec_truthV_magw"] = num_dt / den_magw
        out["I_dec_predV_magw"] = num_dp / den_magw
    for tag, chunks in (("I", i_rel_all), ("Y", y_rel_all)):
        if chunks:
            r = torch.cat(chunks)
            q = torch.quantile(r, torch.tensor([0.5, 0.95], device=r.device, dtype=r.dtype))
            out[f"{tag}_rel_p50"], out[f"{tag}_rel_p95"], out[f"{tag}_rel_max"] = \
                q[0].item(), q[1].item(), r.max().item()
    return out


def percentage_error_sums(batch, preds, clamp: float) -> dict[str, float]:
    """Additive terms for split-level magnitude-weighted percent errors.

    WAPE is used instead of entrywise MAPE because valid grid tensors contain
    many exact and near zeros. The caller sums these terms over the complete
    evaluation split before multiplying num / den by 100.
    """
    out = {f"{k}_{part}": 0.0
           for k in ("V", "V_r", "V_i", "I", "I_r", "I_i", "Y", "Y_r", "Y_i")
           for part in ("num", "den")}

    def add_wape(key: str, pred: torch.Tensor, truth: torch.Tensor,
                 mask: torch.Tensor) -> None:
        """Accumulate a named WAPE without collapsing unlike current roles."""
        out.setdefault(f"{key}_num", 0.0)
        out.setdefault(f"{key}_den", 0.0)
        if mask.any():
            out[f"{key}_num"] += (pred - truth).abs()[mask].sum().item()
            out[f"{key}_den"] += truth.abs()[mask].sum().item()
    nd = batch["node"]
    mv = nd.msk_v
    if mv.any():
        out["V_num"] = (preds["node"].float() - nd.dv)[mv].norm(dim=1).sum().item()
        out["V_den"] = (nd.v_init + nd.dv)[mv].norm(dim=1).sum().item()
        v_err = (preds["node"].float() - nd.dv)[mv]
        v_true = (nd.v_init + nd.dv)[mv]
        out["V_r_num"] = v_err[:, 0].abs().sum().item()
        out["V_i_num"] = v_err[:, 1].abs().sum().item()
        out["V_r_den"] = v_true[:, 0].abs().sum().item()
        out["V_i_den"] = v_true[:, 1].abs().sum().item()
    for store in SPECS:
        st = batch[store]
        if st.num_nodes == 0:
            continue
        ny = y_width(store)
        ni = i_offset(store)
        pu_p = decode(preds[store].float(), st.scale, clamp)
        pu_t = decode_truth(st.x_true, st.scale)
        for key, cols in (("Y", slice(0, ny)), ("I", slice(ny, None))):
            m = st.msk[:, cols]
            if m.any():
                out[f"{key}_num"] += (pu_p[:, cols] - pu_t[:, cols]).abs()[m].sum().item()
                out[f"{key}_den"] += pu_t[:, cols].abs()[m].sum().item()

        # Report current roles separately. The stored terminal target is the SUM
        # I_bus + Icomp (= Y V); physical I_bus = (I_bus + Icomp) - Icomp is
        # derived for KCL only. The vocabulary is just I_bus and Icomp, so the
        # sum is named for what it is. "Ifeat" (jargon) and bare "Ibus" (which
        # read as the physical current and caused the T98 double-Icomp bug) are
        # still emitted as aliases so historical receipts stay comparable.
        mi_comp = st.msk[:, ny:ni]
        mi_bus = st.msk[:, ni:]
        add_wape(f"Y_{store}", pu_p[:, :ny], pu_t[:, :ny], st.msk[:, :ny])
        add_wape("Icomp", pu_p[:, ny:ni], pu_t[:, ny:ni], mi_comp)
        add_wape("Ibus_plus_Icomp", pu_p[:, ni:], pu_t[:, ni:], mi_bus)
        add_wape("Ifeat", pu_p[:, ni:], pu_t[:, ni:], mi_bus)      # legacy alias
        add_wape("Ibus", pu_p[:, ni:], pu_t[:, ni:], mi_bus)       # legacy alias
        add_wape(f"Icomp_{store}", pu_p[:, ny:ni], pu_t[:, ny:ni], mi_comp)
        add_wape(f"Ibus_plus_Icomp_{store}", pu_p[:, ni:], pu_t[:, ni:], mi_bus)
        add_wape(f"Ifeat_{store}", pu_p[:, ni:], pu_t[:, ni:], mi_bus)
        add_wape(f"Ibus_{store}", pu_p[:, ni:], pu_t[:, ni:], mi_bus)
        # Preserve the corpus layout's explicit real/imaginary channels in the
        # report.  This includes Y triangles, Icomp, and every terminal Ibus.
        col = 0
        for field, width in field_layout(store):
            part = "r" if "_r_" in field else "i"
            family = "Y" if col < ny else "I"
            role = "Y" if col < ny else ("Icomp" if col < ni else "Ibus")
            m = st.msk[:, col:col + width]
            if m.any():
                key = f"{family}_{part}"
                out[f"{key}_num"] += (pu_p[:, col:col + width] - pu_t[:, col:col + width]).abs()[m].sum().item()
                out[f"{key}_den"] += pu_t[:, col:col + width].abs()[m].sum().item()
                add_wape(
                    f"{role}_{store}_{part}", pu_p[:, col:col + width],
                    pu_t[:, col:col + width], m,
                )
                add_wape(
                    f"field_{store}_{field}", pu_p[:, col:col + width],
                    pu_t[:, col:col + width], m,
                )
                # A family-scale denominator keeps structurally valid zero and
                # near-zero fields visible in the scorecard without MAPE blowup.
                scale = st.scale[:, col:col + width]
                skey = f"field_{store}_{field}_scale"
                out.setdefault(f"{skey}_num", 0.0)
                out.setdefault(f"{skey}_den", 0.0)
                out[f"{skey}_num"] += (
                    pu_p[:, col:col + width] - pu_t[:, col:col + width]
                ).abs()[m].sum().item()
                out[f"{skey}_den"] += scale[m].sum().item()
            col += width
    return out


def mask_loss_and_metrics(batch, preds, clamp: float, raw_preds=None, huber_i: bool = False,
                          i_weight: float = 1.0):
    """Feat-space masked loss + pu-space MAE metrics + structural-zero error.

    huber_i: Huber (delta=1) instead of MSE on the current columns — with
    decoded currents a masked stiff-line current inherits the Y·V error
    amplification, and the linear tail keeps that from dominating the loss.
    i_weight: weight of the current columns in the masked loss (weighted mean).
    Decoded stiff-element currents start ~5-10 Huber units off and outnumber
    the V/Y terms; at weight 1.0 they throttle V/Y learning (E5 vs E5b).
    """
    dev = preds["node"].device
    sq = torch.zeros((), device=dev)
    cnt = torch.zeros((), device=dev)
    mae = {k: [torch.zeros((), device=dev), torch.zeros((), device=dev)] for k in ("V", "I", "Y")}
    sz_sum = torch.zeros((), device=dev)
    sz_cnt = torch.zeros((), device=dev)

    nd = batch["node"]
    mv = nd.msk_v.unsqueeze(1).to(preds["node"].dtype)
    dv_err = (preds["node"] - nd.dv) * mv
    sq = sq + (dv_err ** 2).sum()
    cnt = cnt + 2 * mv.sum()
    mae["V"][0] += dv_err.abs().sum()
    mae["V"][1] += 2 * mv.sum()

    for store in SPECS:
        st = batch[store]
        if st.num_nodes == 0:
            continue
        m = st.msk.to(preds[store].dtype)
        err = (preds[store] - st.x_true) * m
        ni = i_offset(store)
        i_term = err[:, ni:].abs()
        i_term = torch.where(i_term < 1, 0.5 * i_term ** 2, i_term - 0.5) if huber_i \
            else err[:, ni:] ** 2
        sq = sq + (err[:, :ni] ** 2).sum() + i_weight * i_term.sum()
        cnt = cnt + m[:, :ni].sum() + i_weight * m[:, ni:].sum()
        ny = y_width(store)
        pu_err = (decode(preds[store], st.scale, clamp) - decode_truth(st.x_true, st.scale)).abs() * m
        for key, cols in (("Y", slice(0, ny)), ("I", slice(ny, None))):
            mae[key][0] += pu_err[:, cols].sum()
            mae[key][1] += m[:, cols].sum()
        pad = (~st.act).to(preds[store].dtype)
        sz_source = preds[store] if raw_preds is None else raw_preds[store]
        sz_sum = sz_sum + (sz_source.abs() * pad).sum()
        sz_cnt = sz_cnt + pad.sum()

    loss_mask = sq / cnt.clamp(min=1)
    metrics = {f"{k}_mae_pu": (a / b.clamp(min=1)).item() for k, (a, b) in mae.items()}
    metrics["structzero_err"] = (sz_sum / sz_cnt.clamp(min=1)).item()
    return loss_mask, metrics

#!/usr/bin/env python3
"""Online physics-aware masking for GridFM pretraining.

Adds to each sample, per component store:
    msk [n, W] bool   entries hidden from the model (loss targets); msk ⊆ act
    vis [n, W] bool   entries the model sees: act & ~msk
and per node:
    msk_v, vis_v [n] bool   over the solved-voltage pair (V_init is NEVER masked)

Rules (prompt.md): only connectivity-active entries may be masked; structural
zeros never are; ground is structural (0), and the three slack phase voltages
are known PF boundary conditions, so all four stay visible. Real/imag parts of
one complex entry are masked together — a Y tri
entry masks all its parts (line: Ys_r, Ys_i, Yh_i), an I slot masks (r, i).

Strategies (independent, OR-combined):
    p_voltage    per-node solved-voltage masking
    p_current    per active (terminal, slot) current entry
    p_icomp      per Icomp slot (defaults to p_current when absent from cfg)
    p_admittance per active Y tri entry
    p_terminal   per component: hide one whole terminal's current block
    p_component  per component: hide every physical field (Y + I)

Icomp is the component injection current -- the load/generation itself. pf
mode therefore keeps it VISIBLE (p_icomp=0): "given topology + loads, solve
PF" is well-posed; masking Icomp too made pf underdetermined (the model could
only guess the loading level, ~10% V error compounding with depth).

These p_* rates apply ONLY when cfg has no "mixture", or when the drawn mode is
"random". Every other named mode hardcodes the rates its capability requires --
see _MODE_HONORS_RATES and validate_mask_cfg, which warns when a config sets
rates that its mixture can never read.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData

from .data import FC, SPECS, i_offset, tri_size, y_width


def _bern(rng: np.random.Generator, shape, p: float) -> torch.Tensor:
    if p <= 0.0:
        return torch.zeros(shape, dtype=torch.bool)
    return torch.from_numpy(rng.random(shape) < p)


_RATES = ("p_voltage", "p_current", "p_admittance", "p_terminal", "p_component")
_OFF = dict(p_voltage=0.0, p_current=0.0, p_icomp=0.0, p_admittance=0.0,
            p_terminal=0.0, p_component=0.0)
# "ctrl" masks the parameter blocks of these stores only (source setpoint lives
# in the vsource Y/Icomp, taps in the transformer Y): given the operating
# point, infer the control settings that produce it.
CTRL_STORES = ("vsource", "transformer")
PC_STORES = ("load", "generator", "pvsystem", "storage")


def _mask_identifiable_injections(data: HeteroData, rng: np.random.Generator) -> None:
    """Hide a maximal random set of PC Icomp rows with no shared conductor node."""
    candidates = []
    for store in PC_STORES:
        st, spec = data[store], SPECS[store]
        if st.num_nodes == 0:
            continue
        ny, ni = y_width(store), i_offset(store)
        es = data[(store, "conn", "node")]
        slot_node = torch.full((st.num_nodes, spec.icomp), -1, dtype=torch.long)
        valid = es.slot < spec.icomp
        slot_node[es.edge_index[0, valid], es.slot[valid]] = es.edge_index[1, valid]
        active = st.act[:, ny:ny + spec.icomp] | st.act[:, ny + spec.icomp:ni]
        for row in range(st.num_nodes):
            nodes = slot_node[row].clamp_min(0)
            # Ground is an external reference, not a KCL equation. Its Norton
            # slot cannot be inferred from the modeled network balance.
            eligible = (
                active[row]
                & (slot_node[row] >= 0)
                & data["node"].kcl_mask[nodes]
            )
            slots = torch.where(eligible)[0]
            if slots.numel():
                candidates.append((store, row, slots, slot_node[row, slots]))

    taken = torch.zeros(data["node"].num_nodes, dtype=torch.bool)
    for index in rng.permutation(len(candidates)):
        store, row, slots, nodes = candidates[int(index)]
        if bool(taken[nodes].any()):
            continue
        st = data[store]
        ny, ni = y_width(store), i_offset(store)
        st.msk[row, ny + slots] = True
        st.msk[row, ny + SPECS[store].icomp + slots] = True
        st.vis[row] = st.act[row] & ~st.msk[row]
        taken[nodes] = True


def _effective_rates(cfg: dict, rng: np.random.Generator) -> tuple[dict, str]:
    """Per-sample task mode from cfg["mixture"] (mode -> probability).

    Each mode is one downstream capability expressed as a mask pattern:
      pf     parameters/injections visible, whole solved state (V, Ibus) hidden
      se     partial V/I measurements visible -> infer full state
      se_known partial V/I measurements, with component injections observed
      param_one hide one active complex Y entry per component
      injection hide Icomp with Y, V, and Ibus observed
      random_safe randomly choose one identifiable task above per sample
      random jitter the configured entry/terminal mask rates
      param  state visible, most admittances hidden -> parameter identification
      ctrl   state + other params visible, vsource/transformer params hidden
             -> infer the control settings behind an operating point
      topo   heavy whole-component masking -> damaged-graph inference
      (rest) base rates jittered by U(0.5, 1.5)
    Legacy keys p_pf/p_se are accepted as pf/se.
    """
    mix = cfg.get("mixture")

    def base_rates() -> dict:
        # Built only on the paths that actually read the rates, so a mode-only
        # config (mixture={injection: 1.0}) may omit keys it cannot use rather
        # than list them and imply a control it does not have. The KeyError is
        # kept for the paths that DO read them -- there a missing key is a typo.
        out = {k: float(cfg[k]) for k in _RATES}
        out["p_icomp"] = float(cfg.get("p_icomp", cfg["p_current"]))
        return out

    if not mix:
        return base_rates(), "base"
    modes = {k.removeprefix("p_"): float(v) for k, v in mix.items()}
    u, acc = rng.random(), 0.0
    for mode, p in modes.items():
        acc += p
        if u >= acc:
            continue
        if mode == "pf":
            return {**_OFF, "p_voltage": 1.0, "p_current": 1.0}, mode
        if mode == "se":
            # Observability continuum (physics view: pf IS se with nothing
            # observed). Default ranges keep legacy behavior; cfg se_v_hi/
            # se_i_hi = 1.0 lets se sweep INTO the pf endpoint so the model
            # never faces an untrained masking regime, and se_icomp_free
            # decouples load observability (pf keeps loads visible).
            pv = rng.uniform(float(cfg.get("se_v_lo", 0.4)), float(cfg.get("se_v_hi", 0.9)))
            pc = rng.uniform(float(cfg.get("se_i_lo", 0.2)), float(cfg.get("se_i_hi", 0.6)))
            pi = rng.uniform(0.0, 1.0) if cfg.get("se_icomp_free") else pc
            return {**_OFF, "p_voltage": pv, "p_current": pc, "p_icomp": pi}, mode
        if mode == "se_known":
            pv = rng.uniform(float(cfg.get("se_v_lo", 0.4)), float(cfg.get("se_v_hi", 0.9)))
            pc = rng.uniform(float(cfg.get("se_i_lo", 0.2)), float(cfg.get("se_i_hi", 0.6)))
            return {**_OFF, "p_voltage": pv, "p_current": pc}, mode
        if mode == "param_one":
            # The exact one-entry mask is constructed from each component's
            # active Y pattern in apply_masks.  With V/Ibus/Icomp observed this
            # avoids the grossly underdetermined 50-100% parameter mask.
            return dict(_OFF), mode
        if mode == "injection":
            # A global post-pass selects at most one hidden PC Icomp at each
            # conductor node. Hiding every device at a shared node determines
            # only their sum, not the individual reconstruction targets.
            return dict(_OFF), mode
        if mode == "random_safe":
            # Randomize the requested capability without jointly deleting the
            # variables needed to identify it from one operating snapshot.
            sub = rng.choice(
                ("pf", "se_known", "param_one", "injection"),
                p=(0.35, 0.25, 0.20, 0.20),
            )
            if sub == "pf":
                return {**_OFF, "p_voltage": 1.0, "p_current": 1.0}, sub
            if sub == "se_known":
                pv = rng.uniform(float(cfg.get("se_v_lo", 0.4)),
                                 float(cfg.get("se_v_hi", 0.9)))
                pc = rng.uniform(float(cfg.get("se_i_lo", 0.2)),
                                 float(cfg.get("se_i_hi", 0.6)))
                return {**_OFF, "p_voltage": pv, "p_current": pc}, sub
            if sub == "param_one":
                return dict(_OFF), sub
            return dict(_OFF), sub
        if mode == "random":
            jitter = rng.uniform(0.5, 1.5)
            return {k: min(1.0, v * jitter) for k, v in base_rates().items()}, mode
        if mode == "sysid":
            # parameter/system identification: full state visible, reconstruct
            # component parameters — Y AND the injection Icomp — jointly.
            return {**_OFF, "p_admittance": rng.uniform(0.5, 1.0),
                    "p_icomp": rng.uniform(0.5, 1.0)}, mode
        if mode == "param":
            return {**_OFF, "p_admittance": rng.uniform(0.5, 1.0)}, mode
        if mode == "ctrl":
            return dict(_OFF), mode          # store-targeted; see apply_masks
        if mode == "topo":
            return {**_OFF, "p_component": rng.uniform(0.2, 0.5)}, mode
        raise ValueError(f"unknown mixture mode: {mode}")
    jitter = rng.uniform(0.5, 1.5)
    return {k: min(1.0, v * jitter) for k, v in base_rates().items()}, "jitter"


# Named modes are self-describing: each one hardcodes the rates that make its
# capability identifiable, so the configured p_* rates are DEAD for every mode
# but "random". That is deliberate, but it used to be silent -- a run configured
# with p_icomp=0.3 under mixture={random_safe:1.0} trained with the exact mask
# distribution of a p_icomp=0.0 run, and three GPU-hours went into an arm that
# differed from its baseline only by an ignored key. validate_mask_cfg makes the
# discard audible at dataset-build time.
_MODE_HONORS_RATES = frozenset({"random"})
_RATE_KEYS = ("p_voltage", "p_current", "p_icomp", "p_admittance",
              "p_terminal", "p_component")


def inert_rate_keys(cfg: dict) -> tuple[str, ...]:
    """Rate keys the configured mixture can never read."""
    mix = cfg.get("mixture")
    if not mix:
        return ()
    modes = {str(k).removeprefix("p_") for k in mix}
    if modes & _MODE_HONORS_RATES:
        return ()
    return tuple(k for k in _RATE_KEYS if k in cfg)


def validate_mask_cfg(cfg: dict) -> None:
    """Warn when configured rates cannot reach the chosen mixture modes."""
    inert = inert_rate_keys(cfg)
    if not inert:
        return
    modes = ", ".join(sorted(str(k) for k in cfg["mixture"]))
    detail = ", ".join(f"{k}={cfg[k]}" for k in inert)
    print(
        f"WARNING mask config: {detail} IGNORED -- mixture modes ({modes}) set "
        f"their own rates. Changing these keys will not change this run. Use "
        f"mixture={{random: 1.0}} to drive the rates directly, or select the "
        f"named mode whose capability you want (e.g. injection).",
        flush=True,
    )


def apply_masks(data: HeteroData, cfg: dict, rng: np.random.Generator) -> None:
    cfg, mode = _effective_rates(cfg, rng)
    nd = data["node"]
    # v_init is always an input at every bus.  Solved v_pu may be masked except
    # at the slack phases, whose values are always known boundary conditions.
    protected_v = nd.ground | nd.slack
    msk_v = _bern(rng, (nd.num_nodes,), cfg["p_voltage"]) & ~protected_v
    nd.msk_v = msk_v
    nd.vis_v = ~msk_v
    if bool((nd.msk_v & nd.slack).any()) or not bool(nd.vis_v[nd.slack].all()):
        raise AssertionError("slack v_pu must always remain visible")

    for store, spec in SPECS.items():
        st = data[store]
        n, w = st.act.shape
        tri = tri_size(spec.ydim)
        n_y = y_width(store)
        n_i = i_offset(store)
        if n == 0:
            st.msk = st.act.clone()
            st.vis = st.act.clone()
            continue

        # scalar-entry strategies on complex units
        if mode == "param_one":
            unit_active = st.act[:, :n_y].reshape(n, len(spec.yfields), tri).any(dim=1)
            one = torch.zeros(n, tri, dtype=torch.bool)
            for row in range(n):
                candidates = torch.where(unit_active[row])[0].numpy()
                if candidates.size:
                    one[row, int(rng.choice(candidates))] = True
            y_unit = one.repeat(1, len(spec.yfields))
        else:
            y_unit = _bern(rng, (n, tri), cfg["p_admittance"]).repeat(1, len(spec.yfields))
        if spec.icomp:
            icomp_slot = _bern(rng, (n, spec.icomp), cfg["p_icomp"])
            icomp_unit = icomp_slot.repeat(1, 2)
        else:
            icomp_unit = torch.zeros(n, 0, dtype=torch.bool)
        i_slot = _bern(rng, (n, spec.terms, FC), cfg["p_current"])
        i_unit = torch.cat([i_slot[:, t].repeat(1, 2) for t in range(spec.terms)], dim=1)
        msk = torch.cat([y_unit, icomp_unit, i_unit], dim=1)

        # whole-terminal current blocks
        pick = _bern(rng, (n,), cfg["p_terminal"])
        term = torch.from_numpy(rng.integers(0, spec.terms, n))
        for t in range(spec.terms):
            sel = pick & (term == t)
            msk[sel, n_i + t * 2 * FC: n_i + (t + 1) * 2 * FC] = True

        # whole components
        msk |= _bern(rng, (n, 1), cfg["p_component"])

        # ctrl mode: hide this store's parameter block (Y + Icomp), keep state
        if mode == "ctrl" and store in CTRL_STORES:
            msk[:, :n_i] = True

        st.msk = msk & st.act
        st.vis = st.act & ~st.msk

    if mode == "injection":
        _mask_identifiable_injections(data, rng)

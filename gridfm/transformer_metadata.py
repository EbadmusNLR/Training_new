"""Exact transformer YPrim decoder from causally sufficient datakit metadata."""
from __future__ import annotations

import torch

from datakit.components.transformer import (
    TRANSFORMER_PROPERTY_ORDER,
    reconstruct_transformer_yprim_from_definition,
)


TRANSFORMER_PHYSICS_EXTRA_WIDTH = 14


def _definition(
    row: torch.Tensor, extra: torch.Tensor | None = None,
    extra_mask: torch.Tensor | None = None,
) -> dict:
    p = [float(x) for x in row]
    windings = int(round(p[1]))
    npair = windings * (windings - 1) // 2
    out = {
        "Phases": int(round(p[0])), "Windings": windings,
        "Conns": ["delta" if x > 0.5 else "wye" for x in p[2:2 + windings]],
        "kVs": p[5:5 + windings], "kVAs": p[8:8 + windings],
        "Taps": p[11:11 + windings], "pctRs": p[14:14 + windings],
        "XSCArray": p[17:17 + npair], "pctIMag": p[20],
        "pctNoLoadLoss": p[21], "ppm_Antifloat": p[22],
    }
    if extra is not None and extra_mask is not None:
        e = [float(x) for x in extra]
        m = [bool(x) for x in extra_mask]
        if len(e) != TRANSFORMER_PHYSICS_EXTRA_WIDTH or len(m) != len(e):
            raise RuntimeError(
                "transformer physics schema v2 requires exactly "
                f"{TRANSFORMER_PHYSICS_EXTRA_WIDTH} extra fields"
            )
        if m[0]:
            out["LeadLag"] = "lead" if e[0] > 0.5 else "lag"
        if m[1]:
            out["Core"] = e[1]
        if m[2]:
            out["XRConst"] = bool(e[2] > 0.5)
        if m[3]:
            out["BaseFreq"] = e[3]
        if m[4]:
            out["SolutionFrequency"] = e[4]
        for key, start in (
            ("RNeutsEffective", 5),
            ("XNeutsEffective", 8),
            ("RDCOhmsEffective", 11),
        ):
            active = slice(start, start + windings)
            if all(m[active]):
                out[key] = e[active]
    return out


def _pad(raw: list[list[complex]], windings: int, ncond: int) -> torch.Tensor:
    out = torch.zeros((12, 12), dtype=torch.complex128)
    for old_i in range(windings * ncond):
        new_i = (old_i // ncond) * 4 + old_i % ncond
        for old_j in range(windings * ncond):
            new_j = (old_j // ncond) * 4 + old_j % ncond
            out[new_i, new_j] = raw[old_i][old_j]
    return out


def decode_transformer_metadata(store) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return definition-derived per-unit transformer Y and exactness gate."""
    required = (
        "physics_params", "physics_supported", "terminal_kv_base",
        "system_base_mva",
    )
    if not all(hasattr(store, name) for name in required):
        raise RuntimeError(f"transformer metadata missing one of {required}")
    params = store.physics_params.double().cpu()
    has_v2 = all(hasattr(store, name) for name in (
        "physics_extra_params", "physics_extra_mask", "physics_v2_supported"
    ))
    supported = (
        store.physics_v2_supported if has_v2 else store.physics_supported
    ).reshape(-1).bool().cpu()
    extra = store.physics_extra_params.double().cpu() if has_v2 else None
    extra_mask = store.physics_extra_mask.double().cpu() if has_v2 else None
    terminal_kv = store.terminal_kv_base.double().cpu()
    base_mva = store.system_base_mva.reshape(-1).double().cpu()
    out = torch.zeros((params.shape[0], 12, 12), dtype=torch.complex128)
    for idx in torch.nonzero(supported, as_tuple=False).flatten().tolist():
        definition = _definition(
            params[idx], extra[idx] if extra is not None else None,
            extra_mask[idx] if extra_mask is not None else None,
        )
        windings = int(definition["Windings"])
        ncond = int(round(float(params[idx, 23])))
        raw = reconstruct_transformer_yprim_from_definition(
            definition, TRANSFORMER_PROPERTY_ORDER, [ncond] * windings
        )
        if raw is None:
            raise RuntimeError(
                f"supported transformer metadata row {idx} is not decodable"
            )
        matrix = _pad(raw["Yprim"], windings, ncond)
        volts = terminal_kv[idx] * 1e3
        scale = 3.0 * volts[:, None] * volts[None, :] / (base_mva[idx] * 1e6)
        out[idx] = matrix * scale
    if not bool(torch.isfinite(out.real).all() & torch.isfinite(out.imag).all()):
        raise RuntimeError("non-finite transformer metadata decode")
    return out.real, out.imag, supported

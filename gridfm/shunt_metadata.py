"""Exact capacitor/reactor YPrim decoders from target-free datakit metadata."""
from __future__ import annotations

import torch

from datakit.core.shunt_metadata import (
    decode_capacitor_physics_si,
    decode_reactor_physics_si,
    decode_shunt_physics_pu,
)


def decode_shunt_metadata(store, family: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if family not in {"capacitor", "reactor"}:
        raise ValueError(f"unsupported shunt family {family!r}")
    required = (
        "physics_params", "physics_mask", "physics_supported",
        "terminal_kv_base", "system_base_mva",
    )
    if not all(hasattr(store, name) for name in required):
        raise RuntimeError(f"{family} metadata missing one of {required}")
    params = store.physics_params.double().cpu()
    masks = store.physics_mask.double().cpu()
    supported = store.physics_supported.reshape(-1).bool().cpu()
    terminal_kv = store.terminal_kv_base.double().cpu()
    base_mva = store.system_base_mva.reshape(-1).double().cpu()
    decode_si = (
        decode_capacitor_physics_si if family == "capacitor"
        else decode_reactor_physics_si
    )
    prefix = "Ycap" if family == "capacitor" else "Yreactor"
    out = torch.zeros((params.shape[0], 8, 8), dtype=torch.complex128)
    for idx in torch.nonzero(supported, as_tuple=False).flatten().tolist():
        physical = decode_si(params[idx].tolist(), masks[idx].tolist())
        decoded = decode_shunt_physics_pu(
            physical, terminal_kv[idx].tolist(), float(base_mva[idx])
        )
        if decoded is None:
            raise RuntimeError(f"supported {family} metadata row {idx} is not decodable")
        out[idx] = torch.complex(
            torch.tensor(decoded[f"{prefix}_r_pu"], dtype=torch.float64).reshape(8, 8),
            torch.tensor(decoded[f"{prefix}_i_pu"], dtype=torch.float64).reshape(8, 8),
        )
    if not bool(torch.isfinite(out.real).all() & torch.isfinite(out.imag).all()):
        raise RuntimeError(f"non-finite {family} metadata decode")
    return out.real, out.imag, supported

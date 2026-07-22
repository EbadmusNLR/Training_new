"""Exact Load YPrim decoder from target-free datakit metadata."""
from __future__ import annotations

import torch

from datakit.core.physics_metadata import decode_load_physics_pu


def decode_load_metadata(store) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    required = (
        "physics_params", "physics_mask", "physics_supported",
        "terminal_kv_base", "system_base_mva",
    )
    if not all(hasattr(store, name) for name in required):
        raise RuntimeError(f"load metadata missing one of {required}")
    params = store.physics_params.double().cpu()
    masks = store.physics_mask.double().cpu()
    supported = store.physics_supported.reshape(-1).bool().cpu()
    terminal_kv = store.terminal_kv_base.double().cpu()
    base_mva = store.system_base_mva.reshape(-1).double().cpu()
    out = torch.zeros((params.shape[0], 4, 4), dtype=torch.complex128)
    for idx in torch.nonzero(supported, as_tuple=False).flatten().tolist():
        decoded = decode_load_physics_pu(
            params[idx].tolist(), masks[idx].tolist(), terminal_kv[idx].tolist(),
            float(base_mva[idx]),
        )
        if decoded is None:
            raise RuntimeError(f"supported load metadata row {idx} is not decodable")
        out[idx] = torch.complex(
            torch.tensor(decoded["Yload_r_pu"], dtype=torch.float64).reshape(4, 4),
            torch.tensor(decoded["Yload_i_pu"], dtype=torch.float64).reshape(4, 4),
        )
    if not bool(torch.isfinite(out.real).all() & torch.isfinite(out.imag).all()):
        raise RuntimeError("non-finite load metadata decode")
    return out.real, out.imag, supported

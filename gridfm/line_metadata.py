"""Exact line YPrim decoder from target-free datakit definition metadata."""
from __future__ import annotations

import torch

from datakit.core.passive_metadata import decode_line_physics_pu


def _line_yprim(decoded: dict[str, list[float]]) -> torch.Tensor:
    ys_r = torch.tensor(decoded["Ys_r_pu"], dtype=torch.float64).reshape(4, 4)
    ys_i = torch.tensor(decoded["Ys_i_pu"], dtype=torch.float64).reshape(4, 4)
    yh_i = torch.tensor(decoded["Yh_i_pu"], dtype=torch.float64).reshape(4, 4)
    out = torch.zeros((8, 8), dtype=torch.complex128)
    a = torch.complex(ys_r, ys_i + yh_i)
    b = torch.complex(-ys_r, -ys_i)
    out[:4, :4] = a
    out[:4, 4:] = b
    out[4:, :4] = b.T
    out[4:, 4:] = a
    return out


def decode_line_metadata(store) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return definition-derived per-unit line YPrim and an exactness gate."""
    required = (
        "physics_params", "physics_mask", "physics_supported",
        "terminal_kv_base", "system_base_mva",
    )
    if not all(hasattr(store, name) for name in required):
        raise RuntimeError(f"line metadata missing one of {required}")
    params = store.physics_params.double().cpu()
    masks = store.physics_mask.double().cpu()
    supported = store.physics_supported.reshape(-1).bool().cpu()
    terminal_kv = store.terminal_kv_base.double().cpu()
    base_mva = store.system_base_mva.reshape(-1).double().cpu()
    out = torch.zeros((params.shape[0], 8, 8), dtype=torch.complex128)
    for idx in torch.nonzero(supported, as_tuple=False).flatten().tolist():
        decoded = decode_line_physics_pu(
            params[idx].tolist(), masks[idx].tolist(), terminal_kv[idx].tolist(),
            float(base_mva[idx]),
        )
        if decoded is None:
            raise RuntimeError(f"supported line metadata row {idx} is not decodable")
        out[idx] = _line_yprim(decoded)
    if not bool(torch.isfinite(out.real).all() & torch.isfinite(out.imag).all()):
        raise RuntimeError("non-finite line metadata decode")
    return out.real, out.imag, supported

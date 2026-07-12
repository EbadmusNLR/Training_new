"""Selective local-physics current decoding for non-stiff device families."""
from __future__ import annotations

import torch

from .legacy import i_offset, physics


SAFE_PHYSICS_STORES = ("capacitor", "generator", "load", "pvsystem", "storage")


def decode_hybrid_device_currents(batch, preds, clamp: float, stores=SAFE_PHYSICS_STORES):
    """Use local Ibus=YV-Icomp only where voltage error is not stiffly amplified.

    Lines, transformers, and reactors retain learned currents. Radial line-series
    flow is handled separately by the structural tree decoder.
    """
    decoded = physics.decode_currents(batch, preds, clamp)
    out = dict(preds)
    for store in stores:
        st = batch[store]
        if st.num_nodes == 0:
            continue
        ni = i_offset(store)
        value = out[store].clone()
        take = st.msk[:, ni:]
        value[:, ni:] = torch.where(take, decoded[store][:, ni:], value[:, ni:])
        out[store] = value
    return out

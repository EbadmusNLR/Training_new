"""Cache and apply target-independent exact passive-device Y features."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from .legacy import data as legacy_data
from .line_metadata import decode_line_metadata
from .transformer_metadata import decode_transformer_metadata


def _definition_store(info: dict, family: str) -> SimpleNamespace:
    values = {}
    for name, (static, dynamic, _shape) in info.get("definitions", {}).items():
        if dynamic is not None:
            raise RuntimeError(
                f"exact {family} metadata currently requires definition-static fields; "
                f"{name} is dynamic"
            )
        values[name] = static
    return SimpleNamespace(**values)


def _feature(pu: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.asinh(pu / (scale.double() + legacy_data.EPS)).to(scale.dtype)


def _line_feature(info: dict) -> tuple[torch.Tensor, torch.Tensor]:
    yr, yi, supported = decode_line_metadata(_definition_store(info, "line"))
    rows, cols = legacy_data.tri_rc(4)
    ys_r = -yr[:, :4, 4:]
    ys_i = -yi[:, :4, 4:]
    yh_i = yi[:, :4, :4] - ys_i
    pu = torch.cat(
        (ys_r[:, rows, cols], ys_i[:, rows, cols], yh_i[:, rows, cols]), dim=1
    )
    ny = legacy_data.y_width("line")
    return _feature(pu, info["scale"][:, :ny]), supported


def _transformer_feature(info: dict) -> tuple[torch.Tensor, torch.Tensor]:
    yr, yi, supported = decode_transformer_metadata(
        _definition_store(info, "transformer")
    )
    rows, cols = legacy_data.tri_rc(12)
    pu = torch.cat((yr[:, rows, cols], yi[:, rows, cols]), dim=1)
    ny = legacy_data.y_width("transformer")
    return _feature(pu, info["scale"][:, :ny]), supported


def attach_exact_metadata(caches: list, line: bool, transformer: bool) -> None:
    """Decode once per topology and attach small batch-ready feature tensors."""
    if not line and not transformer:
        return
    requested = []
    if line:
        requested.append(("line", _line_feature))
    if transformer:
        requested.append(("transformer", _transformer_feature))
    counts = {family: 0 for family, _ in requested}
    for cache in caches:
        for family, decoder in requested:
            info = cache.stores.get(family)
            if info is None:
                continue
            feature, supported = decoder(info)
            if feature.shape != (info["n"], legacy_data.y_width(family)):
                raise RuntimeError(
                    f"{cache.name}: invalid exact {family} feature shape {feature.shape}"
                )
            if not bool(supported.all()):
                bad = int((~supported).sum())
                raise RuntimeError(
                    f"{cache.name}: exact {family} requested with {bad} unsupported rows"
                )
            info["definitions"]["metadata_y_feat"] = (
                feature.contiguous(), None, tuple(feature.shape)
            )
            info["definitions"]["metadata_y_supported"] = (
                supported.contiguous(), None, tuple(supported.shape)
            )
            counts[family] += int(supported.sum())
    for family, _ in requested:
        if counts[family] == 0:
            raise RuntimeError(f"exact {family} metadata requested but no rows were decoded")


def apply_exact_metadata(
    batch, preds: dict[str, torch.Tensor], line: bool, transformer: bool,
) -> dict[str, torch.Tensor]:
    """Replace only supported passive-Y predictions; never read x_true."""
    for family, enabled in (("line", line), ("transformer", transformer)):
        if not enabled or batch[family].num_nodes == 0:
            continue
        store = batch[family]
        if not hasattr(store, "metadata_y_feat") or not hasattr(
            store, "metadata_y_supported"
        ):
            raise RuntimeError(f"exact {family} metadata missing from batch")
        supported = store.metadata_y_supported.reshape(-1).bool()
        ny = legacy_data.y_width(family)
        value = preds[family].clone()
        value[supported, :ny] = store.metadata_y_feat[supported].to(value.dtype)
        preds[family] = value
    return preds

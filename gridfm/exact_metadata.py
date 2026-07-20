"""Cache and apply target-independent exact passive-device Y features."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from .legacy import data as legacy_data
from .line_metadata import decode_line_metadata
from .transformer_metadata import decode_transformer_metadata


def _definition_store(
    info: dict, family: str, row=None,
) -> SimpleNamespace:
    values = {}
    for name, (static, dynamic, _shape) in info.get("definitions", {}).items():
        if static is not None:
            values[name] = static
        elif row is None:
            raise RuntimeError(
                f"exact {family} metadata requires a scenario row for dynamic field {name}"
            )
        else:
            offset, numel = dynamic
            values[name] = torch.from_numpy(row[offset:offset + numel]).reshape(_shape)
    return SimpleNamespace(**values)


def _feature(pu: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.asinh(pu / (scale.double() + legacy_data.EPS)).to(scale.dtype)


def _line_feature(info: dict, row=None) -> tuple[torch.Tensor, torch.Tensor]:
    yr, yi, supported = decode_line_metadata(_definition_store(info, "line", row))
    rows, cols = legacy_data.tri_rc(4)
    ys_r = -yr[:, :4, 4:]
    ys_i = -yi[:, :4, 4:]
    yh_i = yi[:, :4, :4] - ys_i
    pu = torch.cat(
        (ys_r[:, rows, cols], ys_i[:, rows, cols], yh_i[:, rows, cols]), dim=1
    )
    ny = legacy_data.y_width("line")
    return _feature(pu, info["scale"][:, :ny]), supported


def _transformer_feature(info: dict, row=None) -> tuple[torch.Tensor, torch.Tensor]:
    yr, yi, supported = decode_transformer_metadata(
        _definition_store(info, "transformer", row)
    )
    rows, cols = legacy_data.tri_rc(12)
    pu = torch.cat((yr[:, rows, cols], yi[:, rows, cols]), dim=1)
    ny = legacy_data.y_width("transformer")
    return _feature(pu, info["scale"][:, :ny]), supported


def attach_exact_metadata(caches: list, line: bool, transformer: bool) -> None:
    """Predecode target-independent Y, retaining scenario-varying definitions.

    Static definitions are decoded once per topology. If any required field is
    dynamic, every scenario is decoded and the compact feature tensor is
    indexed at sample time. This avoids both stale-Y reuse and decoder work in
    every DataLoader epoch.
    """
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
            dynamic = any(
                offset is not None
                for _static, offset, _shape in info.get("definitions", {}).values()
            )
            rows = [None] if not dynamic else list(cache.dyn)
            features, support = [], []
            for variant, row in enumerate(rows):
                feature, supported = decoder(info, row)
                expected = (info["n"], legacy_data.y_width(family))
                if feature.shape != expected:
                    raise RuntimeError(
                        f"{cache.name}: invalid exact {family} feature shape "
                        f"{feature.shape}, expected {expected}"
                    )
                if not bool(supported.all()):
                    bad = int((~supported).sum())
                    label = "static" if row is None else f"variant {variant}"
                    raise RuntimeError(
                        f"{cache.name}: exact {family} {label} has {bad} unsupported rows"
                    )
                features.append(feature.contiguous())
                support.append(supported.contiguous())
            derived = info.setdefault("derived_definitions", {})
            if dynamic:
                derived["metadata_y_feat"] = (None, torch.stack(features))
                derived["metadata_y_supported"] = (None, torch.stack(support))
            else:
                derived["metadata_y_feat"] = (features[0], None)
                derived["metadata_y_supported"] = (support[0], None)
            counts[family] += int(sum(int(value.sum()) for value in support))
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

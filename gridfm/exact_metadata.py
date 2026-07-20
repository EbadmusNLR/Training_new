"""Cache and apply target-independent exact device-Y features."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

from .legacy import data as legacy_data
from .line_metadata import decode_line_metadata
from .transformer_metadata import decode_transformer_metadata
from .generator_metadata import decode_generator_metadata
from .shunt_metadata import decode_shunt_metadata
from .load_metadata import decode_load_metadata
from .pvsystem_metadata import decode_pvsystem_metadata
from .vsource_metadata import decode_vsource_metadata


EXACT_METADATA_CACHE_VERSION = 1


def _codec_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__), Path(__file__).with_name("line_metadata.py"),
        Path(__file__).with_name("transformer_metadata.py"),
        Path(__file__).with_name("generator_metadata.py"),
        Path(__file__).with_name("shunt_metadata.py"),
        Path(__file__).with_name("load_metadata.py"),
        Path(__file__).with_name("pvsystem_metadata.py"),
        Path(__file__).with_name("vsource_metadata.py"),
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


CODEC_FINGERPRINT = _codec_fingerprint()


def _definition_store(
    info: dict, family: str, row=None, variant: int | None = None,
) -> SimpleNamespace:
    values = {}
    for name, (static, dynamic, _shape) in info.get("definitions", {}).items():
        if static is not None:
            values[name] = static
        elif name in info.get("definition_values", {}):
            if variant is None:
                raise RuntimeError(
                    f"exact {family} metadata requires a variant for dynamic field {name}"
                )
            values[name] = torch.from_numpy(
                info["definition_values"][name][variant]
            ).reshape(_shape)
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


def _line_feature(
    info: dict, row=None, variant: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    yr, yi, supported = decode_line_metadata(
        _definition_store(info, "line", row, variant)
    )
    rows, cols = legacy_data.tri_rc(4)
    ys_r = -yr[:, :4, 4:]
    ys_i = -yi[:, :4, 4:]
    yh_i = yi[:, :4, :4] - ys_i
    pu = torch.cat(
        (ys_r[:, rows, cols], ys_i[:, rows, cols], yh_i[:, rows, cols]), dim=1
    )
    ny = legacy_data.y_width("line")
    return _feature(pu, info["scale"][:, :ny]), supported


def _transformer_feature(
    info: dict, row=None, variant: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    yr, yi, supported = decode_transformer_metadata(
        _definition_store(info, "transformer", row, variant)
    )
    rows, cols = legacy_data.tri_rc(12)
    pu = torch.cat((yr[:, rows, cols], yi[:, rows, cols]), dim=1)
    ny = legacy_data.y_width("transformer")
    return _feature(pu, info["scale"][:, :ny]), supported


def _generator_feature(
    info: dict, row=None, variant: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    yr, yi, supported = decode_generator_metadata(
        _definition_store(info, "generator", row, variant)
    )
    rows, cols = legacy_data.tri_rc(4)
    pu = torch.cat((yr[:, rows, cols], yi[:, rows, cols]), dim=1)
    ny = legacy_data.y_width("generator")
    return _feature(pu, info["scale"][:, :ny]), supported


def _load_feature(
    info: dict, row=None, variant: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    yr, yi, supported = decode_load_metadata(
        _definition_store(info, "load", row, variant)
    )
    rows, cols = legacy_data.tri_rc(4)
    pu = torch.cat((yr[:, rows, cols], yi[:, rows, cols]), dim=1)
    ny = legacy_data.y_width("load")
    return _feature(pu, info["scale"][:, :ny]), supported


def _pc_feature(info: dict, family: str, decoder, dim: int, row=None, variant=None):
    yr, yi, supported = decoder(_definition_store(info, family, row, variant))
    rows, cols = legacy_data.tri_rc(dim)
    pu = torch.cat((yr[:, rows, cols], yi[:, rows, cols]), dim=1)
    ny = legacy_data.y_width(family)
    return _feature(pu, info["scale"][:, :ny]), supported


def _pvsystem_feature(info: dict, row=None, variant=None):
    return _pc_feature(info, "pvsystem", decode_pvsystem_metadata, 4, row, variant)


def _vsource_feature(info: dict, row=None, variant=None):
    return _pc_feature(info, "vsource", decode_vsource_metadata, 8, row, variant)


def _shunt_feature(
    info: dict, family: str, row=None, variant: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    yr, yi, supported = decode_shunt_metadata(
        _definition_store(info, family, row, variant), family
    )
    rows, cols = legacy_data.tri_rc(8)
    pu = torch.cat((yr[:, rows, cols], yi[:, rows, cols]), dim=1)
    ny = legacy_data.y_width(family)
    return _feature(pu, info["scale"][:, :ny]), supported


def _capacitor_feature(info: dict, row=None, variant: int | None = None):
    return _shunt_feature(info, "capacitor", row, variant)


def _reactor_feature(info: dict, row=None, variant: int | None = None):
    return _shunt_feature(info, "reactor", row, variant)


def _decode_cache(cache, families: tuple[str, ...]) -> dict:
    decoders = {
        "line": _line_feature,
        "transformer": _transformer_feature,
        "generator": _generator_feature,
        "capacitor": _capacitor_feature,
        "reactor": _reactor_feature,
        "load": _load_feature,
        "pvsystem": _pvsystem_feature,
        "vsource": _vsource_feature,
    }
    result = {}
    for family in families:
        info = cache.stores.get(family)
        if info is None or int(info.get("n", 0)) == 0:
            continue
        dynamic = any(
            offset is not None
            for _static, offset, _shape in info.get("definitions", {}).values()
        )
        variants = [None] if not dynamic else list(range(cache.n_variants))
        features, support = [], []
        for variant in variants:
            row = None if variant is None else cache.dyn[variant]
            try:
                feature, supported = decoders[family](info, row, variant)
            except Exception as exc:
                raise RuntimeError(
                    f"{cache.name}: exact {family} decode failed for "
                    f"n={info.get('n')} variant={variant}; definition fields="
                    f"{sorted(info.get('definitions', {}))}"
                ) from exc
            expected = (info["n"], legacy_data.y_width(family))
            if feature.shape != expected:
                raise RuntimeError(
                    f"{cache.name}: invalid exact {family} feature shape "
                    f"{feature.shape}, expected {expected}"
                )
            if not bool(supported.all()):
                bad = int((~supported).sum())
                label = "static" if variant is None else f"variant {variant}"
                raise RuntimeError(
                    f"{cache.name}: exact {family} {label} has {bad} unsupported rows"
                )
            features.append(feature.contiguous())
            support.append(supported.contiguous())
        feature = torch.stack(features) if dynamic else features[0]
        supported = torch.stack(support) if dynamic else support[0]
        result[family] = (
            dynamic, feature.numpy(), supported.numpy(),
            int(sum(int(value.sum()) for value in support)),
        )
    return result


def _source_fingerprint(cache, families: tuple[str, ...]) -> dict | None:
    feeder_dir = getattr(cache, "_dir", None)
    if feeder_dir is None:
        return None
    files = []
    for name in ("static.pt", "dynamic.npy"):
        path = Path(feeder_dir) / name
        stat = path.stat()
        files.append((name, int(stat.st_size), int(stat.st_mtime_ns)))
    return {
        "version": EXACT_METADATA_CACHE_VERSION,
        "codec": CODEC_FINGERPRINT,
        "families": families,
        "source": tuple(files),
    }


def _disk_cache_path(cache, families: tuple[str, ...], disk_cache_dir) -> Path | None:
    feeder_dir = getattr(cache, "_dir", None)
    if feeder_dir is None or disk_cache_dir is None:
        return None
    identity = hashlib.sha256(str(Path(feeder_dir).resolve()).encode()).hexdigest()[:24]
    suffix = "-".join(families)
    return Path(disk_cache_dir) / f"{identity}-{suffix}.pt"


def _decode_cache_index(
    index: int, cache, families: tuple[str, ...], disk_cache_dir=None,
):
    path = _disk_cache_path(cache, families, disk_cache_dir)
    fingerprint = _source_fingerprint(cache, families)
    if path is not None and path.is_file():
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("fingerprint") == fingerprint:
                return index, payload["result"], True
        except Exception:
            # A stale/interrupted cache is never a correctness fallback: decode
            # from the causal definitions and atomically replace it below.
            pass
    result = _decode_cache(cache, families)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.close(fd)
        try:
            torch.save({"fingerprint": fingerprint, "result": result}, temp_name)
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    return index, result, False


def attach_exact_metadata(
    caches: list, line: bool, transformer: bool, workers: int = 0,
    generator: bool = False, disk_cache_dir=None, capacitor: bool = False,
    reactor: bool = False, load: bool = False, pvsystem: bool = False,
    vsource: bool = False,
) -> None:
    """Predecode target-independent Y, retaining scenario-varying definitions.

    Static definitions are decoded once per topology. If any required field is
    dynamic, every scenario is decoded and the compact feature tensor is
    indexed at sample time. This avoids both stale-Y reuse and decoder work in
    every DataLoader epoch.
    """
    requested = []
    if line:
        requested.append("line")
    if transformer:
        requested.append("transformer")
    if generator:
        requested.append("generator")
    if capacitor:
        requested.append("capacitor")
    if reactor:
        requested.append("reactor")
    if load:
        requested.append("load")
    if pvsystem:
        requested.append("pvsystem")
    if vsource:
        requested.append("vsource")
    families = tuple(requested)
    counts = {family: 0 for family in families}
    cache_hits = cache_misses = 0
    for cache in caches:
        empty = getattr(cache, "empty_derived_definitions", None)
        if empty is None:
            empty = {}
            cache.empty_derived_definitions = empty
        for family in families:
            empty[family] = {
                "metadata_y_feat": torch.zeros(
                    0, legacy_data.y_width(family),
                    dtype=getattr(cache, "dtype", torch.float32),
                ),
                "metadata_y_supported": torch.zeros(0, dtype=torch.bool),
            }
    if not families:
        results = []
    elif workers > 1 and len(caches) > 1:
        # The decoder is tensor-heavy and releases the GIL. Threads avoid
        # copying large per-variant feature arrays through multiprocessing
        # pipes (and the resulting worker/parent pipe deadlock).
        with ThreadPoolExecutor(
            max_workers=min(int(workers), len(caches))
        ) as pool:
            futures = [
                pool.submit(
                    _decode_cache_index, index, cache, families, disk_cache_dir
                )
                for index, cache in enumerate(caches)
            ]
            results = (future.result() for future in as_completed(futures))
    else:
        results = [
            _decode_cache_index(index, cache, families, disk_cache_dir)
            for index, cache in enumerate(caches)
        ]
    for index, result, cache_hit in results:
        cache_hits += int(cache_hit)
        cache_misses += int(not cache_hit)
        cache = caches[index]
        for family, (dynamic, feature_np, supported_np, count) in result.items():
            info = cache.stores[family]
            feature = torch.from_numpy(feature_np)
            supported = torch.from_numpy(supported_np)
            derived = info.setdefault("derived_definitions", {})
            derived["metadata_y_feat"] = (
                (None, feature) if dynamic else (feature, None)
            )
            derived["metadata_y_supported"] = (
                (None, supported) if dynamic else (supported, None)
            )
            # Definitions have served their causal purpose. Keep them in the
            # persisted store for auditability, but do not collate/move the raw
            # fp64 parameter blocks through every training batch.
            info["definitions"] = {}
            info["definition_values"] = {}
            counts[family] += count
    # Raw definition blocks are decoder inputs, never learned model features.
    # Drop unused families too so generic/ablation arms do not collate and copy
    # target-independent fp64 metadata to the GPU on every training batch.
    for cache in caches:
        for family in ("line", "transformer", "generator", "capacitor", "reactor", "load", "pvsystem", "vsource"):
            info = cache.stores.get(family)
            if info is not None:
                info["definitions"] = {}
                info["definition_values"] = {}
    for family in families:
        if counts[family] == 0:
            raise RuntimeError(f"exact {family} metadata requested but no rows were decoded")
    if families and disk_cache_dir is not None:
        print(
            f"exact metadata disk cache hits={cache_hits} misses={cache_misses} "
            f"dir={Path(disk_cache_dir)}",
            flush=True,
        )


def apply_exact_metadata(
    batch, preds: dict[str, torch.Tensor], line: bool, transformer: bool,
    generator: bool = False, capacitor: bool = False, reactor: bool = False,
    load: bool = False, pvsystem: bool = False, vsource: bool = False,
) -> dict[str, torch.Tensor]:
    """Replace only supported device-Y predictions; never read x_true."""
    for family, enabled in (
        ("line", line), ("transformer", transformer), ("generator", generator),
        ("capacitor", capacitor), ("reactor", reactor),
        ("load", load),
        ("pvsystem", pvsystem), ("vsource", vsource),
    ):
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

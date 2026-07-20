"""Contracts for the cached exact-metadata path used by EdgeStateGridFM."""
from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

from gridfm import exact_metadata as em


def _info(family: str, n: int, width: int) -> dict:
    return {
        "n": n,
        "tmpl": torch.full((n, width), 1e20),
        "scale": torch.ones((n, width)),
        "definitions": {"sentinel_definition": (torch.arange(n), None, (n,))},
    }


def test_cached_features_ignore_target_and_apply_only_to_y() -> None:
    line_info = _info("line", 1, 38)
    transformer_info = _info("transformer", 1, 180)
    generator_info = _info("generator", 1, 36)
    cache = SimpleNamespace(
        name="poison", stores={
            "line": line_info, "transformer": transformer_info,
            "generator": generator_info,
        }
    )
    absent = SimpleNamespace(name="absent", stores={}, dtype=torch.float32)

    original_line = em.decode_line_metadata
    original_transformer = em.decode_transformer_metadata
    original_generator = em.decode_generator_metadata
    line_yr = torch.zeros((1, 8, 8), dtype=torch.float64)
    line_yi = torch.zeros_like(line_yr)
    line_yr[:, :4, 4:] = -torch.eye(4)
    line_yi[:, :4, 4:] = -2 * torch.eye(4)
    line_yi[:, :4, :4] = 5 * torch.eye(4)
    transformer_yr = torch.diag_embed(torch.arange(1, 13, dtype=torch.float64)[None])
    transformer_yi = -transformer_yr
    generator_yr = torch.diag_embed(torch.arange(1, 5, dtype=torch.float64)[None])
    generator_yi = -2 * generator_yr
    try:
        em.decode_line_metadata = lambda _store: (
            line_yr, line_yi, torch.ones(1, dtype=torch.bool)
        )
        em.decode_transformer_metadata = lambda _store: (
            transformer_yr, transformer_yi, torch.ones(1, dtype=torch.bool)
        )
        em.decode_generator_metadata = lambda _store: (
            generator_yr, generator_yi, torch.ones(1, dtype=torch.bool)
        )
        em.attach_exact_metadata(
            [cache, absent], line=True, transformer=True, generator=True
        )
        before_line = line_info["derived_definitions"]["metadata_y_feat"][0].clone()
        before_transformer = transformer_info["derived_definitions"]["metadata_y_feat"][0].clone()
        before_generator = generator_info["derived_definitions"]["metadata_y_feat"][0].clone()
        line_info["tmpl"].fill_(-1e30)
        transformer_info["tmpl"].fill_(1e30)
        generator_info["tmpl"].fill_(-1e35)
        em.attach_exact_metadata(
            [cache, absent], line=True, transformer=True, generator=True
        )
    finally:
        em.decode_line_metadata = original_line
        em.decode_transformer_metadata = original_transformer
        em.decode_generator_metadata = original_generator

    assert torch.equal(before_line, line_info["derived_definitions"]["metadata_y_feat"][0])
    assert torch.equal(
        before_transformer, transformer_info["derived_definitions"]["metadata_y_feat"][0]
    )
    assert absent.empty_derived_definitions["line"]["metadata_y_feat"].shape == (
        0, 30,
    )
    assert absent.empty_derived_definitions["transformer"][
        "metadata_y_feat"
    ].shape == (0, 156)
    assert absent.empty_derived_definitions["generator"][
        "metadata_y_feat"
    ].shape == (0, 20)

    line_pred = torch.randn(1, 38)
    transformer_pred = torch.randn(1, 180)
    generator_pred = torch.randn(1, 36)
    line_current = line_pred[:, 30:].clone()
    transformer_current = transformer_pred[:, 156:].clone()
    generator_current = generator_pred[:, 20:].clone()
    batch = {
        "line": SimpleNamespace(
            num_nodes=1, metadata_y_feat=before_line,
            metadata_y_supported=torch.ones(1, dtype=torch.bool),
            x_true=torch.full((1, 38), 1e35),
        ),
        "transformer": SimpleNamespace(
            num_nodes=1, metadata_y_feat=before_transformer,
            metadata_y_supported=torch.ones(1, dtype=torch.bool),
            x_true=torch.full((1, 180), -1e35),
        ),
        "generator": SimpleNamespace(
            num_nodes=1, metadata_y_feat=before_generator,
            metadata_y_supported=torch.ones(1, dtype=torch.bool),
            x_true=torch.full((1, 36), 1e35),
        ),
    }
    out = em.apply_exact_metadata(
        batch, {
            "line": line_pred, "transformer": transformer_pred,
            "generator": generator_pred,
        }, True, True, True
    )
    assert torch.equal(out["line"][:, :30], before_line)
    assert torch.equal(out["transformer"][:, :156], before_transformer)
    assert torch.equal(out["line"][:, 30:], line_current)
    assert torch.equal(out["transformer"][:, 156:], transformer_current)
    assert torch.equal(out["generator"][:, :20], before_generator)
    assert torch.equal(out["generator"][:, 20:], generator_current)


def test_unsupported_rows_fail_closed() -> None:
    cache = SimpleNamespace(name="unsupported", stores={"line": _info("line", 1, 38)})
    original = em.decode_line_metadata
    try:
        em.decode_line_metadata = lambda _store: (
            torch.zeros(1, 8, 8), torch.zeros(1, 8, 8),
            torch.zeros(1, dtype=torch.bool),
        )
        try:
            em.attach_exact_metadata([cache], line=True, transformer=False)
        except RuntimeError as exc:
            assert "unsupported rows" in str(exc)
        else:
            raise AssertionError("unsupported exact line row reached a fallback")
    finally:
        em.decode_line_metadata = original


def test_shunt_metadata_is_exact_and_target_independent() -> None:
    capacitor_info = _info("capacitor", 1, 88)
    reactor_info = _info("reactor", 1, 88)
    cache = SimpleNamespace(
        name="shunts", stores={"capacitor": capacitor_info, "reactor": reactor_info}
    )
    original = em.decode_shunt_metadata
    yr = torch.diag_embed(torch.arange(1, 9, dtype=torch.float64)[None])
    yi = -2 * yr
    try:
        em.decode_shunt_metadata = lambda _store, _family: (
            yr, yi, torch.ones(1, dtype=torch.bool)
        )
        em.attach_exact_metadata(
            [cache], line=False, transformer=False, capacitor=True, reactor=True
        )
    finally:
        em.decode_shunt_metadata = original
    cap = capacitor_info["derived_definitions"]["metadata_y_feat"][0]
    reactor = reactor_info["derived_definitions"]["metadata_y_feat"][0]
    assert cap.shape == reactor.shape == (1, 72)
    cap_pred = torch.randn(1, 88)
    reactor_pred = torch.randn(1, 88)
    cap_current = cap_pred[:, 72:].clone()
    reactor_current = reactor_pred[:, 72:].clone()
    batch = {
        "capacitor": SimpleNamespace(
            num_nodes=1, metadata_y_feat=cap,
            metadata_y_supported=torch.ones(1, dtype=torch.bool),
        ),
        "reactor": SimpleNamespace(
            num_nodes=1, metadata_y_feat=reactor,
            metadata_y_supported=torch.ones(1, dtype=torch.bool),
        ),
    }
    out = em.apply_exact_metadata(
        batch, {"capacitor": cap_pred, "reactor": reactor_pred},
        False, False, False, True, True,
    )
    assert torch.equal(out["capacitor"][:, :72], cap)
    assert torch.equal(out["reactor"][:, :72], reactor)
    assert torch.equal(out["capacitor"][:, 72:], cap_current)
    assert torch.equal(out["reactor"][:, 72:], reactor_current)


def test_load_metadata_is_exact_and_preserves_current() -> None:
    info = _info("load", 1, 36)
    cache = SimpleNamespace(name="load", stores={"load": info})
    original = em.decode_load_metadata
    yr = torch.diag_embed(torch.arange(1, 5, dtype=torch.float64)[None])
    yi = -3 * yr
    try:
        em.decode_load_metadata = lambda _store: (
            yr, yi, torch.ones(1, dtype=torch.bool)
        )
        em.attach_exact_metadata(
            [cache], line=False, transformer=False, load=True
        )
    finally:
        em.decode_load_metadata = original
    exact_y = info["derived_definitions"]["metadata_y_feat"][0]
    assert exact_y.shape == (1, 20)
    pred = torch.randn(1, 36)
    current = pred[:, 20:].clone()
    batch = {
        "load": SimpleNamespace(
            num_nodes=1, metadata_y_feat=exact_y,
            metadata_y_supported=torch.ones(1, dtype=torch.bool),
            x_true=torch.full((1, 36), 1e35),
        ),
    }
    out = em.apply_exact_metadata(
        batch, {"load": pred}, False, False, load=True,
    )
    assert torch.equal(out["load"][:, :20], exact_y)
    assert torch.equal(out["load"][:, 20:], current)


def test_pvsystem_and_vsource_exact_metadata() -> None:
    pv_info, source_info = _info("pvsystem", 1, 21), _info("vsource", 1, 73)
    cache = SimpleNamespace(name="pc", stores={"pvsystem": pv_info, "vsource": source_info})
    original_pv, original_source = em.decode_pvsystem_metadata, em.decode_vsource_metadata
    pv_y = torch.diag_embed(torch.arange(1, 5, dtype=torch.float64)[None])
    source_y = torch.diag_embed(torch.arange(1, 9, dtype=torch.float64)[None])
    try:
        em.decode_pvsystem_metadata = lambda _store: (pv_y, -pv_y, torch.ones(1, dtype=torch.bool))
        em.decode_vsource_metadata = lambda _store: (source_y, -source_y, torch.ones(1, dtype=torch.bool))
        em.attach_exact_metadata([cache], False, False, pvsystem=True, vsource=True)
    finally:
        em.decode_pvsystem_metadata, em.decode_vsource_metadata = original_pv, original_source
    pv_exact = pv_info["derived_definitions"]["metadata_y_feat"][0]
    source_exact = source_info["derived_definitions"]["metadata_y_feat"][0]
    assert pv_exact.shape == (1, 20) and source_exact.shape == (1, 72)
    batch = {
        "pvsystem": SimpleNamespace(num_nodes=1, metadata_y_feat=pv_exact, metadata_y_supported=torch.ones(1, dtype=torch.bool)),
        "vsource": SimpleNamespace(num_nodes=1, metadata_y_feat=source_exact, metadata_y_supported=torch.ones(1, dtype=torch.bool)),
    }
    pv_pred, source_pred = torch.randn(1, 21), torch.randn(1, 73)
    pv_tail, source_tail = pv_pred[:, 20:].clone(), source_pred[:, 72:].clone()
    out = em.apply_exact_metadata(batch, {"pvsystem": pv_pred, "vsource": source_pred}, False, False, pvsystem=True, vsource=True)
    assert torch.equal(out["pvsystem"][:, :20], pv_exact) and torch.equal(out["pvsystem"][:, 20:], pv_tail)
    assert torch.equal(out["vsource"][:, :72], source_exact) and torch.equal(out["vsource"][:, 72:], source_tail)


def test_storage_exact_metadata_preserves_current() -> None:
    info = _info("storage", 1, 36)
    cache = SimpleNamespace(name="storage", stores={"storage": info})
    original = em.decode_storage_metadata
    yr = torch.diag_embed(torch.arange(1, 5, dtype=torch.float64)[None])
    try:
        em.decode_storage_metadata = lambda _store: (
            yr, -yr, torch.ones(1, dtype=torch.bool)
        )
        em.attach_exact_metadata(
            [cache], False, False, storage=True
        )
    finally:
        em.decode_storage_metadata = original
    exact_y = info["derived_definitions"]["metadata_y_feat"][0]
    assert exact_y.shape == (1, 20)
    pred = torch.randn(1, 36)
    current = pred[:, 20:].clone()
    batch = {
        "storage": SimpleNamespace(
            num_nodes=1, metadata_y_feat=exact_y,
            metadata_y_supported=torch.ones(1, dtype=torch.bool),
        ),
    }
    out = em.apply_exact_metadata(
        batch, {"storage": pred}, False, False, storage=True,
    )
    assert torch.equal(out["storage"][:, :20], exact_y)
    assert torch.equal(out["storage"][:, 20:], current)


def test_unused_definitions_are_not_collated() -> None:
    line_info = _info("line", 1, 38)
    line_info["definition_values"] = {"dynamic": torch.ones(2, 1).numpy()}
    cache = SimpleNamespace(name="generic", stores={"line": line_info})
    em.attach_exact_metadata([cache], line=False, transformer=False)
    assert line_info["definitions"] == {}
    assert line_info["definition_values"] == {}


def test_dynamic_definitions_are_variant_specific() -> None:
    info = _info("line", 1, 38)
    info["definitions"] = {
        "sentinel": (None, (0, 1), (1, 1)),
    }
    cache = SimpleNamespace(
        name="dynamic", stores={"line": info},
        dyn=torch.tensor([[2.0], [7.0]], dtype=torch.float64).numpy(),
        n_variants=2,
    )
    original = em.decode_line_metadata
    try:
        def decode(store):
            value = float(store.sentinel.item())
            yr = torch.zeros(1, 8, 8, dtype=torch.float64)
            yi = torch.zeros_like(yr)
            yr[:, :4, 4:] = -value * torch.eye(4)
            return yr, yi, torch.ones(1, dtype=torch.bool)

        em.decode_line_metadata = decode
        em.attach_exact_metadata([cache], line=True, transformer=False)
    finally:
        em.decode_line_metadata = original
    values = info["derived_definitions"]["metadata_y_feat"][1]
    assert values.shape == (2, 1, 30)
    assert not torch.equal(values[0], values[1])


def test_parallel_predecode_matches_serial() -> None:
    def make_caches(prefix):
        caches = []
        for index in range(2):
            info = _info("line", 1, 38)
            info["definitions"] = {
                "sentinel": (torch.tensor([[float(index + 1)]]), None, (1, 1)),
            }
            caches.append(
                SimpleNamespace(name=f"{prefix}-{index}", stores={"line": info})
            )
        return caches

    serial = make_caches("serial")
    parallel = make_caches("parallel")
    original = em.decode_line_metadata
    try:
        def decode(store):
            value = float(store.sentinel.item())
            yr = torch.zeros(1, 8, 8, dtype=torch.float64)
            yi = torch.zeros_like(yr)
            yr[:, :4, 4:] = -value * torch.eye(4)
            return yr, yi, torch.ones(1, dtype=torch.bool)

        em.decode_line_metadata = decode
        em.attach_exact_metadata(
            serial, line=True, transformer=False, workers=0
        )
        em.attach_exact_metadata(
            parallel, line=True, transformer=False, workers=2
        )
    finally:
        em.decode_line_metadata = original
    for serial_cache, parallel_cache in zip(serial, parallel):
        serial_value = serial_cache.stores["line"]["derived_definitions"][
            "metadata_y_feat"
        ][0]
        parallel_value = parallel_cache.stores["line"]["derived_definitions"][
            "metadata_y_feat"
        ][0]
        assert torch.equal(serial_value, parallel_value)
    assert not torch.equal(
        serial[0].stores["line"]["derived_definitions"]["metadata_y_feat"][0],
        serial[1].stores["line"]["derived_definitions"]["metadata_y_feat"][0],
    )


def test_disk_cache_reuses_and_invalidates_source() -> None:
    with tempfile.TemporaryDirectory() as root_s:
        root = Path(root_s)
        feeder = root / "feeder"
        feeder.mkdir()
        (feeder / "static.pt").write_bytes(b"static-v1")
        (feeder / "dynamic.npy").write_bytes(b"dynamic-v1")
        cache = SimpleNamespace(_dir=feeder)
        calls = []
        original = em._decode_cache
        try:
            em._decode_cache = lambda _cache, _families: calls.append(1) or {"sentinel": 7}
            _, first, first_hit = em._decode_cache_index(0, cache, ("line",), root / "cache")
            _, second, second_hit = em._decode_cache_index(0, cache, ("line",), root / "cache")
            (feeder / "dynamic.npy").write_bytes(b"dynamic-v2-longer")
            _, third, third_hit = em._decode_cache_index(0, cache, ("line",), root / "cache")
        finally:
            em._decode_cache = original
        assert first == second == third == {"sentinel": 7}
        assert (first_hit, second_hit, third_hit) == (False, True, False)
        assert len(calls) == 2


if __name__ == "__main__":
    test_cached_features_ignore_target_and_apply_only_to_y()
    test_unsupported_rows_fail_closed()
    test_shunt_metadata_is_exact_and_target_independent()
    test_load_metadata_is_exact_and_preserves_current()
    test_pvsystem_and_vsource_exact_metadata()
    test_storage_exact_metadata_preserves_current()
    test_unused_definitions_are_not_collated()
    test_dynamic_definitions_are_variant_specific()
    test_parallel_predecode_matches_serial()
    test_disk_cache_reuses_and_invalidates_source()
    print("EDGE_EXACT_METADATA_TEST_OK")

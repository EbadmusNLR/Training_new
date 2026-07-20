"""Contracts for the cached exact-metadata path used by EdgeStateGridFM."""
from __future__ import annotations

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
    cache = SimpleNamespace(
        name="poison", stores={"line": line_info, "transformer": transformer_info}
    )

    original_line = em.decode_line_metadata
    original_transformer = em.decode_transformer_metadata
    line_yr = torch.zeros((1, 8, 8), dtype=torch.float64)
    line_yi = torch.zeros_like(line_yr)
    line_yr[:, :4, 4:] = -torch.eye(4)
    line_yi[:, :4, 4:] = -2 * torch.eye(4)
    line_yi[:, :4, :4] = 5 * torch.eye(4)
    transformer_yr = torch.diag_embed(torch.arange(1, 13, dtype=torch.float64)[None])
    transformer_yi = -transformer_yr
    try:
        em.decode_line_metadata = lambda _store: (
            line_yr, line_yi, torch.ones(1, dtype=torch.bool)
        )
        em.decode_transformer_metadata = lambda _store: (
            transformer_yr, transformer_yi, torch.ones(1, dtype=torch.bool)
        )
        em.attach_exact_metadata([cache], line=True, transformer=True)
        before_line = line_info["definitions"]["metadata_y_feat"][0].clone()
        before_transformer = transformer_info["definitions"]["metadata_y_feat"][0].clone()
        line_info["tmpl"].fill_(-1e30)
        transformer_info["tmpl"].fill_(1e30)
        em.attach_exact_metadata([cache], line=True, transformer=True)
    finally:
        em.decode_line_metadata = original_line
        em.decode_transformer_metadata = original_transformer

    assert torch.equal(before_line, line_info["definitions"]["metadata_y_feat"][0])
    assert torch.equal(
        before_transformer, transformer_info["definitions"]["metadata_y_feat"][0]
    )

    line_pred = torch.randn(1, 38)
    transformer_pred = torch.randn(1, 180)
    line_current = line_pred[:, 30:].clone()
    transformer_current = transformer_pred[:, 156:].clone()
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
    }
    out = em.apply_exact_metadata(
        batch, {"line": line_pred, "transformer": transformer_pred}, True, True
    )
    assert torch.equal(out["line"][:, :30], before_line)
    assert torch.equal(out["transformer"][:, :156], before_transformer)
    assert torch.equal(out["line"][:, 30:], line_current)
    assert torch.equal(out["transformer"][:, 156:], transformer_current)


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


if __name__ == "__main__":
    test_cached_features_ignore_target_and_apply_only_to_y()
    test_unsupported_rows_fail_closed()
    print("EDGE_EXACT_METADATA_TEST_OK")

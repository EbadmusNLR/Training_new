"""Configured mask rates that the mixture cannot read must be reported.

Named mixture modes hardcode their own rates -- that is deliberate, since each
mode encodes the visibility pattern that makes one capability identifiable. But
the discard was silent: an arm configured with p_icomp=0.3 under
mixture={random_safe: 1.0} produced the exact mask distribution of the
p_icomp=0.0 baseline it was meant to be compared against, and the two configs
differed by nothing else. Three GPU-hours went into a duplicate run.

These tests pin which modes read the configured rates, so the next person to add
a mode has to decide explicitly rather than inherit the silence.
"""
from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]

import numpy as np  # noqa: E402

from gridfm.core.masking import (  # noqa: E402
    _effective_rates, inert_rate_keys, validate_mask_cfg,
)

MIX900 = {
    "mixture": {"random_safe": 1.0},
    "p_voltage": 0.3, "p_current": 0.15,
    "p_icomp": 0.3, "p_admittance": 0.0,
    "p_terminal": 0.0, "p_component": 0.0,
}


class InertRateKeys(unittest.TestCase):
    def test_random_safe_reads_no_configured_rate(self):
        self.assertEqual(
            inert_rate_keys(MIX900),
            ("p_voltage", "p_current", "p_icomp", "p_admittance",
             "p_terminal", "p_component"),
        )

    def test_random_mode_reads_them(self):
        self.assertEqual(inert_rate_keys({**MIX900, "mixture": {"random": 1.0}}), ())

    def test_no_mixture_reads_them(self):
        cfg = {k: v for k, v in MIX900.items() if k != "mixture"}
        self.assertEqual(inert_rate_keys(cfg), ())

    def test_partial_config_reports_only_present_keys(self):
        self.assertEqual(
            inert_rate_keys({"mixture": {"pf": 1.0}, "p_icomp": 0.3}), ("p_icomp",)
        )


class ValidateWarns(unittest.TestCase):
    def _capture(self, cfg) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            validate_mask_cfg(cfg)
        return buf.getvalue()

    def test_warns_and_names_the_dead_key(self):
        out = self._capture(MIX900)
        self.assertIn("p_icomp=0.3", out)
        self.assertIn("IGNORED", out)

    def test_silent_when_rates_are_live(self):
        self.assertEqual(self._capture({**MIX900, "mixture": {"random": 1.0}}), "")


class RatesAreActuallyDiscarded(unittest.TestCase):
    """The behaviour the warning describes, measured rather than assumed."""

    def _mode_rates(self, cfg, draws=4000):
        seen = {}
        for seed in range(draws):
            rates, mode = _effective_rates(cfg, np.random.default_rng(seed))
            seen.setdefault(mode, rates)
        return seen

    def test_p_icomp_does_not_change_the_drawn_rates(self):
        zero = self._mode_rates(MIX900 | {"p_icomp": 0.0})
        three = self._mode_rates(MIX900 | {"p_icomp": 0.3})
        self.assertEqual(zero, three)

    def test_random_safe_still_reaches_the_injection_mode(self):
        # The capability was exercised all along -- via the mode, not the rate.
        self.assertIn("injection", self._mode_rates(MIX900))


if __name__ == "__main__":
    unittest.main()

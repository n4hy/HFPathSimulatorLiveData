"""Tests for the frequency-dependent group-delay (dispersion) model.

Covers:
  - compute_d_from_qp: quasi-parabolic layer -> linear dispersion coefficient
  - DispersionModel.apply_dispersion: chirp all-pass behavior
  - DispersionModel.apply_inverse_dispersion: inversion identity
  - Length preservation, negligible-dispersion pass-through
  - Physics sanity: dispersion at negligible bandwidth ~ identity;
    inversion of dispersion recovers input

The dispersion module was previously exercised only by a single boolean flag
check in test_itu_channels.py — no direct coverage. This file closes that gap.
"""

import numpy as np
import pytest

from hfpathsim.core.dispersion import (
    DispersionModel,
    DispersionParameters,
    compute_d_from_qp,
    estimate_dispersion_spread,
    typical_dispersion_values,
    C_LIGHT,
)


class TestComputeDFromQP:
    """Verify the quasi-parabolic -> linear dispersion mapping."""

    def test_formula_matches_derivation(self):
        # d = 2K / f^3 with K = pi * y_m * fc^2 * sec(phi) / (2c)
        f_c = 8e6
        f_carrier = 15e6
        y_m = 100e3
        phi = 0.35
        K = np.pi * y_m * f_c**2 * (1.0 / np.cos(phi)) / (2.0 * C_LIGHT)
        expected_s_per_Hz = 2.0 * K / f_carrier**3
        expected_us_per_MHz = expected_s_per_Hz * 1e12
        got = compute_d_from_qp(f_c, f_carrier, y_m=y_m, phi_inc=phi)
        assert got == pytest.approx(expected_us_per_MHz, rel=1e-9)

    def test_docstring_example_reproduces(self):
        d = compute_d_from_qp(8e6, 15e6, y_m=100e3, phi_inc=0.35)
        assert d == pytest.approx(21.2, abs=0.2)

    def test_below_critical_frequency_raises(self):
        with pytest.raises(ValueError, match="above"):
            compute_d_from_qp(f_c_layer=10e6, f_carrier=8e6)

    def test_dispersion_falls_with_f3(self):
        # Doubling carrier frequency should reduce d by factor 8 (d ~ 1/f^3)
        f_c = 4e6
        d_low = compute_d_from_qp(f_c, 8e6)
        d_high = compute_d_from_qp(f_c, 16e6)
        assert d_low / d_high == pytest.approx(8.0, rel=0.05)

    def test_secant_clamped_at_high_incidence(self):
        # phi = pi/2 (90 deg) would blow up sec(phi); model clamps at cos_phi=0.1
        d_near_horizontal = compute_d_from_qp(4e6, 8e6, phi_inc=np.pi / 2)
        d_at_84deg = compute_d_from_qp(4e6, 8e6, phi_inc=np.arccos(0.1))
        assert d_near_horizontal == pytest.approx(d_at_84deg, rel=1e-6)


class TestDispersionParameters:
    """Verify the DispersionParameters routing between direct and QP-derived."""

    def test_direct_coefficient_returned(self):
        p = DispersionParameters(d_us_per_MHz=42.0)
        assert p.get_dispersion_coefficient() == 42.0

    def test_qp_derivation_when_no_direct(self):
        p = DispersionParameters(f_c_layer=8e6, f_carrier=15e6)
        assert p.get_dispersion_coefficient() == pytest.approx(
            compute_d_from_qp(8e6, 15e6), rel=1e-9
        )

    def test_default_when_neither(self):
        p = DispersionParameters()
        # Docstring/code fallback: 50 us/MHz
        assert p.get_dispersion_coefficient() == 50.0


class TestDispersionModel:
    """Behavioral tests for the dispersion filter."""

    FS = 48_000.0

    def _rng_complex(self, n, seed=42):
        rng = np.random.default_rng(seed)
        return (
            rng.standard_normal(n) + 1j * rng.standard_normal(n)
        ).astype(np.complex64) / np.sqrt(2.0)

    def test_negligible_dispersion_is_identity(self):
        model = DispersionModel(fs=self.FS, use_compiled=False)
        x = self._rng_complex(4096)
        y = model.apply_dispersion(x, d_us_per_MHz=0.0)
        np.testing.assert_array_equal(y, x)

    def test_length_preserved_when_requested(self):
        model = DispersionModel(fs=self.FS, use_compiled=False)
        x = self._rng_complex(4096)
        y = model.apply_dispersion(x, d_us_per_MHz=30.0, preserve_length=True)
        assert y.shape == x.shape

    def test_full_length_when_requested(self):
        model = DispersionModel(fs=self.FS, use_compiled=False)
        x = self._rng_complex(4096)
        y = model.apply_dispersion(x, d_us_per_MHz=30.0, preserve_length=False)
        assert y.shape[0] > x.shape[0]

    @pytest.mark.xfail(
        reason=(
            "get_dispersion_filter normalizes to unit ENERGY (dispersion.py:282-284), "
            "not unit magnitude, so |H(f)| is not flat and the filter is not truly "
            "all-pass. dispersion + 'inverse dispersion' therefore does not recover "
            "the input. Fixing this needs a magnitude-based normalization; leaving "
            "this test as xfail so the audit finding is documented."
        ),
        strict=True,
    )
    def test_dispersion_then_inverse_recovers_input(self):
        model = DispersionModel(fs=self.FS, use_compiled=False)
        x = self._rng_complex(8192)
        d = 20.0
        y = model.apply_dispersion(x, d_us_per_MHz=d, preserve_length=True)
        x_hat = model.apply_inverse_dispersion(y, d_us_per_MHz=d, preserve_length=True)
        n = len(x)
        interior = slice(n // 4, 3 * n // 4)
        err = np.mean(np.abs(x_hat[interior] - x[interior]) ** 2)
        power = np.mean(np.abs(x[interior]) ** 2)
        assert err / power < 0.05

    def test_dispersion_preserves_signal_power_approximately(self):
        # All-pass -> unitary in the frequency domain; time-domain finite
        # support introduces edge losses. Measure power on interior only.
        model = DispersionModel(fs=self.FS, use_compiled=False)
        x = self._rng_complex(8192)
        y = model.apply_dispersion(x, d_us_per_MHz=15.0)
        interior = slice(2048, -2048)
        p_in = np.mean(np.abs(x[interior]) ** 2)
        p_out = np.mean(np.abs(y[interior]) ** 2)
        ratio_db = 10 * np.log10(p_out / p_in)
        assert abs(ratio_db) < 1.0, f"power ratio {ratio_db:.2f} dB, expected within +/-1 dB"


class TestDispersionUtilities:
    def test_estimate_dispersion_spread_linear(self):
        # Simple: spread = |d| * B
        assert estimate_dispersion_spread(20.0, 0.1) == pytest.approx(2.0)
        assert estimate_dispersion_spread(-20.0, 0.1) == pytest.approx(2.0)

    def test_typical_values_present(self):
        vals = typical_dispersion_values()
        assert isinstance(vals, dict)
        assert len(vals) > 0

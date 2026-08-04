"""Regression tests guarding recent bug fixes in channel models.

These tests exist to detect reverts of specific fixes:
  - Watterson tap-placement math (commit bcd2a81)
  - Vogler-Hoffmeyer AR(1) Box-Muller and sigma_f mapping (commit 62f59e1)

If any of these tests fail after a change, examine the change before assuming
the test is wrong — the RMS delay spread and Rayleigh envelope statistics
have well-defined values that should not drift.
"""

import numpy as np
import pytest

from hfpathsim.core.watterson import (
    WattersonChannel,
    WattersonConfig,
    WattersonTap,
)
from hfpathsim.core.vogler_hoffmeyer import (
    VoglerHoffmeyerChannel,
    VoglerHoffmeyerConfig,
    ModeParameters,
    CorrelationType,
    GPU_AVAILABLE,
)
from hfpathsim.validation.statistics import compute_delay_spread


SAMPLE_RATE_HZ = 48_000.0


def _watterson_pdp_delay_spread(taps, sample_rate_hz=SAMPLE_RATE_HZ):
    """Build a Watterson channel and measure its RMS delay spread from IR."""
    cfg = WattersonConfig(sample_rate_hz=sample_rate_hz, taps=taps)
    ch = WattersonChannel(cfg, seed=1234, use_compiled=False)
    ir_len = max(1024, int(max(t.delay_ms for t in taps) * sample_rate_hz / 1000) + 128)
    h = ch.get_impulse_response(length=ir_len)
    return compute_delay_spread(h, sample_rate_hz).rms_delay_spread_ms


class TestWattersonTapPlacementFormula:
    """Guards commit bcd2a81. For N equal-power taps at 0..tau_max:
        N=2  -> RMS = tau_max / 2
        N=3  -> RMS = tau_max / sqrt(6)
    """

    def test_two_taps_rms_equals_tau_max_over_2(self):
        tau_max_ms = 2.0
        taps = [
            WattersonTap(delay_ms=0.0, amplitude=1.0, doppler_spread_hz=0.5),
            WattersonTap(delay_ms=tau_max_ms, amplitude=1.0, doppler_spread_hz=0.5),
        ]
        expected = tau_max_ms / 2.0
        measured = _watterson_pdp_delay_spread(taps)
        # 5% tolerance to accommodate the instantaneous |gain|^2 weighting
        # from randomized fading. Grows tighter as duration/averaging grows.
        assert abs(measured - expected) < 0.05 * expected, (
            f"2-tap RMS delay spread {measured:.4f} ms differs from expected "
            f"{expected:.4f} ms by more than 5%"
        )

    def test_three_taps_rms_equals_tau_max_over_sqrt6(self):
        tau_max_ms = 3.0
        mid = tau_max_ms / 2.0
        taps = [
            WattersonTap(delay_ms=0.0, amplitude=1.0, doppler_spread_hz=0.5),
            WattersonTap(delay_ms=mid, amplitude=1.0, doppler_spread_hz=0.5),
            WattersonTap(delay_ms=tau_max_ms, amplitude=1.0, doppler_spread_hz=0.5),
        ]
        expected = tau_max_ms / np.sqrt(6.0)
        measured = _watterson_pdp_delay_spread(taps)
        assert abs(measured - expected) < 0.05 * expected, (
            f"3-tap RMS delay spread {measured:.4f} ms differs from expected "
            f"{expected:.4f} ms by more than 5%"
        )


class TestVHRayleighEnvelope:
    """Guards commit 62f59e1. VH CW output should have envelope ratio
    Mean(|g|)/RMS(|g|) close to sqrt(pi/4) ~= 0.886 for a Rayleigh process.
    """

    RAYLEIGH_RATIO = np.sqrt(np.pi / 4.0)

    def _measure_envelope_ratio(self, correlation_type, use_gpu=False):
        sigma_tau_us = 1000.0
        mode = ModeParameters(
            name="rayleigh_probe",
            amplitude=1.0,
            floor_amplitude=0.01,
            tau_L=0.0,
            sigma_tau=sigma_tau_us,
            sigma_c=sigma_tau_us / 2.0,
            sigma_D=1.0,
            doppler_shift=0.0,
            correlation_type=correlation_type,
        )
        cfg = VoglerHoffmeyerConfig(
            sample_rate=SAMPLE_RATE_HZ,
            modes=[mode],
            use_gpu=use_gpu,
        )
        ch = VoglerHoffmeyerChannel(cfg)
        # 10-second CW; downsample to 100 Hz before envelope stats
        n = int(10.0 * SAMPLE_RATE_HZ)
        cw = np.ones(n, dtype=np.complex128)
        y = ch.process(cw)
        env = np.abs(y[::480])  # 100 Hz sampling
        mean_env = float(np.mean(env))
        rms_env = float(np.sqrt(np.mean(env**2)))
        return mean_env / rms_env if rms_env > 0 else 0.0

    def test_gaussian_correlation_envelope_ratio(self):
        ratio = self._measure_envelope_ratio(CorrelationType.GAUSSIAN)
        # 15% tolerance around Rayleigh reference (matches validator threshold)
        assert abs(ratio - self.RAYLEIGH_RATIO) < 0.15 * self.RAYLEIGH_RATIO, (
            f"VH Gaussian envelope ratio {ratio:.4f} not within 15% of "
            f"Rayleigh reference {self.RAYLEIGH_RATIO:.4f} — AR(1)/Box-Muller "
            f"innovation may be broken"
        )

    def test_exponential_correlation_envelope_ratio(self):
        ratio = self._measure_envelope_ratio(CorrelationType.EXPONENTIAL)
        assert abs(ratio - self.RAYLEIGH_RATIO) < 0.15 * self.RAYLEIGH_RATIO, (
            f"VH Exponential envelope ratio {ratio:.4f} not within 15% of "
            f"Rayleigh reference {self.RAYLEIGH_RATIO:.4f}"
        )

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU module not available")
    def test_gaussian_envelope_ratio_gpu_path(self):
        # Exercises the GPU path independently. Audit noted CPU path had
        # essentially no coverage; this pairs the CPU test above with the
        # explicit GPU counterpart.
        ratio = self._measure_envelope_ratio(CorrelationType.GAUSSIAN, use_gpu=True)
        assert abs(ratio - self.RAYLEIGH_RATIO) < 0.15 * self.RAYLEIGH_RATIO

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU module not available")
    def test_exponential_envelope_ratio_gpu_path(self):
        ratio = self._measure_envelope_ratio(CorrelationType.EXPONENTIAL, use_gpu=True)
        assert abs(ratio - self.RAYLEIGH_RATIO) < 0.15 * self.RAYLEIGH_RATIO


class TestVHTapDecimationWarns:
    """The GPU code path is limited to 16 taps. When decimated from a larger
    tap set, total power is preserved but the RMS delay spread is NOT — the
    audit flagged that this was happening silently. Warning must fire.
    """

    def test_gpu_decimation_emits_warning(self):
        # Choose sigma_tau big enough vs sample rate that we exceed 16 taps.
        # At 48 kHz, delta_t ~= 20.83 us; sigma_tau = 500 us gives ~25 taps.
        mode = ModeParameters(
            name="decimation_probe",
            amplitude=1.0,
            floor_amplitude=0.01,
            tau_L=0.0,
            sigma_tau=500.0,
            sigma_c=250.0,
            sigma_D=1.0,
            correlation_type=CorrelationType.EXPONENTIAL,
        )
        cfg = VoglerHoffmeyerConfig(
            sample_rate=SAMPLE_RATE_HZ,
            modes=[mode],
            use_gpu=True,
        )
        if not GPU_AVAILABLE:
            pytest.skip("GPU module not available; decimation path unreachable")
        with pytest.warns(RuntimeWarning, match=r"GPU tap decimation"):
            VoglerHoffmeyerChannel(cfg)


class TestVHSigmaFMapping:
    """Guards the sigma_f = sigma_D mapping for the exponential branch
    (audit: the previous 2*pi*sigma_D was ~6.28x too fast).
    """

    def test_exponential_sigma_f_equals_sigma_d(self):
        sigma_D = 3.0
        mode = ModeParameters(
            name="sigma_f_probe",
            amplitude=1.0,
            floor_amplitude=0.01,
            tau_L=0.0,
            sigma_tau=1000.0,
            sigma_c=500.0,
            sigma_D=sigma_D,
            correlation_type=CorrelationType.EXPONENTIAL,
        )
        cfg = VoglerHoffmeyerConfig(sample_rate=SAMPLE_RATE_HZ, modes=[mode])
        ch = VoglerHoffmeyerChannel(cfg)
        sigma_f = ch._compute_sigma_f(mode)
        assert sigma_f == pytest.approx(sigma_D, rel=1e-6), (
            f"Exponential sigma_f = {sigma_f}, expected {sigma_D} "
            f"(reverting to 2*pi*sigma_D reintroduces a 2*pi bug)"
        )

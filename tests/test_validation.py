"""Tests for HF channel validation module."""

import numpy as np
import pytest
from scipy import stats

from hfpathsim.validation import (
    # Reference data
    ReferenceDataset,
    NTIAMeasurement,
    ITUReference,
    NTIA_MIDLATITUDE_QUIET,
    NTIA_MIDLATITUDE_DISTURBED,
    NTIA_AURORAL,
    NTIA_SPREAD_F,
    ITU_F1487_QUIET,
    ITU_F1487_MODERATE,
    ITU_F1487_DISTURBED,
    ITU_F1487_FLUTTER,
    WATTERSON_1970_GOOD,
    WATTERSON_1970_MODERATE,
    WATTERSON_1970_POOR,
    get_reference_dataset,
    list_reference_datasets,
    # Statistics
    compute_delay_spread,
    compute_doppler_spread,
    compute_coherence_bandwidth,
    compute_coherence_time,
    compute_scattering_function,
    compare_scattering_functions,
    compute_fading_statistics,
    rayleigh_fit_test,
    compute_level_crossing_rate,
    compute_average_fade_duration,
    compute_fade_depth,
    # Validator
    ChannelValidator,
    ValidationResult,
    ValidationReport,
    validate_channel,
)
from hfpathsim.validation.validator import ValidationStatus


# =============================================================================
# Reference Data Tests
# =============================================================================

class TestReferenceDatasets:
    """Tests for reference dataset definitions."""

    def test_ntia_datasets_exist(self):
        """All NTIA datasets should be defined."""
        assert NTIA_MIDLATITUDE_QUIET is not None
        assert NTIA_MIDLATITUDE_DISTURBED is not None
        assert NTIA_AURORAL is not None
        assert NTIA_SPREAD_F is not None

    def test_itu_datasets_exist(self):
        """All ITU-R datasets should be defined."""
        assert ITU_F1487_QUIET is not None
        assert ITU_F1487_MODERATE is not None
        assert ITU_F1487_DISTURBED is not None
        assert ITU_F1487_FLUTTER is not None

    def test_watterson_datasets_exist(self):
        """All Watterson datasets should be defined."""
        assert WATTERSON_1970_GOOD is not None
        assert WATTERSON_1970_MODERATE is not None
        assert WATTERSON_1970_POOR is not None

    def test_ntia_quiet_values(self):
        """NTIA quiet dataset should have expected values."""
        ds = NTIA_MIDLATITUDE_QUIET
        assert ds.delay_spread_ms == 0.5
        assert ds.doppler_spread_hz == 0.2
        assert ds.source == "NTIA TR-90-255"
        assert ds.year == 1988

    def test_itu_disturbed_values(self):
        """ITU disturbed dataset should have expected values."""
        ds = ITU_F1487_DISTURBED
        assert ds.delay_spread_ms == 4.0
        assert ds.doppler_spread_hz == 2.0

    def test_reference_dataset_coherence_bandwidth(self):
        """Coherence bandwidth calculation should be correct."""
        ds = NTIA_MIDLATITUDE_QUIET
        bc = ds.get_coherence_bandwidth_khz()
        # Bc = 1/(2π × τ_rms) = 1/(2π × 0.5ms) ≈ 0.318 kHz
        assert 0.3 < bc < 0.35

    def test_reference_dataset_coherence_time(self):
        """Coherence time calculation should be correct."""
        ds = NTIA_MIDLATITUDE_QUIET
        tc = ds.get_coherence_time_ms()
        # Tc = 1000/(2π × fd) = 1000/(2π × 0.2) ≈ 795.8 ms
        assert 700 < tc < 900

    def test_get_reference_dataset(self):
        """Should retrieve dataset by name."""
        ds = get_reference_dataset("ntia_midlatitude_quiet")
        assert ds == NTIA_MIDLATITUDE_QUIET

        ds = get_reference_dataset("itu_f1487_moderate")
        assert ds == ITU_F1487_MODERATE

    def test_get_reference_dataset_case_insensitive(self):
        """Should be case-insensitive."""
        ds = get_reference_dataset("NTIA_MIDLATITUDE_QUIET")
        assert ds == NTIA_MIDLATITUDE_QUIET

    def test_get_reference_dataset_invalid(self):
        """Should return None for invalid name."""
        ds = get_reference_dataset("nonexistent")
        assert ds is None

    def test_list_reference_datasets(self):
        """Should list all dataset names."""
        names = list_reference_datasets()
        assert len(names) == 11
        assert "ntia_midlatitude_quiet" in names
        assert "itu_f1487_flutter" in names
        assert "watterson_1970_poor" in names

    def test_reference_dataset_to_dict(self):
        """Should convert to dictionary."""
        d = NTIA_MIDLATITUDE_QUIET.to_dict()
        assert d["name"] == "NTIA Midlatitude Quiet"
        assert d["delay_spread_ms"] == 0.5
        assert "coherence_bandwidth_khz" in d


# =============================================================================
# Statistics Tests
# =============================================================================

class TestDelaySpread:
    """Tests for delay spread computation."""

    def test_compute_delay_spread_simple(self):
        """Compute delay spread from simple impulse response."""
        # Single tap - zero delay spread
        h = np.array([1.0 + 0j])
        result = compute_delay_spread(h, sample_rate_hz=48000)
        assert result.rms_delay_spread_ms == 0.0

    def test_compute_delay_spread_two_taps(self):
        """Two equal taps should give known delay spread."""
        # Two taps at 0 and 1ms with equal power
        sample_rate_hz = 1000  # 1ms per sample
        h = np.array([1.0 + 0j, 1.0 + 0j])  # Equal power at 0 and 1ms
        result = compute_delay_spread(h, sample_rate_hz)

        # Mean delay = 0.5ms, RMS spread = sqrt(0.25) = 0.5ms
        assert 0.4 < result.rms_delay_spread_ms < 0.6
        assert 0.4 < result.mean_delay_ms < 0.6

    def test_compute_delay_spread_exponential(self):
        """Exponential decay profile."""
        sample_rate_hz = 10000  # 0.1ms per sample
        n_taps = 100
        tau_decay = 0.5  # 0.5ms decay constant

        # Exponential power delay profile
        t = np.arange(n_taps) / sample_rate_hz * 1000  # in ms
        h = np.exp(-t / tau_decay).astype(complex)

        result = compute_delay_spread(h, sample_rate_hz)

        # For truncated exponential, RMS spread is less than tau_decay
        # The exact value depends on truncation but should be positive
        assert 0.1 < result.rms_delay_spread_ms < 0.5

    def test_compute_delay_spread_zero_energy(self):
        """Zero energy should return zeros."""
        h = np.zeros(10, dtype=complex)
        result = compute_delay_spread(h, sample_rate_hz=48000)
        assert result.rms_delay_spread_ms == 0.0


class TestDopplerSpread:
    """Tests for Doppler spread computation."""

    def test_compute_doppler_spread_static(self):
        """Static channel should have near-zero Doppler."""
        # Constant fading coefficient
        fading = np.ones(1000, dtype=complex)
        result = compute_doppler_spread(fading, sample_rate_hz=100)
        assert result.rms_doppler_spread_hz < 0.1

    def test_compute_doppler_spread_sinusoidal(self):
        """Sinusoidal fading should show Doppler at that frequency."""
        sample_rate_hz = 100.0
        duration_sec = 10.0
        fd = 2.0  # 2 Hz Doppler

        t = np.arange(int(duration_sec * sample_rate_hz)) / sample_rate_hz
        fading = np.exp(1j * 2 * np.pi * fd * t)

        result = compute_doppler_spread(fading, sample_rate_hz)

        # Pure tone has mean shift at fd but zero RMS spread (single frequency)
        # Mean Doppler should be at 2 Hz
        assert 1.5 < result.mean_doppler_shift_hz < 2.5
        # RMS spread around mean is near zero for pure tone
        assert result.rms_doppler_spread_hz < 1.0

    def test_compute_doppler_spread_rayleigh(self):
        """Rayleigh fading should have spread related to Doppler."""
        sample_rate_hz = 100.0
        n_samples = 1000

        # Generate complex Gaussian fading
        fading = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) / np.sqrt(2)

        result = compute_doppler_spread(fading, sample_rate_hz)

        # White noise has spread related to bandwidth
        assert result.rms_doppler_spread_hz > 0


class TestCoherence:
    """Tests for coherence bandwidth and time."""

    def test_coherence_bandwidth(self):
        """Coherence bandwidth from delay spread."""
        # τ_rms = 0.5ms -> Bc = 1/(2π×0.5) ≈ 0.318 kHz
        bc = compute_coherence_bandwidth(0.5)
        assert 0.3 < bc < 0.35

    def test_coherence_bandwidth_zero(self):
        """Zero delay spread gives infinite coherence bandwidth."""
        bc = compute_coherence_bandwidth(0.0)
        assert bc == float('inf')

    def test_coherence_time(self):
        """Coherence time from Doppler spread."""
        # fd = 1 Hz -> Tc = 1000/(2π×1) ≈ 159.15 ms
        tc = compute_coherence_time(1.0)
        assert 150 < tc < 170

    def test_coherence_time_zero(self):
        """Zero Doppler spread gives infinite coherence time."""
        tc = compute_coherence_time(0.0)
        assert tc == float('inf')


class TestScatteringFunction:
    """Tests for scattering function computation."""

    def test_compute_scattering_function_shape(self):
        """Scattering function should have correct shape."""
        n_snapshots = 50
        n_delay = 32
        channel = np.random.randn(n_snapshots, n_delay) + 1j * np.random.randn(n_snapshots, n_delay)

        delay_axis, doppler_axis, S = compute_scattering_function(
            channel,
            sample_rate_hz=48000,
            snapshot_rate_hz=100,
            n_delay_bins=16,
            n_doppler_bins=32,
        )

        assert S.shape == (32, 16)
        assert len(delay_axis) == 16
        assert len(doppler_axis) == 32

    def test_compare_scattering_functions_identical(self):
        """Identical functions should have correlation 1."""
        S = np.random.rand(32, 32).astype(np.float32)
        delay = np.arange(32) * 0.1
        doppler = np.arange(32) - 16

        result = compare_scattering_functions(
            S, S, delay, delay, doppler, doppler
        )

        assert result.correlation > 0.99
        assert result.rmse < 0.01

    def test_compare_scattering_functions_orthogonal(self):
        """Very different functions should have low correlation."""
        S1 = np.zeros((32, 32), dtype=np.float32)
        S1[0:10, 0:10] = 1.0

        S2 = np.zeros((32, 32), dtype=np.float32)
        S2[20:32, 20:32] = 1.0

        delay = np.arange(32) * 0.1
        doppler = np.arange(32) - 16

        result = compare_scattering_functions(
            S1, S2, delay, delay, doppler, doppler
        )

        assert result.correlation < 0.5


class TestFadingStatistics:
    """Tests for fading statistics computation."""

    def test_rayleigh_fit_rayleigh_samples(self):
        """Rayleigh samples should pass Rayleigh test."""
        # Generate Rayleigh samples
        n_samples = 10000
        sigma = 1.0
        envelope = stats.rayleigh.rvs(scale=sigma, size=n_samples)

        pvalue = rayleigh_fit_test(envelope)

        # Should have high p-value (accept null hypothesis)
        assert pvalue > 0.01

    def test_rayleigh_fit_uniform_samples(self):
        """Uniform samples should fail Rayleigh test."""
        envelope = np.random.uniform(0.5, 1.5, size=10000)

        pvalue = rayleigh_fit_test(envelope)

        # Should have low p-value (reject null hypothesis)
        assert pvalue < 0.05

    def test_level_crossing_rate(self):
        """Level crossing rate calculation."""
        sample_rate_hz = 1000.0
        # Sinusoidal signal crosses level upward once per period
        t = np.arange(1000) / sample_rate_hz
        envelope = 1 + np.sin(2 * np.pi * 10 * t)  # 10 Hz -> 10 upward crossings/sec at level 1

        lcr = compute_level_crossing_rate(envelope, sample_rate_hz, level=1.0)

        # Should be close to 10 Hz (one upward crossing per cycle)
        assert 8 < lcr < 12

    def test_average_fade_duration(self):
        """Average fade duration calculation."""
        sample_rate_hz = 1000.0

        # Signal that's below threshold half the time
        envelope = np.concatenate([
            np.zeros(50),   # 50ms below
            np.ones(50),    # 50ms above
            np.zeros(100),  # 100ms below
            np.ones(100),   # 100ms above
        ])

        afd = compute_average_fade_duration(envelope, sample_rate_hz, level=0.5)

        # Average of 50ms and 100ms = 75ms
        assert 60 < afd < 90

    def test_fade_depth(self):
        """Fade depth calculation."""
        envelope = np.array([0.1, 0.5, 1.0, 0.5, 0.2])

        depth = compute_fade_depth(envelope, percentile=1.0)

        # 20*log10(1.0/0.1) = 20 dB
        assert 15 < depth < 25

    def test_compute_fading_statistics(self):
        """Full fading statistics computation."""
        # Generate Rayleigh fading
        n_samples = 10000
        envelope = stats.rayleigh.rvs(scale=1.0, size=n_samples)

        result = compute_fading_statistics(envelope, sample_rate_hz=1000)

        assert result.mean_envelope > 0
        assert result.std_envelope > 0
        assert result.fade_depth_db > 0
        assert result.rayleigh_fit_pvalue > 0


# =============================================================================
# Validator Tests
# =============================================================================

class TestChannelValidator:
    """Tests for ChannelValidator class."""

    def test_validator_initialization(self):
        """Validator should initialize with reference."""
        validator = ChannelValidator(reference=NTIA_MIDLATITUDE_QUIET)
        assert validator.reference == NTIA_MIDLATITUDE_QUIET
        assert validator.delay_tolerance_pct == 50.0

    def test_validation_with_impulse_response(self):
        """Should validate delay spread from impulse response."""
        validator = ChannelValidator(reference=NTIA_MIDLATITUDE_QUIET)

        # Generate impulse response matching reference
        sample_rate_hz = 48000
        n_taps = 100
        tau_rms_ms = 0.5  # Match reference

        # Exponential decay
        t = np.arange(n_taps) / sample_rate_hz * 1000
        h = np.exp(-t / tau_rms_ms).astype(complex)

        # Single snapshot
        h = h.reshape(1, -1)

        report = validator.validate(
            impulse_responses=h,
            sample_rate_hz=sample_rate_hz,
        )

        assert report.delay_spread_result is not None
        assert len(report.results) > 0

    def test_validation_with_fading(self):
        """Should validate Doppler spread from fading coefficients."""
        validator = ChannelValidator(reference=NTIA_MIDLATITUDE_QUIET)

        # Generate Rayleigh fading
        n_samples = 10000
        fading = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) / np.sqrt(2)

        report = validator.validate(
            fading_coefficients=fading,
            snapshot_rate_hz=100,
        )

        assert report.doppler_spread_result is not None
        assert report.fading_stats is not None
        assert len(report.results) > 0

    def test_validation_report_methods(self):
        """ValidationReport should have working methods."""
        validator = ChannelValidator(reference=NTIA_MIDLATITUDE_QUIET)

        fading = np.random.randn(1000) + 1j * np.random.randn(1000)
        report = validator.validate(fading_coefficients=fading, snapshot_rate_hz=100)

        # Test methods
        assert 0 <= report.get_pass_rate() <= 100
        assert isinstance(report.get_failed_tests(), list)

        # Test serialization
        d = report.to_dict()
        assert "reference" in d
        assert "results" in d

        json_str = report.to_json()
        assert isinstance(json_str, str)


class TestValidateChannel:
    """Tests for validate_channel convenience function."""

    def test_validate_channel_by_name(self):
        """Should accept reference by name."""
        fading = np.random.randn(1000) + 1j * np.random.randn(1000)

        report = validate_channel(
            fading_coefficients=fading,
            reference="ntia_midlatitude_quiet",
        )

        assert report.reference == NTIA_MIDLATITUDE_QUIET

    def test_validate_channel_by_dataset(self):
        """Should accept reference dataset object."""
        fading = np.random.randn(1000) + 1j * np.random.randn(1000)

        report = validate_channel(
            fading_coefficients=fading,
            reference=ITU_F1487_MODERATE,
        )

        assert report.reference == ITU_F1487_MODERATE

    def test_validate_channel_invalid_reference(self):
        """Should raise error for invalid reference name."""
        fading = np.random.randn(1000) + 1j * np.random.randn(1000)

        with pytest.raises(ValueError, match="Unknown reference dataset"):
            validate_channel(
                fading_coefficients=fading,
                reference="nonexistent_dataset",
            )


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_validation_result_creation(self):
        """Should create ValidationResult correctly."""
        result = ValidationResult(
            name="Test",
            status=ValidationStatus.PASS,
            measured_value=1.0,
            reference_value=1.0,
            tolerance_pct=10.0,
            error_pct=0.0,
            details="Test passed",
        )

        assert result.status == ValidationStatus.PASS
        assert result.error_pct == 0.0

    def test_validation_result_to_dict(self):
        """Should convert to dictionary."""
        result = ValidationResult(
            name="Test",
            status=ValidationStatus.FAIL,
            measured_value=2.0,
            reference_value=1.0,
            tolerance_pct=10.0,
            error_pct=100.0,
        )

        d = result.to_dict()
        assert d["status"] == "fail"
        assert d["error_pct"] == 100.0


class TestValidationStatus:
    """Tests for ValidationStatus enum."""

    def test_status_values(self):
        """All status values should exist."""
        assert ValidationStatus.PASS.value == "pass"
        assert ValidationStatus.FAIL.value == "fail"
        assert ValidationStatus.WARN.value == "warn"
        assert ValidationStatus.SKIP.value == "skip"


# =============================================================================
# Integration Tests
# =============================================================================

class TestValidationIntegration:
    """Integration tests for validation workflow."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        # Generate synthetic channel matching NTIA quiet conditions
        sample_rate_hz = 48000
        snapshot_rate_hz = 100
        duration_sec = 5.0

        n_snapshots = int(duration_sec * snapshot_rate_hz)
        n_taps = 50

        # Generate impulse responses with appropriate delay spread
        tau_rms_ms = 0.5  # Target 0.5ms
        t = np.arange(n_taps) / sample_rate_hz * 1000

        impulse_responses = np.zeros((n_snapshots, n_taps), dtype=complex)
        for i in range(n_snapshots):
            # Exponential decay with random phase
            h = np.exp(-t / tau_rms_ms) * np.exp(1j * np.random.uniform(0, 2*np.pi, n_taps))
            impulse_responses[i] = h

        # Generate fading coefficients (Rayleigh)
        fading = (np.random.randn(n_snapshots) + 1j * np.random.randn(n_snapshots)) / np.sqrt(2)

        # Validate
        report = validate_channel(
            impulse_responses=impulse_responses,
            fading_coefficients=fading,
            sample_rate_hz=sample_rate_hz,
            snapshot_rate_hz=snapshot_rate_hz,
            reference=NTIA_MIDLATITUDE_QUIET,
        )

        # Check report
        assert report.reference == NTIA_MIDLATITUDE_QUIET
        assert report.delay_spread_result is not None
        assert report.doppler_spread_result is not None
        assert report.fading_stats is not None
        assert len(report.results) >= 3

    def test_validation_against_multiple_references(self):
        """Validate against multiple references."""
        fading = (np.random.randn(5000) + 1j * np.random.randn(5000)) / np.sqrt(2)

        references = [
            NTIA_MIDLATITUDE_QUIET,
            ITU_F1487_QUIET,
            WATTERSON_1970_GOOD,
        ]

        for ref in references:
            report = validate_channel(
                fading_coefficients=fading,
                reference=ref,
            )

            assert report.reference == ref
            assert report.overall_status in [
                ValidationStatus.PASS,
                ValidationStatus.WARN,
                ValidationStatus.FAIL,
            ]

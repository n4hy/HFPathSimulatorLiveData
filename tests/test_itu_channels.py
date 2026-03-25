"""Tests for ITU-R standardized HF channel models."""

import numpy as np
import pytest

from hfpathsim.core.itu_channels import (
    # Enums
    CCIR520Condition,
    ITURF1289Condition,
    ITURF1487Condition,
    # Presets
    CCIR520_PRESETS,
    ITURF1289_PRESETS,
    ITURF1487_PRESETS,
    # Channel classes
    CCIR520Channel,
    ITURF1289Channel,
    ITURF1487Channel,
    # Utilities
    list_ccir520_presets,
    list_iturf1289_presets,
    list_iturf1487_presets,
    get_preset_info,
    create_channel,
)


class TestCCIR520Presets:
    """Tests for CCIR 520 / ITU-R F.520 presets."""

    def test_all_conditions_have_presets(self):
        """Verify all CCIR520 conditions have defined presets."""
        for condition in CCIR520Condition:
            assert condition in CCIR520_PRESETS

    def test_preset_parameters_valid(self):
        """Verify preset parameters are physically valid."""
        for condition, spec in CCIR520_PRESETS.items():
            # Delay spread should be non-negative
            assert spec.delay_spread_ms >= 0

            # Doppler spread should be positive
            assert spec.doppler_spread_hz > 0

            # Number of paths should match arrays
            assert len(spec.path_delays_ms) == spec.num_paths
            assert len(spec.path_amplitudes) == spec.num_paths
            assert len(spec.path_doppler_spreads_hz) == spec.num_paths

            # Amplitudes should be positive
            for amp in spec.path_amplitudes:
                assert amp > 0

    def test_list_presets(self):
        """Test listing CCIR520 presets."""
        presets = list_ccir520_presets()
        assert len(presets) == len(CCIR520Condition)
        assert "good_low_latency" in presets
        assert "moderate" in presets
        assert "poor" in presets


class TestITURF1289Presets:
    """Tests for ITU-R F.1289 wideband presets."""

    def test_all_conditions_have_presets(self):
        """Verify all F.1289 conditions have presets."""
        for condition in ITURF1289Condition:
            assert condition in ITURF1289_PRESETS

    def test_wideband_parameters(self):
        """Verify wideband-specific parameters."""
        for condition, spec in ITURF1289_PRESETS.items():
            # Coherence bandwidth should be positive
            assert spec.coherence_bandwidth_khz > 0

            # Disturbed conditions should have lower coherence BW
            if "disturbed" in condition.value.lower():
                assert spec.coherence_bandwidth_khz <= 100

    def test_list_presets(self):
        """Test listing F.1289 presets."""
        presets = list_iturf1289_presets()
        assert len(presets) == len(ITURF1289Condition)
        assert "mid_latitude_moderate" in presets


class TestITURF1487Presets:
    """Tests for ITU-R F.1487 presets."""

    def test_table1_values(self):
        """Verify ITU-R F.1487 Table 1 values."""
        # Quiet
        quiet = ITURF1487_PRESETS[ITURF1487Condition.QUIET]
        assert quiet.delay_spread_ms == 0.5
        assert quiet.doppler_spread_hz == 0.1

        # Moderate
        mod = ITURF1487_PRESETS[ITURF1487Condition.MODERATE]
        assert mod.delay_spread_ms == 2.0
        assert mod.doppler_spread_hz == 1.0

        # Disturbed
        dist = ITURF1487_PRESETS[ITURF1487Condition.DISTURBED]
        assert dist.delay_spread_ms == 4.0
        assert dist.doppler_spread_hz == 2.0

        # Flutter
        flutter = ITURF1487_PRESETS[ITURF1487Condition.FLUTTER]
        assert flutter.delay_spread_ms == 7.0
        assert flutter.doppler_spread_hz == 10.0


class TestCCIR520Channel:
    """Tests for CCIR 520 channel implementation."""

    def test_create_from_preset(self):
        """Test channel creation from preset."""
        channel = CCIR520Channel.from_preset(CCIR520Condition.MODERATE)
        assert channel is not None
        assert channel.spec.condition == CCIR520Condition.MODERATE

    def test_convenience_constructors(self):
        """Test convenience class methods."""
        good = CCIR520Channel.good()
        assert good.spec.doppler_spread_hz == 0.1

        moderate = CCIR520Channel.moderate()
        assert moderate.spec.doppler_spread_hz == 0.5

        poor = CCIR520Channel.poor()
        assert poor.spec.doppler_spread_hz == 1.0

        flutter = CCIR520Channel.flutter()
        assert flutter.spec.doppler_spread_hz == 10.0

    def test_process_signal(self):
        """Test signal processing through channel."""
        channel = CCIR520Channel.from_preset(
            CCIR520Condition.MODERATE,
            sample_rate_hz=48000,
            seed=42,
        )

        # Generate test signal
        n_samples = 4096
        input_signal = (
            np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        ).astype(np.complex64)

        output = channel.process_block(input_signal)

        assert output.shape == input_signal.shape
        assert output.dtype == np.complex64
        assert np.all(np.isfinite(output))

    def test_deterministic_with_seed(self):
        """Test reproducibility with seed."""
        input_signal = (
            np.random.randn(1024) + 1j * np.random.randn(1024)
        ).astype(np.complex64)

        channel1 = CCIR520Channel.from_preset(
            CCIR520Condition.POOR, seed=12345
        )
        output1 = channel1.process_block(input_signal.copy())

        channel2 = CCIR520Channel.from_preset(
            CCIR520Condition.POOR, seed=12345
        )
        output2 = channel2.process_block(input_signal.copy())

        np.testing.assert_array_almost_equal(output1, output2, decimal=5)

    def test_channel_state(self):
        """Test channel state retrieval."""
        channel = CCIR520Channel.from_preset(CCIR520Condition.MODERATE)

        # Process some samples to advance state
        input_signal = np.ones(1024, dtype=np.complex64)
        channel.process_block(input_signal)

        state = channel.get_state()

        assert "time" in state
        assert "taps" in state
        assert len(state["taps"]) == channel.spec.num_paths


class TestITURF1289Channel:
    """Tests for ITU-R F.1289 wideband channel."""

    def test_create_from_preset(self):
        """Test channel creation from preset."""
        channel = ITURF1289Channel.from_preset(
            ITURF1289Condition.MID_LATITUDE_MODERATE
        )
        assert channel is not None

    def test_wideband_parameters(self):
        """Test wideband-specific functionality."""
        channel = ITURF1289Channel.from_preset(
            ITURF1289Condition.MID_LATITUDE_MODERATE,
            bandwidth_khz=12.0,
        )

        coherence_bw = channel.get_coherence_bandwidth()
        assert coherence_bw > 0

        # Check frequency-selectivity
        is_selective = channel.is_frequency_selective()
        assert isinstance(is_selective, bool)

    def test_process_wideband_signal(self):
        """Test processing wideband signal."""
        channel = ITURF1289Channel.from_preset(
            ITURF1289Condition.HIGH_LATITUDE_MODERATE,
            sample_rate_hz=48000,
            bandwidth_khz=24.0,
            seed=42,
        )

        n_samples = 4096
        input_signal = (
            np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        ).astype(np.complex64)

        output = channel.process_block(input_signal)

        assert output.shape == input_signal.shape
        assert np.all(np.isfinite(output))

    def test_dispersion_enabled(self):
        """Test that disturbed conditions enable dispersion."""
        channel = ITURF1289Channel.from_preset(
            ITURF1289Condition.MID_LATITUDE_DISTURBED
        )
        assert channel._dispersion_enabled
        assert channel._dispersion_factor > 0


class TestITURF1487Channel:
    """Tests for ITU-R F.1487 channel."""

    def test_convenience_constructors(self):
        """Test convenience class methods."""
        quiet = ITURF1487Channel.quiet()
        assert quiet.itu_condition == ITURF1487Condition.QUIET

        moderate = ITURF1487Channel.moderate()
        assert moderate.itu_condition == ITURF1487Condition.MODERATE

        disturbed = ITURF1487Channel.disturbed()
        assert disturbed.itu_condition == ITURF1487Condition.DISTURBED

        flutter = ITURF1487Channel.flutter()
        assert flutter.itu_condition == ITURF1487Condition.FLUTTER

    def test_process_signal(self):
        """Test signal processing."""
        channel = ITURF1487Channel.moderate(seed=42)

        input_signal = np.exp(
            2j * np.pi * 1000 * np.arange(2048) / 48000
        ).astype(np.complex64)

        output = channel.process_block(input_signal)

        assert output.shape == input_signal.shape
        assert np.all(np.isfinite(output))


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_preset_info_ccir520(self):
        """Test getting CCIR520 preset info."""
        info = get_preset_info("moderate")
        assert info is not None
        assert info["type"] == "CCIR520"
        assert info["delay_spread_ms"] == 1.0
        assert info["doppler_spread_hz"] == 0.5

    def test_get_preset_info_f1289(self):
        """Test getting F.1289 preset info."""
        info = get_preset_info("mid_latitude_moderate")
        assert info is not None
        assert info["type"] == "ITURF1289"
        assert "coherence_bandwidth_khz" in info

    def test_get_preset_info_f1487(self):
        """Test getting F.1487 preset info."""
        info = get_preset_info("quiet")
        # CCIR520 has a 'good_low_latency' with similar params
        # F.1487 'quiet' should also exist
        assert info is not None

    def test_get_preset_info_invalid(self):
        """Test invalid preset name."""
        info = get_preset_info("nonexistent_preset")
        assert info is None

    def test_create_channel_ccir520(self):
        """Test creating channel from CCIR520 preset name."""
        channel = create_channel("moderate")
        assert isinstance(channel, CCIR520Channel)

    def test_create_channel_f1289(self):
        """Test creating channel from F.1289 preset name."""
        channel = create_channel("mid_latitude_moderate")
        assert isinstance(channel, ITURF1289Channel)

    def test_create_channel_invalid(self):
        """Test creating channel with invalid name."""
        with pytest.raises(ValueError):
            create_channel("invalid_preset_name")


class TestChannelBehavior:
    """Tests for channel physical behavior."""

    def test_multipath_creates_delay(self):
        """Verify multipath creates appropriate delay structure."""
        channel = CCIR520Channel.from_preset(
            CCIR520Condition.MODERATE_MULTIPATH,
            sample_rate_hz=48000,
            seed=42,
        )

        h = channel.get_impulse_response(256)

        # Should have non-zero values at multiple delays
        nonzero_indices = np.where(np.abs(h) > 0.01)[0]
        assert len(nonzero_indices) >= 2

    def test_doppler_causes_time_variation(self):
        """Verify Doppler spread causes time-varying fading."""
        channel = CCIR520Channel.from_preset(
            CCIR520Condition.FLUTTER,  # High Doppler
            sample_rate_hz=48000,
            seed=42,
        )

        # Get impulse response at two different times
        h1 = channel.get_impulse_response(64)

        # Process some samples to advance time
        _ = channel.process_block(np.ones(4800, dtype=np.complex64))

        h2 = channel.get_impulse_response(64)

        # Impulse responses should be different due to fading
        diff = np.max(np.abs(h1 - h2))
        assert diff > 0.01  # Some variation expected

    def test_frequency_response(self):
        """Test frequency response computation."""
        channel = CCIR520Channel.from_preset(
            CCIR520Condition.POOR_MULTIPATH,
            sample_rate_hz=48000,
        )

        freq, H = channel.get_frequency_response(1024)

        assert len(freq) == 1024
        assert len(H) == 1024
        assert H.dtype == np.complex64

        # Check for frequency-selective nulls (poor multipath should have some)
        H_mag = np.abs(H)
        variation_db = 20 * np.log10(np.max(H_mag) / (np.min(H_mag) + 1e-10))
        # Some variation expected for multipath channel
        assert variation_db > 0


@pytest.mark.benchmark
class TestChannelPerformance:
    """Performance benchmarks for ITU channel models."""

    def test_ccir520_throughput(self):
        """Benchmark CCIR520 processing throughput."""
        import time

        channel = CCIR520Channel.moderate(sample_rate_hz=48000, seed=42)
        n_samples = 48000  # 1 second at 48 kHz
        input_signal = (
            np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        ).astype(np.complex64)

        # Warmup
        _ = channel.process_block(input_signal[:4096].copy())

        # Benchmark
        start = time.time()
        for _ in range(10):
            _ = channel.process_block(input_signal.copy())
        elapsed = time.time() - start

        throughput_sps = (n_samples * 10) / elapsed
        print(f"\nCCIR520 throughput: {throughput_sps / 1000:.1f} ksps")
        assert throughput_sps > 10000  # At least 10 ksps

    def test_iturf1289_throughput(self):
        """Benchmark F.1289 wideband processing throughput."""
        import time

        channel = ITURF1289Channel.from_preset(
            ITURF1289Condition.MID_LATITUDE_MODERATE,
            sample_rate_hz=48000,
            seed=42,
        )
        n_samples = 48000
        input_signal = (
            np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        ).astype(np.complex64)

        # Warmup
        _ = channel.process_block(input_signal[:4096].copy())

        # Benchmark
        start = time.time()
        for _ in range(10):
            _ = channel.process_block(input_signal.copy())
        elapsed = time.time() - start

        throughput_sps = (n_samples * 10) / elapsed
        print(f"\nF.1289 throughput: {throughput_sps / 1000:.1f} ksps")
        assert throughput_sps > 5000  # At least 5 ksps (with dispersion)

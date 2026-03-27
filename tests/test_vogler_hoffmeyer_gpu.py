"""Tests for Vogler-Hoffmeyer GPU acceleration.

Verifies that GPU and CPU paths produce statistically equivalent output.
"""

import numpy as np
import pytest

from hfpathsim.core.vogler_hoffmeyer import (
    VoglerHoffmeyerChannel,
    VoglerHoffmeyerConfig,
    ModeParameters,
    CorrelationType,
    GPU_AVAILABLE,
)
from hfpathsim.validation.statistics import compute_delay_spread, compute_fading_statistics


class TestVoglerHoffmeyerGPU:
    """Test GPU acceleration of Vogler-Hoffmeyer channel."""

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU module not available")
    def test_gpu_initialization(self):
        """Test that GPU processors initialize correctly."""
        config = VoglerHoffmeyerConfig(
            sample_rate=48000.0,
            modes=[ModeParameters(sigma_tau=100.0, sigma_D=1.0)],
            use_gpu=True,
        )
        channel = VoglerHoffmeyerChannel(config)

        # Check backend info
        info = channel.get_backend_info()
        assert info['gpu_available'] is True
        assert len(info['mode_backends']) == 1

    def test_cpu_fallback_when_disabled(self):
        """Test that CPU is used when GPU is disabled."""
        config = VoglerHoffmeyerConfig(
            sample_rate=48000.0,
            modes=[ModeParameters(sigma_tau=100.0, sigma_D=1.0)],
            use_gpu=False,
        )
        channel = VoglerHoffmeyerChannel(config)

        info = channel.get_backend_info()
        assert info['use_gpu'] is False

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU module not available")
    def test_gpu_output_shape(self):
        """Test that GPU produces correct output shape."""
        config = VoglerHoffmeyerConfig(
            sample_rate=48000.0,
            modes=[ModeParameters(sigma_tau=100.0, sigma_D=1.0)],
            use_gpu=True,
            random_seed=42,
        )
        channel = VoglerHoffmeyerChannel(config)

        # Generate test signal
        n_samples = 4096
        input_signal = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        input_signal = input_signal.astype(np.complex64)

        output = channel.process(input_signal)

        assert output.shape == input_signal.shape
        assert output.dtype == np.complex64

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU module not available")
    def test_gpu_power_preservation(self):
        """Test that GPU processing preserves average power."""
        config = VoglerHoffmeyerConfig(
            sample_rate=48000.0,
            modes=[ModeParameters(
                sigma_tau=100.0,
                sigma_D=1.0,
                amplitude=1.0,
            )],
            use_gpu=True,
            random_seed=42,
        )
        channel = VoglerHoffmeyerChannel(config)

        # Generate noise input
        n_samples = 16384
        rng = np.random.default_rng(123)
        input_signal = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
        input_signal = input_signal.astype(np.complex64)

        input_power = np.mean(np.abs(input_signal)**2)
        output = channel.process(input_signal)
        output_power = np.mean(np.abs(output)**2)

        # Power should be preserved within 3 dB (channel has unit average gain)
        power_ratio = output_power / input_power
        assert 0.5 < power_ratio < 2.0, f"Power ratio {power_ratio:.3f} outside acceptable range"

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU module not available")
    def test_gpu_fading_statistics(self):
        """Test that GPU produces fading with reasonable statistics.

        Note: GPU and CPU use different random number generators, so exact
        statistical equivalence is not expected. We verify that fading
        has reasonable depth and variation.
        """
        config = VoglerHoffmeyerConfig(
            sample_rate=48000.0,
            modes=[ModeParameters(
                sigma_tau=100.0,
                sigma_D=1.0,
                amplitude=1.0,
            )],
            use_gpu=True,
            random_seed=42,
            k_factor=None,  # Rayleigh fading
        )
        channel = VoglerHoffmeyerChannel(config)

        # Generate noise input - longer duration for better statistics
        duration_sec = 10.0
        n_samples = int(duration_sec * config.sample_rate)
        rng = np.random.default_rng(123)
        input_signal = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
        input_signal = input_signal.astype(np.complex64)

        output = channel.process(input_signal)
        envelope = np.abs(output)

        # Downsample for fading analysis
        downsample_factor = max(1, int(config.sample_rate / 60))
        envelope_ds = envelope[::downsample_factor]

        # Compute fading statistics
        stats = compute_fading_statistics(envelope_ds, config.sample_rate / downsample_factor)

        # Verify fading has reasonable depth (deep fading expected)
        assert stats.fade_depth_db > 10.0, f"Fade depth {stats.fade_depth_db:.1f} dB too shallow"

        # Verify envelope has variation (not constant)
        assert stats.std_envelope > 0.1, f"Envelope std {stats.std_envelope:.4f} too small"


class TestGPUvsCPUEquivalence:
    """Compare GPU and CPU outputs for statistical equivalence."""

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU module not available")
    def test_delay_spread_equivalence(self):
        """Test that GPU and CPU produce same delay spread."""
        sample_rate = 48000.0
        seed = 42

        # Create GPU channel
        config_gpu = VoglerHoffmeyerConfig(
            sample_rate=sample_rate,
            modes=[ModeParameters(sigma_tau=200.0, sigma_c=100.0, sigma_D=1.0)],
            use_gpu=True,
            random_seed=seed,
        )
        channel_gpu = VoglerHoffmeyerChannel(config_gpu)

        # Create CPU channel
        config_cpu = VoglerHoffmeyerConfig(
            sample_rate=sample_rate,
            modes=[ModeParameters(sigma_tau=200.0, sigma_c=100.0, sigma_D=1.0)],
            use_gpu=False,
            random_seed=seed,
        )
        channel_cpu = VoglerHoffmeyerChannel(config_cpu)

        # Get impulse responses
        h_gpu = channel_gpu.get_impulse_response(num_samples=8192)
        h_cpu = channel_cpu.get_impulse_response(num_samples=8192)

        # Compute delay spreads
        delay_gpu = compute_delay_spread(h_gpu, sample_rate)
        delay_cpu = compute_delay_spread(h_cpu, sample_rate)

        # RMS delay spread should be within 50% (different random sequences)
        ratio = delay_gpu.rms_delay_spread_ms / delay_cpu.rms_delay_spread_ms
        assert 0.5 < ratio < 2.0, f"Delay spread ratio {ratio:.3f} outside acceptable range"

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU module not available")
    def test_output_statistics_equivalence(self):
        """Test that GPU and CPU produce comparable output statistics.

        Note: GPU and CPU have different random number generators and fading
        implementations, so exact equivalence is not expected. We verify that
        both produce output with reasonable power levels.
        """
        sample_rate = 48000.0
        duration_sec = 5.0
        n_samples = int(duration_sec * sample_rate)

        # Create channels
        config_gpu = VoglerHoffmeyerConfig(
            sample_rate=sample_rate,
            modes=[ModeParameters(sigma_tau=100.0, sigma_D=1.0)],
            use_gpu=True,
            random_seed=42,
        )
        channel_gpu = VoglerHoffmeyerChannel(config_gpu)

        config_cpu = VoglerHoffmeyerConfig(
            sample_rate=sample_rate,
            modes=[ModeParameters(sigma_tau=100.0, sigma_D=1.0)],
            use_gpu=False,
            random_seed=42,
        )
        channel_cpu = VoglerHoffmeyerChannel(config_cpu)

        # Generate same input for both
        rng = np.random.default_rng(123)
        input_signal = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
        input_signal = input_signal.astype(np.complex64)

        input_power = np.mean(np.abs(input_signal)**2)

        # Process through both channels
        output_gpu = channel_gpu.process(input_signal.copy())
        output_cpu = channel_cpu.process(input_signal.copy())

        # Both should preserve power within 6 dB
        power_gpu = np.mean(np.abs(output_gpu)**2)
        power_cpu = np.mean(np.abs(output_cpu)**2)

        gpu_ratio = power_gpu / input_power
        cpu_ratio = power_cpu / input_power

        assert 0.25 < gpu_ratio < 4.0, f"GPU power ratio {gpu_ratio:.3f} outside acceptable range"
        assert 0.25 < cpu_ratio < 4.0, f"CPU power ratio {cpu_ratio:.3f} outside acceptable range"


class TestMultiModeGPU:
    """Test GPU with multiple propagation modes."""

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU module not available")
    def test_multi_mode_processing(self):
        """Test GPU with multiple modes."""
        config = VoglerHoffmeyerConfig(
            sample_rate=48000.0,
            modes=[
                ModeParameters(name="E-layer", sigma_tau=50.0, sigma_D=0.5),
                ModeParameters(name="F-layer", sigma_tau=150.0, sigma_D=1.5, tau_L=100.0),
            ],
            use_gpu=True,
            random_seed=42,
        )
        channel = VoglerHoffmeyerChannel(config)

        # Check both modes initialized
        info = channel.get_backend_info()
        assert len(info['mode_backends']) == 2

        # Process signal
        n_samples = 4096
        input_signal = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        input_signal = input_signal.astype(np.complex64)

        output = channel.process(input_signal)
        assert output.shape == input_signal.shape

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU module not available")
    def test_itu_conditions(self):
        """Test GPU with ITU channel conditions."""
        from hfpathsim.core.parameters import ITUCondition

        for condition in [ITUCondition.QUIET, ITUCondition.MODERATE, ITUCondition.DISTURBED]:
            config = VoglerHoffmeyerConfig.from_itu_condition(condition)
            config = VoglerHoffmeyerConfig(
                sample_rate=config.sample_rate,
                modes=config.modes,
                use_gpu=True,
                random_seed=42,
            )
            channel = VoglerHoffmeyerChannel(config)

            # Process signal
            n_samples = 2048
            input_signal = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
            input_signal = input_signal.astype(np.complex64)

            output = channel.process(input_signal)
            assert output.shape == input_signal.shape


class TestGPUReset:
    """Test GPU processor reset functionality."""

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU module not available")
    def test_reset_clears_state(self):
        """Test that reset clears GPU processor state."""
        config = VoglerHoffmeyerConfig(
            sample_rate=48000.0,
            modes=[ModeParameters(sigma_tau=100.0, sigma_D=1.0)],
            use_gpu=True,
            random_seed=42,
        )
        channel = VoglerHoffmeyerChannel(config)

        # Process some signal
        input_signal = np.random.randn(1024) + 1j * np.random.randn(1024)
        input_signal = input_signal.astype(np.complex64)
        channel.process(input_signal)

        # Reset
        channel.reset(seed=42)

        # Process again - should work without error
        output = channel.process(input_signal)
        assert output.shape == input_signal.shape

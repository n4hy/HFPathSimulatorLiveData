"""Tests for GPU acceleration module."""

import numpy as np
import pytest

from hfpathsim.gpu import (
    get_device_info,
    is_available,
    vogler_transfer_function,
    apply_channel,
    compute_scattering_function,
)


class TestGPUModule:
    """Test GPU module functionality."""

    def test_get_device_info(self):
        """Test device info retrieval."""
        info = get_device_info()

        assert "name" in info
        assert "compute_capability" in info
        assert "total_memory_gb" in info
        assert "backend" in info

        # Backend should be one of known types
        backend = info["backend"]
        assert (
            backend in ["cuda", "cupy", "numpy"]
            or backend.startswith("numpy")  # "numpy (cupy kernels unavailable)"
        )

    def test_is_available(self):
        """Test GPU availability check."""
        available = is_available()
        assert isinstance(available, bool)

    def test_vogler_transfer_function_shape(self):
        """Test transfer function output shape."""
        freq = np.linspace(-1e6, 1e6, 4096).astype(np.float64)

        R = vogler_transfer_function(
            freq_hz=freq,
            foF2_mhz=7.5,
            hmF2_km=300.0,
            sigma=0.1,
            chi=0.3,
            t0_sec=0.004,
        )

        assert R.shape == (4096,)
        assert R.dtype == np.complex64

    def test_vogler_transfer_function_values(self):
        """Test transfer function produces valid values."""
        freq = np.linspace(-500e3, 500e3, 1024).astype(np.float64)

        R = vogler_transfer_function(
            freq_hz=freq,
            foF2_mhz=10.0,
            hmF2_km=300.0,
            sigma=0.1,
            chi=0.2,
            t0_sec=0.003,
        )

        # Should have finite values
        assert np.all(np.isfinite(R))

        # Should have non-zero magnitude
        assert np.max(np.abs(R)) > 0

    def test_apply_channel_shape(self):
        """Test channel application output shape."""
        input_signal = (
            np.random.randn(8192) + 1j * np.random.randn(8192)
        ).astype(np.complex64)

        H = np.ones(4096, dtype=np.complex64)

        output = apply_channel(input_signal, H, block_size=4096, overlap=1024)

        assert output.shape == input_signal.shape
        assert output.dtype == np.complex64

    def test_apply_channel_unity(self):
        """Test channel application with unity transfer function."""
        # Simple tone
        t = np.arange(4096) / 1e6
        freq = 10000  # 10 kHz
        input_signal = np.exp(2j * np.pi * freq * t).astype(np.complex64)

        # Unity transfer function
        H = np.ones(4096, dtype=np.complex64)

        output = apply_channel(input_signal, H, block_size=4096, overlap=1024)

        # Output should be close to input (allowing for edge effects)
        # Check middle portion
        mid = len(output) // 2
        window = slice(mid - 1000, mid + 1000)

        # Should maintain signal structure
        assert np.std(np.abs(output[window])) < 0.5

    def test_apply_channel_filtering(self):
        """Test channel application applies frequency filtering."""
        # Generate wideband noise
        np.random.seed(42)  # Reproducibility
        input_signal = (
            np.random.randn(8192) + 1j * np.random.randn(8192)
        ).astype(np.complex64)

        # Create bandpass transfer function (centered)
        H = np.zeros(4096, dtype=np.complex64)
        # FFT-shifted: center is at index 2048
        H[2048 - 256 : 2048 + 256] = 1.0  # Pass center 1/8 of bandwidth

        output = apply_channel(input_signal, H, block_size=4096, overlap=1024)

        # Check output has been modified (not all zeros, but reduced power)
        input_power = np.mean(np.abs(input_signal[:4096]) ** 2)
        output_power = np.mean(np.abs(output[:4096]) ** 2)

        # Output should have less power than input (we filtered out most)
        # With 512/4096 passband, expect ~1/8 power ratio
        assert output_power < input_power
        assert output_power > 0  # But not zero

    def test_scattering_function_shape(self):
        """Test scattering function output shape."""
        delay_axis = np.linspace(0, 10, 64)
        doppler_axis = np.linspace(-5, 5, 32)

        S = compute_scattering_function(
            delay_axis_ms=delay_axis,
            doppler_axis_hz=doppler_axis,
            delay_spread_ms=2.0,
            doppler_spread_hz=1.0,
        )

        assert S.shape == (32, 64)
        assert S.dtype == np.float32

    def test_scattering_function_normalization(self):
        """Test scattering function is normalized."""
        delay_axis = np.linspace(0, 10, 64)
        doppler_axis = np.linspace(-5, 5, 32)

        S = compute_scattering_function(
            delay_axis_ms=delay_axis,
            doppler_axis_hz=doppler_axis,
            delay_spread_ms=2.0,
            doppler_spread_hz=1.0,
        )

        # Maximum should be 1.0
        assert np.max(S) == pytest.approx(1.0)

        # All values should be non-negative
        assert np.min(S) >= 0

    def test_scattering_function_peak_location(self):
        """Test scattering function peak is at tau=0, nu=0."""
        delay_axis = np.linspace(0, 10, 65)  # 0 is at index 0
        doppler_axis = np.linspace(-5, 5, 33)  # 0 is at index 16

        S = compute_scattering_function(
            delay_axis_ms=delay_axis,
            doppler_axis_hz=doppler_axis,
            delay_spread_ms=2.0,
            doppler_spread_hz=1.0,
        )

        # Peak should be near (tau=0, nu=0)
        peak_idx = np.unravel_index(np.argmax(S), S.shape)

        # Doppler peak at center (index 16)
        assert abs(peak_idx[0] - 16) <= 1

        # Delay peak at start (index 0)
        assert peak_idx[1] == 0


class TestGPUPerformance:
    """Performance tests for GPU module."""

    @pytest.mark.slow
    def test_large_transfer_function(self):
        """Test transfer function with large array."""
        N = 1024 * 1024  # 1M points
        freq = np.linspace(-10e6, 10e6, N).astype(np.float64)

        R = vogler_transfer_function(
            freq_hz=freq,
            foF2_mhz=10.0,
            hmF2_km=300.0,
            sigma=0.1,
            chi=0.2,
            t0_sec=0.003,
        )

        assert R.shape == (N,)

    @pytest.mark.slow
    def test_streaming_throughput(self):
        """Test sustained processing throughput."""
        # Simulate 1 second of 2 Msps data
        sample_rate = 2e6
        duration = 0.1  # 100 ms for test
        total_samples = int(sample_rate * duration)
        block_size = 4096

        input_data = (
            np.random.randn(total_samples) + 1j * np.random.randn(total_samples)
        ).astype(np.complex64)

        H = np.ones(block_size, dtype=np.complex64)

        # Process in blocks
        output = np.zeros_like(input_data)
        n_blocks = total_samples // block_size

        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            block_output = apply_channel(
                input_data[start:end], H,
                block_size=block_size, overlap=1024
            )
            output[start:end] = block_output

        # Should complete without error
        assert len(output) == len(input_data)


class TestGPUFallback:
    """Test fallback behavior when GPU unavailable."""

    def test_numpy_fallback(self):
        """Test that functions work with NumPy fallback."""
        # These should work regardless of GPU availability
        freq = np.linspace(-100e3, 100e3, 256).astype(np.float64)

        R = vogler_transfer_function(
            freq_hz=freq,
            foF2_mhz=7.5,
            hmF2_km=300.0,
            sigma=0.1,
            chi=0.3,
            t0_sec=0.004,
        )

        assert R is not None
        assert len(R) == 256

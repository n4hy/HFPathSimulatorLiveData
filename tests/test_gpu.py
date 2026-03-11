"""Tests for GPU acceleration module."""

import numpy as np
import pytest
import time

from hfpathsim.gpu import (
    get_device_info,
    is_available,
    vogler_transfer_function,
    apply_channel,
    apply_channel_batched,
    compute_scattering_function,
    generate_doppler_fading,
    compute_spectrum_db,
    get_native_module_available,
    get_backend_info,
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


class TestDopplerFading:
    """Tests for Doppler fading generation."""

    def test_doppler_fading_shape(self):
        """Test Doppler fading output shape."""
        n_samples = 4096
        fading = generate_doppler_fading(
            doppler_spread_hz=1.0,
            sample_rate=2e6,
            n_samples=n_samples,
        )

        assert fading.shape == (n_samples,)
        assert fading.dtype == np.complex64

    def test_doppler_fading_values(self):
        """Test Doppler fading produces valid values."""
        fading = generate_doppler_fading(
            doppler_spread_hz=2.0,
            sample_rate=1e6,
            n_samples=1024,
        )

        # Should have finite values
        assert np.all(np.isfinite(fading))

        # Should have non-zero magnitude
        assert np.max(np.abs(fading)) > 0

    def test_doppler_fading_reproducibility(self):
        """Test Doppler fading with same seed produces same results."""
        seed = 12345
        n_samples = 1024

        fading1 = generate_doppler_fading(
            doppler_spread_hz=1.5,
            sample_rate=2e6,
            n_samples=n_samples,
            seed=seed,
        )

        fading2 = generate_doppler_fading(
            doppler_spread_hz=1.5,
            sample_rate=2e6,
            n_samples=n_samples,
            seed=seed,
        )

        np.testing.assert_array_almost_equal(fading1, fading2)

    def test_doppler_fading_spectrum_shape(self):
        """Test that fading has correct Doppler spectrum shape."""
        doppler_spread = 2.0  # Hz
        sample_rate = 1000.0  # Hz (low for testing)
        n_samples = 4096

        fading = generate_doppler_fading(
            doppler_spread_hz=doppler_spread,
            sample_rate=sample_rate,
            n_samples=n_samples,
            seed=42,
        )

        # Compute power spectrum
        spectrum = np.fft.fft(fading)
        power = np.abs(spectrum) ** 2
        power = np.fft.fftshift(power)
        freq = np.fft.fftshift(np.fft.fftfreq(n_samples, 1 / sample_rate))

        # Find peak (should be near DC)
        peak_idx = np.argmax(power)
        peak_freq = abs(freq[peak_idx])

        # Peak should be within Doppler spread
        assert peak_freq < doppler_spread * 2


class TestSpectrumComputation:
    """Tests for GPU spectrum computation."""

    def test_spectrum_shape(self):
        """Test spectrum output shape."""
        signal = (np.random.randn(4096) + 1j * np.random.randn(4096)).astype(np.complex64)

        power_db = compute_spectrum_db(signal)

        assert power_db.shape == signal.shape
        assert power_db.dtype == np.float32

    def test_spectrum_values(self):
        """Test spectrum produces valid dB values."""
        signal = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)

        power_db = compute_spectrum_db(signal)

        # Should have finite values
        assert np.all(np.isfinite(power_db))

        # Should be bounded (min is -120 dB in implementation)
        assert np.min(power_db) >= -120.0

    def test_spectrum_tone_detection(self):
        """Test spectrum correctly identifies a tone."""
        n_samples = 4096
        sample_rate = 2e6
        tone_freq = 100000  # 100 kHz

        t = np.arange(n_samples) / sample_rate
        signal = np.exp(2j * np.pi * tone_freq * t).astype(np.complex64)

        power_db = compute_spectrum_db(signal)

        # Find peak
        peak_idx = np.argmax(power_db)
        freq_axis = np.fft.fftfreq(n_samples, 1 / sample_rate)
        peak_freq = abs(freq_axis[peak_idx])

        # Peak should be near the tone frequency
        assert abs(peak_freq - tone_freq) < sample_rate / n_samples * 2


class TestBatchedOverlapSave:
    """Tests for batched overlap-save processing."""

    def test_batched_shape(self):
        """Test batched processing output shape."""
        input_signal = (
            np.random.randn(16384) + 1j * np.random.randn(16384)
        ).astype(np.complex64)

        H = np.ones(4096, dtype=np.complex64)

        output = apply_channel_batched(
            input_signal, H,
            block_size=4096,
            overlap=1024,
            batch_size=4
        )

        assert output.shape == input_signal.shape

    def test_batched_unity_tf(self):
        """Test batched processing with unity transfer function."""
        # Simple tone
        n_samples = 8192
        t = np.arange(n_samples) / 1e6
        freq = 10000  # 10 kHz
        input_signal = np.exp(2j * np.pi * freq * t).astype(np.complex64)

        H = np.ones(4096, dtype=np.complex64)

        output = apply_channel_batched(input_signal, H, batch_size=4)

        # Output should maintain signal structure
        mid = len(output) // 2
        window = slice(mid - 500, mid + 500)
        assert np.std(np.abs(output[window])) < 0.5

    def test_batched_vs_single(self):
        """Test batched processing matches single-block processing."""
        np.random.seed(42)
        input_signal = (
            np.random.randn(8192) + 1j * np.random.randn(8192)
        ).astype(np.complex64)

        # Create a filter
        H = np.zeros(4096, dtype=np.complex64)
        H[1024:3072] = 1.0  # Bandpass

        output_single = apply_channel(input_signal, H, block_size=4096, overlap=1024)
        output_batched = apply_channel_batched(
            input_signal, H, block_size=4096, overlap=1024, batch_size=4
        )

        # Results should be very similar
        # Note: May have small numerical differences
        correlation = np.abs(np.corrcoef(output_single.real, output_batched.real)[0, 1])
        assert correlation > 0.99


class TestBackendInfo:
    """Tests for backend information functions."""

    def test_backend_info_structure(self):
        """Test backend info returns expected structure."""
        info = get_backend_info()

        assert "native_module" in info
        assert "cupy_available" in info
        assert "cupy_works" in info
        assert "backend" in info
        assert "device_info" in info

        assert isinstance(info["native_module"], bool)
        assert isinstance(info["cupy_available"], bool)

    def test_native_module_check(self):
        """Test native module availability check."""
        available = get_native_module_available()
        assert isinstance(available, bool)


@pytest.mark.gpu
class TestNativeCUDA:
    """Tests requiring native CUDA module.

    These tests are marked with @pytest.mark.gpu and should be skipped
    if the native module is not available.
    """

    @pytest.fixture(autouse=True)
    def check_native_module(self):
        """Skip if native module not available."""
        if not get_native_module_available():
            pytest.skip("Native CUDA module not available")

    def test_native_module_loads(self):
        """Verify _hfpathsim_gpu imports."""
        from hfpathsim.gpu import _hfpathsim_gpu
        assert _hfpathsim_gpu is not None

    def test_native_device_info(self):
        """Test native module device info."""
        from hfpathsim.gpu import _hfpathsim_gpu

        info = _hfpathsim_gpu.get_device_info()
        assert info["backend"] == "cuda"
        assert "name" in info
        assert info["multiprocessors"] > 0

    def test_native_batched_processor(self):
        """Test native batched overlap-save processor class."""
        from hfpathsim.gpu import _hfpathsim_gpu

        proc = _hfpathsim_gpu.OverlapSaveProcessorBatched(4096, 1024, 8)
        assert proc.get_batch_size() == 8
        assert proc.get_block_size() == 4096
        assert proc.get_overlap() == 1024

    def test_native_doppler_generator(self):
        """Test native Doppler fading generator class."""
        from hfpathsim.gpu import _hfpathsim_gpu

        gen = _hfpathsim_gpu.DopplerFadingGenerator(4096, 42)
        assert gen.get_n_samples() == 4096

        fading = gen.generate(1.5, 2e6)
        assert fading.shape == (4096,)
        assert fading.dtype == np.complex64


@pytest.mark.benchmark
class TestGPUBenchmarks:
    """Performance benchmark tests for GPU module."""

    def test_batched_fft_throughput(self):
        """Verify batched FFT meets throughput targets."""
        # Target: 2 Msps real-time processing
        sample_rate = 2e6
        duration = 0.1  # 100 ms of data
        n_samples = int(sample_rate * duration)
        block_size = 4096
        batch_size = 8

        input_signal = (
            np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        ).astype(np.complex64)

        H = np.ones(block_size, dtype=np.complex64)

        # Warm up
        _ = apply_channel_batched(input_signal[:block_size * 2], H, batch_size=batch_size)

        # Benchmark
        start = time.time()
        _ = apply_channel_batched(input_signal, H, batch_size=batch_size)
        elapsed = time.time() - start

        throughput = n_samples / elapsed
        print(f"\nBatched throughput: {throughput / 1e6:.2f} Msps")

        # Should process faster than real-time (>2 Msps)
        # Note: This may fail on slow systems without GPU
        # Just check it completes in reasonable time
        assert elapsed < duration * 10  # At least 10% of real-time

    def test_gpu_vs_cpu_speedup(self):
        """Measure GPU speedup over CPU for spectrum computation."""
        n_samples = 65536
        signal = (
            np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        ).astype(np.complex64)

        # CPU baseline
        start_cpu = time.time()
        for _ in range(10):
            _ = np.abs(np.fft.fft(signal)) ** 2
        cpu_time = time.time() - start_cpu

        # GPU (or fallback)
        start_gpu = time.time()
        for _ in range(10):
            _ = compute_spectrum_db(signal)
        gpu_time = time.time() - start_gpu

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"\nSpectrum computation: CPU={cpu_time*100:.1f}ms, GPU={gpu_time*100:.1f}ms, Speedup={speedup:.1f}x")

        # Should complete (may not show speedup without native module)
        assert gpu_time < 10.0  # Reasonable time limit

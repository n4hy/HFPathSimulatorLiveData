"""Tests for SpectrumWidget."""

import os
import sys
import numpy as np
import pytest

# Check if display is available for GUI tests
HAS_DISPLAY = os.environ.get("DISPLAY") is not None or sys.platform == "darwin"


class TestSpectrumComputationCPU:
    """Test spectrum computation logic (no Qt required)."""

    def test_fft_spectrum_tone(self):
        """Test FFT spectrum of a pure tone."""
        # Generate tone at Fs/4
        fs = 1e6
        fft_size = 1024
        f_tone = fs / 4
        t = np.arange(fft_size) / fs
        signal = np.exp(2j * np.pi * f_tone * t).astype(np.complex64)

        # Apply window
        window = np.blackman(fft_size)
        windowed = signal * window

        # Compute spectrum
        spectrum = np.fft.fftshift(np.fft.fft(windowed))
        power_db = 20 * np.log10(np.abs(spectrum) + 1e-10)

        # Peak should be at Fs/4 = bin 768 after fftshift
        peak_bin = np.argmax(power_db)
        expected_bin = fft_size // 2 + fft_size // 4  # 512 + 256 = 768

        assert abs(peak_bin - expected_bin) <= 1

    def test_fft_spectrum_dc(self):
        """Test FFT spectrum of DC signal."""
        fft_size = 1024
        signal = np.ones(fft_size, dtype=np.complex64)

        window = np.blackman(fft_size)
        windowed = signal * window

        spectrum = np.fft.fftshift(np.fft.fft(windowed))
        power_db = 20 * np.log10(np.abs(spectrum) + 1e-10)

        # Peak should be at DC (center after fftshift)
        peak_bin = np.argmax(power_db)
        assert peak_bin == fft_size // 2

    def test_window_normalization(self):
        """Test window power normalization."""
        fft_size = 4096
        window = np.blackman(fft_size)
        window_power = np.sum(window**2)

        # For Blackman window, power should be around 0.305 * N
        expected_power = 0.305 * fft_size
        assert window_power == pytest.approx(expected_power, rel=0.1)

    def test_spectrum_symmetry_real_signal(self):
        """Test that spectrum of real signal is conjugate symmetric."""
        fs = 1e6
        fft_size = 1024
        t = np.arange(fft_size) / fs

        # Real signal (cosine)
        signal = np.cos(2 * np.pi * 1000 * t).astype(np.complex64)

        spectrum = np.fft.fft(signal)

        # Magnitude should be symmetric
        mag = np.abs(spectrum)
        for k in range(1, fft_size // 2):
            assert mag[k] == pytest.approx(mag[fft_size - k], rel=1e-5)

    def test_frequency_axis_calculation(self):
        """Test frequency axis matches expected values."""
        fft_size = 1024
        sample_rate = 1e6

        freq_axis = np.fft.fftshift(
            np.fft.fftfreq(fft_size, 1 / sample_rate)
        )

        # Should span -fs/2 to +fs/2
        assert freq_axis[0] == pytest.approx(-sample_rate / 2, rel=1e-6)
        assert freq_axis[-1] == pytest.approx(
            sample_rate / 2 - sample_rate / fft_size, rel=1e-6
        )

    def test_zero_padding_effect(self):
        """Test that zero-padding increases frequency resolution."""
        signal = np.ones(512, dtype=np.complex64)

        # Original spectrum
        spectrum1 = np.fft.fft(signal)
        assert len(spectrum1) == 512

        # Zero-padded spectrum
        padded = np.zeros(1024, dtype=np.complex64)
        padded[:512] = signal
        spectrum2 = np.fft.fft(padded)
        assert len(spectrum2) == 1024

    def test_blackman_window_properties(self):
        """Test Blackman window has expected properties."""
        fft_size = 4096
        window = np.blackman(fft_size)

        # Symmetric
        assert np.allclose(window, window[::-1])

        # Near zero at edges
        assert window[0] < 1e-5
        assert window[-1] < 1e-5

        # Peak at center
        assert window[fft_size // 2] == pytest.approx(1.0, rel=0.01)

    def test_power_db_conversion(self):
        """Test power dB conversion is correct."""
        # Unit amplitude complex signal
        signal = np.ones(100, dtype=np.complex64)
        power_db = 20 * np.log10(np.abs(signal) + 1e-10)

        # Should be 0 dB
        assert np.allclose(power_db, 0.0, atol=1e-6)

        # Half amplitude
        signal_half = signal * 0.5
        power_db_half = 20 * np.log10(np.abs(signal_half) + 1e-10)

        # Should be -6 dB
        assert np.allclose(power_db_half, -6.02, atol=0.1)

    def test_averaging_computation(self):
        """Test averaging multiple spectra."""
        fft_size = 1024
        n_averages = 4

        # Generate multiple spectra with noise
        np.random.seed(42)
        spectra = []
        for _ in range(n_averages):
            noise = np.random.randn(fft_size) + 1j * np.random.randn(fft_size)
            noise = noise.astype(np.complex64)
            power_db = 20 * np.log10(np.abs(noise) + 1e-10)
            spectra.append(power_db)

        # Average
        avg_power = np.mean(spectra, axis=0)

        assert avg_power.shape == (fft_size,)
        # Averaged power should have lower variance
        single_variance = np.var(spectra[0])
        avg_variance = np.var(avg_power)
        assert avg_variance < single_variance

    def test_peak_hold_tracking(self):
        """Test peak hold algorithm."""
        fft_size = 100

        # Initialize peak data
        peak_data = np.full(fft_size, -100.0)

        # Update with increasing values
        for i in range(5):
            new_data = np.full(fft_size, -50.0 + i * 10)
            peak_data = np.maximum(peak_data, new_data)

        # Peak should track maximum
        assert np.allclose(peak_data, -10.0)

    def test_peak_hold_decay(self):
        """Test peak hold decay behavior."""
        fft_size = 100
        decay_rate = 0.1

        peak_data = np.full(fft_size, 0.0)

        # Apply decay
        for _ in range(100):
            peak_data -= decay_rate

        # Should have decayed by 10 dB
        assert np.allclose(peak_data, -10.0)


@pytest.mark.skipif(not HAS_DISPLAY, reason="No display available")
class TestSpectrumWidgetGUI:
    """Test SpectrumWidget GUI components (requires display)."""

    @pytest.fixture
    def qtbot_fixture(self, qtbot):
        """Provide qtbot fixture."""
        return qtbot

    def test_widget_creation(self, qtbot_fixture):
        """Test widget can be created."""
        from hfpathsim.gui.widgets.spectrum import SpectrumWidget

        widget = SpectrumWidget()
        qtbot_fixture.addWidget(widget)

        assert widget._title == "Spectrum"
        assert widget._fft_size == 4096

    def test_widget_custom_title(self, qtbot_fixture):
        """Test widget with custom title."""
        from hfpathsim.gui.widgets.spectrum import SpectrumWidget

        widget = SpectrumWidget(title="Input Spectrum")
        qtbot_fixture.addWidget(widget)

        assert widget._title == "Input Spectrum"

    def test_sample_rate_setter(self, qtbot_fixture):
        """Test sample rate setter."""
        from hfpathsim.gui.widgets.spectrum import SpectrumWidget

        widget = SpectrumWidget()
        qtbot_fixture.addWidget(widget)

        widget.set_sample_rate(1e6)
        assert widget._sample_rate == 1e6

    def test_update_data(self, qtbot_fixture):
        """Test data update."""
        from hfpathsim.gui.widgets.spectrum import SpectrumWidget

        widget = SpectrumWidget(use_gpu=False)
        qtbot_fixture.addWidget(widget)

        samples = np.ones(4096, dtype=np.complex64)
        widget.update_data(samples)

        assert len(widget._avg_buffer) == 1
        assert widget._peak_data is not None

    def test_reset_view(self, qtbot_fixture):
        """Test reset_view clears state."""
        from hfpathsim.gui.widgets.spectrum import SpectrumWidget

        widget = SpectrumWidget(use_gpu=False)
        qtbot_fixture.addWidget(widget)

        # Add some data
        samples = np.ones(4096, dtype=np.complex64)
        widget.update_data(samples)

        # Reset
        widget.reset_view()

        assert widget._avg_buffer == []
        assert widget._peak_data is None

    def test_clear_peak_hold(self, qtbot_fixture):
        """Test clear_peak_hold."""
        from hfpathsim.gui.widgets.spectrum import SpectrumWidget

        widget = SpectrumWidget(use_gpu=False)
        qtbot_fixture.addWidget(widget)

        # Add some data
        samples = np.ones(4096, dtype=np.complex64)
        widget.update_data(samples)
        assert widget._peak_data is not None

        # Clear peak
        widget.clear_peak_hold()
        assert widget._peak_data is None

    def test_averaging_change(self, qtbot_fixture):
        """Test averaging parameter change."""
        from hfpathsim.gui.widgets.spectrum import SpectrumWidget

        widget = SpectrumWidget(use_gpu=False)
        qtbot_fixture.addWidget(widget)

        # Add some buffered data
        samples = np.ones(4096, dtype=np.complex64)
        widget.update_data(samples)
        widget.update_data(samples)

        assert len(widget._avg_buffer) == 2

        # Change averaging (clears buffer)
        widget._on_avg_changed("8")

        assert widget._averaging == 8
        assert widget._avg_buffer == []

    def test_fft_size_change(self, qtbot_fixture):
        """Test FFT size change."""
        from hfpathsim.gui.widgets.spectrum import SpectrumWidget

        widget = SpectrumWidget(use_gpu=False)
        qtbot_fixture.addWidget(widget)

        assert widget._fft_size == 4096
        assert len(widget._window) == 4096

        widget._on_fft_changed("2048")

        assert widget._fft_size == 2048
        assert len(widget._window) == 2048

    def test_short_samples_padding(self, qtbot_fixture):
        """Test handling of samples shorter than FFT size."""
        from hfpathsim.gui.widgets.spectrum import SpectrumWidget

        widget = SpectrumWidget(use_gpu=False)
        qtbot_fixture.addWidget(widget)

        # Samples shorter than FFT size
        samples = np.ones(1024, dtype=np.complex64)
        widget.update_data(samples)

        # Should still work (zero-padded internally)
        assert len(widget._avg_buffer) == 1

    def test_gpu_flag_when_unavailable(self, qtbot_fixture):
        """Test GPU flag behavior when GPU unavailable."""
        from hfpathsim.gui.widgets.spectrum import SpectrumWidget

        widget = SpectrumWidget(use_gpu=True)
        qtbot_fixture.addWidget(widget)

        # GPU may or may not be available depending on system
        # Just verify the flag is set appropriately
        assert isinstance(widget.is_using_gpu(), bool)

"""Tests for Phase 3 channel models and impairments."""

import numpy as np
import pytest
import tempfile
import os

from hfpathsim.core.watterson import (
    WattersonChannel,
    WattersonConfig,
    WattersonTap,
)
from hfpathsim.core.noise import (
    NoiseGenerator,
    NoiseConfig,
    NoiseType,
    ManMadeEnvironment,
    estimate_noise_floor,
)
from hfpathsim.core.impairments import (
    AGC,
    AGCConfig,
    AGCMode,
    Limiter,
    LimiterConfig,
    FrequencyOffset,
    FrequencyOffsetConfig,
    ImpairmentChain,
)
from hfpathsim.core.recording import (
    ChannelRecorder,
    ChannelPlayer,
    ChannelSnapshot,
    RecordingMetadata,
)


class TestWattersonChannel:
    """Tests for Watterson tapped delay line model."""

    def test_default_config(self):
        """Test channel creation with default config."""
        channel = WattersonChannel()
        assert channel.config is not None
        assert channel.config.sample_rate_hz == 2_000_000
        assert len(channel.config.taps) == 2

    def test_custom_config(self):
        """Test channel with custom configuration."""
        config = WattersonConfig(
            sample_rate_hz=16000,
            taps=[
                WattersonTap(delay_ms=0.0, amplitude=1.0, doppler_spread_hz=0.5),
                WattersonTap(delay_ms=1.0, amplitude=0.5, doppler_spread_hz=1.0),
            ],
        )
        channel = WattersonChannel(config=config)
        assert channel.config.sample_rate_hz == 16000
        assert len(channel.config.taps) == 2

    def test_process_block_shape(self):
        """Test that output shape matches input shape."""
        channel = WattersonChannel(seed=42)
        input_signal = np.random.randn(1024) + 1j * np.random.randn(1024)
        input_signal = input_signal.astype(np.complex64)

        output = channel.process_block(input_signal)
        assert output.shape == input_signal.shape
        assert output.dtype == np.complex64

    def test_process_block_deterministic(self):
        """Test that same seed produces same output."""
        input_signal = np.random.randn(512) + 1j * np.random.randn(512)
        input_signal = input_signal.astype(np.complex64)

        channel1 = WattersonChannel(seed=123)
        channel2 = WattersonChannel(seed=123)

        out1 = channel1.process_block(input_signal.copy())
        out2 = channel2.process_block(input_signal.copy())

        np.testing.assert_array_almost_equal(out1, out2)

    def test_impulse_response(self):
        """Test impulse response generation."""
        channel = WattersonChannel(seed=42)
        ir = channel.get_impulse_response(length=256)

        assert len(ir) == 256
        assert ir.dtype == np.complex64
        # Should have some energy
        assert np.sum(np.abs(ir) ** 2) > 0

    def test_channel_causes_fading(self):
        """Test that channel introduces fading."""
        channel = WattersonChannel(seed=42)

        # Continuous tone input
        t = np.arange(4096) / 8000
        input_signal = np.exp(1j * 2 * np.pi * 1000 * t).astype(np.complex64)

        output = channel.process_block(input_signal)

        # Output power should vary (fading)
        input_power = np.abs(input_signal) ** 2
        output_power = np.abs(output) ** 2

        # Fading causes power variation
        assert np.std(output_power) > np.std(input_power) * 0.01

    def test_reset(self):
        """Test channel reset."""
        channel = WattersonChannel(seed=42)
        input_signal = np.random.randn(512).astype(np.complex64)

        channel.process_block(input_signal)
        channel.reset(seed=42)

        # After reset with same seed, should produce consistent output
        channel2 = WattersonChannel(seed=42)
        out1 = channel.process_block(input_signal.copy())
        out2 = channel2.process_block(input_signal.copy())

        np.testing.assert_array_almost_equal(out1, out2)


class TestNoiseGenerator:
    """Tests for noise generation."""

    def test_default_config(self):
        """Test noise generator with default config."""
        gen = NoiseGenerator()
        assert gen.config.snr_db == 20.0
        assert gen.sample_rate == 2_000_000

    def test_awgn_generation(self):
        """Test AWGN generation."""
        gen = NoiseGenerator(seed=42)
        noise = gen.generate_awgn(10000)

        assert len(noise) == 10000
        assert noise.dtype == np.complex64

        # Should be zero-mean
        assert abs(np.mean(noise)) < 0.1

    def test_awgn_power(self):
        """Test that AWGN has correct power for SNR."""
        config = NoiseConfig(snr_db=10.0)
        gen = NoiseGenerator(config=config, seed=42)

        # Signal power = 1, SNR = 10 dB => noise power = 0.1
        noise = gen.generate_awgn(100000)
        noise_power = np.mean(np.abs(noise) ** 2)
        expected_power = 0.1

        assert abs(noise_power - expected_power) < 0.02

    def test_atmospheric_noise(self):
        """Test atmospheric noise generation."""
        config = NoiseConfig(
            enable_atmospheric=True,
            frequency_mhz=5.0,
            season="summer",
            time_of_day="night",
        )
        gen = NoiseGenerator(config=config, seed=42)
        noise = gen.generate_atmospheric(10000)

        assert len(noise) == 10000
        # Should have some energy
        assert np.sum(np.abs(noise) ** 2) > 0

    def test_manmade_noise(self):
        """Test man-made noise generation."""
        config = NoiseConfig(
            enable_manmade=True,
            environment=ManMadeEnvironment.CITY,
            frequency_mhz=10.0,
        )
        gen = NoiseGenerator(config=config, seed=42)
        noise = gen.generate_manmade(10000)

        assert len(noise) == 10000

    def test_impulse_noise(self):
        """Test impulse noise generation."""
        config = NoiseConfig(
            enable_impulse=True,
            impulse_rate_hz=100.0,
            impulse_amplitude_db=20.0,
            impulse_duration_us=50.0,
        )
        gen = NoiseGenerator(config=config, sample_rate_hz=100000, seed=42)
        noise = gen.generate_impulse(100000)

        assert len(noise) == 100000
        # Should have impulses (non-zero samples)
        assert np.sum(np.abs(noise) > 0) > 0

    def test_combined_noise(self):
        """Test combined noise generation."""
        config = NoiseConfig(
            snr_db=20.0,
            enable_atmospheric=True,
            enable_manmade=True,
            enable_impulse=True,
        )
        gen = NoiseGenerator(config=config, seed=42)
        noise = gen.generate(10000)

        assert len(noise) == 10000

    def test_add_noise_to_signal(self):
        """Test adding noise to a signal."""
        config = NoiseConfig(snr_db=10.0)
        gen = NoiseGenerator(config=config, seed=42)

        # Unit power signal
        signal = np.ones(10000, dtype=np.complex64)
        noisy = gen.add_noise(signal, normalize=True)

        assert len(noisy) == len(signal)
        # Should be different from original
        assert not np.allclose(noisy, signal)

    def test_set_snr(self):
        """Test SNR adjustment."""
        gen = NoiseGenerator(seed=42)
        gen.set_snr(30.0)
        assert gen.config.snr_db == 30.0

    def test_estimate_noise_floor(self):
        """Test noise floor estimation."""
        # Create signal with known SNR
        signal = np.random.randn(10000) + 1j * np.random.randn(10000)
        signal = signal.astype(np.complex64)

        estimate = estimate_noise_floor(signal, sample_rate_hz=10000)

        assert hasattr(estimate, 'noise_power_dbm')
        assert hasattr(estimate, 'snr_estimate_db')
        assert hasattr(estimate, 'noise_figure_db')


class TestAGC:
    """Tests for Automatic Gain Control."""

    def test_default_config(self):
        """Test AGC with default config."""
        agc = AGC()
        assert agc.config.mode == AGCMode.MEDIUM
        assert agc.config.target_level_db == -10.0

    def test_mode_presets(self):
        """Test AGC mode presets."""
        for mode in AGCMode:
            config = AGCConfig(mode=mode)
            agc = AGC(config=config)
            assert agc.config.mode == mode
            assert agc.config.attack_time_ms is not None
            assert agc.config.release_time_ms is not None

    def test_process_shape(self):
        """Test that AGC preserves signal shape."""
        agc = AGC()
        signal = np.random.randn(1024) + 1j * np.random.randn(1024)
        signal = signal.astype(np.complex64)

        output = agc.process(signal)
        assert output.shape == signal.shape

    def test_process_block(self):
        """Test block-based processing."""
        agc = AGC()
        signal = np.random.randn(1024) + 1j * np.random.randn(1024)
        signal = signal.astype(np.complex64)

        output = agc.process_block(signal)
        assert output.shape == signal.shape
        assert output.dtype == np.complex64

    def test_gain_control(self):
        """Test that AGC controls gain toward target."""
        config = AGCConfig(mode=AGCMode.FAST, target_level_db=-10.0)
        agc = AGC(config=config)

        # Very weak signal
        weak_signal = 0.001 * (np.random.randn(4096) + 1j * np.random.randn(4096))
        weak_signal = weak_signal.astype(np.complex64)

        output = agc.process(weak_signal)

        # Output should be stronger than input
        input_power = np.mean(np.abs(weak_signal) ** 2)
        output_power = np.mean(np.abs(output) ** 2)

        assert output_power > input_power

    def test_manual_mode(self):
        """Test manual gain mode."""
        config = AGCConfig(mode=AGCMode.MANUAL)
        agc = AGC(config=config)
        agc.set_gain(10.0)

        signal = np.ones(100, dtype=np.complex64)
        output = agc.process(signal)

        # Should apply fixed gain of 10 dB (~3.16x)
        expected_gain = 10 ** (10.0 / 20)
        np.testing.assert_array_almost_equal(
            np.abs(output), expected_gain, decimal=2
        )

    def test_reset(self):
        """Test AGC reset."""
        agc = AGC()
        signal = np.random.randn(1024).astype(np.complex64)
        agc.process(signal)

        agc.reset()
        assert agc._gain_db == 0.0
        assert agc._envelope == 0.0

    def test_current_gain(self):
        """Test current gain property."""
        agc = AGC()
        signal = np.random.randn(1024).astype(np.complex64)
        agc.process(signal)

        gain = agc.current_gain_db
        assert isinstance(gain, float)


class TestLimiter:
    """Tests for signal limiter."""

    def test_default_config(self):
        """Test limiter with default config."""
        limiter = Limiter()
        assert limiter.config.threshold_db == -3.0
        assert limiter.config.mode == "soft"

    def test_hard_limiting(self):
        """Test hard clipping mode."""
        config = LimiterConfig(threshold_db=-6.0, mode="hard")
        limiter = Limiter(config=config)

        # Signal with peaks above threshold
        signal = 2.0 * (np.random.randn(1024) + 1j * np.random.randn(1024))
        signal = signal.astype(np.complex64)

        output = limiter.process(signal)

        threshold = 10 ** (-6.0 / 20)
        assert np.all(np.abs(output) <= threshold + 1e-6)

    def test_soft_limiting(self):
        """Test soft limiting mode."""
        config = LimiterConfig(threshold_db=-6.0, mode="soft")
        limiter = Limiter(config=config)

        signal = 2.0 * (np.random.randn(1024) + 1j * np.random.randn(1024))
        signal = signal.astype(np.complex64)

        output = limiter.process(signal)

        # Soft limiting keeps envelope below threshold (roughly)
        threshold = 10 ** (-6.0 / 20)
        assert np.mean(np.abs(output)) < threshold

    def test_cubic_limiting(self):
        """Test cubic soft clipping mode."""
        config = LimiterConfig(threshold_db=-6.0, mode="cubic")
        limiter = Limiter(config=config)

        signal = 2.0 * (np.random.randn(1024) + 1j * np.random.randn(1024))
        signal = signal.astype(np.complex64)

        output = limiter.process(signal)
        assert len(output) == len(signal)

    def test_gain_reduction_tracking(self):
        """Test gain reduction property."""
        config = LimiterConfig(threshold_db=-6.0, mode="hard")
        limiter = Limiter(config=config)

        # Large signal
        signal = 10.0 * np.ones(100, dtype=np.complex64)
        limiter.process(signal)

        gr = limiter.gain_reduction_db
        assert gr < 0  # Should be negative (reduction)


class TestFrequencyOffset:
    """Tests for frequency offset simulation."""

    def test_default_config(self):
        """Test frequency offset with default config."""
        fo = FrequencyOffset()
        assert fo.config.offset_hz == 0.0
        assert fo.config.drift_rate_hz_per_sec == 0.0

    def test_static_offset(self):
        """Test static frequency offset."""
        config = FrequencyOffsetConfig(offset_hz=100.0)
        fo = FrequencyOffset(config=config, sample_rate_hz=10000)

        # DC signal
        signal = np.ones(1000, dtype=np.complex64)
        output = fo.process(signal)

        # Should have 100 Hz component
        spectrum = np.abs(np.fft.fft(output))
        freq_bins = np.fft.fftfreq(len(output), 1/10000)

        peak_idx = np.argmax(spectrum)
        peak_freq = abs(freq_bins[peak_idx])

        assert abs(peak_freq - 100.0) < 20  # Allow some tolerance

    def test_frequency_drift(self):
        """Test frequency drift."""
        config = FrequencyOffsetConfig(
            offset_hz=0.0,
            drift_rate_hz_per_sec=10.0,
        )
        fo = FrequencyOffset(config=config, sample_rate_hz=1000)

        signal = np.ones(2000, dtype=np.complex64)
        output = fo.process(signal)

        # Phase should increase quadratically due to drift
        phase = np.unwrap(np.angle(output))

        # Check that phase is increasing
        assert phase[-1] > phase[0]

    def test_reset(self):
        """Test frequency offset reset."""
        fo = FrequencyOffset()
        signal = np.ones(1000, dtype=np.complex64)
        fo.process(signal)

        fo.reset()
        assert fo._phase == 0.0
        assert fo._time == 0.0

    def test_set_offset(self):
        """Test setting offset dynamically."""
        fo = FrequencyOffset()
        fo.set_offset(50.0)
        assert fo.config.offset_hz == 50.0


class TestImpairmentChain:
    """Tests for impairment chain."""

    def test_empty_chain(self):
        """Test chain with no impairments."""
        chain = ImpairmentChain()
        signal = np.random.randn(100).astype(np.complex64)

        output = chain.process(signal)
        np.testing.assert_array_almost_equal(output, signal)

    def test_chain_with_agc(self):
        """Test chain with AGC only."""
        agc = AGC(AGCConfig(mode=AGCMode.FAST))
        chain = ImpairmentChain(agc=agc)

        signal = np.random.randn(1024).astype(np.complex64)
        output = chain.process(signal)

        assert output.shape == signal.shape

    def test_full_chain(self):
        """Test chain with all impairments."""
        agc = AGC(AGCConfig(mode=AGCMode.FAST))
        limiter = Limiter(LimiterConfig(threshold_db=-6.0))
        freq_offset = FrequencyOffset(FrequencyOffsetConfig(offset_hz=10.0))
        noise_gen = NoiseGenerator(NoiseConfig(snr_db=30.0))

        chain = ImpairmentChain(
            agc=agc,
            limiter=limiter,
            freq_offset=freq_offset,
            noise_generator=noise_gen,
        )

        signal = np.random.randn(1024).astype(np.complex64)
        output = chain.process(signal)

        assert output.shape == signal.shape

    def test_chain_reset(self):
        """Test chain reset."""
        agc = AGC()
        freq_offset = FrequencyOffset()
        chain = ImpairmentChain(agc=agc, freq_offset=freq_offset)

        signal = np.random.randn(1024).astype(np.complex64)
        chain.process(signal)
        chain.reset()

        assert agc._gain_db == 0.0
        assert freq_offset._phase == 0.0

    def test_chain_status(self):
        """Test chain status reporting."""
        agc = AGC()
        limiter = Limiter()
        freq_offset = FrequencyOffset(FrequencyOffsetConfig(offset_hz=25.0))

        chain = ImpairmentChain(
            agc=agc, limiter=limiter, freq_offset=freq_offset
        )

        signal = np.random.randn(1024).astype(np.complex64)
        chain.process(signal)

        status = chain.get_status()
        assert "agc_gain_db" in status
        assert "limiter_gr_db" in status
        assert "freq_offset_hz" in status
        assert status["freq_offset_hz"] == 25.0


class TestChannelRecording:
    """Tests for channel state recording and playback."""

    def test_snapshot_creation(self):
        """Test creating a channel snapshot."""
        H = np.random.randn(64) + 1j * np.random.randn(64)
        H = H.astype(np.complex64)

        snapshot = ChannelSnapshot(timestamp=1.0, transfer_function=H)

        assert snapshot.timestamp == 1.0
        assert len(snapshot.transfer_function) == 64

    def test_player_direct_creation(self):
        """Test creating a player directly."""
        timestamps = np.array([0.0, 0.1, 0.2])
        H_array = np.random.randn(3, 64) + 1j * np.random.randn(3, 64)
        H_array = H_array.astype(np.complex64)

        player = ChannelPlayer(timestamps, H_array)

        assert player.num_snapshots == 3
        assert player.fft_size == 64
        assert player.duration == pytest.approx(0.2, rel=0.01)

    def test_player_save_load_npz(self):
        """Test saving and loading recordings in NPZ format."""
        import json
        from datetime import datetime

        timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        fft_size = 64
        H_array = np.random.randn(5, fft_size) + 1j * np.random.randn(5, fft_size)
        H_array = H_array.astype(np.complex64)

        metadata = RecordingMetadata(
            created=datetime.now().isoformat(),
            duration_sec=0.4,
            num_snapshots=5,
            snapshot_rate_hz=10.0,
            sample_rate_hz=8000.0,
            fft_size=fft_size,
            channel_model="test",
        )

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            filepath = f.name

        try:
            # Save manually in the expected format
            from dataclasses import asdict
            np.savez_compressed(
                filepath,
                timestamps=timestamps,
                transfer_functions=H_array,
                metadata=np.array([json.dumps(asdict(metadata))]),
            )
            assert os.path.exists(filepath)

            player = ChannelPlayer.load(filepath)
            assert player.duration == pytest.approx(0.4, rel=0.01)
            assert player.num_snapshots == 5
        finally:
            os.unlink(filepath)

    def test_player_interpolation(self):
        """Test channel player interpolation."""
        timestamps = np.array([0.0, 0.1, 0.2])
        fft_size = 32
        H_array = np.random.randn(3, fft_size) + 1j * np.random.randn(3, fft_size)
        H_array = H_array.astype(np.complex64)

        player = ChannelPlayer(timestamps, H_array)

        # Get interpolated response at t=0.05
        H = player.get_at_time(0.05, interpolate=True)
        assert len(H) == fft_size

        # Get nearest response (no interpolation)
        H_nearest = player.get_at_time(0.05, interpolate=False)
        assert len(H_nearest) == fft_size

    def test_player_iteration(self):
        """Test channel player iteration."""
        timestamps = np.array([0.0, 0.1, 0.2, 0.3])
        fft_size = 32
        H_array = np.random.randn(4, fft_size) + 1j * np.random.randn(4, fft_size)
        H_array = H_array.astype(np.complex64)

        player = ChannelPlayer(timestamps, H_array)

        # Iterate without looping
        count = 0
        for H in player.iterate(rate_hz=20.0, loop=False):
            count += 1
            assert len(H) == fft_size
            if count > 10:  # Safety limit
                break

        assert count > 0

    def test_player_get_snapshot(self):
        """Test getting snapshot by index."""
        timestamps = np.array([0.0, 0.1, 0.2])
        fft_size = 32
        H_array = np.random.randn(3, fft_size) + 1j * np.random.randn(3, fft_size)
        H_array = H_array.astype(np.complex64)

        player = ChannelPlayer(timestamps, H_array)

        snapshot = player.get_snapshot(1)
        assert snapshot.timestamp == 0.1
        np.testing.assert_array_equal(snapshot.transfer_function, H_array[1])

    def test_recording_metadata(self):
        """Test recording metadata."""
        from datetime import datetime

        metadata = RecordingMetadata(
            created=datetime.now().isoformat(),
            duration_sec=10.0,
            num_snapshots=100,
            snapshot_rate_hz=10.0,
            sample_rate_hz=8000.0,
            fft_size=64,
            channel_model="watterson",
            description="Test recording",
        )

        assert metadata.sample_rate_hz == 8000.0
        assert metadata.duration_sec == 10.0
        assert metadata.channel_model == "watterson"

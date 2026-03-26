#!/usr/bin/env python3
"""Test phase continuity through the entire processing chain.

Injects a continuous tone and verifies no discontinuities are introduced
by any processing stage.
"""

import numpy as np
import sounddevice as sd
import sys
import time

# Add parent to path for imports
sys.path.insert(0, '/home/n4hy/HFPathSimulatorLiveData/src')

SAMPLE_RATE = 8000
TONE_FREQ = 440.0
BLOCK_SIZE = 4096
TEST_DURATION = 5  # seconds per test


class PhaseContinuousTone:
    """Generate phase-continuous tone."""

    def __init__(self, freq_hz, sample_rate):
        self.freq = freq_hz
        self.sample_rate = sample_rate
        self.phase = 0.0
        self.phase_inc = 2 * np.pi * freq_hz / sample_rate

    def generate(self, n):
        """Generate n samples maintaining phase."""
        t = np.arange(n)
        phases = self.phase + self.phase_inc * t
        samples = 0.5 * np.exp(1j * phases).astype(np.complex64)
        self.phase = (self.phase + self.phase_inc * n) % (2 * np.pi)
        return samples

    def reset(self):
        self.phase = 0.0


class RingBuffer:
    """Simple ring buffer with single read/write pointers."""

    def __init__(self, size):
        self.buffer = np.zeros(size, dtype=np.complex64)
        self.size = size
        self.write_ptr = 0
        self.read_ptr = 0
        self.count = 0

    def write(self, samples):
        """Write samples to buffer. Returns number written."""
        n = len(samples)
        space = self.size - self.count
        to_write = min(n, space)

        if to_write == 0:
            return 0

        # Write with wrap
        end = self.write_ptr + to_write
        if end <= self.size:
            self.buffer[self.write_ptr:end] = samples[:to_write]
        else:
            first = self.size - self.write_ptr
            self.buffer[self.write_ptr:] = samples[:first]
            self.buffer[:to_write - first] = samples[first:to_write]

        self.write_ptr = end % self.size
        self.count += to_write
        return to_write

    def read(self, n):
        """Read n samples from buffer."""
        to_read = min(n, self.count)
        if to_read == 0:
            return np.zeros(0, dtype=np.complex64)

        end = self.read_ptr + to_read
        if end <= self.size:
            samples = self.buffer[self.read_ptr:end].copy()
        else:
            first = self.size - self.read_ptr
            samples = np.concatenate([
                self.buffer[self.read_ptr:],
                self.buffer[:to_read - first]
            ])

        self.read_ptr = end % self.size
        self.count -= to_read
        return samples

    @property
    def available(self):
        return self.count


def check_phase_continuity(samples, sample_rate, freq_hz, tolerance_deg=10):
    """Check if samples maintain phase continuity.

    Returns (is_continuous, max_phase_jump_degrees, jump_locations)
    """
    if len(samples) < 2:
        return True, 0, []

    # Extract phase
    phase = np.angle(samples)

    # Calculate expected phase increment per sample
    expected_inc = 2 * np.pi * freq_hz / sample_rate

    # Calculate actual phase differences
    phase_diff = np.diff(phase)

    # Unwrap (handle -pi to pi transitions)
    phase_diff = np.where(phase_diff > np.pi, phase_diff - 2*np.pi, phase_diff)
    phase_diff = np.where(phase_diff < -np.pi, phase_diff + 2*np.pi, phase_diff)

    # Check deviation from expected
    deviation = np.abs(phase_diff - expected_inc)
    max_deviation = np.max(deviation) * 180 / np.pi

    # Find jump locations
    tolerance_rad = tolerance_deg * np.pi / 180
    jumps = np.where(deviation > tolerance_rad)[0]

    return len(jumps) == 0, max_deviation, jumps.tolist()


def play_through_ringbuffer(tone_gen, duration, ring_size=65536):
    """Test: tone -> ring buffer -> audio. Should be perfect."""
    print(f"\n{'='*60}")
    print("TEST: Tone -> Ring Buffer -> Audio")
    print(f"{'='*60}")

    ring = RingBuffer(ring_size)
    tone_gen.reset()

    # Pre-fill buffer
    prefill = tone_gen.generate(ring_size // 2)
    ring.write(prefill)
    print(f"Pre-filled {ring.count} samples")

    status_issues = []

    def callback(outdata, frames, time_info, status):
        if status:
            status_issues.append(str(status))

        samples = ring.read(frames)
        n = len(samples)

        if n > 0:
            outdata[:n, 0] = np.real(samples).astype(np.float32)
            outdata[:n, 1] = np.imag(samples).astype(np.float32)
        if n < frames:
            outdata[n:] = 0

    # Writer thread
    import threading
    stop_event = threading.Event()

    def writer():
        while not stop_event.is_set():
            if ring.count < ring.size - BLOCK_SIZE:
                samples = tone_gen.generate(BLOCK_SIZE)
                ring.write(samples)
            time.sleep(0.01)

    writer_thread = threading.Thread(target=writer)
    writer_thread.start()

    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=2, dtype='float32',
                         blocksize=256, latency='high', callback=callback):
        print(f"Playing for {duration} seconds... Listen for clicks!")
        time.sleep(duration)

    stop_event.set()
    writer_thread.join()

    print(f"Status issues: {len(status_issues)}")
    if status_issues:
        print(f"  Issues: {status_issues[:5]}...")

    return len(status_issues) == 0


def test_component(name, process_func, tone_gen, duration):
    """Test a single processing component."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    ring = RingBuffer(65536)
    tone_gen.reset()

    # Pre-fill
    for _ in range(8):
        samples = tone_gen.generate(BLOCK_SIZE)
        processed = process_func(samples)
        ring.write(processed)

    print(f"Pre-filled {ring.count} samples")

    status_issues = []
    all_output = []

    def callback(outdata, frames, time_info, status):
        if status:
            status_issues.append(str(status))

        samples = ring.read(frames)
        n = len(samples)

        if n > 0:
            all_output.append(samples.copy())
            outdata[:n, 0] = np.real(samples).astype(np.float32)
            outdata[:n, 1] = np.imag(samples).astype(np.float32)
        if n < frames:
            outdata[n:] = 0

    import threading
    stop_event = threading.Event()

    def writer():
        while not stop_event.is_set():
            if ring.count < ring.size - BLOCK_SIZE:
                samples = tone_gen.generate(BLOCK_SIZE)
                processed = process_func(samples)
                ring.write(processed)
            time.sleep(0.01)

    writer_thread = threading.Thread(target=writer)
    writer_thread.start()

    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=2, dtype='float32',
                         blocksize=256, latency='high', callback=callback):
        print(f"Playing for {duration} seconds... Listen for clicks!")
        time.sleep(duration)

    stop_event.set()
    writer_thread.join()

    # Check phase continuity of output
    if all_output:
        full_output = np.concatenate(all_output)
        is_cont, max_jump, jumps = check_phase_continuity(
            full_output, SAMPLE_RATE, TONE_FREQ, tolerance_deg=15
        )
        print(f"Phase continuity: {'PASS' if is_cont else 'FAIL'}")
        print(f"Max phase deviation: {max_jump:.1f} degrees")
        if not is_cont:
            print(f"Jumps at samples: {jumps[:10]}...")

    print(f"Audio status issues: {len(status_issues)}")

    return len(status_issues) == 0


def main():
    print("="*60)
    print("PHASE CONTINUITY TEST SUITE")
    print("="*60)
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Tone frequency: {TONE_FREQ} Hz")
    print(f"Block size: {BLOCK_SIZE}")
    print()

    tone_gen = PhaseContinuousTone(TONE_FREQ, SAMPLE_RATE)

    # Test 1: Pure ring buffer (baseline)
    print("\n" + "="*60)
    print("TEST 1: BASELINE - Ring buffer only (no processing)")
    print("="*60)
    play_through_ringbuffer(tone_gen, TEST_DURATION)
    input("Press Enter to continue...")

    # Test 2: Passthrough (identity function)
    test_component(
        "PASSTHROUGH (identity)",
        lambda x: x,
        tone_gen,
        TEST_DURATION
    )
    input("Press Enter to continue...")

    # Test 3: Channel model (if no channel, should be passthrough)
    try:
        from hfpathsim.core.channel import VoglerChannel, ChannelConfig
        config = ChannelConfig(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
        channel = VoglerChannel(config)

        test_component(
            "VOGLER CHANNEL",
            lambda x: channel.process(x),
            tone_gen,
            TEST_DURATION
        )
        input("Press Enter to continue...")
    except Exception as e:
        print(f"Skipping channel test: {e}")

    # Test 4: Noise generator
    try:
        from hfpathsim.core.noise import NoiseGenerator, NoiseConfig
        noise_gen = NoiseGenerator(NoiseConfig(
            awgn_enabled=True,
            snr_db=30,  # High SNR so we can still hear the tone
        ))

        test_component(
            "NOISE GENERATOR (30dB SNR)",
            lambda x: noise_gen.add_noise(x),
            tone_gen,
            TEST_DURATION
        )
        input("Press Enter to continue...")
    except Exception as e:
        print(f"Skipping noise test: {e}")

    # Test 5: AGC
    try:
        from hfpathsim.core.impairments import AGC, AGCConfig
        agc = AGC(AGCConfig())

        test_component(
            "AGC",
            lambda x: agc.process_block(x),
            tone_gen,
            TEST_DURATION
        )
        input("Press Enter to continue...")
    except Exception as e:
        print(f"Skipping AGC test: {e}")

    # Test 6: Limiter
    try:
        from hfpathsim.core.impairments import Limiter, LimiterConfig
        limiter = Limiter(LimiterConfig())

        test_component(
            "LIMITER",
            lambda x: limiter.process(x),
            tone_gen,
            TEST_DURATION
        )
        input("Press Enter to continue...")
    except Exception as e:
        print(f"Skipping limiter test: {e}")

    # Test 7: Frequency offset
    try:
        from hfpathsim.core.impairments import FrequencyOffset, FrequencyOffsetConfig
        freq_off = FrequencyOffset(FrequencyOffsetConfig(offset_hz=10))

        test_component(
            "FREQUENCY OFFSET (+10 Hz)",
            lambda x: freq_off.process(x),
            tone_gen,
            TEST_DURATION
        )
        input("Press Enter to continue...")
    except Exception as e:
        print(f"Skipping frequency offset test: {e}")

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

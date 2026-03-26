#!/usr/bin/env python3
"""Test the EXACT dashboard signal chain for phase continuity.

This test uses the actual components from the codebase, configured
exactly as the dashboard configures them.
"""

import numpy as np
import sys
import time

sys.path.insert(0, '/home/n4hy/HFPathSimulatorLiveData/src')

# Import actual dashboard components
from hfpathsim.input.siggen import SignalGenerator, WaveformType
from hfpathsim.core.watterson import WattersonChannel, WattersonConfig
from hfpathsim.core.parameters import ITUCondition
from hfpathsim.output.audio import AudioOutputSink

SAMPLE_RATE = 8000
DURATION = 5  # seconds


def generate_test_tone(n_samples: int, freq_hz: float = 440.0) -> np.ndarray:
    """Generate a simple phase-continuous test tone."""
    t = np.arange(n_samples) / SAMPLE_RATE
    # Simple complex exponential - perfectly phase continuous
    return (0.5 * np.exp(2j * np.pi * freq_hz * t)).astype(np.complex64)


class PhaseContinuousToneSource:
    """Phase-continuous tone generator matching InputSource interface."""

    def __init__(self, freq_hz: float, sample_rate: float):
        self.freq_hz = freq_hz
        self.sample_rate = sample_rate
        self.phase = 0.0
        self.phase_inc = 2 * np.pi * freq_hz / sample_rate
        self._is_open = False

    def open(self) -> bool:
        self._is_open = True
        self.phase = 0.0
        return True

    def close(self):
        self._is_open = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    def read(self, n: int) -> np.ndarray:
        """Generate n samples maintaining exact phase continuity."""
        t = np.arange(n)
        phases = self.phase + self.phase_inc * t
        samples = (0.5 * np.exp(1j * phases)).astype(np.complex64)
        self.phase = (self.phase + self.phase_inc * n) % (2 * np.pi)
        return samples


def test_watterson_isolation():
    """Test Watterson channel in isolation with a tone."""
    print("\n" + "=" * 60)
    print("TEST: Watterson Channel Isolation")
    print("=" * 60)

    # Create Watterson with CORRECT sample rate
    config = WattersonConfig.from_itu_condition(
        ITUCondition.MODERATE,
        sample_rate_hz=SAMPLE_RATE,
    )
    print(f"Watterson config: sample_rate={config.sample_rate_hz}, taps={len(config.taps)}")
    for i, tap in enumerate(config.taps):
        print(f"  Tap {i}: delay={tap.delay_ms}ms, amp={tap.amplitude}, doppler={tap.doppler_spread_hz}Hz")

    channel = WattersonChannel(config, seed=42)

    # Check tap states
    print("\nTap states after init:")
    for i, state in enumerate(channel._tap_states):
        print(f"  Tap {i}: delay_samples={state['delay_samples']}, "
              f"delay_buffer_size={len(state['delay_buffer'])}, "
              f"gain={state['current_gain']:.4f}")

    # Generate blocks and check for discontinuities
    tone_gen = PhaseContinuousToneSource(440.0, SAMPLE_RATE)
    tone_gen.open()

    block_size = 400  # 50ms at 8kHz - matches dashboard
    all_output = []

    print(f"\nProcessing {DURATION}s of tone in {block_size}-sample blocks...")

    n_blocks = int(SAMPLE_RATE * DURATION / block_size)
    for i in range(n_blocks):
        samples = tone_gen.read(block_size)
        output = channel.process_block(samples)
        all_output.append(output)

    # Concatenate and analyze
    full_output = np.concatenate(all_output)
    print(f"Total samples: {len(full_output)}")

    # Check for discontinuities at block boundaries
    print("\nChecking for discontinuities at block boundaries...")
    discontinuities = []

    for i in range(1, n_blocks):
        boundary = i * block_size
        if boundary >= len(full_output):
            break

        # Check phase jump at boundary
        before = full_output[boundary - 1]
        after = full_output[boundary]

        phase_before = np.angle(before)
        phase_after = np.angle(after)

        # Expected phase change for 440Hz tone
        expected_phase_inc = 2 * np.pi * 440.0 / SAMPLE_RATE
        actual_phase_diff = phase_after - phase_before

        # Unwrap
        while actual_phase_diff > np.pi:
            actual_phase_diff -= 2 * np.pi
        while actual_phase_diff < -np.pi:
            actual_phase_diff += 2 * np.pi

        # The Watterson channel applies fading, so we can't check exact phase
        # But we CAN check for sudden amplitude jumps
        amp_before = np.abs(before)
        amp_after = np.abs(after)
        amp_ratio = amp_after / (amp_before + 1e-10)

        if amp_ratio > 1.5 or amp_ratio < 0.67:
            discontinuities.append((boundary, amp_ratio))

    if discontinuities:
        print(f"FOUND {len(discontinuities)} amplitude discontinuities:")
        for boundary, ratio in discontinuities[:10]:
            print(f"  Sample {boundary}: amplitude ratio = {ratio:.3f}")
    else:
        print("No amplitude discontinuities found at block boundaries")

    # Check for clicks by looking at sudden sample-to-sample changes
    print("\nChecking for clicks (sudden amplitude changes)...")
    diff = np.abs(np.diff(full_output))
    threshold = np.mean(diff) + 5 * np.std(diff)
    clicks = np.where(diff > threshold)[0]

    if len(clicks) > 0:
        print(f"FOUND {len(clicks)} potential clicks (threshold={threshold:.4f}):")
        for idx in clicks[:20]:
            print(f"  Sample {idx}: diff={diff[idx]:.4f}")
    else:
        print("No clicks detected")

    return len(discontinuities) == 0 and len(clicks) < 10


def test_audio_output():
    """Test audio output with a simple tone - no processing."""
    print("\n" + "=" * 60)
    print("TEST: Audio Output Only (bypass all processing)")
    print("=" * 60)

    tone_gen = PhaseContinuousToneSource(440.0, SAMPLE_RATE)
    tone_gen.open()

    # Create audio sink exactly as dashboard does
    audio = AudioOutputSink(
        sample_rate_hz=SAMPLE_RATE,
        buffer_size=1048576,
        blocksize=256,
        latency="high",
    )

    if not audio.open():
        print("Failed to open audio output")
        return False

    print(f"Audio opened: {audio.get_device_info()}")
    print(f"Playing {DURATION}s of 440Hz tone...")

    block_size = 400  # 50ms blocks
    start_time = time.time()

    while time.time() - start_time < DURATION:
        samples = tone_gen.read(block_size)
        written = audio.write(samples)

        # Match dashboard timing - 50ms per block
        time.sleep(0.05)

    # Let buffer drain briefly
    time.sleep(0.5)

    print(f"Underruns: {audio.underruns}")
    print(f"Buffer fill: {audio.buffer_fill:.1f}%")

    audio.close()

    return audio.underruns == 0


def test_full_chain_with_tone():
    """Test the full chain: Tone -> Watterson -> Audio."""
    print("\n" + "=" * 60)
    print("TEST: Full Chain (Tone -> Watterson -> Audio)")
    print("=" * 60)

    tone_gen = PhaseContinuousToneSource(440.0, SAMPLE_RATE)
    tone_gen.open()

    # Create Watterson with correct sample rate
    wconfig = WattersonConfig.from_itu_condition(
        ITUCondition.MODERATE,
        sample_rate_hz=SAMPLE_RATE,
    )
    watterson = WattersonChannel(wconfig, seed=42)

    # Create audio sink
    audio = AudioOutputSink(
        sample_rate_hz=SAMPLE_RATE,
        buffer_size=1048576,
        blocksize=256,
        latency="high",
    )

    if not audio.open():
        print("Failed to open audio output")
        return False

    print(f"Playing {DURATION}s through Watterson channel...")
    print("Listen for clicks/pops!")

    block_size = 400  # 50ms blocks - exact dashboard timing
    start_time = time.time()

    while time.time() - start_time < DURATION:
        # Read from tone generator
        samples = tone_gen.read(block_size)

        # Process through Watterson
        processed = watterson.process_block(samples)

        # Write to audio
        written = audio.write(processed)

        # Match dashboard timing
        time.sleep(0.05)

    time.sleep(0.5)

    print(f"Underruns: {audio.underruns}")

    audio.close()

    return audio.underruns == 0


def test_gain_continuity():
    """Directly test gain interpolation across blocks."""
    print("\n" + "=" * 60)
    print("TEST: Gain Interpolation Verification")
    print("=" * 60)

    config = WattersonConfig.from_itu_condition(
        ITUCondition.MODERATE,
        sample_rate_hz=SAMPLE_RATE,
    )
    channel = WattersonChannel(config, seed=42)

    # Track gains across blocks
    gains_at_boundaries = []

    block_size = 400
    n_blocks = 20

    for block_idx in range(n_blocks):
        # Get gains before processing
        old_gains = [state["current_gain"] for state in channel._tap_states]

        # Process a block of ones to see the gain directly
        ones = np.ones(block_size, dtype=np.complex64)
        output = channel.process_block(ones)

        # Get gains after processing
        new_gains = [state["current_gain"] for state in channel._tap_states]

        # Check output at block boundary
        # First sample should use mostly old_gain
        # Last sample should use mostly new_gain
        first_sample = output[0]
        last_sample = output[-1]

        gains_at_boundaries.append({
            'block': block_idx,
            'old_gain_0': old_gains[0],
            'new_gain_0': new_gains[0],
            'first_output': first_sample,
            'last_output': last_sample,
        })

    print("Block | Old Gain (tap0) | New Gain (tap0) | First Out | Last Out")
    print("-" * 70)
    for g in gains_at_boundaries[:10]:
        print(f"{g['block']:5d} | {abs(g['old_gain_0']):14.4f} | {abs(g['new_gain_0']):14.4f} | "
              f"{abs(g['first_output']):9.4f} | {abs(g['last_output']):8.4f}")

    # Check for sudden jumps between blocks
    print("\nChecking for gain jumps between consecutive blocks...")
    jumps = []
    for i in range(1, len(gains_at_boundaries)):
        prev_last = gains_at_boundaries[i-1]['last_output']
        curr_first = gains_at_boundaries[i]['first_output']

        ratio = abs(curr_first) / (abs(prev_last) + 1e-10)
        if ratio > 1.2 or ratio < 0.8:
            jumps.append((i, ratio))

    if jumps:
        print(f"FOUND {len(jumps)} gain jumps:")
        for block, ratio in jumps:
            print(f"  Block {block}: ratio = {ratio:.3f}")
        return False
    else:
        print("No significant gain jumps between blocks")
        return True


def main():
    print("=" * 60)
    print("DASHBOARD SIGNAL CHAIN PHASE CONTINUITY TEST")
    print("=" * 60)
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Block Size: 400 samples (50ms)")
    print(f"Duration: {DURATION}s per test")
    print()

    results = {}

    # Test 1: Gain interpolation verification
    results['Gain Interpolation'] = test_gain_continuity()

    # Test 2: Watterson in isolation
    results['Watterson Isolation'] = test_watterson_isolation()

    # Test 3: Audio output only (bypass processing)
    print("\nStarting audio test - listen for any clicks...")
    input("Press Enter to start audio output test...")
    results['Audio Only'] = test_audio_output()

    # Test 4: Full chain
    print("\nStarting full chain test - listen for clicks...")
    input("Press Enter to start full chain test...")
    results['Full Chain'] = test_full_chain_with_tone()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()

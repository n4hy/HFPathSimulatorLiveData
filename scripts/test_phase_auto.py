#!/usr/bin/env python3
"""Automated phase continuity test - no user input needed."""

import numpy as np
import sounddevice as sd
import sys
import time
import threading

sys.path.insert(0, '/home/n4hy/HFPathSimulatorLiveData/src')

SAMPLE_RATE = 8000
TONE_FREQ = 440.0
BLOCK_SIZE = 4096
TEST_DURATION = 3


class PhaseContinuousTone:
    def __init__(self, freq_hz, sample_rate):
        self.freq = freq_hz
        self.sample_rate = sample_rate
        self.phase = 0.0
        self.phase_inc = 2 * np.pi * freq_hz / sample_rate

    def generate(self, n):
        t = np.arange(n)
        phases = self.phase + self.phase_inc * t
        samples = 0.5 * np.exp(1j * phases).astype(np.complex64)
        self.phase = (self.phase + self.phase_inc * n) % (2 * np.pi)
        return samples

    def reset(self):
        self.phase = 0.0


class RingBuffer:
    def __init__(self, size):
        self.buffer = np.zeros(size, dtype=np.complex64)
        self.size = size
        self.write_ptr = 0
        self.read_ptr = 0
        self.count = 0
        self.lock = threading.Lock()

    def write(self, samples):
        with self.lock:
            n = len(samples)
            space = self.size - self.count
            to_write = min(n, space)
            if to_write == 0:
                return 0

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
        with self.lock:
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
        with self.lock:
            return self.count


def test_component(name, process_func, duration=TEST_DURATION):
    """Test a component and report results."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    tone_gen = PhaseContinuousTone(TONE_FREQ, SAMPLE_RATE)
    ring = RingBuffer(65536)

    # Pre-fill with processed samples
    for _ in range(8):
        samples = tone_gen.generate(BLOCK_SIZE)
        processed = process_func(samples)
        ring.write(processed)

    print(f"Pre-filled {ring.available} samples")

    status_issues = 0
    underruns = 0

    def callback(outdata, frames, time_info, status):
        nonlocal status_issues, underruns
        if status:
            status_issues += 1
            if status.output_underflow:
                underruns += 1

        samples = ring.read(frames)
        n = len(samples)

        if n > 0:
            outdata[:n, 0] = np.real(samples).astype(np.float32)
            outdata[:n, 1] = np.imag(samples).astype(np.float32)
        if n < frames:
            outdata[n:] = 0

    stop_event = threading.Event()

    def writer():
        while not stop_event.is_set():
            if ring.available < ring.size - BLOCK_SIZE:
                samples = tone_gen.generate(BLOCK_SIZE)
                processed = process_func(samples)
                written = ring.write(processed)
                if written < len(processed):
                    print(f"  WARNING: Buffer full, dropped {len(processed)-written} samples")
            time.sleep(0.01)

    writer_thread = threading.Thread(target=writer)
    writer_thread.start()

    try:
        with sd.OutputStream(samplerate=SAMPLE_RATE, channels=2, dtype='float32',
                             blocksize=256, latency='high', callback=callback):
            print(f"Playing for {duration} seconds...")
            time.sleep(duration)
    finally:
        stop_event.set()
        writer_thread.join()

    result = "PASS" if status_issues == 0 else "FAIL"
    print(f"Result: {result}")
    print(f"  Status callbacks: {status_issues}")
    print(f"  Underruns: {underruns}")

    return status_issues == 0


def main():
    print("="*60)
    print("AUTOMATED PHASE CONTINUITY TESTS")
    print("="*60)
    print(f"Each test plays a {TONE_FREQ}Hz tone for {TEST_DURATION}s")
    print("Listen for clicks/pops in each test")
    print()

    results = {}

    # Test 1: Baseline
    results['Baseline (passthrough)'] = test_component(
        "BASELINE - Ring buffer only",
        lambda x: x
    )
    time.sleep(0.5)

    # Test 2: HF Channel (Vogler)
    try:
        from hfpathsim.core.channel import HFChannel, ProcessingConfig
        config = ProcessingConfig(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
        channel = HFChannel(config=config, use_gpu=False)
        results['HF Channel'] = test_component(
            "HF CHANNEL (Vogler)",
            lambda x: channel.process(x)
        )
        time.sleep(0.5)
    except Exception as e:
        print(f"\nSkipping HF Channel: {e}")
        results['HF Channel'] = 'SKIP'

    # Test 3: Watterson
    try:
        from hfpathsim.core.watterson import WattersonChannel, WattersonConfig
        wconfig = WattersonConfig(sample_rate_hz=SAMPLE_RATE, block_size=BLOCK_SIZE)
        watterson = WattersonChannel(wconfig)
        results['Watterson Channel'] = test_component(
            "WATTERSON CHANNEL",
            lambda x: watterson.process_block(x)
        )
        time.sleep(0.5)
    except Exception as e:
        print(f"\nSkipping Watterson: {e}")
        results['Watterson Channel'] = 'SKIP'

    # Test 4: Noise
    try:
        from hfpathsim.core.noise import NoiseGenerator, NoiseConfig
        noise = NoiseGenerator(NoiseConfig(snr_db=30))
        results['Noise (30dB)'] = test_component(
            "NOISE GENERATOR",
            lambda x: noise.add_noise(x)
        )
        time.sleep(0.5)
    except Exception as e:
        print(f"\nSkipping Noise: {e}")
        results['Noise'] = 'SKIP'

    # Test 5: AGC
    try:
        from hfpathsim.core.impairments import AGC, AGCConfig
        agc = AGC(AGCConfig())
        results['AGC'] = test_component(
            "AGC",
            lambda x: agc.process_block(x)
        )
        time.sleep(0.5)
    except Exception as e:
        print(f"\nSkipping AGC: {e}")
        results['AGC'] = 'SKIP'

    # Test 6: Limiter
    try:
        from hfpathsim.core.impairments import Limiter, LimiterConfig
        limiter = Limiter(LimiterConfig())
        results['Limiter'] = test_component(
            "LIMITER",
            lambda x: limiter.process(x)
        )
        time.sleep(0.5)
    except Exception as e:
        print(f"\nSkipping Limiter: {e}")
        results['Limiter'] = 'SKIP'

    # Test 7: Frequency offset
    try:
        from hfpathsim.core.impairments import FrequencyOffset, FrequencyOffsetConfig
        freq_off = FrequencyOffset(FrequencyOffsetConfig(offset_hz=10))
        results['Freq Offset'] = test_component(
            "FREQUENCY OFFSET",
            lambda x: freq_off.process(x)
        )
        time.sleep(0.5)
    except Exception as e:
        print(f"\nSkipping Freq Offset: {e}")
        results['Freq Offset'] = 'SKIP'

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, result in results.items():
        status = "PASS" if result is True else ("FAIL" if result is False else result)
        print(f"  {name}: {status}")

    print("\nListen to each test - any that clicked need fixing!")


if __name__ == "__main__":
    main()

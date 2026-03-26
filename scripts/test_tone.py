"""Simple phase-continuous tone test - direct sounddevice, no buffers."""
import numpy as np
import sounddevice as sd
import time
import sys

SAMPLE_RATE = 8000
TONE_FREQ = 440.0

class ToneGenerator:
    """Phase-continuous tone generator."""

    def __init__(self, freq_hz, sample_rate):
        self.phase = 0.0
        self.phase_inc = 2 * np.pi * freq_hz / sample_rate

    def generate(self, n):
        """Generate n samples, maintaining phase continuity."""
        phases = self.phase + self.phase_inc * np.arange(n)
        self.phase = (self.phase + self.phase_inc * n) % (2 * np.pi)
        return 0.5 * np.exp(1j * phases).astype(np.complex64)


def main():
    print("=" * 60)
    print("Phase-Continuous Tone Test")
    print("=" * 60)
    print(f"Frequency: {TONE_FREQ} Hz")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Duration: 10 seconds")
    print()
    print("This uses direct sounddevice callback - NO intermediate buffer.")
    print("If you hear clicks, the problem is system-level.")
    print("If smooth, the problem is in our buffer code.")
    print()
    print("Playing...")

    gen = ToneGenerator(TONE_FREQ, SAMPLE_RATE)
    status_count = 0

    def callback(outdata, frames, time_info, status):
        nonlocal status_count
        if status:
            status_count += 1
            print(f"  AUDIO STATUS: {status}", file=sys.stderr)
        samples = gen.generate(frames)
        outdata[:, 0] = np.real(samples)
        outdata[:, 1] = np.imag(samples)

    try:
        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=2,
            dtype='float32',
            blocksize=512,
            latency='high',
            callback=callback
        ):
            time.sleep(10)

        print()
        print("=" * 60)
        print(f"Done. Status callbacks: {status_count}")
        if status_count == 0:
            print("No audio issues reported by sounddevice.")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

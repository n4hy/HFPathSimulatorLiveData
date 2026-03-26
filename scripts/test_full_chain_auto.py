#!/usr/bin/env python3
"""Automated full signal chain test - simulates exactly what the dashboard does."""

import numpy as np
import sys
sys.path.insert(0, '/home/n4hy/HFPathSimulatorLiveData/src')

from hfpathsim.input.siggen import SignalGenerator, WaveformType
from hfpathsim.core.watterson import WattersonChannel, WattersonConfig
from hfpathsim.core.channel import HFChannel, ProcessingConfig
from hfpathsim.core.parameters import VoglerParameters, ITUCondition
from hfpathsim.core.noise import NoiseGenerator, NoiseConfig
from hfpathsim.core.impairments import AGC, AGCConfig

SAMPLE_RATE = 8000


def test_with_model(model_name, channel):
    """Test a specific channel model."""
    print(f"\n{'=' * 60}")
    print(f"TESTING: {model_name} at {SAMPLE_RATE}Hz")
    print(f"{'=' * 60}")

    # Create signal generator
    siggen = SignalGenerator(WaveformType.RTTY, sample_rate_hz=SAMPLE_RATE)
    if not siggen.open():
        print("ERROR: Failed to open signal generator")
        return False

    # Create noise and AGC at correct sample rate
    noise_gen = NoiseGenerator(NoiseConfig(snr_db=20.0), sample_rate_hz=SAMPLE_RATE)
    agc = AGC(AGCConfig(), sample_rate_hz=SAMPLE_RATE)

    print(f"Noise sample_rate: {noise_gen.sample_rate}Hz")
    print(f"AGC sample_rate: {agc.sample_rate}Hz")

    # Process 10 blocks
    results = []
    for i in range(10):
        # Read samples
        samples = siggen.read(400)  # 50ms at 8kHz

        # Channel
        if channel is not None:
            try:
                if hasattr(channel, 'process_block'):
                    output = channel.process_block(samples)
                else:
                    output = channel.process(samples)
            except Exception as e:
                print(f"ERROR in channel: {e}")
                return False
        else:
            output = samples

        after_channel_mag = np.max(np.abs(output))

        # Noise
        output = noise_gen.add_noise(output)
        after_noise_mag = np.max(np.abs(output))

        # AGC
        output = agc.process_block(output)
        after_agc_mag = np.max(np.abs(output))
        agc_gain = agc.current_gain_db

        results.append({
            'input_mag': np.max(np.abs(samples)),
            'channel_mag': after_channel_mag,
            'noise_mag': after_noise_mag,
            'agc_mag': after_agc_mag,
            'agc_gain': agc_gain,
        })

    siggen.close()

    # Print results
    print("\nBlock | Input | Channel | Noise | AGC Out | AGC Gain")
    print("-" * 60)
    for i, r in enumerate(results):
        print(f"{i:5d} | {r['input_mag']:.3f} | {r['channel_mag']:.3f}   | "
              f"{r['noise_mag']:.3f} | {r['agc_mag']:.3f}   | {r['agc_gain']:+.1f}dB")

    # Check final results
    final = results[-1]
    gain_ok = abs(final['agc_gain']) < 40  # Reasonable gain
    output_ok = 0.05 < final['agc_mag'] < 5.0  # Reasonable output

    print(f"\nFinal AGC gain reasonable: {gain_ok} ({final['agc_gain']:.1f}dB)")
    print(f"Final output reasonable: {output_ok} ({final['agc_mag']:.3f})")

    return gain_ok and output_ok


def main():
    print("FULL CHAIN AUTOMATED TEST")
    print("=" * 60)
    print(f"Sample rate: {SAMPLE_RATE}Hz")
    print(f"Block size: 400 samples (50ms)")
    print()

    results = {}

    # Test 1: Passthrough (no channel)
    results['Passthrough'] = test_with_model("Passthrough", None)

    # Test 2: Watterson channel
    wconfig = WattersonConfig.from_itu_condition(ITUCondition.MODERATE, sample_rate_hz=SAMPLE_RATE)
    watterson = WattersonChannel(wconfig)
    print(f"\nWatterson config: sample_rate={wconfig.sample_rate_hz}Hz, taps={len(wconfig.taps)}")
    results['Watterson'] = test_with_model("Watterson TDL", watterson)

    # Test 3: Vogler channel at 8kHz
    vogler_config = ProcessingConfig(
        sample_rate_hz=SAMPLE_RATE,
        block_size=int(SAMPLE_RATE * 0.05),
        overlap=int(SAMPLE_RATE * 0.0125),
    )
    vogler_params = VoglerParameters.from_itu_condition(ITUCondition.MODERATE)
    vogler = HFChannel(vogler_params, vogler_config, use_gpu=False)
    print(f"\nVogler config: sample_rate={vogler_config.sample_rate_hz}Hz, "
          f"block_size={vogler_config.block_size}")
    results['Vogler'] = test_with_model("Vogler IPM", vogler)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

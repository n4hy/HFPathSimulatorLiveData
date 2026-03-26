#!/usr/bin/env python3
"""Test that impairments use correct sample rate.

This test verifies the fix for the sample rate mismatch bug where
impairments defaulted to 2MHz but input was at 8kHz.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/n4hy/HFPathSimulatorLiveData/src')

from hfpathsim.core.noise import NoiseGenerator, NoiseConfig
from hfpathsim.core.impairments import AGC, AGCConfig, AGCMode

SAMPLE_RATE = 8000  # Same as signal generator default


def test_agc_tracking():
    """Test that AGC tracks envelope correctly at 8kHz."""
    print("=" * 60)
    print("TEST: AGC Envelope Tracking at 8kHz vs 2MHz")
    print("=" * 60)

    # Create test signal
    n_samples = 400  # 50ms block at 8kHz
    signal = np.ones(n_samples, dtype=np.complex64) * 0.5

    # Test with WRONG sample rate (the bug)
    agc_wrong = AGC(AGCConfig(mode=AGCMode.MEDIUM), sample_rate_hz=2_000_000)

    # Process a few blocks
    for i in range(5):
        output_wrong = agc_wrong.process_block(signal)

    print(f"\nAGC at 2MHz (WRONG):")
    print(f"  Envelope: {agc_wrong._envelope:.4f}")
    print(f"  Gain: {agc_wrong.current_gain_db:.1f} dB")
    print(f"  Output mag: {np.max(np.abs(output_wrong)):.4f}")

    # Test with CORRECT sample rate
    agc_correct = AGC(AGCConfig(mode=AGCMode.MEDIUM), sample_rate_hz=SAMPLE_RATE)

    # Process same blocks
    for i in range(5):
        output_correct = agc_correct.process_block(signal)

    print(f"\nAGC at 8kHz (CORRECT):")
    print(f"  Envelope: {agc_correct._envelope:.4f}")
    print(f"  Gain: {agc_correct.current_gain_db:.1f} dB")
    print(f"  Output mag: {np.max(np.abs(output_correct)):.4f}")

    # Check results
    # At 2MHz, envelope tracking is 250x slower, so gain is wrong
    # At 8kHz, envelope should track properly

    envelope_ok = agc_correct._envelope > 0.3  # Should track to ~0.5
    gain_ok = abs(agc_correct.current_gain_db) < 30  # Shouldn't be huge

    print(f"\nEnvelope tracked properly: {envelope_ok}")
    print(f"Gain reasonable: {gain_ok}")

    return envelope_ok and gain_ok


def test_noise_with_signal():
    """Test noise generation at correct sample rate."""
    print("\n" + "=" * 60)
    print("TEST: Noise with 20dB SNR")
    print("=" * 60)

    config = NoiseConfig(snr_db=20.0)

    # Test signal
    n_samples = 400
    signal = np.ones(n_samples, dtype=np.complex64) * 0.5

    # Add noise at correct sample rate
    noise_gen = NoiseGenerator(config, sample_rate_hz=SAMPLE_RATE)
    noisy = noise_gen.add_noise(signal)

    # Measure SNR
    signal_power = np.mean(np.abs(signal) ** 2)
    noise = noisy - signal
    noise_power = np.mean(np.abs(noise) ** 2)
    measured_snr = 10 * np.log10(signal_power / noise_power)

    print(f"Signal power: {signal_power:.4f}")
    print(f"Noise power: {noise_power:.6f}")
    print(f"Measured SNR: {measured_snr:.1f} dB (target: 20.0 dB)")
    print(f"Output mean magnitude: {np.mean(np.abs(noisy)):.4f}")

    # Check result
    snr_ok = abs(measured_snr - 20.0) < 2.0  # Within 2dB

    print(f"\nSNR correct: {snr_ok}")

    return snr_ok


def test_full_chain():
    """Test the full processing chain."""
    print("\n" + "=" * 60)
    print("TEST: Full Chain (Channel -> Noise -> AGC)")
    print("=" * 60)

    from hfpathsim.core.watterson import WattersonChannel, WattersonConfig
    from hfpathsim.core.parameters import ITUCondition

    # Create Watterson channel at correct sample rate
    wconfig = WattersonConfig.from_itu_condition(
        ITUCondition.MODERATE,
        sample_rate_hz=SAMPLE_RATE,
    )
    channel = WattersonChannel(wconfig)

    # Create noise at correct sample rate
    noise_config = NoiseConfig(snr_db=20.0)
    noise_gen = NoiseGenerator(noise_config, sample_rate_hz=SAMPLE_RATE)

    # Create AGC at correct sample rate
    agc_config = AGCConfig(mode=AGCMode.MEDIUM)
    agc = AGC(agc_config, sample_rate_hz=SAMPLE_RATE)

    # Create test signal (simple tone)
    n_samples = 400
    t = np.arange(n_samples) / SAMPLE_RATE
    signal = (0.5 * np.exp(2j * np.pi * 1000 * t)).astype(np.complex64)

    # Process through chain
    results = []
    for block in range(10):
        # Channel
        after_channel = channel.process_block(signal)

        # Noise
        after_noise = noise_gen.add_noise(after_channel)

        # AGC
        after_agc = agc.process_block(after_noise)

        results.append({
            'channel_mag': np.max(np.abs(after_channel)),
            'noise_mag': np.max(np.abs(after_noise)),
            'agc_mag': np.max(np.abs(after_agc)),
            'agc_gain': agc.current_gain_db,
        })

    print("\nBlock | After Channel | After Noise | After AGC | AGC Gain")
    print("-" * 60)
    for i, r in enumerate(results):
        print(f"{i:5d} | {r['channel_mag']:13.4f} | {r['noise_mag']:11.4f} | "
              f"{r['agc_mag']:9.4f} | {r['agc_gain']:+7.1f} dB")

    # Check final AGC gain is reasonable
    final_gain = results[-1]['agc_gain']
    final_mag = results[-1]['agc_mag']

    gain_reasonable = abs(final_gain) < 30
    mag_reasonable = 0.1 < final_mag < 10

    print(f"\nFinal AGC gain reasonable (|{final_gain:.1f}| < 30 dB): {gain_reasonable}")
    print(f"Final output magnitude reasonable (0.1 < {final_mag:.4f} < 10): {mag_reasonable}")

    return gain_reasonable and mag_reasonable


def main():
    print("SAMPLE RATE FIX VERIFICATION")
    print("=" * 60)
    print(f"Test sample rate: {SAMPLE_RATE} Hz")
    print()

    results = {}

    results['AGC Tracking'] = test_agc_tracking()
    results['Noise SNR'] = test_noise_with_signal()
    results['Full Chain'] = test_full_chain()

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

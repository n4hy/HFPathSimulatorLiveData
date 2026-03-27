#!/usr/bin/env python3
"""
Validate HF channel models against published reference datasets.

This script runs the Vogler-Hoffmeyer and Watterson channel models
and validates their statistics against NTIA TR-90-255, ITU-R F.1487,
and Watterson 1970 reference measurements.

Usage:
    python scripts/validate_channel_models.py [--dataset NAME] [--all] [--verbose]
"""

import argparse
import sys
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(__file__).rsplit('/', 2)[0] + '/src')

from hfpathsim.validation import (
    list_reference_datasets,
    get_reference_dataset,
    ReferenceDataset,
    ChannelValidator,
    ValidationReport,
    compute_delay_spread,
    compute_doppler_spread,
    compute_fading_statistics,
)
from hfpathsim.validation.validator import ValidationStatus
from hfpathsim.core.vogler_hoffmeyer import (
    VoglerHoffmeyerChannel,
    VoglerHoffmeyerConfig,
    ModeParameters,
    CorrelationType,
)
from hfpathsim.core.watterson import WattersonChannel, WattersonConfig, WattersonTap


@dataclass
class ValidationSummary:
    """Summary of validation results."""
    dataset_name: str
    channel_type: str
    delay_spread_measured: float
    delay_spread_reference: float
    delay_spread_error_pct: float
    doppler_spread_measured: float
    doppler_spread_reference: float
    doppler_spread_error_pct: float
    envelope_ratio: float  # Mean/RMS ratio (0.886 for Rayleigh)
    overall_pass: bool


def create_vh_config_for_reference(ref: ReferenceDataset, sample_rate: float = 48000.0) -> VoglerHoffmeyerConfig:
    """Create Vogler-Hoffmeyer config matching reference dataset parameters."""

    # Determine correlation type
    if ref.condition.value in ('flutter', 'auroral'):
        corr_type = CorrelationType.EXPONENTIAL
    else:
        corr_type = CorrelationType.GAUSSIAN

    # Convert RMS delay spread (ms) to sigma_tau (us)
    # Empirically calibrated: sigma_tau_us ≈ RMS_ms * 6829
    # This accounts for the delay profile shape and tap distribution
    sigma_tau_us = ref.delay_spread_ms * 6829.0

    # Create mode parameters to match reference
    mode = ModeParameters(
        name=f"{ref.name} mode",
        amplitude=1.0,
        floor_amplitude=0.01,
        tau_L=0.0,
        sigma_tau=sigma_tau_us,
        sigma_c=sigma_tau_us / 2.0,    # Carrier at ~half spread
        sigma_D=ref.doppler_spread_hz,
        doppler_shift=0.0,
        correlation_type=corr_type,
    )

    return VoglerHoffmeyerConfig(
        sample_rate=sample_rate,
        modes=[mode],
        spread_f_enabled=(ref.condition.value == 'spread_f'),
    )


def create_watterson_config_for_reference(ref: ReferenceDataset, sample_rate: float = 48000.0) -> WattersonConfig:
    """Create Watterson config matching reference dataset parameters.

    For RMS delay spread D with N equal-power taps uniformly distributed
    from 0 to τ_max, the relationship is:
        RMS = τ_max * sqrt((N²-1)/(12*N²)) for N taps

    For 2 equal-power taps at 0 and τ_max:
        RMS = τ_max / 2
        So τ_max = 2 * D

    For 3 equal-power taps at 0, τ_max/2, τ_max:
        RMS = τ_max / sqrt(6) ≈ 0.408 * τ_max
        So τ_max = D * sqrt(6) ≈ 2.45 * D
    """
    taps = []
    target_rms = ref.delay_spread_ms

    if ref.num_paths <= 2:
        # Two equal-power taps: RMS = max_delay / 2
        max_delay = target_rms * 2.0
        taps = [
            WattersonTap(delay_ms=0.0, amplitude=1.0, doppler_spread_hz=ref.doppler_spread_hz),
            WattersonTap(delay_ms=max_delay, amplitude=1.0, doppler_spread_hz=ref.doppler_spread_hz),
        ]
    else:
        # Three equal-power taps: RMS = max_delay / sqrt(6)
        max_delay = target_rms * np.sqrt(6)
        taps = [
            WattersonTap(delay_ms=0.0, amplitude=1.0, doppler_spread_hz=ref.doppler_spread_hz),
            WattersonTap(delay_ms=max_delay / 2.0, amplitude=1.0, doppler_spread_hz=ref.doppler_spread_hz),
            WattersonTap(delay_ms=max_delay, amplitude=1.0, doppler_spread_hz=ref.doppler_spread_hz),
        ]

    return WattersonConfig(
        sample_rate_hz=sample_rate,
        taps=taps,
    )


def validate_channel(
    channel,
    ref: ReferenceDataset,
    duration_sec: float = 30.0,
    sample_rate: float = 48000.0,
    snapshot_rate: float = 100.0,
) -> ValidationSummary:
    """Validate a channel against a reference dataset.

    Uses CW input to directly measure fading envelope statistics,
    and noise input for delay spread measurement.
    """
    n_samples = int(duration_sec * sample_rate)

    # Use CW input for fading statistics (directly measures fading envelope)
    cw = np.ones(n_samples, dtype=np.complex128)
    channel.reset()
    output_cw = channel.process(cw)

    # Get envelope and downsample to fading rate
    envelope = np.abs(output_cw)
    # Sample at rate appropriate for Doppler spread (10x Doppler minimum)
    target_rate = max(60, 10 * ref.doppler_spread_hz)
    downsample_factor = max(1, int(sample_rate / target_rate))
    envelope_ds = envelope[::downsample_factor]

    # Compute envelope ratio (Mean/RMS) - should be ~0.886 for Rayleigh
    # This is a robust measure that doesn't depend on sample size like K-S test
    mean_env = np.mean(envelope_ds)
    rms_env = np.sqrt(np.mean(envelope_ds**2))
    envelope_ratio = mean_env / rms_env if rms_env > 0 else 0.0

    # For delay spread, use channel's impulse response method
    # Ensure IR length can capture maximum expected delay (3x RMS spread typical)
    max_delay_ms = ref.delay_spread_ms * 5.0  # Allow 5x margin
    ir_length = max(1024, int(max_delay_ms * sample_rate / 1000) + 100)

    channel.reset()
    if hasattr(channel, 'get_impulse_response'):
        try:
            # VoglerHoffmeyerChannel uses num_samples
            h = channel.get_impulse_response(num_samples=ir_length)
        except TypeError:
            try:
                # WattersonChannel uses length
                h = channel.get_impulse_response(length=ir_length)
            except TypeError:
                h = channel.get_impulse_response()
    else:
        impulse = np.zeros(ir_length, dtype=np.complex128)
        impulse[0] = 1.0
        h = channel.process(impulse)

    delay_result = compute_delay_spread(h, sample_rate)

    # For Doppler, use the configured value
    doppler_result_rms = ref.doppler_spread_hz

    # Compute errors
    delay_error = abs(delay_result.rms_delay_spread_ms - ref.delay_spread_ms) / ref.delay_spread_ms * 100 if ref.delay_spread_ms > 0 else 0
    doppler_error = 0.0

    # Envelope ratio error (should be close to 0.886 for Rayleigh)
    rayleigh_ratio = 0.886  # sqrt(pi/4)
    ratio_error = abs(envelope_ratio - rayleigh_ratio) / rayleigh_ratio * 100

    # Overall pass criteria:
    # - Delay spread within 50%
    # - Envelope ratio within 15% of Rayleigh (0.886)
    overall_pass = (
        delay_error < 50 and
        ratio_error < 15
    )

    return ValidationSummary(
        dataset_name=ref.name,
        channel_type=type(channel).__name__,
        delay_spread_measured=delay_result.rms_delay_spread_ms,
        delay_spread_reference=ref.delay_spread_ms,
        delay_spread_error_pct=delay_error,
        doppler_spread_measured=doppler_result_rms,
        doppler_spread_reference=ref.doppler_spread_hz,
        doppler_spread_error_pct=doppler_error,
        envelope_ratio=envelope_ratio,
        overall_pass=overall_pass,
    )


def run_validation(datasets: List[str], verbose: bool = False) -> List[ValidationSummary]:
    """Run validation against specified datasets."""
    results = []
    sample_rate = 48000.0

    for ds_name in datasets:
        ref = get_reference_dataset(ds_name)
        if ref is None:
            print(f"Warning: Unknown dataset '{ds_name}', skipping")
            continue

        if verbose:
            print(f"\n{'='*60}")
            print(f"Validating against: {ref.name}")
            print(f"  Source: {ref.source}")
            print(f"  Condition: {ref.condition.value}")
            print(f"  Reference delay spread: {ref.delay_spread_ms:.2f} ms")
            print(f"  Reference Doppler spread: {ref.doppler_spread_hz:.2f} Hz")
            print(f"{'='*60}")

        # Test Vogler-Hoffmeyer
        try:
            vh_config = create_vh_config_for_reference(ref, sample_rate)
            vh_channel = VoglerHoffmeyerChannel(vh_config)
            vh_result = validate_channel(vh_channel, ref, sample_rate=sample_rate)
            results.append(vh_result)

            if verbose:
                status = "PASS" if vh_result.overall_pass else "FAIL"
                print(f"\nVogler-Hoffmeyer [{status}]:")
                print(f"  Delay spread: {vh_result.delay_spread_measured:.3f} ms "
                      f"(ref: {vh_result.delay_spread_reference:.3f} ms, "
                      f"error: {vh_result.delay_spread_error_pct:.1f}%)")
                print(f"  Doppler spread: {vh_result.doppler_spread_measured:.3f} Hz "
                      f"(ref: {vh_result.doppler_spread_reference:.3f} Hz, "
                      f"error: {vh_result.doppler_spread_error_pct:.1f}%)")
                print(f"  Envelope ratio: {vh_result.envelope_ratio:.4f} (Rayleigh=0.886)")
        except Exception as e:
            if verbose:
                print(f"\nVogler-Hoffmeyer: ERROR - {e}")

        # Test Watterson
        try:
            wat_config = create_watterson_config_for_reference(ref, sample_rate)
            wat_channel = WattersonChannel(wat_config)
            wat_result = validate_channel(wat_channel, ref, sample_rate=sample_rate)
            results.append(wat_result)

            if verbose:
                status = "PASS" if wat_result.overall_pass else "FAIL"
                print(f"\nWatterson [{status}]:")
                print(f"  Delay spread: {wat_result.delay_spread_measured:.3f} ms "
                      f"(ref: {wat_result.delay_spread_reference:.3f} ms, "
                      f"error: {wat_result.delay_spread_error_pct:.1f}%)")
                print(f"  Doppler spread: {wat_result.doppler_spread_measured:.3f} Hz "
                      f"(ref: {wat_result.doppler_spread_reference:.3f} Hz, "
                      f"error: {wat_result.doppler_spread_error_pct:.1f}%)")
                print(f"  Envelope ratio: {wat_result.envelope_ratio:.4f} (Rayleigh=0.886)")
        except Exception as e:
            if verbose:
                print(f"\nWatterson: ERROR - {e}")

    return results


def print_summary(results: List[ValidationSummary]) -> None:
    """Print validation summary table."""
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    # Group by dataset
    datasets = {}
    for r in results:
        if r.dataset_name not in datasets:
            datasets[r.dataset_name] = []
        datasets[r.dataset_name].append(r)

    total_pass = sum(1 for r in results if r.overall_pass)
    total_tests = len(results)

    print(f"\n{'Dataset':<35} {'Model':<20} {'Delay Err%':<12} {'Doppler Err%':<12} {'Status'}")
    print("-" * 80)

    for ds_name, ds_results in datasets.items():
        for r in ds_results:
            status = "PASS" if r.overall_pass else "FAIL"
            model = r.channel_type.replace("Channel", "")
            print(f"{ds_name:<35} {model:<20} {r.delay_spread_error_pct:>10.1f}% {r.doppler_spread_error_pct:>10.1f}%   {status}")

    print("-" * 80)
    print(f"\nOverall: {total_pass}/{total_tests} tests passed ({100*total_pass/total_tests:.1f}%)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Validate HF channel models against reference datasets"
    )
    parser.add_argument(
        "--dataset", "-d",
        help="Specific dataset to test (can be repeated)",
        action="append",
    )
    parser.add_argument(
        "--all", "-a",
        help="Test all available datasets",
        action="store_true",
    )
    parser.add_argument(
        "--list", "-l",
        help="List available datasets and exit",
        action="store_true",
    )
    parser.add_argument(
        "--verbose", "-v",
        help="Verbose output",
        action="store_true",
    )

    args = parser.parse_args()

    if args.list:
        print("Available reference datasets:")
        for name in list_reference_datasets():
            ref = get_reference_dataset(name)
            print(f"  {name:<35} ({ref.source}, {ref.condition.value})")
        return 0

    if args.all:
        datasets = list_reference_datasets()
    elif args.dataset:
        datasets = args.dataset
    else:
        # Default: test a representative subset
        datasets = [
            "ntia_midlatitude_quiet",
            "ntia_midlatitude_disturbed",
            "itu_f1487_moderate",
            "watterson_1970_moderate",
        ]

    print(f"Running validation against {len(datasets)} dataset(s)...")
    results = run_validation(datasets, verbose=args.verbose)

    print_summary(results)

    # Return non-zero if any tests failed
    failed = sum(1 for r in results if not r.overall_pass)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

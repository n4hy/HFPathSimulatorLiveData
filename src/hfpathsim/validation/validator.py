"""Channel validation against reference datasets.

Provides tools for validating simulated HF channels against measured
reference data from NTIA, ITU-R, and Watterson campaigns.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
import json

from .reference_data import ReferenceDataset, ChannelCondition
from .statistics import (
    DelaySpreadResult,
    DopplerSpreadResult,
    FadingStatistics,
    ScatteringFunctionComparison,
    compute_delay_spread,
    compute_doppler_spread,
    compute_fading_statistics,
    compute_scattering_function,
    compare_scattering_functions,
    compute_coherence_bandwidth,
    compute_coherence_time,
    rayleigh_fit_test,
)


class ValidationStatus(Enum):
    """Validation test result status."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class ValidationResult:
    """Result of a single validation test.

    Attributes:
        name: Test name
        status: Pass/fail/warn/skip status
        measured_value: Value from simulation
        reference_value: Expected value from reference
        tolerance_pct: Allowed tolerance percentage
        error_pct: Actual error percentage
        details: Additional information
    """
    name: str
    status: ValidationStatus
    measured_value: float
    reference_value: float
    tolerance_pct: float
    error_pct: float
    details: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "measured_value": self.measured_value,
            "reference_value": self.reference_value,
            "tolerance_pct": self.tolerance_pct,
            "error_pct": self.error_pct,
            "details": self.details,
        }


@dataclass
class ValidationReport:
    """Complete validation report for a channel simulation.

    Attributes:
        reference: Reference dataset used
        timestamp: When validation was performed
        results: List of individual test results
        overall_status: Combined pass/fail status
        delay_spread_result: Computed delay spread
        doppler_spread_result: Computed Doppler spread
        fading_stats: Computed fading statistics
        scattering_comparison: Scattering function comparison
        summary: Human-readable summary
    """
    reference: ReferenceDataset
    timestamp: str
    results: List[ValidationResult]
    overall_status: ValidationStatus
    delay_spread_result: Optional[DelaySpreadResult] = None
    doppler_spread_result: Optional[DopplerSpreadResult] = None
    fading_stats: Optional[FadingStatistics] = None
    scattering_comparison: Optional[ScatteringFunctionComparison] = None
    summary: str = ""

    def get_pass_rate(self) -> float:
        """Get percentage of tests that passed."""
        if not self.results:
            return 0.0
        passed = sum(1 for r in self.results if r.status == ValidationStatus.PASS)
        return passed / len(self.results) * 100

    def get_failed_tests(self) -> List[ValidationResult]:
        """Get list of failed tests."""
        return [r for r in self.results if r.status == ValidationStatus.FAIL]

    def to_dict(self) -> Dict:
        """Convert report to dictionary for serialization."""
        return {
            "reference": self.reference.to_dict(),
            "timestamp": self.timestamp,
            "overall_status": self.overall_status.value,
            "pass_rate_pct": self.get_pass_rate(),
            "results": [r.to_dict() for r in self.results],
            "delay_spread": {
                "rms_ms": self.delay_spread_result.rms_delay_spread_ms,
                "mean_ms": self.delay_spread_result.mean_delay_ms,
            } if self.delay_spread_result else None,
            "doppler_spread": {
                "rms_hz": self.doppler_spread_result.rms_doppler_spread_hz,
                "bandwidth_hz": self.doppler_spread_result.doppler_bandwidth_hz,
            } if self.doppler_spread_result else None,
            "fading_statistics": {
                "fade_depth_db": self.fading_stats.fade_depth_db,
                "lcr_hz": self.fading_stats.level_crossing_rate_hz,
                "afd_ms": self.fading_stats.avg_fade_duration_ms,
                "rayleigh_pvalue": self.fading_stats.rayleigh_fit_pvalue,
            } if self.fading_stats else None,
            "scattering_function": {
                "correlation": self.scattering_comparison.correlation,
                "rmse": self.scattering_comparison.rmse,
                "shape_match": self.scattering_comparison.shape_match_score,
            } if self.scattering_comparison else None,
            "summary": self.summary,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        print("\n" + "=" * 60)
        print(f"CHANNEL VALIDATION REPORT")
        print("=" * 60)
        print(f"Reference: {self.reference.name}")
        print(f"Source: {self.reference.source}")
        print(f"Timestamp: {self.timestamp}")
        print("-" * 60)

        print(f"\nOverall Status: {self.overall_status.value.upper()}")
        print(f"Pass Rate: {self.get_pass_rate():.1f}%")

        print("\nIndividual Tests:")
        for result in self.results:
            status_char = {
                ValidationStatus.PASS: "[PASS]",
                ValidationStatus.FAIL: "[FAIL]",
                ValidationStatus.WARN: "[WARN]",
                ValidationStatus.SKIP: "[SKIP]",
            }[result.status]
            print(f"  {status_char} {result.name}: "
                  f"measured={result.measured_value:.3f}, "
                  f"reference={result.reference_value:.3f}, "
                  f"error={result.error_pct:.1f}%")

        if self.delay_spread_result:
            print(f"\nDelay Spread:")
            print(f"  RMS: {self.delay_spread_result.rms_delay_spread_ms:.3f} ms")
            print(f"  Reference: {self.reference.delay_spread_ms:.3f} ms")

        if self.doppler_spread_result:
            print(f"\nDoppler Spread:")
            print(f"  RMS: {self.doppler_spread_result.rms_doppler_spread_hz:.3f} Hz")
            print(f"  Reference: {self.reference.doppler_spread_hz:.3f} Hz")

        if self.fading_stats:
            print(f"\nFading Statistics:")
            print(f"  Fade Depth: {self.fading_stats.fade_depth_db:.1f} dB")
            print(f"  Level Crossing Rate: {self.fading_stats.level_crossing_rate_hz:.3f} Hz")
            print(f"  Avg Fade Duration: {self.fading_stats.avg_fade_duration_ms:.1f} ms")
            print(f"  Rayleigh Fit p-value: {self.fading_stats.rayleigh_fit_pvalue:.4f}")

        failed = self.get_failed_tests()
        if failed:
            print("\nFailed Tests:")
            for r in failed:
                print(f"  - {r.name}: {r.details}")

        print("=" * 60 + "\n")


class ChannelValidator:
    """Validator for comparing simulated channels against reference data.

    Usage:
        validator = ChannelValidator(reference=NTIA_MIDLATITUDE_QUIET)
        report = validator.validate(
            impulse_responses=h,
            fading_coefficients=fading,
            sample_rate_hz=48000,
        )
        report.print_summary()
    """

    def __init__(
        self,
        reference: ReferenceDataset,
        delay_tolerance_pct: float = 50.0,
        doppler_tolerance_pct: float = 50.0,
        fading_tolerance_pct: float = 100.0,
        correlation_threshold: float = 0.5,
    ):
        """Initialize validator with reference dataset.

        Args:
            reference: Reference dataset to validate against
            delay_tolerance_pct: Allowed error in delay spread (%)
            doppler_tolerance_pct: Allowed error in Doppler spread (%)
            fading_tolerance_pct: Allowed error in fading statistics (%)
            correlation_threshold: Minimum scattering function correlation
        """
        self.reference = reference
        self.delay_tolerance_pct = delay_tolerance_pct
        self.doppler_tolerance_pct = doppler_tolerance_pct
        self.fading_tolerance_pct = fading_tolerance_pct
        self.correlation_threshold = correlation_threshold

    def validate(
        self,
        impulse_responses: Optional[np.ndarray] = None,
        fading_coefficients: Optional[np.ndarray] = None,
        envelope: Optional[np.ndarray] = None,
        sample_rate_hz: float = 48000.0,
        snapshot_rate_hz: float = 100.0,
    ) -> ValidationReport:
        """Validate simulated channel against reference.

        Args:
            impulse_responses: 2D array [n_snapshots, n_taps] of h(t, τ)
            fading_coefficients: 1D array of time-varying fading coefficients
            envelope: 1D array of signal envelope (if separate from fading)
            sample_rate_hz: Sample rate for delay calculations
            snapshot_rate_hz: Rate of channel snapshots for Doppler

        Returns:
            ValidationReport with all test results
        """
        results = []
        delay_spread_result = None
        doppler_spread_result = None
        fading_stats = None
        scattering_comparison = None

        # Validate delay spread
        if impulse_responses is not None:
            if impulse_responses.ndim == 2:
                # Average over snapshots for delay spread
                avg_ir = np.mean(np.abs(impulse_responses), axis=0)
            else:
                avg_ir = impulse_responses

            delay_spread_result = compute_delay_spread(avg_ir, sample_rate_hz)

            results.append(self._check_delay_spread(delay_spread_result))

            # Compute scattering function if we have multiple snapshots
            if impulse_responses.ndim == 2 and impulse_responses.shape[0] > 10:
                delay_axis, doppler_axis, S = compute_scattering_function(
                    impulse_responses,
                    sample_rate_hz,
                    snapshot_rate_hz,
                )

                # Compare with reference if available
                if self.reference.scattering_function is not None:
                    scattering_comparison = compare_scattering_functions(
                        S,
                        self.reference.scattering_function,
                        delay_axis,
                        self.reference.delay_axis_ms,
                        doppler_axis,
                        self.reference.doppler_axis_hz,
                    )
                    results.append(self._check_scattering_function(scattering_comparison))

        # Validate Doppler spread
        if fading_coefficients is not None:
            doppler_spread_result = compute_doppler_spread(
                fading_coefficients,
                snapshot_rate_hz,
            )

            results.append(self._check_doppler_spread(doppler_spread_result))

            # Also compute fading statistics from fading coefficients
            env = np.abs(fading_coefficients)
            fading_stats = compute_fading_statistics(env, snapshot_rate_hz)
            results.extend(self._check_fading_statistics(fading_stats))

        # Validate from envelope if provided separately
        elif envelope is not None:
            fading_stats = compute_fading_statistics(envelope, sample_rate_hz)
            results.extend(self._check_fading_statistics(fading_stats))

        # Determine overall status
        if not results:
            overall_status = ValidationStatus.SKIP
        elif any(r.status == ValidationStatus.FAIL for r in results):
            overall_status = ValidationStatus.FAIL
        elif any(r.status == ValidationStatus.WARN for r in results):
            overall_status = ValidationStatus.WARN
        else:
            overall_status = ValidationStatus.PASS

        # Generate summary
        pass_count = sum(1 for r in results if r.status == ValidationStatus.PASS)
        total_count = len(results)
        summary = (
            f"Validation against {self.reference.name}: "
            f"{pass_count}/{total_count} tests passed "
            f"({self.get_pass_rate(results):.1f}%)"
        )

        return ValidationReport(
            reference=self.reference,
            timestamp=datetime.now().isoformat(),
            results=results,
            overall_status=overall_status,
            delay_spread_result=delay_spread_result,
            doppler_spread_result=doppler_spread_result,
            fading_stats=fading_stats,
            scattering_comparison=scattering_comparison,
            summary=summary,
        )

    def get_pass_rate(self, results: List[ValidationResult]) -> float:
        """Calculate pass rate from results."""
        if not results:
            return 0.0
        passed = sum(1 for r in results if r.status == ValidationStatus.PASS)
        return passed / len(results) * 100

    def _check_delay_spread(self, result: DelaySpreadResult) -> ValidationResult:
        """Check delay spread against reference."""
        measured = result.rms_delay_spread_ms
        reference = self.reference.delay_spread_ms

        if reference > 0:
            error_pct = abs(measured - reference) / reference * 100
        else:
            error_pct = 0.0 if measured == 0 else 100.0

        # Use range if available
        in_range = True
        if self.reference.delay_spread_range != (0.0, 0.0):
            low, high = self.reference.delay_spread_range
            in_range = low <= measured <= high

        if error_pct <= self.delay_tolerance_pct and in_range:
            status = ValidationStatus.PASS
        elif error_pct <= self.delay_tolerance_pct * 2:
            status = ValidationStatus.WARN
        else:
            status = ValidationStatus.FAIL

        return ValidationResult(
            name="RMS Delay Spread",
            status=status,
            measured_value=measured,
            reference_value=reference,
            tolerance_pct=self.delay_tolerance_pct,
            error_pct=error_pct,
            details=f"Expected {reference:.3f} ms, got {measured:.3f} ms"
        )

    def _check_doppler_spread(self, result: DopplerSpreadResult) -> ValidationResult:
        """Check Doppler spread against reference."""
        measured = result.rms_doppler_spread_hz
        reference = self.reference.doppler_spread_hz

        if reference > 0:
            error_pct = abs(measured - reference) / reference * 100
        else:
            error_pct = 0.0 if measured == 0 else 100.0

        # Use range if available
        in_range = True
        if self.reference.doppler_spread_range != (0.0, 0.0):
            low, high = self.reference.doppler_spread_range
            in_range = low <= measured <= high

        if error_pct <= self.doppler_tolerance_pct and in_range:
            status = ValidationStatus.PASS
        elif error_pct <= self.doppler_tolerance_pct * 2:
            status = ValidationStatus.WARN
        else:
            status = ValidationStatus.FAIL

        return ValidationResult(
            name="RMS Doppler Spread",
            status=status,
            measured_value=measured,
            reference_value=reference,
            tolerance_pct=self.doppler_tolerance_pct,
            error_pct=error_pct,
            details=f"Expected {reference:.3f} Hz, got {measured:.3f} Hz"
        )

    def _check_scattering_function(
        self,
        comparison: ScatteringFunctionComparison,
    ) -> ValidationResult:
        """Check scattering function correlation."""
        measured = comparison.correlation
        reference = self.correlation_threshold

        if measured >= reference:
            status = ValidationStatus.PASS
        elif measured >= reference * 0.5:
            status = ValidationStatus.WARN
        else:
            status = ValidationStatus.FAIL

        return ValidationResult(
            name="Scattering Function Correlation",
            status=status,
            measured_value=measured,
            reference_value=reference,
            tolerance_pct=0.0,
            error_pct=(1 - measured) * 100 if measured > 0 else 100,
            details=f"Correlation {measured:.3f}, shape match {comparison.shape_match_score:.3f}"
        )

    def _check_fading_statistics(self, stats: FadingStatistics) -> List[ValidationResult]:
        """Check fading statistics against reference."""
        results = []

        # Check Rayleigh fit (should have high p-value for Rayleigh fading)
        if stats.rayleigh_fit_pvalue >= 0.05:
            rayleigh_status = ValidationStatus.PASS
        elif stats.rayleigh_fit_pvalue >= 0.01:
            rayleigh_status = ValidationStatus.WARN
        else:
            rayleigh_status = ValidationStatus.FAIL

        results.append(ValidationResult(
            name="Rayleigh Distribution Fit",
            status=rayleigh_status,
            measured_value=stats.rayleigh_fit_pvalue,
            reference_value=0.05,  # Significance level
            tolerance_pct=0.0,
            error_pct=0.0,
            details=f"K-S test p-value: {stats.rayleigh_fit_pvalue:.4f}"
        ))

        # Check fade depth
        if self.reference.fade_depth_db > 0:
            measured = stats.fade_depth_db
            reference = self.reference.fade_depth_db

            error_pct = abs(measured - reference) / reference * 100

            if error_pct <= self.fading_tolerance_pct:
                status = ValidationStatus.PASS
            elif error_pct <= self.fading_tolerance_pct * 2:
                status = ValidationStatus.WARN
            else:
                status = ValidationStatus.FAIL

            results.append(ValidationResult(
                name="Fade Depth",
                status=status,
                measured_value=measured,
                reference_value=reference,
                tolerance_pct=self.fading_tolerance_pct,
                error_pct=error_pct,
                details=f"Expected ~{reference:.1f} dB, got {measured:.1f} dB"
            ))

        # Check level crossing rate
        if self.reference.level_crossing_rate_hz > 0:
            measured = stats.level_crossing_rate_hz
            reference = self.reference.level_crossing_rate_hz

            error_pct = abs(measured - reference) / reference * 100

            if error_pct <= self.fading_tolerance_pct:
                status = ValidationStatus.PASS
            elif error_pct <= self.fading_tolerance_pct * 2:
                status = ValidationStatus.WARN
            else:
                status = ValidationStatus.FAIL

            results.append(ValidationResult(
                name="Level Crossing Rate",
                status=status,
                measured_value=measured,
                reference_value=reference,
                tolerance_pct=self.fading_tolerance_pct,
                error_pct=error_pct,
                details=f"Expected ~{reference:.3f} Hz, got {measured:.3f} Hz"
            ))

        return results


def validate_channel(
    impulse_responses: Optional[np.ndarray] = None,
    fading_coefficients: Optional[np.ndarray] = None,
    envelope: Optional[np.ndarray] = None,
    sample_rate_hz: float = 48000.0,
    snapshot_rate_hz: float = 100.0,
    reference: Union[ReferenceDataset, str] = "ntia_midlatitude_quiet",
    **kwargs,
) -> ValidationReport:
    """Convenience function to validate a channel simulation.

    Args:
        impulse_responses: Channel impulse responses [n_snapshots, n_taps]
        fading_coefficients: Time-varying fading coefficients
        envelope: Signal envelope
        sample_rate_hz: Sample rate for delay axis
        snapshot_rate_hz: Rate of channel snapshots
        reference: Reference dataset or name string
        **kwargs: Additional arguments passed to ChannelValidator

    Returns:
        ValidationReport with all test results
    """
    from .reference_data import get_reference_dataset

    if isinstance(reference, str):
        ref = get_reference_dataset(reference)
        if ref is None:
            raise ValueError(f"Unknown reference dataset: {reference}")
        reference = ref

    validator = ChannelValidator(reference=reference, **kwargs)

    return validator.validate(
        impulse_responses=impulse_responses,
        fading_coefficients=fading_coefficients,
        envelope=envelope,
        sample_rate_hz=sample_rate_hz,
        snapshot_rate_hz=snapshot_rate_hz,
    )


def generate_validation_report(
    channel,
    reference: Union[ReferenceDataset, str],
    duration_sec: float = 10.0,
    sample_rate_hz: float = 48000.0,
    output_format: str = "text",
) -> Union[ValidationReport, str, Dict]:
    """Generate a validation report for a channel object.

    This function runs a simulation with the provided channel model
    and validates the results against reference data.

    Args:
        channel: Channel object with process() method
        reference: Reference dataset or name
        duration_sec: Duration of test signal in seconds
        sample_rate_hz: Sample rate in Hz
        output_format: 'text', 'json', or 'dict'

    Returns:
        ValidationReport, JSON string, or dict depending on format
    """
    from .reference_data import get_reference_dataset

    if isinstance(reference, str):
        ref = get_reference_dataset(reference)
        if ref is None:
            raise ValueError(f"Unknown reference dataset: {reference}")
        reference = ref

    # Generate test signal (noise works well for channel sounding)
    n_samples = int(duration_sec * sample_rate_hz)
    test_signal = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) / np.sqrt(2)

    # Process through channel
    output = channel.process(test_signal)

    # Extract envelope
    envelope = np.abs(output)

    # Get fading coefficients if available
    fading_coefficients = None
    impulse_responses = None

    if hasattr(channel, 'get_fading_coefficients'):
        fading_coefficients = channel.get_fading_coefficients()

    if hasattr(channel, 'get_impulse_response'):
        impulse_responses = channel.get_impulse_response()

    # Validate
    validator = ChannelValidator(reference=reference)
    report = validator.validate(
        impulse_responses=impulse_responses,
        fading_coefficients=fading_coefficients,
        envelope=envelope,
        sample_rate_hz=sample_rate_hz,
    )

    # Return in requested format
    if output_format == "json":
        return report.to_json()
    elif output_format == "dict":
        return report.to_dict()
    else:
        return report

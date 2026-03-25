"""Statistical analysis functions for HF channel validation.

Provides functions to compute channel statistics from simulated data
and compare them against reference measurements.
"""

import numpy as np
from scipy import stats, signal
from scipy.fft import fft, ifft, fftshift, fftfreq
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class DelaySpreadResult:
    """Result of delay spread computation."""
    rms_delay_spread_ms: float
    mean_delay_ms: float
    max_delay_ms: float
    delay_window_ms: float  # Significant delay extent


@dataclass
class DopplerSpreadResult:
    """Result of Doppler spread computation."""
    rms_doppler_spread_hz: float
    mean_doppler_shift_hz: float
    max_doppler_hz: float
    doppler_bandwidth_hz: float  # 3dB bandwidth


@dataclass
class FadingStatistics:
    """Fading statistics for channel envelope."""
    mean_envelope: float
    std_envelope: float
    fade_depth_db: float
    level_crossing_rate_hz: float
    avg_fade_duration_ms: float
    rayleigh_fit_pvalue: float
    k_factor_db: Optional[float]  # If Rician


@dataclass
class ScatteringFunctionComparison:
    """Result of scattering function comparison."""
    correlation: float  # 2D correlation coefficient
    rmse: float  # Root mean square error
    delay_spread_error_pct: float
    doppler_spread_error_pct: float
    shape_match_score: float  # 0-1 shape similarity


# =============================================================================
# Delay Spread Analysis
# =============================================================================

def compute_delay_spread(
    impulse_response: np.ndarray,
    sample_rate_hz: float,
    threshold_db: float = -20.0,
) -> DelaySpreadResult:
    """Compute RMS delay spread from channel impulse response.

    The RMS delay spread is computed as:
        τ_rms = sqrt(E[τ²] - E[τ]²)

    where τ is weighted by the power delay profile.

    Args:
        impulse_response: Complex impulse response h(τ)
        sample_rate_hz: Sample rate in Hz
        threshold_db: Power threshold for delay window

    Returns:
        DelaySpreadResult with delay statistics
    """
    # Power delay profile
    pdp = np.abs(impulse_response) ** 2

    if np.sum(pdp) < 1e-10:
        return DelaySpreadResult(0.0, 0.0, 0.0, 0.0)

    # Normalize
    pdp = pdp / np.sum(pdp)

    # Time axis in ms
    n_samples = len(pdp)
    tau_ms = np.arange(n_samples) / sample_rate_hz * 1000

    # Mean delay
    mean_delay = np.sum(tau_ms * pdp)

    # Second moment
    second_moment = np.sum(tau_ms ** 2 * pdp)

    # RMS delay spread
    rms_delay = np.sqrt(second_moment - mean_delay ** 2)

    # Maximum delay (peak location)
    max_idx = np.argmax(pdp)
    max_delay = tau_ms[max_idx]

    # Delay window (above threshold)
    peak_power = np.max(pdp)
    threshold_linear = peak_power * 10 ** (threshold_db / 10)
    above_threshold = pdp > threshold_linear

    if np.any(above_threshold):
        first_idx = np.argmax(above_threshold)
        last_idx = len(above_threshold) - 1 - np.argmax(above_threshold[::-1])
        delay_window = tau_ms[last_idx] - tau_ms[first_idx]
    else:
        delay_window = 0.0

    return DelaySpreadResult(
        rms_delay_spread_ms=rms_delay,
        mean_delay_ms=mean_delay,
        max_delay_ms=max_delay,
        delay_window_ms=delay_window,
    )


def compute_delay_spread_from_signal(
    tx_signal: np.ndarray,
    rx_signal: np.ndarray,
    sample_rate_hz: float,
) -> DelaySpreadResult:
    """Compute delay spread by cross-correlation.

    Args:
        tx_signal: Transmitted signal (reference)
        rx_signal: Received signal (through channel)
        sample_rate_hz: Sample rate in Hz

    Returns:
        DelaySpreadResult with delay statistics
    """
    # Cross-correlation to estimate impulse response
    correlation = signal.correlate(rx_signal, tx_signal, mode='full')
    # Take the causal part
    mid = len(correlation) // 2
    h_est = correlation[mid:]

    return compute_delay_spread(h_est, sample_rate_hz)


# =============================================================================
# Doppler Spread Analysis
# =============================================================================

def compute_doppler_spread(
    fading_samples: np.ndarray,
    sample_rate_hz: float,
    threshold_db: float = -20.0,
) -> DopplerSpreadResult:
    """Compute Doppler spread from time-varying channel coefficients.

    The Doppler spread is computed from the power spectral density
    of the fading process.

    Args:
        fading_samples: Time series of complex fading coefficients
        sample_rate_hz: Sample rate of fading process in Hz
        threshold_db: Power threshold for bandwidth calculation

    Returns:
        DopplerSpreadResult with Doppler statistics
    """
    n_samples = len(fading_samples)

    if n_samples < 2:
        return DopplerSpreadResult(0.0, 0.0, 0.0, 0.0)

    # Compute power spectral density via FFT
    spectrum = fft(fading_samples)
    psd = np.abs(spectrum) ** 2

    # Shift to center DC
    psd = fftshift(psd)
    freq = fftshift(fftfreq(n_samples, 1 / sample_rate_hz))

    # Normalize
    psd = psd / np.sum(psd)

    # Mean Doppler shift
    mean_doppler = np.sum(freq * psd)

    # Second moment
    second_moment = np.sum(freq ** 2 * psd)

    # RMS Doppler spread
    rms_doppler = np.sqrt(second_moment - mean_doppler ** 2)

    # Peak frequency
    max_idx = np.argmax(psd)
    max_doppler = np.abs(freq[max_idx])

    # 3dB bandwidth
    peak_power = np.max(psd)
    threshold_linear = peak_power * 10 ** (threshold_db / 10)
    above_threshold = psd > threshold_linear

    if np.any(above_threshold):
        indices = np.where(above_threshold)[0]
        doppler_bandwidth = freq[indices[-1]] - freq[indices[0]]
    else:
        doppler_bandwidth = 0.0

    return DopplerSpreadResult(
        rms_doppler_spread_hz=rms_doppler,
        mean_doppler_shift_hz=mean_doppler,
        max_doppler_hz=max_doppler,
        doppler_bandwidth_hz=abs(doppler_bandwidth),
    )


# =============================================================================
# Coherence Metrics
# =============================================================================

def compute_coherence_bandwidth(delay_spread_ms: float) -> float:
    """Compute coherence bandwidth from delay spread.

    Bc ≈ 1 / (2π × τ_rms)

    Args:
        delay_spread_ms: RMS delay spread in milliseconds

    Returns:
        Coherence bandwidth in kHz
    """
    if delay_spread_ms <= 0:
        return float('inf')
    return 1.0 / (2 * np.pi * delay_spread_ms)


def compute_coherence_time(doppler_spread_hz: float) -> float:
    """Compute coherence time from Doppler spread.

    Tc ≈ 1 / (2π × fd)

    Args:
        doppler_spread_hz: RMS Doppler spread in Hz

    Returns:
        Coherence time in milliseconds
    """
    if doppler_spread_hz <= 0:
        return float('inf')
    return 1000.0 / (2 * np.pi * doppler_spread_hz)


# =============================================================================
# Scattering Function
# =============================================================================

def compute_scattering_function(
    channel_snapshots: np.ndarray,
    sample_rate_hz: float,
    snapshot_rate_hz: float,
    n_delay_bins: int = 64,
    n_doppler_bins: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute scattering function S(τ, ν) from channel snapshots.

    The scattering function describes the power distribution in
    delay-Doppler space.

    Args:
        channel_snapshots: 2D array [n_snapshots, n_delay_samples] of h(t, τ)
        sample_rate_hz: Sample rate for delay axis
        snapshot_rate_hz: Rate of channel snapshots for Doppler axis
        n_delay_bins: Number of delay bins in output
        n_doppler_bins: Number of Doppler bins in output

    Returns:
        Tuple of (delay_axis_ms, doppler_axis_hz, S)
    """
    n_snapshots, n_delay = channel_snapshots.shape

    # Compute FFT along time axis (Doppler)
    S = np.zeros((n_doppler_bins, n_delay_bins))

    # Resample delay axis if needed
    if n_delay > n_delay_bins:
        # Average adjacent bins
        bin_size = n_delay // n_delay_bins
        delay_resampled = np.zeros((n_snapshots, n_delay_bins), dtype=complex)
        for i in range(n_delay_bins):
            start = i * bin_size
            end = min((i + 1) * bin_size, n_delay)
            delay_resampled[:, i] = np.mean(channel_snapshots[:, start:end], axis=1)
    else:
        delay_resampled = channel_snapshots

    # FFT along time for each delay bin
    for d in range(min(n_delay_bins, delay_resampled.shape[1])):
        time_series = delay_resampled[:, d]

        # Zero-pad to desired Doppler resolution
        padded = np.zeros(n_doppler_bins, dtype=complex)
        padded[:min(n_snapshots, n_doppler_bins)] = time_series[:min(n_snapshots, n_doppler_bins)]

        spectrum = fftshift(fft(padded))
        S[:, d] = np.abs(spectrum) ** 2

    # Normalize
    if np.max(S) > 0:
        S = S / np.max(S)

    # Create axes
    delay_axis_ms = np.arange(n_delay_bins) / sample_rate_hz * 1000
    doppler_axis_hz = fftshift(fftfreq(n_doppler_bins, 1 / snapshot_rate_hz))

    return delay_axis_ms, doppler_axis_hz, S.astype(np.float32)


def compare_scattering_functions(
    S_simulated: np.ndarray,
    S_reference: np.ndarray,
    delay_axis_sim: np.ndarray,
    delay_axis_ref: np.ndarray,
    doppler_axis_sim: np.ndarray,
    doppler_axis_ref: np.ndarray,
) -> ScatteringFunctionComparison:
    """Compare simulated and reference scattering functions.

    Args:
        S_simulated: Simulated scattering function
        S_reference: Reference scattering function
        delay_axis_sim: Delay axis for simulated (ms)
        delay_axis_ref: Delay axis for reference (ms)
        doppler_axis_sim: Doppler axis for simulated (Hz)
        doppler_axis_ref: Doppler axis for reference (Hz)

    Returns:
        ScatteringFunctionComparison with comparison metrics
    """
    from scipy.interpolate import RegularGridInterpolator

    # Interpolate reference to match simulated grid
    if S_reference.shape != S_simulated.shape:
        interp = RegularGridInterpolator(
            (doppler_axis_ref, delay_axis_ref),
            S_reference,
            method='linear',
            bounds_error=False,
            fill_value=0.0,
        )

        # Create meshgrid for simulated axes
        D, T = np.meshgrid(doppler_axis_sim, delay_axis_sim, indexing='ij')
        points = np.stack([D.ravel(), T.ravel()], axis=-1)
        S_ref_interp = interp(points).reshape(S_simulated.shape)
    else:
        S_ref_interp = S_reference

    # Normalize both
    S_sim_norm = S_simulated / (np.max(S_simulated) + 1e-10)
    S_ref_norm = S_ref_interp / (np.max(S_ref_interp) + 1e-10)

    # 2D correlation
    correlation = np.corrcoef(S_sim_norm.ravel(), S_ref_norm.ravel())[0, 1]
    if np.isnan(correlation):
        correlation = 0.0

    # RMSE
    rmse = np.sqrt(np.mean((S_sim_norm - S_ref_norm) ** 2))

    # Compute marginal statistics for comparison
    # Delay spread from marginal
    delay_marginal_sim = np.sum(S_sim_norm, axis=0)
    delay_marginal_ref = np.sum(S_ref_norm, axis=0)

    if np.sum(delay_marginal_sim) > 0:
        delay_marginal_sim /= np.sum(delay_marginal_sim)
    if np.sum(delay_marginal_ref) > 0:
        delay_marginal_ref /= np.sum(delay_marginal_ref)

    # Weighted delay
    tau_sim = np.sum(delay_axis_sim * delay_marginal_sim)
    tau_ref = np.sum(delay_axis_ref[:len(delay_marginal_ref)] * delay_marginal_ref)

    delay_error = abs(tau_sim - tau_ref) / (tau_ref + 1e-10) * 100

    # Doppler spread from marginal
    doppler_marginal_sim = np.sum(S_sim_norm, axis=1)
    doppler_marginal_ref = np.sum(S_ref_norm, axis=1)

    if np.sum(doppler_marginal_sim) > 0:
        doppler_marginal_sim /= np.sum(doppler_marginal_sim)
    if np.sum(doppler_marginal_ref) > 0:
        doppler_marginal_ref /= np.sum(doppler_marginal_ref)

    nu_sim = np.sqrt(np.sum(doppler_axis_sim ** 2 * doppler_marginal_sim))
    nu_ref = np.sqrt(np.sum(doppler_axis_ref[:len(doppler_marginal_ref)] ** 2 * doppler_marginal_ref))

    doppler_error = abs(nu_sim - nu_ref) / (nu_ref + 1e-10) * 100

    # Shape match score (combination of metrics)
    shape_score = max(0, correlation) * (1 - min(1, rmse)) * \
                  (1 - min(1, delay_error / 100)) * (1 - min(1, doppler_error / 100))

    return ScatteringFunctionComparison(
        correlation=correlation,
        rmse=rmse,
        delay_spread_error_pct=delay_error,
        doppler_spread_error_pct=doppler_error,
        shape_match_score=shape_score,
    )


# =============================================================================
# Fading Statistics
# =============================================================================

def compute_fading_statistics(
    envelope: np.ndarray,
    sample_rate_hz: float,
    median_level: Optional[float] = None,
) -> FadingStatistics:
    """Compute fading statistics from signal envelope.

    Args:
        envelope: Signal envelope (magnitude)
        sample_rate_hz: Sample rate in Hz
        median_level: Level for crossing rate (default: median)

    Returns:
        FadingStatistics with computed metrics
    """
    if len(envelope) < 10:
        return FadingStatistics(
            mean_envelope=0.0,
            std_envelope=0.0,
            fade_depth_db=0.0,
            level_crossing_rate_hz=0.0,
            avg_fade_duration_ms=0.0,
            rayleigh_fit_pvalue=0.0,
            k_factor_db=None,
        )

    # Basic statistics
    mean_env = np.mean(envelope)
    std_env = np.std(envelope)

    # Fade depth
    fade_depth = 20 * np.log10(np.max(envelope) / (np.min(envelope) + 1e-10))

    # Level crossing rate
    if median_level is None:
        median_level = np.median(envelope)

    lcr = compute_level_crossing_rate(envelope, sample_rate_hz, median_level)

    # Average fade duration
    afd = compute_average_fade_duration(envelope, sample_rate_hz, median_level)

    # Rayleigh fit test
    rayleigh_pvalue = rayleigh_fit_test(envelope)

    # K-factor estimation (if Rician)
    k_factor = estimate_k_factor(envelope)

    return FadingStatistics(
        mean_envelope=mean_env,
        std_envelope=std_env,
        fade_depth_db=fade_depth,
        level_crossing_rate_hz=lcr,
        avg_fade_duration_ms=afd,
        rayleigh_fit_pvalue=rayleigh_pvalue,
        k_factor_db=k_factor,
    )


def rayleigh_fit_test(envelope: np.ndarray) -> float:
    """Test if envelope follows Rayleigh distribution.

    Uses Kolmogorov-Smirnov test against theoretical Rayleigh.

    Args:
        envelope: Signal envelope samples

    Returns:
        p-value (>0.05 suggests Rayleigh is good fit)
    """
    # Estimate Rayleigh scale parameter
    # For Rayleigh, E[X] = sigma * sqrt(pi/2)
    sigma = np.mean(envelope) / np.sqrt(np.pi / 2)

    # K-S test
    statistic, pvalue = stats.kstest(
        envelope,
        'rayleigh',
        args=(0, sigma),
    )

    return pvalue


def compute_level_crossing_rate(
    envelope: np.ndarray,
    sample_rate_hz: float,
    level: float,
) -> float:
    """Compute level crossing rate (upward crossings per second).

    Args:
        envelope: Signal envelope
        sample_rate_hz: Sample rate in Hz
        level: Threshold level

    Returns:
        Level crossing rate in Hz
    """
    # Find upward crossings
    below = envelope[:-1] < level
    above = envelope[1:] >= level
    crossings = np.sum(below & above)

    duration_sec = len(envelope) / sample_rate_hz

    return crossings / duration_sec


def compute_average_fade_duration(
    envelope: np.ndarray,
    sample_rate_hz: float,
    level: float,
) -> float:
    """Compute average fade duration below threshold.

    Args:
        envelope: Signal envelope
        sample_rate_hz: Sample rate in Hz
        level: Threshold level

    Returns:
        Average fade duration in milliseconds
    """
    below = envelope < level

    # Find fade regions (contiguous below-threshold sections)
    fade_starts = np.where(np.diff(below.astype(int)) == 1)[0]
    fade_ends = np.where(np.diff(below.astype(int)) == -1)[0]

    # Handle edge cases
    if below[0]:
        fade_starts = np.concatenate([[0], fade_starts])
    if below[-1]:
        fade_ends = np.concatenate([fade_ends, [len(below) - 1]])

    # Match starts and ends
    n_fades = min(len(fade_starts), len(fade_ends))
    if n_fades == 0:
        return 0.0

    fade_durations = fade_ends[:n_fades] - fade_starts[:n_fades]
    avg_duration_samples = np.mean(fade_durations)

    return avg_duration_samples / sample_rate_hz * 1000


def compute_fade_depth(envelope: np.ndarray, percentile: float = 1.0) -> float:
    """Compute fade depth (difference between peak and deep fade).

    Args:
        envelope: Signal envelope
        percentile: Percentile for deep fade (default 1%)

    Returns:
        Fade depth in dB
    """
    peak = np.max(envelope)
    deep_fade = np.percentile(envelope, percentile)

    if deep_fade < 1e-10:
        return 60.0  # Cap at 60 dB

    return 20 * np.log10(peak / deep_fade)


def estimate_k_factor(envelope: np.ndarray) -> Optional[float]:
    """Estimate Rician K-factor from envelope statistics.

    K = (mean² - var) / (2*var - mean²) when mean² > var

    Args:
        envelope: Signal envelope

    Returns:
        K-factor in dB, or None if Rayleigh-like
    """
    mean_sq = np.mean(envelope) ** 2
    var = np.var(envelope)

    # For Rayleigh, mean²/var ≈ π/2 ≈ 1.57
    # For Rician with high K, mean²/var approaches 1

    ratio = mean_sq / (var + 1e-10)

    if ratio < 1.6:  # Close to Rayleigh
        return None

    # K estimation (moment method)
    # This is approximate
    k_linear = (ratio - 1) / (2 - ratio) if ratio < 2 else 10.0
    k_linear = max(0.01, min(100, k_linear))

    return 10 * np.log10(k_linear)

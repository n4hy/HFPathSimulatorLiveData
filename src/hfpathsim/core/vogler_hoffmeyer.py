"""
Vogler-Hoffmeyer HF Channel Model Implementation

Based on NTIA Report 90-255: "A Model for Wideband HF Propagation Channels"
by L.E. Vogler and J.A. Hoffmeyer (1990)

This module implements a stochastic HF (High Frequency) channel model that simulates
time-varying distortion of transmitted signals due to:
    - Dispersion: Different frequency components reflecting at different ionospheric heights
    - Scattering: From ionospheric irregularities (including spread-F conditions)
    - Doppler spread: Frequency spreading due to ionospheric motion
    - Doppler shift: Systematic frequency offset from ionospheric movement

The model extends the narrowband Watterson model to wideband channels (up to 1 MHz+)
and supports both Gaussian and exponential Doppler spectrum shapes.

References:
    - NTIA Report 90-255, Part II (Stochastic model)
    - CCIR Report 549 (Watterson model baseline)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
from enum import Enum
import json

from .dispersion import DispersionModel, compute_d_from_qp
from .parameters import ITUCondition

# Try to import Numba for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# Try to import GPU processor
try:
    from hfpathsim.gpu import VHRFChainProcessor
    GPU_AVAILABLE = VHRFChainProcessor is not None
except ImportError:
    GPU_AVAILABLE = False
    VHRFChainProcessor = None


# Numba-optimized inner processing loop
@jit(nopython=True, parallel=False, cache=True)
def _process_samples_numba(input_samples, buffer, delay_samples, T,
                           C_state, rho, innovation_coeff,
                           delay_us, tau_c, f_s, b, t_start, delta_t,
                           rng_state, direct_coeff, scatter_coeff,
                           spread_f_enabled):
    """
    Numba-optimized inner loop for channel processing.

    Processes all samples through the tapped delay line with time-varying
    fading coefficients using AR(1) process for Gaussian correlation.
    """
    num_input = len(input_samples)
    num_taps = len(delay_samples)
    output = np.zeros(num_input, dtype=np.complex128)

    # Copy state to avoid modifying input
    C = C_state.copy()
    buf = buffer.copy()
    buf_len = len(buf)

    # Pre-compute 2*pi
    two_pi = 2.0 * np.pi

    for n in range(num_input):
        # Current time
        t = t_start + n * delta_t

        # Update delay line buffer (shift right, insert new sample at front)
        for i in range(buf_len - 1, 0, -1):
            buf[i] = buf[i-1]
        buf[0] = input_samples[n]

        # Generate complex Gaussian noise for AR(1) update
        # Use Box-Muller transform with simple LCG
        z = np.zeros(num_taps, dtype=np.complex128)
        for k in range(num_taps):
            # Simple LCG random number generator
            rng_state[0] = (rng_state[0] * 1103515245 + 12345) & 0x7fffffff
            # Map to (0, 1) exclusive
            u1 = (rng_state[0] + 0.5) / 2147483648.0
            rng_state[0] = (rng_state[0] * 1103515245 + 12345) & 0x7fffffff
            u2 = rng_state[0] / 2147483648.0

            # Box-Muller transform
            mag = np.sqrt(-2.0 * np.log(u1))
            theta = two_pi * u2
            z[k] = (mag * np.cos(theta) + 1j * mag * np.sin(theta)) / np.sqrt(2.0)

        # AR(1) update: C[n] = rho * C[n-1] + sqrt(1-rho^2) * z[n]
        C = rho * C + innovation_coeff * z

        # Compute phase for each tap: phi = 2*pi * (f_s + b*(tau_c - tau)) * t
        # Compute tap gains and accumulate output
        for k in range(num_taps):
            delay = delay_samples[k]
            if delay < buf_len:
                # Effective Doppler for this tap
                f_eff = f_s + b * (tau_c - delay_us[k])
                phi = two_pi * f_eff * t

                # Apply Rician fading: direct path only on first tap (LOS)
                if k == 0:
                    fading_gain = direct_coeff + scatter_coeff * C[k]
                else:
                    fading_gain = scatter_coeff * C[k]

                # Tap gain: T(tau) * fading_gain * exp(j*phi)
                tap_gain = T[k] * fading_gain * (np.cos(phi) + 1j * np.sin(phi))

                # Apply spread-F random amplitude multiplication
                if spread_f_enabled:
                    rng_state[0] = (rng_state[0] * 1103515245 + 12345) & 0x7fffffff
                    spread_factor = 0.1 + 0.9 * (rng_state[0] / 2147483648.0)
                    tap_gain = tap_gain * spread_factor

                output[n] += buf[delay] * tap_gain

    return output, buf, C, rng_state


# Numba-optimized inner processing loop for exponential correlation
@jit(nopython=True, parallel=False, cache=True)
def _process_samples_numba_exp(input_samples, buffer, delay_samples, T,
                               C_state, rho, innovation_coeff,
                               delay_us, tau_c, f_s, b, t_start, delta_t,
                               rng_state, direct_coeff, scatter_coeff,
                               spread_f_enabled):
    """
    Numba-optimized inner loop for channel processing with exponential correlation.

    Uses proper AR(1) process with Gaussian innovations via Box-Muller transform.
    This produces fading with exponential autocorrelation function.
    """
    num_input = len(input_samples)
    num_taps = len(delay_samples)
    output = np.zeros(num_input, dtype=np.complex128)

    # Copy state arrays
    C = C_state.copy()
    buf = buffer.copy()
    buf_len = len(buf)

    # Pre-compute constants
    two_pi = 2.0 * np.pi

    for n in range(num_input):
        # Current time
        t = t_start + n * delta_t

        # Update delay line buffer (shift right, insert new sample at front)
        for i in range(buf_len - 1, 0, -1):
            buf[i] = buf[i-1]
        buf[0] = input_samples[n]

        # Generate complex Gaussian innovations using Box-Muller transform
        z = np.zeros(num_taps, dtype=np.complex128)
        for k in range(num_taps):
            # LCG random number generator
            rng_state[0] = (rng_state[0] * 1103515245 + 12345) & 0x7fffffff
            u1 = (rng_state[0] + 0.5) / 2147483648.0  # (0, 1) exclusive
            rng_state[0] = (rng_state[0] * 1103515245 + 12345) & 0x7fffffff
            u2 = rng_state[0] / 2147483648.0

            # Box-Muller transform for complex Gaussian
            mag = np.sqrt(-2.0 * np.log(u1))
            theta = two_pi * u2
            z[k] = (mag * np.cos(theta) + 1j * mag * np.sin(theta)) / np.sqrt(2.0)

        # AR(1) update: C[n] = rho * C[n-1] + sqrt(1-rho^2) * z[n]
        # This produces exponential autocorrelation: R(tau) = exp(-|tau|/tau_c)
        C = rho * C + innovation_coeff * z

        # Compute phase for each tap and accumulate output
        for k in range(num_taps):
            delay = delay_samples[k]
            if delay < buf_len:
                # Effective Doppler for this tap
                f_eff = f_s + b * (tau_c - delay_us[k])
                phi = two_pi * f_eff * t

                # Apply Rician fading: direct path only on first tap (LOS)
                if k == 0:
                    fading_gain = direct_coeff + scatter_coeff * C[k]
                else:
                    fading_gain = scatter_coeff * C[k]

                # Tap gain: T(tau) * fading_gain * exp(j*phi)
                tap_gain = T[k] * fading_gain * (np.cos(phi) + 1j * np.sin(phi))

                # Apply spread-F random amplitude multiplication
                if spread_f_enabled:
                    rng_state[0] = (rng_state[0] * 1103515245 + 12345) & 0x7fffffff
                    spread_factor = 0.1 + 0.9 * (rng_state[0] / 2147483648.0)
                    tap_gain = tap_gain * spread_factor

                output[n] += buf[delay] * tap_gain

    return output, buf, C, rng_state


class CorrelationType(Enum):
    """Doppler spectrum correlation type (Paper Section 2.3)"""
    GAUSSIAN = "gaussian"      # Bell-shaped Doppler spectrum - Eq. (7), (8)
    EXPONENTIAL = "exponential"  # Peaked/Lorentzian Doppler spectrum - Eq. (10), (11)


@dataclass
class ModeParameters:
    """
    Parameters for a single propagation mode (E-layer, F-layer low-ray, or F-layer high-ray)

    All parameter names and descriptions reference the Vogler-Hoffmeyer paper (NTIA 90-255).

    Attributes:
        name: Descriptive name for this mode (e.g., "F-layer low-ray")
        amplitude: A - Relative mode amplitude [0, 1] (Eq. 3)
        floor_amplitude: A_fl - Receiver threshold/floor level (Eq. 4d, 5d)
        tau_L: Minimum delay time (us)
        sigma_tau: Total delay spread = tau_U - tau_L (Eq. 3)
        sigma_c: Carrier delay subinterval = tau_c - tau_L (Eq. 3)
        sigma_D: Half-width of Doppler spread at floor level (Eq. 7, 10)
        doppler_shift: f_s - Doppler shift at carrier delay (Eq. 15)
        doppler_shift_min_delay: f_sL - Doppler shift at minimum delay (Eq. 15b)
        correlation_type: Gaussian or Exponential Doppler spectrum shape
    """
    name: str = "F-layer"
    amplitude: float = 1.0
    floor_amplitude: float = 0.01
    tau_L: float = 0.0
    sigma_tau: float = 100.0
    sigma_c: float = 50.0
    sigma_D: float = 1.0
    doppler_shift: float = 0.0
    doppler_shift_min_delay: float = 0.0
    correlation_type: CorrelationType = CorrelationType.GAUSSIAN
    dispersion_us_per_MHz: Optional[float] = None
    f_c_layer: Optional[float] = None
    y_m: float = 100e3
    phi_inc: float = 0.0

    @property
    def tau_c(self) -> float:
        """tau_c: Delay time at carrier frequency = tau_L + sigma_c (us)"""
        return self.tau_L + self.sigma_c

    @property
    def tau_U(self) -> float:
        """tau_U: Maximum delay time = tau_L + sigma_tau (us)"""
        return self.tau_L + self.sigma_tau

    @property
    def doppler_delay_coupling(self) -> float:
        """b: Delay-Doppler coupling coefficient (Hz/us)"""
        if self.sigma_c == 0:
            return 0.0
        return (self.doppler_shift_min_delay - self.doppler_shift) / self.sigma_c


@dataclass
class VoglerHoffmeyerConfig:
    """
    Complete channel model configuration

    Attributes:
        sample_rate: Sample rate in Hz
        modes: List of propagation mode parameters
        spread_f_enabled: Enable spread-F random multiplication (Paper Section 3)
        random_seed: Optional seed for reproducibility
        k_factor: Rician K-factor for fading control (None=Rayleigh)
        use_gpu: Enable GPU acceleration when available (default True)
    """
    sample_rate: float = 1e6
    modes: List[ModeParameters] = field(default_factory=lambda: [ModeParameters()])
    spread_f_enabled: bool = False
    random_seed: Optional[int] = None
    k_factor: Optional[float] = None
    dispersion_enabled: bool = False
    carrier_frequency: float = 10e6
    use_gpu: bool = True

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for JSON serialization"""
        return {
            'sample_rate': self.sample_rate,
            'spread_f_enabled': self.spread_f_enabled,
            'random_seed': self.random_seed,
            'k_factor': self.k_factor,
            'dispersion_enabled': self.dispersion_enabled,
            'carrier_frequency': self.carrier_frequency,
            'use_gpu': self.use_gpu,
            'modes': [
                {
                    'name': m.name,
                    'amplitude': m.amplitude,
                    'floor_amplitude': m.floor_amplitude,
                    'tau_L': m.tau_L,
                    'sigma_tau': m.sigma_tau,
                    'sigma_c': m.sigma_c,
                    'sigma_D': m.sigma_D,
                    'doppler_shift': m.doppler_shift,
                    'doppler_shift_min_delay': m.doppler_shift_min_delay,
                    'correlation_type': m.correlation_type.value,
                    'dispersion_us_per_MHz': m.dispersion_us_per_MHz,
                    'f_c_layer': m.f_c_layer,
                    'y_m': m.y_m,
                    'phi_inc': m.phi_inc
                }
                for m in self.modes
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'VoglerHoffmeyerConfig':
        """Create configuration from dictionary (e.g., loaded from JSON)"""
        modes = [
            ModeParameters(
                name=m.get('name', 'mode'),
                amplitude=m.get('amplitude', 1.0),
                floor_amplitude=m.get('floor_amplitude', 0.01),
                tau_L=m.get('tau_L', 0.0),
                sigma_tau=m.get('sigma_tau', 100.0),
                sigma_c=m.get('sigma_c', 50.0),
                sigma_D=m.get('sigma_D', 1.0),
                doppler_shift=m.get('doppler_shift', 0.0),
                doppler_shift_min_delay=m.get('doppler_shift_min_delay', 0.0),
                correlation_type=CorrelationType(m.get('correlation_type', 'gaussian')),
                dispersion_us_per_MHz=m.get('dispersion_us_per_MHz', None),
                f_c_layer=m.get('f_c_layer', None),
                y_m=m.get('y_m', 100e3),
                phi_inc=m.get('phi_inc', 0.0)
            )
            for m in data.get('modes', [])
        ]
        return cls(
            sample_rate=data.get('sample_rate', 1e6),
            modes=modes if modes else [ModeParameters()],
            spread_f_enabled=data.get('spread_f_enabled', False),
            random_seed=data.get('random_seed', None),
            k_factor=data.get('k_factor', None),
            dispersion_enabled=data.get('dispersion_enabled', False),
            carrier_frequency=data.get('carrier_frequency', 10e6),
            use_gpu=data.get('use_gpu', True)
        )

    @classmethod
    def from_itu_condition(
        cls,
        condition: ITUCondition,
        sample_rate: float = 2_000_000,
    ) -> 'VoglerHoffmeyerConfig':
        """Create config from ITU-R F.1487 condition."""
        if condition == ITUCondition.QUIET:
            modes = [
                ModeParameters(
                    name="F-layer (quiet)",
                    amplitude=1.0,
                    sigma_tau=50.0,
                    sigma_c=25.0,
                    sigma_D=0.1,
                    correlation_type=CorrelationType.GAUSSIAN
                )
            ]
        elif condition == ITUCondition.MODERATE:
            modes = [
                ModeParameters(
                    name="F-layer (moderate)",
                    amplitude=1.0,
                    sigma_tau=100.0,
                    sigma_c=50.0,
                    sigma_D=1.0,
                    correlation_type=CorrelationType.GAUSSIAN
                )
            ]
        elif condition == ITUCondition.DISTURBED:
            modes = [
                ModeParameters(
                    name="F-layer (disturbed)",
                    amplitude=1.0,
                    sigma_tau=500.0,
                    sigma_c=125.0,
                    sigma_D=3.0,
                    correlation_type=CorrelationType.GAUSSIAN
                )
            ]
        elif condition == ITUCondition.FLUTTER:
            modes = [
                ModeParameters(
                    name="F-layer (flutter)",
                    amplitude=1.0,
                    sigma_tau=200.0,
                    sigma_c=50.0,
                    sigma_D=10.0,
                    correlation_type=CorrelationType.EXPONENTIAL
                )
            ]
        else:
            modes = [ModeParameters()]

        return cls(sample_rate=sample_rate, modes=modes)


class VoglerHoffmeyerChannel:
    """
    Vogler-Hoffmeyer HF Channel Model

    Implements the stochastic channel model from NTIA Report 90-255.
    The channel is modeled as a tapped delay line with time-varying complex gains.

    Channel effects modeled:
        1. Multipath delay spread (via delay amplitude function T(tau))
        2. Doppler spread (via correlation factor C(t))
        3. Doppler shift (via phase function phi_s)
        4. Multiple propagation modes (E-layer, F-layer low/high rays)
        5. Spread-F conditions (optional random multiplication)
    """

    def __init__(self, config: VoglerHoffmeyerConfig):
        """Initialize the channel model."""
        self.config = config
        self.sample_rate = config.sample_rate
        self.delta_t = 1.0 / config.sample_rate

        # Initialize random number generator
        self.rng = np.random.default_rng(config.random_seed)

        # Initialize dispersion model if enabled
        self.dispersion_model = None
        if config.dispersion_enabled:
            self.dispersion_model = DispersionModel(config.sample_rate)

        # Pre-compute parameters for each mode
        self._setup_modes()

        # Time index for phase continuity
        self.time_index = 0

        # Callbacks for state updates
        self._state_callbacks: List[Callable] = []

        # Initialize GPU processors if requested and available
        self.gpu_processors: List[Optional[any]] = []
        self.use_gpu = False
        if config.use_gpu and GPU_AVAILABLE:
            self._setup_gpu_processors()
        elif config.use_gpu and not GPU_AVAILABLE:
            import warnings
            warnings.warn("GPU requested but not available, using CPU fallback")

    def _setup_modes(self) -> None:
        """Pre-compute delay taps and parameters for each mode"""
        self.mode_data = []

        for mode in self.config.modes:
            # Convert delay parameters from microseconds to samples
            tau_L_samples = int(mode.tau_L * 1e-6 * self.sample_rate)
            tau_U_samples = int(mode.tau_U * 1e-6 * self.sample_rate)
            num_taps = max(1, tau_U_samples - tau_L_samples + 1)

            # Create delay grid (in microseconds for calculations)
            delay_samples = np.arange(tau_L_samples, tau_U_samples + 1)
            delay_us = delay_samples / self.sample_rate * 1e6

            # Compute alpha and beta for delay amplitude function
            alpha_low, beta_low = self._compute_alpha_beta_low(mode)
            alpha_high, beta_high = self._compute_alpha_beta_high(mode)

            # Compute sigma_f for Doppler spread
            sigma_f = self._compute_sigma_f(mode)

            # Initialize correlation parameters for both Gaussian and Exponential
            # Gaussian: R(tau) = exp(-pi*(sigma_f*tau)^2) -> rho = exp(-pi*(sigma_f*delta_t)^2)
            gauss_rho = np.exp(-np.pi * (sigma_f * self.delta_t)**2)
            z_init = self.rng.standard_normal(num_taps) + 1j * self.rng.standard_normal(num_taps)
            gauss_C_state = z_init / np.sqrt(2)

            # Exponential: R(tau) = exp(-2*pi*sigma_f*|tau|) -> rho = exp(-2*pi*sigma_f*delta_t)
            exp_rho = np.exp(-2 * np.pi * sigma_f * self.delta_t)
            z_init_exp = self.rng.standard_normal(num_taps) + 1j * self.rng.standard_normal(num_taps)
            exp_C_state = z_init_exp / np.sqrt(2)

            mode_data = {
                'mode': mode,
                'tau_L_samples': tau_L_samples,
                'num_taps': num_taps,
                'delay_samples': delay_samples,
                'delay_us': delay_us,
                'alpha_low': alpha_low,
                'beta_low': beta_low,
                'alpha_high': alpha_high,
                'beta_high': beta_high,
                'sigma_f': sigma_f,
                'gauss_rho': gauss_rho,
                'gauss_C_state': gauss_C_state,
                'exp_rho': exp_rho,
                'exp_C_state': exp_C_state,
                'buffer': np.zeros(max(1, tau_U_samples + 1), dtype=np.complex128),
                'norm_factor': 1.0,
                'dispersion_d': 0.0
            }

            # Compute power normalization factor
            T = self._compute_delay_amplitude(delay_us, mode_data)
            sum_T_squared = np.sum(T**2)
            if sum_T_squared > 0:
                mode_data['norm_factor'] = mode.amplitude / np.sqrt(sum_T_squared)
            else:
                mode_data['norm_factor'] = mode.amplitude

            # Compute dispersion coefficient for this mode if enabled
            if self.config.dispersion_enabled:
                if mode.dispersion_us_per_MHz is not None:
                    mode_data['dispersion_d'] = mode.dispersion_us_per_MHz
                elif mode.f_c_layer is not None:
                    mode_data['dispersion_d'] = compute_d_from_qp(
                        mode.f_c_layer,
                        self.config.carrier_frequency,
                        mode.y_m,
                        mode.phi_inc
                    )

            self.mode_data.append(mode_data)

    def _setup_gpu_processors(self) -> None:
        """Initialize GPU processors for each mode.

        Creates a VHRFChainProcessor per mode and configures it with the
        appropriate tap delays, amplitudes, and Doppler shifts.
        """
        self.gpu_processors = []

        # Determine maximum samples to process (for GPU buffer allocation)
        # Use larger buffer for high sample rates, smaller for low rates
        self.gpu_block_size = min(32768, max(4096, int(self.sample_rate / 10)))
        max_input_samples = self.gpu_block_size

        for mode_idx, mode_data in enumerate(self.mode_data):
            mode = mode_data['mode']

            try:
                # Compute coherence time from sigma_f (AR(1) correlation parameter)
                sigma_f = mode_data['sigma_f']
                coherence_time_sec = 1.0 / sigma_f if sigma_f > 0 else 1.0

                # K-factor for Rician fading
                k_factor = self.config.k_factor if self.config.k_factor is not None else 0.0

                # Create GPU processor (uses baseband rate for input and RF)
                # When rf_rate == input_rate, the polyphase stages pass through
                processor = VHRFChainProcessor(
                    input_rate=int(self.sample_rate),
                    rf_rate=int(self.sample_rate),  # Baseband-only processing
                    max_input_samples=max_input_samples,
                    carrier_freq_hz=float(self.config.carrier_frequency),
                    coherence_time_sec=float(coherence_time_sec),
                    k_factor=float(k_factor),
                    seed=self.config.random_seed if self.config.random_seed else 42
                )

                # Configure taps with delay, amplitude, and Doppler
                self._configure_gpu_taps(processor, mode_idx)

                self.gpu_processors.append(processor)

            except Exception as e:
                import warnings
                warnings.warn(f"Failed to initialize GPU processor for mode {mode_idx}: {e}")
                self.gpu_processors.append(None)

        # Check if any GPU processor was successfully created
        self.use_gpu = any(p is not None for p in self.gpu_processors)

        if self.use_gpu:
            # Report which backend is being used
            for i, p in enumerate(self.gpu_processors):
                if p is not None and hasattr(p, 'is_using_gpu'):
                    backend = "GPU" if p.is_using_gpu() else "CPU"

    def _configure_gpu_taps(self, processor, mode_idx: int) -> None:
        """Configure GPU processor taps from mode parameters.

        Maps the Python mode_data to GPU format:
        - delays: integer sample delays
        - amplitudes: T(tau) * norm_factor (float32)
        - doppler_hz: per-tap Doppler shift (float32)

        Note: GPU kernel supports max 16 taps. If more taps needed,
        they are decimated (sampled at regular intervals).
        """
        GPU_MAX_TAPS = 16  # Hardcoded limit in CUDA kernel

        mode_data = self.mode_data[mode_idx]
        mode = mode_data['mode']

        # Get delay samples (integer) - relative to minimum delay
        delay_samples = mode_data['delay_samples']
        delays = (delay_samples - delay_samples[0]).astype(np.int32)

        # Compute tap amplitudes: T(tau) * normalization factor
        delay_us = mode_data['delay_us']
        T = self._compute_delay_amplitude(delay_us, mode_data)
        amplitudes = (T * mode_data['norm_factor']).astype(np.float32)

        # Compute per-tap Doppler shifts
        # f_D_effective(tau) = doppler_shift + b * (tau_c - tau)
        tau_c = mode.tau_L + mode.sigma_c
        b = mode.doppler_delay_coupling
        doppler_hz = (mode.doppler_shift + b * (tau_c - delay_us)).astype(np.float32)

        # Decimate taps if more than GPU limit
        n_taps = len(delays)
        if n_taps > GPU_MAX_TAPS:
            # Select evenly spaced subset of taps
            indices = np.linspace(0, n_taps - 1, GPU_MAX_TAPS, dtype=int)
            delays = delays[indices]
            amplitudes = amplitudes[indices]
            doppler_hz = doppler_hz[indices]

            # Re-normalize amplitudes to preserve power
            amp_sum_sq_original = np.sum((T * mode_data['norm_factor'])**2)
            amp_sum_sq_new = np.sum(amplitudes**2)
            if amp_sum_sq_new > 0:
                scale = np.sqrt(amp_sum_sq_original / amp_sum_sq_new)
                amplitudes *= scale

        # Configure the GPU processor
        processor.configure_taps(delays, amplitudes, doppler_hz)

    def _compute_alpha_beta_low(self, mode: ModeParameters) -> Tuple[float, float]:
        """Compute alpha and beta for delay amplitude when y <= 1"""
        y1 = 0.01
        y2 = 0.5
        A1 = mode.floor_amplitude

        numerator = np.log(A1) * (1 - y2 + np.log(y2))
        denominator = 1 - y1 + np.log(y1)
        if abs(denominator) < 1e-10:
            A2 = A1
        else:
            A2 = np.exp(numerator / denominator)

        A2 = max(A2, 1e-10)
        d = (1 - y2) * np.log(y1) - (1 - y1) * np.log(y2)

        if abs(d) < 1e-10:
            return 0.0, 0.0

        alpha = ((1 - y2) * np.log(A1) - (1 - y1) * np.log(A2)) / d
        beta = (np.log(y1) * np.log(A2) - np.log(y2) * np.log(A1)) / d

        return alpha, beta

    def _compute_alpha_beta_high(self, mode: ModeParameters) -> Tuple[float, float]:
        """Compute alpha and beta for delay amplitude when y > 1"""
        if mode.sigma_c == 0:
            return 0.0, 0.0

        y2 = mode.sigma_tau / mode.sigma_c
        A_fl_ratio = mode.floor_amplitude / mode.amplitude if mode.amplitude > 0 else 0.01
        A_fl_ratio = max(A_fl_ratio, 1e-10)
        y2 = max(y2, 1.001)

        denom = np.log(y2) + 1 - y2
        if abs(denom) < 1e-10:
            return 1.0, 1.0

        alpha = np.log(A_fl_ratio) / denom
        beta = alpha

        return alpha, beta

    def _compute_sigma_f(self, mode: ModeParameters) -> float:
        """Compute sigma_f spectral width parameter for correlation factor C(t).

        sigma_f controls the fading rate through the AR(1) coefficient:
        - Gaussian: rho = exp(-pi*(sigma_f*delta_t)^2)
        - Exponential: rho = exp(-2*pi*sigma_f*delta_t)

        For proper Rayleigh fading statistics, sigma_f should be based on
        the Doppler spread sigma_D. The relationship depends on the Doppler
        spectrum shape:
        - Gaussian spectrum: sigma_f = sigma_D (direct mapping)
        - Exponential spectrum: sigma_f = 2*pi*sigma_D (to match fading rate)
        """
        sigma_D = mode.sigma_D

        if sigma_D <= 0:
            return 1.0

        if mode.correlation_type == CorrelationType.GAUSSIAN:
            # Gaussian Doppler spectrum: direct mapping
            sigma_f = sigma_D
        else:
            # Exponential Doppler spectrum: scale for proper fading rate
            # The exponential autocorrelation R(τ) = exp(-|τ|/τc) has
            # coherence time τc = 1/(2*pi*fd) for Doppler spread fd
            sigma_f = 2 * np.pi * sigma_D

        return max(sigma_f, 1e-6)

    def _compute_delay_amplitude(self, tau_us: np.ndarray, mode_data: dict) -> np.ndarray:
        """Compute delay amplitude factor T(tau) for all delay taps"""
        mode = mode_data['mode']
        sigma_c = mode.sigma_c
        if sigma_c == 0:
            return np.ones_like(tau_us) * mode.amplitude

        y = (tau_us - mode.tau_L) / sigma_c
        T = np.zeros_like(tau_us, dtype=float)

        # For y <= 1 (tau <= tau_c): use low parameters
        mask_low = (y > 0) & (y <= 1)
        if np.any(mask_low):
            alpha = mode_data['alpha_low']
            beta = mode_data['beta_low']
            y_low = y[mask_low]
            T[mask_low] = mode.amplitude * np.power(y_low, alpha) * np.exp(beta * (1 - y_low))

        # For y > 1 (tau > tau_c): use high parameters
        mask_high = y > 1
        if np.any(mask_high):
            alpha = mode_data['alpha_high']
            beta = mode_data['beta_high']
            y_high = y[mask_high]
            T[mask_high] = mode.amplitude * np.power(y_high, alpha) * np.exp(beta * (1 - y_high))

        T = np.maximum(T, mode.floor_amplitude)
        return T

    def _process_mode(self, input_samples: np.ndarray, mode_idx: int) -> np.ndarray:
        """Process input samples through a single propagation mode"""
        mode_data = self.mode_data[mode_idx]
        mode = mode_data['mode']
        delay_us = mode_data['delay_us']
        delay_samples = mode_data['delay_samples']

        # Get delay amplitude with normalization
        T = self._compute_delay_amplitude(delay_us, mode_data) * mode_data['norm_factor']

        # Use Numba-optimized path when available
        if NUMBA_AVAILABLE:
            if mode.correlation_type == CorrelationType.GAUSSIAN:
                return self._process_mode_numba(input_samples, mode_idx, T)
            else:
                return self._process_mode_numba_exp(input_samples, mode_idx, T)

        return self._process_mode_python(input_samples, mode_idx, T)

    def _process_mode_numba(self, input_samples: np.ndarray, mode_idx: int,
                           T: np.ndarray) -> np.ndarray:
        """Numba-optimized processing for Gaussian correlation"""
        mode_data = self.mode_data[mode_idx]
        mode = mode_data['mode']
        delay_us = mode_data['delay_us']
        delay_samples = mode_data['delay_samples']

        rho = mode_data['gauss_rho']
        innovation_coeff = np.sqrt(1 - rho**2)
        C_state = mode_data['gauss_C_state'].copy()

        tau_c = mode.tau_L + mode.sigma_c
        f_s = mode.doppler_shift
        b = mode.doppler_delay_coupling
        t_start = self.time_index * self.delta_t

        k_factor = self.config.k_factor
        if k_factor is not None and k_factor > 0:
            direct_coeff = np.sqrt(k_factor / (k_factor + 1))
            scatter_coeff = np.sqrt(1 / (k_factor + 1))
        else:
            direct_coeff = 0.0
            scatter_coeff = 1.0

        if 'numba_rng_state' not in mode_data:
            mode_data['numba_rng_state'] = np.array([self.rng.integers(1, 2**31-1)], dtype=np.int64)

        output, new_buffer, new_C, new_rng = _process_samples_numba(
            input_samples,
            mode_data['buffer'],
            delay_samples.astype(np.int64),
            T,
            C_state,
            rho,
            innovation_coeff,
            delay_us,
            tau_c,
            f_s,
            b,
            t_start,
            self.delta_t,
            mode_data['numba_rng_state'],
            direct_coeff,
            scatter_coeff,
            self.config.spread_f_enabled
        )

        mode_data['buffer'] = new_buffer
        mode_data['gauss_C_state'] = new_C
        mode_data['numba_rng_state'] = new_rng

        return output

    def _process_mode_numba_exp(self, input_samples: np.ndarray, mode_idx: int,
                                T: np.ndarray) -> np.ndarray:
        """Numba-optimized processing for exponential correlation.

        Uses proper AR(1) process with Gaussian innovations (Box-Muller).
        Same structure as Gaussian correlation but with exp(-|tau|/tau_c)
        autocorrelation function.
        """
        mode_data = self.mode_data[mode_idx]
        mode = mode_data['mode']
        delay_us = mode_data['delay_us']
        delay_samples = mode_data['delay_samples']

        # Use same AR(1) structure as Gaussian correlation
        rho = mode_data['exp_rho']
        innovation_coeff = np.sqrt(1 - rho**2)
        C_state = mode_data['exp_C_state'].copy()

        tau_c = mode.tau_L + mode.sigma_c
        f_s = mode.doppler_shift
        b = mode.doppler_delay_coupling
        t_start = self.time_index * self.delta_t

        k_factor = self.config.k_factor
        if k_factor is not None and k_factor > 0:
            direct_coeff = np.sqrt(k_factor / (k_factor + 1))
            scatter_coeff = np.sqrt(1 / (k_factor + 1))
        else:
            direct_coeff = 0.0
            scatter_coeff = 1.0

        if 'numba_rng_state_exp' not in mode_data:
            mode_data['numba_rng_state_exp'] = np.array([self.rng.integers(1, 2**31-1)], dtype=np.int64)

        output, new_buffer, new_C, new_rng = _process_samples_numba_exp(
            input_samples,
            mode_data['buffer'],
            delay_samples.astype(np.int64),
            T,
            C_state,
            rho,
            innovation_coeff,
            delay_us,
            tau_c,
            f_s,
            b,
            t_start,
            self.delta_t,
            mode_data['numba_rng_state_exp'],
            direct_coeff,
            scatter_coeff,
            self.config.spread_f_enabled
        )

        mode_data['buffer'] = new_buffer
        mode_data['exp_C_state'] = new_C
        mode_data['numba_rng_state_exp'] = new_rng

        return output

    def _process_mode_python(self, input_samples: np.ndarray, mode_idx: int,
                            T: np.ndarray) -> np.ndarray:
        """Python implementation (fallback when Numba unavailable)"""
        mode_data = self.mode_data[mode_idx]
        mode = mode_data['mode']
        num_taps = mode_data['num_taps']
        delay_us = mode_data['delay_us']
        delay_samples = mode_data['delay_samples']

        num_input = len(input_samples)
        output = np.zeros(num_input, dtype=np.complex128)

        k_factor = self.config.k_factor
        if k_factor is not None and k_factor > 0:
            direct_coeff = np.sqrt(k_factor / (k_factor + 1))
            scatter_coeff = np.sqrt(1 / (k_factor + 1))
        else:
            direct_coeff = 0.0
            scatter_coeff = 1.0

        buffer = mode_data['buffer'].copy()
        rho = mode_data['gauss_rho']
        C_state = mode_data['gauss_C_state'].copy()

        for n in range(num_input):
            t = (self.time_index + n) * self.delta_t

            # Update delay line buffer
            buffer = np.roll(buffer, 1)
            buffer[0] = input_samples[n]

            # Generate fading
            z = self.rng.standard_normal(num_taps) + 1j * self.rng.standard_normal(num_taps)
            z /= np.sqrt(2)
            innovation_coeff = np.sqrt(1 - rho**2)
            C_state = rho * C_state + innovation_coeff * z

            fading_gain = scatter_coeff * C_state
            fading_gain[0] = direct_coeff + scatter_coeff * C_state[0]

            # Compute phase for Doppler shift
            tau_c = mode.tau_L + mode.sigma_c
            b = mode.doppler_delay_coupling
            f_D_effective = mode.doppler_shift + b * (tau_c - delay_us)
            phi = 2 * np.pi * f_D_effective * t

            tap_gains = T * fading_gain * np.exp(1j * phi)

            if self.config.spread_f_enabled:
                spread_factor = self.rng.uniform(0.1, 1.0, num_taps)
                tap_gains *= spread_factor

            for k, delay in enumerate(delay_samples):
                if delay < len(buffer):
                    output[n] += buffer[delay] * tap_gains[k]

        mode_data['buffer'] = buffer
        mode_data['gauss_C_state'] = C_state

        return output

    def _process_mode_gpu(self, input_samples: np.ndarray, mode_idx: int) -> np.ndarray:
        """Process samples through GPU-accelerated TDL for a single mode.

        Args:
            input_samples: Complex I/Q input samples
            mode_idx: Index of the propagation mode

        Returns:
            Complex output samples after channel effects
        """
        processor = self.gpu_processors[mode_idx]
        if processor is None:
            # Fallback to CPU if GPU processor not available for this mode
            return self._process_mode(input_samples, mode_idx)

        # Convert to complex64 for GPU processing
        input_c64 = input_samples.astype(np.complex64)
        n_samples = len(input_c64)

        # Process in blocks if input exceeds GPU buffer size
        if n_samples <= self.gpu_block_size:
            # Single block - direct processing
            output = processor.process(input_c64)
        else:
            # Block-based processing for large inputs
            output = np.zeros(n_samples, dtype=np.complex64)
            for start in range(0, n_samples, self.gpu_block_size):
                end = min(start + self.gpu_block_size, n_samples)
                block_output = processor.process(input_c64[start:end])
                output[start:end] = block_output

        return output.astype(np.complex128)

    def process(self, input_samples: np.ndarray) -> np.ndarray:
        """
        Apply channel model to I/Q samples

        This is the main processing function. It applies all configured
        propagation modes to the input signal and sums the results.

        Uses GPU acceleration when available, with automatic CPU fallback.

        Args:
            input_samples: Complex I/Q input samples (numpy array)

        Returns:
            Complex I/Q output samples after channel effects
        """
        output = np.zeros(len(input_samples), dtype=np.complex128)

        # Process each mode and sum contributions
        for mode_idx in range(len(self.config.modes)):
            mode_data = self.mode_data[mode_idx]

            # Apply dispersion if enabled for this mode
            if self.dispersion_model is not None and mode_data['dispersion_d'] != 0.0:
                dispersed_input = self.dispersion_model.apply_dispersion(
                    input_samples, mode_data['dispersion_d']
                )
            else:
                dispersed_input = input_samples

            # Choose GPU or CPU processing path
            if self.use_gpu and mode_idx < len(self.gpu_processors) and self.gpu_processors[mode_idx] is not None:
                mode_output = self._process_mode_gpu(dispersed_input, mode_idx)
            else:
                mode_output = self._process_mode(dispersed_input, mode_idx)

            output += mode_output

        # Update time index for phase continuity across blocks
        self.time_index += len(input_samples)

        # Notify callbacks
        for callback in self._state_callbacks:
            callback(self.get_state())

        return output.astype(np.complex64)

    def process_block(self, input_samples: np.ndarray, block_size: int = 1024) -> np.ndarray:
        """Process input in blocks for memory efficiency"""
        num_samples = len(input_samples)
        output = np.zeros(num_samples, dtype=np.complex128)

        for start in range(0, num_samples, block_size):
            end = min(start + block_size, num_samples)
            output[start:end] = self.process(input_samples[start:end])

        return output.astype(np.complex64)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset channel state for processing a new signal"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.time_index = 0
        self._setup_modes()

        # Reset GPU processors if active
        if self.use_gpu:
            for processor in self.gpu_processors:
                if processor is not None:
                    processor.reset()
            # Reconfigure taps after reset
            for mode_idx, processor in enumerate(self.gpu_processors):
                if processor is not None:
                    self._configure_gpu_taps(processor, mode_idx)

    def get_state(self) -> dict:
        """Get current channel state"""
        mode_info = []
        for mode_data in self.mode_data:
            mode = mode_data['mode']
            mode_info.append({
                'name': mode.name,
                'sigma_tau': mode.sigma_tau,
                'sigma_D': mode.sigma_D,
                'num_taps': mode_data['num_taps'],
            })

        return {
            'time': self.time_index * self.delta_t,
            'modes': mode_info,
            'num_modes': len(self.config.modes),
            'sample_rate': self.sample_rate,
        }

    def get_backend_info(self) -> dict:
        """Get GPU/CPU backend information.

        Returns:
            Dictionary with:
            - use_gpu: Whether GPU is being used
            - num_gpu_modes: Number of modes using GPU
            - mode_backends: List of backend per mode ("GPU", "CPU", or "unavailable")
            - gpu_available: Whether GPU module is available
        """
        mode_backends = []
        for i, processor in enumerate(self.gpu_processors):
            if processor is None:
                mode_backends.append("CPU")
            elif hasattr(processor, 'is_using_gpu'):
                mode_backends.append("GPU" if processor.is_using_gpu() else "CPU")
            else:
                mode_backends.append("unknown")

        return {
            'use_gpu': self.use_gpu,
            'num_gpu_modes': sum(1 for p in self.gpu_processors if p is not None and hasattr(p, 'is_using_gpu') and p.is_using_gpu()),
            'mode_backends': mode_backends,
            'gpu_available': GPU_AVAILABLE,
        }

    def get_impulse_response(self, num_samples: int = 1024) -> np.ndarray:
        """Get the instantaneous channel impulse response"""
        impulse = np.zeros(num_samples, dtype=np.complex128)
        impulse[0] = 1.0

        saved_time_index = self.time_index
        saved_buffers = [md['buffer'].copy() for md in self.mode_data]

        self.reset()
        response = self.process(impulse)

        self.time_index = saved_time_index
        for md, buf in zip(self.mode_data, saved_buffers):
            md['buffer'] = buf

        return response

    def compute_scattering_function(self, num_delay_bins: int = 64,
                                    num_doppler_bins: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the THEORETICAL channel scattering function S(tau, f_D)

        Returns:
            Tuple of (delay_axis, doppler_axis, scattering_function)
        """
        max_delay_us = max(m.tau_U for m in self.config.modes)
        max_doppler_hz = max(m.sigma_D * 3 + abs(m.doppler_shift) for m in self.config.modes)

        delay_axis = np.linspace(0, max_delay_us, num_delay_bins)
        doppler_axis = np.linspace(-max_doppler_hz, max_doppler_hz, num_doppler_bins)

        S = np.zeros((num_delay_bins, num_doppler_bins))

        for mode_data in self.mode_data:
            mode = mode_data['mode']

            for i, tau in enumerate(delay_axis):
                T = self._compute_delay_amplitude(np.array([tau]), mode_data)[0]

                tau_c = mode.tau_L + mode.sigma_c
                if mode.sigma_c > 0:
                    b = (mode.doppler_shift_min_delay - mode.doppler_shift) / mode.sigma_c
                else:
                    b = 0.0
                f_s_tau = mode.doppler_shift + b * (tau_c - tau)

                sigma_D = mode.sigma_D

                if mode.correlation_type == CorrelationType.GAUSSIAN:
                    if sigma_D > 0:
                        doppler_spectrum = np.exp(-np.pi * ((doppler_axis - f_s_tau) / sigma_D)**2)
                    else:
                        doppler_spectrum = np.zeros_like(doppler_axis)
                        closest_idx = np.argmin(np.abs(doppler_axis - f_s_tau))
                        doppler_spectrum[closest_idx] = 1.0
                else:
                    if sigma_D > 0:
                        f_D_shifted = doppler_axis - f_s_tau
                        doppler_spectrum = sigma_D / (sigma_D**2 + (np.pi * f_D_shifted)**2)
                    else:
                        doppler_spectrum = np.zeros_like(doppler_axis)
                        closest_idx = np.argmin(np.abs(doppler_axis - f_s_tau))
                        doppler_spectrum[closest_idx] = 1.0

                S[i, :] += (T**2) * doppler_spectrum

        if S.max() > 0:
            S /= S.max()

        return delay_axis, doppler_axis, S

    def add_state_callback(self, callback: Callable):
        """Register callback for state updates."""
        self._state_callbacks.append(callback)

    def remove_state_callback(self, callback: Callable):
        """Remove state callback."""
        if callback in self._state_callbacks:
            self._state_callbacks.remove(callback)


# ============================================================================
# Preset Configurations
# ============================================================================

def create_equatorial_config(sample_rate: float = 1e6) -> VoglerHoffmeyerConfig:
    """Create equatorial path configuration"""
    mode = ModeParameters(
        name="F-layer (equatorial)",
        amplitude=1.0,
        floor_amplitude=0.01,
        tau_L=0.0,
        sigma_tau=880.0,
        sigma_c=220.0,
        sigma_D=2.0,
        doppler_shift=0.0,
        doppler_shift_min_delay=0.0,
        correlation_type=CorrelationType.GAUSSIAN
    )
    return VoglerHoffmeyerConfig(sample_rate=sample_rate, modes=[mode])


def create_polar_config(sample_rate: float = 1e6) -> VoglerHoffmeyerConfig:
    """Create polar path configuration"""
    e_layer = ModeParameters(
        name="E-layer (polar)",
        amplitude=0.7,
        floor_amplitude=0.01,
        tau_L=0.0,
        sigma_tau=250.0,
        sigma_c=100.0,
        sigma_D=16.0,
        doppler_shift=0.0,
        doppler_shift_min_delay=0.0,
        correlation_type=CorrelationType.GAUSSIAN
    )
    f_low = ModeParameters(
        name="F-layer low-ray (polar)",
        amplitude=1.0,
        floor_amplitude=0.01,
        tau_L=300.0,
        sigma_tau=400.0,
        sigma_c=135.0,
        sigma_D=7.0,
        doppler_shift=0.0,
        doppler_shift_min_delay=0.0,
        correlation_type=CorrelationType.GAUSSIAN
    )
    return VoglerHoffmeyerConfig(sample_rate=sample_rate, modes=[e_layer, f_low])


def create_midlatitude_config(sample_rate: float = 1e6) -> VoglerHoffmeyerConfig:
    """Create mid-latitude path configuration"""
    mode = ModeParameters(
        name="F-layer (mid-latitude)",
        amplitude=1.0,
        floor_amplitude=0.01,
        tau_L=0.0,
        sigma_tau=50.0,
        sigma_c=25.0,
        sigma_D=0.1,
        doppler_shift=0.2,
        doppler_shift_min_delay=-0.2,
        correlation_type=CorrelationType.GAUSSIAN
    )
    return VoglerHoffmeyerConfig(sample_rate=sample_rate, modes=[mode])


def create_auroral_spread_f_config(sample_rate: float = 1e6) -> VoglerHoffmeyerConfig:
    """Create auroral spread-F configuration"""
    mode = ModeParameters(
        name="F-layer (auroral spread-F)",
        amplitude=1.0,
        floor_amplitude=0.01,
        tau_L=0.0,
        sigma_tau=2000.0,
        sigma_c=500.0,
        sigma_D=5.0,
        doppler_shift=0.0,
        doppler_shift_min_delay=0.0,
        correlation_type=CorrelationType.EXPONENTIAL
    )
    return VoglerHoffmeyerConfig(
        sample_rate=sample_rate,
        modes=[mode],
        spread_f_enabled=True
    )


VOGLER_HOFFMEYER_PRESETS = {
    'equatorial': create_equatorial_config,
    'polar': create_polar_config,
    'midlatitude': create_midlatitude_config,
    'auroral_spread_f': create_auroral_spread_f_config,
}


def get_vogler_hoffmeyer_preset(name: str, sample_rate: float = 1e6) -> VoglerHoffmeyerConfig:
    """Get a preset channel configuration by name"""
    if name not in VOGLER_HOFFMEYER_PRESETS:
        available = ', '.join(VOGLER_HOFFMEYER_PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return VOGLER_HOFFMEYER_PRESETS[name](sample_rate)


def list_vogler_hoffmeyer_presets() -> List[str]:
    """Return list of available preset names"""
    return list(VOGLER_HOFFMEYER_PRESETS.keys())

# Vogler-Hoffmeyer HF Channel Model: Processing Implementation

This document describes the implementation of the Vogler-Hoffmeyer wideband HF channel model in the HF Path Simulator, based on NTIA Report 90-255: "A Model for Wideband HF Propagation Channels" by L.E. Vogler and J.A. Hoffmeyer (1990).

## Overview

The Vogler-Hoffmeyer model extends the classical narrowband Watterson model to wideband channels (up to 1 MHz+). It simulates time-varying distortion due to:

- **Dispersion**: Frequency-dependent group delay from ionospheric refraction
- **Multipath**: Multiple propagation modes with different delays
- **Doppler spread**: Frequency spreading from ionospheric motion
- **Doppler shift**: Systematic frequency offset from bulk ionospheric movement
- **Spread-F**: Scattering from ionospheric irregularities (auroral/equatorial)

## The Computational Challenge

At 1 MHz bandwidth, the computational burden comes from three interlocking pieces:

| Component | Complexity | 1 MHz Example |
|-----------|------------|---------------|
| Dispersion filtering | O(N log N) | FFT convolution per mode |
| Tapped delay line | O(N × M) | N=1024 samples × M=500 taps |
| Multi-mode processing | O(K × above) | K=2-3 propagation modes |

For real-time processing at 1 Msps with 500 µs delay spread, the inner loop must handle ~500,000 tap-sample operations per millisecond per mode.

## Implementation Architecture

### File Structure

```
src/hfpathsim/core/
├── vogler_hoffmeyer.py   # Main channel model (~1064 lines)
├── dispersion.py         # Frequency-dependent delay (~470 lines)
└── parameters.py         # ITU condition definitions

src/hfpathsim/gpu/
├── kernels/signal_proc.cu  # CUDA kernels for dispersion
└── bindings.cpp            # pybind11 Python interface
```

### Processing Chain

```
Input I/Q samples
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  For each propagation mode (E-layer, F-layer low/high)  │
│  ┌────────────────────────────────────────────────────┐  │
│  │  1. Apply dispersion (if enabled)                 │  │
│  │     └─ Chirp all-pass filter via FFT convolution  │  │
│  │                                                    │  │
│  │  2. Tapped delay line processing                  │  │
│  │     ├─ Update delay buffer                        │  │
│  │     ├─ Generate AR(1) fading coefficients         │  │
│  │     ├─ Apply delay amplitude T(τ)                 │  │
│  │     ├─ Apply Doppler phase rotation               │  │
│  │     └─ Accumulate tap contributions               │  │
│  └────────────────────────────────────────────────────┘  │
│                          │                               │
│                          ▼                               │
│              Sum mode contributions                      │
└──────────────────────────────────────────────────────────┘
       │
       ▼
Output I/Q samples
```

## Component Details

### 1. Frequency-Dependent Delay (Dispersion)

**Physics**: The ionosphere causes group delay τ_g(f) ~ K/f² due to the refractive index varying with electron density.

**Linear Approximation**: For bandwidth B << carrier frequency f_c:

```
τ(f) = τ₀ + d·(f - f_c)
```

where `d` is the dispersion coefficient in µs/MHz.

**Implementation** (`dispersion.py`):

The linear dispersion is implemented via a chirp all-pass filter with impulse response:

```
h(t) = exp(j·π·t²/d) / √(j·d)
```

**Coefficient Derivation** (quasi-parabolic layer model):

```python
K = (π · y_m · f_c² · sec(φ)) / (2c)
d = 2K / f_carrier³
```

Where:
- `y_m` = layer semi-thickness (typically 50-150 km)
- `f_c` = layer critical frequency (4-12 MHz typical)
- `φ` = incidence angle
- `c` = speed of light

**Optimizations**:
- Filter caching by coefficient value
- GPU/CPU compiled convolution when available
- SciPy overlap-add (`oaconvolve`) for long signals

**Typical Values**:

| Condition | Dispersion (µs/MHz) |
|-----------|---------------------|
| Quiet, high frequency | 10-30 |
| Moderate | 30-80 |
| Disturbed | 80-150 |
| Spread-F | 150-240 |
| Severe | 200-400 |

### 2. Tapped Delay Line with Time-Varying Gains

**Physics**: The multipath channel impulse response h(t, τ) varies in both time and delay. Each tap has a time-varying complex gain following a specified Doppler spectrum.

**Delay Amplitude Function T(τ)**:

The delay power profile follows NTIA 90-255 Equation 3-4:

```
T(τ) = A · y^α · exp(β·(1-y))
```

where `y = (τ - τ_L) / σ_c` is the normalized delay.

Two regions with different (α, β) parameters:
- **y ≤ 1** (τ ≤ τ_c): Rising portion toward carrier delay
- **y > 1** (τ > τ_c): Falling portion beyond carrier delay

**Parameters**:
- `τ_L` = minimum delay (µs)
- `τ_c` = carrier delay (µs)
- `τ_U` = maximum delay (µs)
- `σ_τ` = total delay spread = τ_U - τ_L
- `σ_c` = carrier delay offset = τ_c - τ_L
- `A` = mode amplitude
- `A_fl` = floor amplitude (receiver threshold)

### 3. Stochastic Fading Generation

**Gaussian Doppler Spectrum** (NTIA 90-255 Eq. 7-8):

AR(1) process with correlation:
```
C[n] = ρ · C[n-1] + √(1-ρ²) · z[n]
```

where:
- `ρ = exp(-π·(σ_f·Δt)²)`
- `z[n]` = complex Gaussian noise with unit variance
- `σ_f` = spectral width parameter

This produces a bell-shaped (Gaussian) Doppler spectrum.

**Exponential Doppler Spectrum** (NTIA 90-255 Eq. 10-11):

Filtered uniform process:
```
x[n] = u[n] + (x[n-1] - u[n]) · λ
```

where:
- `λ = exp(-Δt · σ_f)`
- `u[n]` = uniform random in [-0.5, 0.5]

This produces a peaked (Lorentzian) Doppler spectrum, appropriate for flutter fading conditions.

**Rician Fading**:

When K-factor is specified, the first tap (LOS component) has a direct path:
```
fading_gain[0] = √(K/(K+1)) + √(1/(K+1)) · C[0]
fading_gain[k] = √(1/(K+1)) · C[k]   for k > 0
```

### 4. Doppler Phase Rotation

**Physics**: Ionospheric motion causes frequency shifts that vary with delay (delay-Doppler coupling).

**Implementation**:

```
f_eff(τ) = f_s + b·(τ_c - τ)
φ(t, τ) = 2π · f_eff(τ) · t
tap_gain = T(τ) · fading_gain · exp(j·φ)
```

where:
- `f_s` = Doppler shift at carrier delay
- `b` = delay-Doppler coupling coefficient (Hz/µs)

## Performance Optimizations

### Numba JIT Compilation

The inner processing loop is JIT-compiled using Numba:

```python
@jit(nopython=True, parallel=False, cache=True)
def _process_samples_numba(input_samples, buffer, delay_samples, T,
                           C_state, rho, innovation_coeff, ...):
```

**Key optimizations**:

1. **Explicit buffer shift** instead of `np.roll()`:
   ```python
   for i in range(buf_len - 1, 0, -1):
       buf[i] = buf[i-1]
   buf[0] = input_samples[n]
   ```

2. **Inline complex exponential**:
   ```python
   tap_gain = T[k] * fading_gain * (np.cos(phi) + 1j * np.sin(phi))
   ```

3. **Simple LCG random generator** (deterministic, no GIL):
   ```python
   rng_state[0] = (rng_state[0] * 1103515245 + 12345) & 0x7fffffff
   u = rng_state[0] / 2147483648.0
   ```

4. **Pre-computed parameters**: All mode setup done once in `_setup_modes()`

### GPU Acceleration

Dispersion filtering uses CUDA when available:

```cpp
// signal_proc.cu
__global__ void complex_multiply(cuComplex* a, const cuComplex* b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a[idx] = cuCmulf(a[idx], b[idx]);
    }
}
```

The GPU path uses overlap-save convolution with cuFFT.

### Power Normalization

Unit average power is maintained regardless of tap count:

```python
T = self._compute_delay_amplitude(delay_us, mode_data)
sum_T_squared = np.sum(T**2)
mode_data['norm_factor'] = mode.amplitude / np.sqrt(sum_T_squared)
```

## Mode Configurations

### Preset Configurations

| Preset | Modes | σ_τ (µs) | σ_D (Hz) | Notes |
|--------|-------|----------|----------|-------|
| `midlatitude` | 1 F-layer | 50 | 0.1 | Quiet conditions |
| `equatorial` | 1 F-layer | 880 | 2.0 | High delay spread |
| `polar` | E + F-low | 250/400 | 16/7 | Two-mode propagation |
| `auroral_spread_f` | 1 F-layer | 2000 | 5.0 | Exponential spectrum + scatter |

### ITU-R F.1487 Conditions

| Condition | σ_τ (µs) | σ_D (Hz) | Correlation |
|-----------|----------|----------|-------------|
| Quiet | 50 | 0.1 | Gaussian |
| Moderate | 100 | 1.0 | Gaussian |
| Disturbed | 500 | 3.0 | Gaussian |
| Flutter | 200 | 10.0 | Exponential |

## API Usage

### Basic Usage

```python
from hfpathsim.core.vogler_hoffmeyer import (
    VoglerHoffmeyerChannel,
    VoglerHoffmeyerConfig,
    ModeParameters,
    CorrelationType
)

# Create configuration
config = VoglerHoffmeyerConfig(
    sample_rate=1e6,
    modes=[
        ModeParameters(
            name="F-layer",
            sigma_tau=100.0,      # µs delay spread
            sigma_c=50.0,         # µs carrier delay offset
            sigma_D=1.0,          # Hz Doppler spread
            correlation_type=CorrelationType.GAUSSIAN
        )
    ]
)

# Create channel
channel = VoglerHoffmeyerChannel(config)

# Process samples
output = channel.process(input_samples)
```

### Using Presets

```python
from hfpathsim.core.vogler_hoffmeyer import get_vogler_hoffmeyer_preset

config = get_vogler_hoffmeyer_preset('polar', sample_rate=2e6)
channel = VoglerHoffmeyerChannel(config)
```

### With Dispersion

```python
config = VoglerHoffmeyerConfig(
    sample_rate=1e6,
    carrier_frequency=15e6,
    dispersion_enabled=True,
    modes=[
        ModeParameters(
            name="F-layer",
            sigma_tau=100.0,
            f_c_layer=8e6,        # Critical frequency for QP model
            y_m=100e3,            # Layer semi-thickness
            phi_inc=0.35          # Incidence angle (radians)
        )
    ]
)
```

### Scattering Function Visualization

```python
delay_axis, doppler_axis, S = channel.compute_scattering_function(
    num_delay_bins=64,
    num_doppler_bins=64
)

import matplotlib.pyplot as plt
plt.pcolormesh(delay_axis, doppler_axis, S.T, shading='auto')
plt.xlabel('Delay (µs)')
plt.ylabel('Doppler (Hz)')
plt.colorbar(label='S(τ, f_D)')
```

## Performance Benchmarks

Measured on Intel i7-12700K with NVIDIA RTX 3080:

| Configuration | Sample Rate | Block Size | Processing Time | Real-time Margin |
|---------------|-------------|------------|-----------------|------------------|
| 1 mode, 100 µs spread | 1 MHz | 1024 | 0.3 ms | 3.4× |
| 2 modes, 500 µs spread | 1 MHz | 1024 | 1.8 ms | 0.6× |
| 2 modes + dispersion | 1 MHz | 4096 | 5.2 ms | 0.8× |

Note: Real-time margin < 1× requires GPU acceleration for the dispersion component.

## References

1. Vogler, L.E. and Hoffmeyer, J.A., "A Model for Wideband HF Propagation Channels," NTIA Report 90-255, 1990.

2. Vogler, L.E. and Hoffmeyer, J.A., "A New Approach to HF Channel Modeling and Simulation," NTIA Report 88-240, 1988.

3. Watterson, C.C., Juroshek, J.R., and Bensema, W.D., "Experimental Confirmation of an HF Channel Model," IEEE Trans. Comm. Tech., Vol. COM-18, 1970.

4. CCIR Report 549-3, "HF Ionospheric Channel Simulators," 1990.

5. ITU-R Recommendation F.1487, "Testing of HF modems with bandwidths of up to about 12 kHz using ionospheric channel simulators," 2000.

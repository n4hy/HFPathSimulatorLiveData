# Tutorial 6: Channel Validation

**Time to complete:** 20 minutes
**Prerequisites:** Basic simulation (Tutorial 1), understanding of HF channel statistics

In this tutorial, you'll learn how to:
- Validate simulated channels against published reference data
- Measure delay spread, Doppler spread, and fading statistics
- Compare against NTIA, ITU-R, and Watterson 1970 references
- Diagnose and debug channel model issues

---

## Why Validate?

A channel simulator is only useful if it produces **physically realistic** output. Validation ensures:

1. **Delay spread** matches real ionospheric multipath measurements
2. **Doppler spread** matches observed fading rates
3. **Fading distribution** follows Rayleigh/Rician as expected
4. **Envelope statistics** match published measurement campaigns

The HF Path Simulator includes validation against three authoritative sources:

| Source | Description | Year |
|--------|-------------|------|
| NTIA TR-90-255 | Vogler-Hoffmeyer field measurements | 1990 |
| ITU-R F.1487 | International HF modem testing standard | 2000 |
| Watterson 1970 | Original IEEE channel model validation | 1970 |

---

## Step 1: List Available Reference Datasets

```python
from hfpathsim.validation import list_reference_datasets, get_reference_dataset

# See all available references
datasets = list_reference_datasets()
print("Available reference datasets:")
for name in datasets:
    ref = get_reference_dataset(name)
    print(f"  {name}")
    print(f"    Source: {ref.source}")
    print(f"    Delay spread: {ref.delay_spread_ms:.2f} ms")
    print(f"    Doppler spread: {ref.doppler_spread_hz:.2f} Hz")
    print()
```

Expected output:

```
Available reference datasets:
  ntia_midlatitude_quiet
    Source: NTIA TR-90-255
    Delay spread: 0.50 ms
    Doppler spread: 0.10 Hz

  ntia_midlatitude_disturbed
    Source: NTIA TR-90-255
    Delay spread: 2.00 ms
    Doppler spread: 1.00 Hz

  itu_f1487_moderate
    Source: ITU-R F.1487
    Delay spread: 2.00 ms
    Doppler spread: 1.00 Hz
  ...
```

---

## Step 2: Understand Key Metrics

### RMS Delay Spread

**What it measures:** Time dispersion of multipath arrivals.

**Physical meaning:** If delay spread is 2 ms, signal copies arrive spread over ~2 ms, causing inter-symbol interference for symbols shorter than this.

**Formula:**
```
τ_rms = sqrt(E[(τ - τ_mean)²])
```

where τ is the delay of each path weighted by power.

### RMS Doppler Spread

**What it measures:** Frequency dispersion from ionospheric motion.

**Physical meaning:** If Doppler spread is 1 Hz, the signal spectrum is smeared by ~1 Hz, limiting coherence time to ~1 second.

**Formula:**
```
ν_rms = sqrt(E[(f - f_mean)²])
```

### Rayleigh Envelope Ratio

**What it measures:** Whether fading follows the expected Rayleigh distribution.

**Physical meaning:** For many scattered paths with no dominant component, the envelope should follow Rayleigh distribution with:
```
Mean/RMS = sqrt(π/4) ≈ 0.886
```

If this ratio differs significantly, fading generation may be incorrect.

---

## Step 3: Validate a Channel Model

Let's validate the Watterson channel against the ITU-R F.1487 moderate reference:

```python
import numpy as np
from hfpathsim.core.watterson import WattersonChannel, WattersonConfig, WattersonTap
from hfpathsim.validation import (
    get_reference_dataset,
    compute_delay_spread,
    compute_fading_statistics,
)

# Get reference parameters
ref = get_reference_dataset("itu_f1487_moderate")
print(f"Reference: {ref.name}")
print(f"  Target delay spread: {ref.delay_spread_ms:.2f} ms")
print(f"  Target Doppler spread: {ref.doppler_spread_hz:.2f} Hz")

# Create channel matching reference
# For 2 equal-power taps: RMS delay = max_delay / 2
sample_rate = 48000.0
max_delay_ms = ref.delay_spread_ms * 2.0  # For 2-tap model

config = WattersonConfig(
    sample_rate_hz=sample_rate,
    taps=[
        WattersonTap(delay_ms=0.0, amplitude=1.0, doppler_spread_hz=ref.doppler_spread_hz),
        WattersonTap(delay_ms=max_delay_ms, amplitude=1.0, doppler_spread_hz=ref.doppler_spread_hz),
    ],
)
channel = WattersonChannel(config)

# Generate test signal (CW for clean fading measurement)
duration_sec = 30.0
n_samples = int(duration_sec * sample_rate)
cw_input = np.ones(n_samples, dtype=np.complex128)

# Process through channel
output = channel.process(cw_input)
envelope = np.abs(output)

print(f"\nProcessed {n_samples} samples ({duration_sec} seconds)")
```

---

## Step 4: Measure Delay Spread

```python
# Get impulse response
ir_length = max(1024, int(ref.delay_spread_ms * 5.0 * sample_rate / 1000))
channel.reset()
h = channel.get_impulse_response(length=ir_length)

# Compute delay spread
delay_result = compute_delay_spread(h, sample_rate)

print(f"\nDelay Spread Measurement:")
print(f"  RMS delay spread: {delay_result.rms_delay_spread_ms:.3f} ms")
print(f"  Reference:        {ref.delay_spread_ms:.3f} ms")

error_pct = abs(delay_result.rms_delay_spread_ms - ref.delay_spread_ms) / ref.delay_spread_ms * 100
print(f"  Error:            {error_pct:.1f}%")

if error_pct < 10:
    print("  Status: EXCELLENT")
elif error_pct < 25:
    print("  Status: GOOD")
elif error_pct < 50:
    print("  Status: ACCEPTABLE")
else:
    print("  Status: NEEDS INVESTIGATION")
```

---

## Step 5: Measure Fading Statistics

```python
# Downsample envelope for fading analysis
# Sample at ~10x Doppler spread for accurate measurement
target_rate = max(60, 10 * ref.doppler_spread_hz)
downsample_factor = max(1, int(sample_rate / target_rate))
envelope_ds = envelope[::downsample_factor]

# Compute fading statistics
stats = compute_fading_statistics(envelope_ds, sample_rate / downsample_factor)

print(f"\nFading Statistics:")
print(f"  Fade depth:     {stats.fade_depth_db:.1f} dB")
print(f"  Mean envelope:  {stats.mean_envelope:.4f}")
print(f"  Std envelope:   {stats.std_envelope:.4f}")

# Check Rayleigh envelope ratio
mean_env = np.mean(envelope_ds)
rms_env = np.sqrt(np.mean(envelope_ds**2))
envelope_ratio = mean_env / rms_env

rayleigh_expected = 0.886  # sqrt(pi/4)
ratio_error_pct = abs(envelope_ratio - rayleigh_expected) / rayleigh_expected * 100

print(f"\nRayleigh Envelope Test:")
print(f"  Measured Mean/RMS: {envelope_ratio:.4f}")
print(f"  Expected (Rayleigh): {rayleigh_expected:.4f}")
print(f"  Error: {ratio_error_pct:.1f}%")

if ratio_error_pct < 5:
    print("  Status: EXCELLENT - Rayleigh fading confirmed")
elif ratio_error_pct < 15:
    print("  Status: GOOD - Acceptable Rayleigh match")
else:
    print("  Status: INVESTIGATE - Possible fading generation issue")
```

---

## Step 6: Run Full Validation Suite

The validation script runs all tests against all reference datasets:

```bash
python scripts/validate_channel_models.py --all --verbose
```

Expected output:

```
Running validation against 11 dataset(s)...

============================================================
Validating against: NTIA Midlatitude Quiet
  Source: NTIA TR-90-255
  Condition: quiet
  Reference delay spread: 0.50 ms
  Reference Doppler spread: 0.10 Hz
============================================================

Vogler-Hoffmeyer [PASS]:
  Delay spread: 0.419 ms (ref: 0.500 ms, error: 16.2%)
  Doppler spread: 0.100 Hz (ref: 0.100 Hz, error: 0.0%)
  Envelope ratio: 0.8842 (Rayleigh=0.886)

Watterson [PASS]:
  Delay spread: 0.500 ms (ref: 0.500 ms, error: 0.0%)
  Doppler spread: 0.100 Hz (ref: 0.100 Hz, error: 0.0%)
  Envelope ratio: 0.8857 (Rayleigh=0.886)

...

================================================================================
VALIDATION SUMMARY
================================================================================

Dataset                             Model                Delay Err%   Doppler Err% Status
--------------------------------------------------------------------------------
NTIA Midlatitude Quiet              VoglerHoffmeyer            16.2%        0.0%   PASS
NTIA Midlatitude Quiet              Watterson                   0.0%        0.0%   PASS
...

Overall: 22/22 tests passed (100.0%)
================================================================================
```

---

## Step 7: Diagnose Common Issues

### Issue: High Delay Spread Error

**Symptoms:** Measured delay spread >> or << reference.

**Diagnosis:**

```python
# Check impulse response length
h = channel.get_impulse_response(length=8192)
power_profile = np.abs(h)**2

# Find where power drops to noise floor
threshold = np.max(power_profile) * 0.001  # -30 dB
significant_taps = np.where(power_profile > threshold)[0]

if len(significant_taps) > 0:
    max_significant_delay = significant_taps[-1] / sample_rate * 1000
    print(f"Maximum significant delay: {max_significant_delay:.2f} ms")

    if max_significant_delay > ir_length / sample_rate * 1000 * 0.8:
        print("WARNING: IR length may be too short - increase it")
```

**Common causes:**
- IR length too short to capture full delay spread
- Unequal tap amplitudes skewing the RMS calculation
- Sample rate mismatch

### Issue: Envelope Ratio Far From 0.886

**Symptoms:** Mean/RMS ratio significantly different from Rayleigh expectation.

**Diagnosis:**

```python
# Check if fading is occurring at all
envelope = np.abs(output)
fade_depth = 20 * np.log10(np.max(envelope) / np.min(envelope))
print(f"Fade depth: {fade_depth:.1f} dB")

if fade_depth < 10:
    print("WARNING: Insufficient fading - check Doppler spread configuration")

# Check for numerical issues
if np.any(np.isnan(envelope)) or np.any(np.isinf(envelope)):
    print("ERROR: NaN or Inf in output - numerical instability")

# Check for constant output (no fading)
if np.std(envelope) < 1e-6:
    print("ERROR: Constant envelope - fading not being applied")
```

**Common causes:**
- Uniform random instead of Gaussian (wrong distribution)
- AR(1) correlation coefficient too close to 1 (slow fading)
- Doppler spread parameter misinterpreted (Hz vs rad/s)

### Issue: Power Not Preserved

**Symptoms:** Output power very different from input power.

**Diagnosis:**

```python
input_power = np.mean(np.abs(cw_input)**2)
output_power = np.mean(np.abs(output)**2)
power_ratio = output_power / input_power

print(f"Input power:  {input_power:.4f}")
print(f"Output power: {output_power:.4f}")
print(f"Power ratio:  {power_ratio:.4f}")

if power_ratio < 0.5 or power_ratio > 2.0:
    print("WARNING: Power not preserved - check normalization")
```

---

## Step 8: Validate Vogler-Hoffmeyer Model

The Vogler-Hoffmeyer model requires careful parameter mapping:

```python
from hfpathsim.core.vogler_hoffmeyer import (
    VoglerHoffmeyerChannel,
    VoglerHoffmeyerConfig,
    ModeParameters,
    CorrelationType,
)

ref = get_reference_dataset("ntia_midlatitude_disturbed")

# Map reference delay spread to sigma_tau
# Empirically calibrated: sigma_tau_us ≈ RMS_ms * 6829
sigma_tau_us = ref.delay_spread_ms * 6829.0

mode = ModeParameters(
    name=f"{ref.name} mode",
    amplitude=1.0,
    sigma_tau=sigma_tau_us,
    sigma_c=sigma_tau_us / 2.0,
    sigma_D=ref.doppler_spread_hz,
    correlation_type=CorrelationType.GAUSSIAN,
)

config = VoglerHoffmeyerConfig(
    sample_rate=sample_rate,
    modes=[mode],
)
channel = VoglerHoffmeyerChannel(config)

# Validate as before
output = channel.process(cw_input)
# ... measure and compare statistics
```

---

## Complete Validation Script

Here's a complete script that validates a channel and produces a report:

```python
#!/usr/bin/env python3
"""Complete channel validation example."""

import numpy as np
from hfpathsim.core.watterson import WattersonChannel, WattersonConfig, WattersonTap
from hfpathsim.validation import (
    get_reference_dataset,
    compute_delay_spread,
    compute_fading_statistics,
)

def validate_channel(channel, ref, sample_rate=48000.0, duration_sec=30.0):
    """Validate a channel against a reference dataset."""

    print(f"\n{'='*60}")
    print(f"Validating: {type(channel).__name__}")
    print(f"Reference:  {ref.name}")
    print(f"{'='*60}")

    # Generate CW test signal
    n_samples = int(duration_sec * sample_rate)
    cw_input = np.ones(n_samples, dtype=np.complex128)

    # Process and get envelope
    channel.reset()
    output = channel.process(cw_input)
    envelope = np.abs(output)

    # Measure delay spread from IR
    ir_length = max(1024, int(ref.delay_spread_ms * 5.0 * sample_rate / 1000) + 100)
    channel.reset()

    try:
        h = channel.get_impulse_response(length=ir_length)
    except TypeError:
        h = channel.get_impulse_response(num_samples=ir_length)

    delay_result = compute_delay_spread(h, sample_rate)
    delay_error = abs(delay_result.rms_delay_spread_ms - ref.delay_spread_ms) / ref.delay_spread_ms * 100

    # Measure envelope ratio
    target_rate = max(60, 10 * ref.doppler_spread_hz)
    downsample_factor = max(1, int(sample_rate / target_rate))
    envelope_ds = envelope[::downsample_factor]

    mean_env = np.mean(envelope_ds)
    rms_env = np.sqrt(np.mean(envelope_ds**2))
    envelope_ratio = mean_env / rms_env if rms_env > 0 else 0

    rayleigh_expected = 0.886
    ratio_error = abs(envelope_ratio - rayleigh_expected) / rayleigh_expected * 100

    # Print results
    print(f"\nDelay Spread:")
    print(f"  Measured: {delay_result.rms_delay_spread_ms:.3f} ms")
    print(f"  Expected: {ref.delay_spread_ms:.3f} ms")
    print(f"  Error:    {delay_error:.1f}%")

    print(f"\nEnvelope Ratio (Rayleigh test):")
    print(f"  Measured: {envelope_ratio:.4f}")
    print(f"  Expected: {rayleigh_expected:.4f}")
    print(f"  Error:    {ratio_error:.1f}%")

    # Overall assessment
    passed = delay_error < 50 and ratio_error < 15
    status = "PASS" if passed else "FAIL"
    print(f"\nOverall: {status}")

    return passed


# Run validation
if __name__ == "__main__":
    ref = get_reference_dataset("itu_f1487_moderate")

    # Create Watterson channel
    max_delay_ms = ref.delay_spread_ms * 2.0
    config = WattersonConfig(
        sample_rate_hz=48000.0,
        taps=[
            WattersonTap(delay_ms=0.0, amplitude=1.0, doppler_spread_hz=ref.doppler_spread_hz),
            WattersonTap(delay_ms=max_delay_ms, amplitude=1.0, doppler_spread_hz=ref.doppler_spread_hz),
        ],
    )
    channel = WattersonChannel(config)

    validate_channel(channel, ref)
```

---

## What's Next?

Now that you understand channel validation:

- **[Tutorial 7: HF Modem Testing](07-modem-testing.md)** - Test your HF modem designs
- **[HF Propagation Theory](../hf-propagation-theory.md)** - Deep dive into ionospheric physics
- **[User Guide: Real-World Validation](../user-guide.md#real-world-validation)** - Full validation API reference

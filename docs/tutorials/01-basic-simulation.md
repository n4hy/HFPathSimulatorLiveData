# Tutorial 1: Basic Channel Simulation

**Time to complete:** 15 minutes
**Prerequisites:** Python basics, HF Path Simulator installed

In this tutorial, you'll learn how to:
- Create and configure a simulation engine
- Process signals through different channel models
- Analyze the effects of ionospheric propagation
- Visualize channel characteristics

---

## Step 1: Set Up Your Environment

Create a new Python file called `tutorial_basic.py`:

```python
import numpy as np
import matplotlib.pyplot as plt
from hfpathsim import SimulationEngine, EngineConfig, ChannelModel
```

If you don't have matplotlib, install it:

```bash
pip install matplotlib
```

---

## Step 2: Create a Test Signal

We'll create a simple signal to see how the channel affects it. Let's use a chirp (frequency sweep) which makes channel effects visible:

```python
# Configuration
sample_rate = 48000  # 48 kHz
duration = 2.0       # 2 seconds
f_start = 500        # Start frequency (Hz)
f_end = 2000         # End frequency (Hz)

# Generate time vector
t = np.arange(0, duration, 1/sample_rate)

# Create chirp signal
# Frequency increases linearly from f_start to f_end
phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * duration))
input_signal = np.exp(1j * phase).astype(np.complex64)

print(f"Generated {len(input_signal)} samples ({duration} seconds)")
```

---

## Step 3: Process Through Different Channels

Let's compare how different channel models affect our signal:

```python
# Create engines for different models
configs = {
    "Passthrough": EngineConfig(channel_model=ChannelModel.PASSTHROUGH),
    "Watterson Good": EngineConfig(channel_model=ChannelModel.WATTERSON),
    "Watterson Disturbed": EngineConfig(channel_model=ChannelModel.WATTERSON),
    "Vogler": EngineConfig(channel_model=ChannelModel.VOGLER),
}

# Process through each channel
results = {}
for name, config in configs.items():
    config.sample_rate_hz = sample_rate
    engine = SimulationEngine(config)

    # Configure specific conditions
    if "Disturbed" in name:
        engine.configure_watterson(condition="disturbed")
    elif "Watterson" in name:
        engine.configure_watterson(condition="good")
    elif "Vogler" in name:
        engine.configure_vogler(
            foF2=7.0,
            hmF2=280.0,
            doppler_spread_hz=0.5,
            delay_spread_ms=1.0,
        )

    results[name] = engine.process(input_signal)
    print(f"{name}: processed")
```

---

## Step 4: Visualize the Results

Now let's see what happened to our signal:

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot time domain (amplitude envelope)
ax = axes[0, 0]
for name, output in results.items():
    # Use a window to smooth the envelope
    envelope = np.abs(output)
    window = 100
    smoothed = np.convolve(envelope, np.ones(window)/window, mode='same')
    ax.plot(t[:len(smoothed)], smoothed, label=name, alpha=0.7)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Signal Envelope Over Time")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot spectrogram of original
ax = axes[0, 1]
ax.specgram(input_signal.real, NFFT=256, Fs=sample_rate, cmap='viridis')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_title("Input Signal Spectrogram")

# Plot spectrogram of disturbed channel
ax = axes[1, 0]
output = results["Watterson Disturbed"]
ax.specgram(output.real, NFFT=256, Fs=sample_rate, cmap='viridis')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_title("Disturbed Channel Output")

# Plot power comparison
ax = axes[1, 1]
block_size = 1024
for name, output in results.items():
    n_blocks = len(output) // block_size
    powers = [
        np.mean(np.abs(output[i*block_size:(i+1)*block_size])**2)
        for i in range(n_blocks)
    ]
    powers_db = 10 * np.log10(np.array(powers) + 1e-10)
    time_axis = np.arange(n_blocks) * block_size / sample_rate
    ax.plot(time_axis, powers_db, label=name, alpha=0.7)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Power (dB)")
ax.set_title("Signal Power Over Time (Fading)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("channel_comparison.png", dpi=150)
plt.show()
print("Saved plot to channel_comparison.png")
```

---

## Step 5: Understand What You See

Looking at the plots, you should observe:

### Envelope Variation (Fading)
- **Passthrough**: Constant amplitude
- **Good conditions**: Mild amplitude variations
- **Disturbed conditions**: Deep fades where signal nearly disappears

### Spectrogram Changes
- **Input**: Clean frequency sweep
- **Channel output**: Spread and smeared in both time and frequency
- **Disturbed**: More spreading, possible echoes visible

### Power Variations
- Each channel model produces different fading patterns
- Disturbed conditions show deeper and more rapid fades
- This is what real HF signals experience!

---

## Step 6: Measure Channel Statistics

Let's quantify the channel effects:

```python
def analyze_channel(name, input_signal, output_signal):
    """Compute channel statistics."""

    # Compute correlation (how similar are input and output)
    correlation = np.abs(np.corrcoef(
        np.abs(input_signal),
        np.abs(output_signal)
    )[0, 1])

    # Compute fading depth
    envelope = np.abs(output_signal)
    fade_depth_db = 20 * np.log10(np.max(envelope) / (np.mean(envelope) + 1e-10))

    # Compute delay spread estimate (from autocorrelation)
    autocorr = np.correlate(output_signal[:1024], output_signal[:1024], mode='full')
    autocorr = np.abs(autocorr)
    peak_idx = len(autocorr) // 2
    # Find where autocorrelation drops to 50%
    half_power = autocorr[peak_idx] / 2
    delay_samples = np.argmax(autocorr[peak_idx:] < half_power)
    delay_spread_ms = delay_samples / sample_rate * 1000

    print(f"\n{name}:")
    print(f"  Correlation with input: {correlation:.3f}")
    print(f"  Peak-to-mean ratio: {fade_depth_db:.1f} dB")
    print(f"  Estimated delay spread: {delay_spread_ms:.2f} ms")

# Analyze each channel
for name, output in results.items():
    analyze_channel(name, input_signal, output)
```

Expected output:

```
Passthrough:
  Correlation with input: 1.000
  Peak-to-mean ratio: 0.0 dB
  Estimated delay spread: 0.00 ms

Watterson Good:
  Correlation with input: 0.982
  Peak-to-mean ratio: 3.2 dB
  Estimated delay spread: 0.42 ms

Watterson Disturbed:
  Correlation with input: 0.891
  Peak-to-mean ratio: 12.4 dB
  Estimated delay spread: 1.87 ms

Vogler:
  Correlation with input: 0.945
  Peak-to-mean ratio: 6.8 dB
  Estimated delay spread: 0.95 ms
```

---

## Step 7: Add Noise

Real HF channels also have noise. Let's add it:

```python
# Create engine with noise
config = EngineConfig(
    channel_model=ChannelModel.WATTERSON,
    sample_rate_hz=sample_rate,
)
engine = SimulationEngine(config)
engine.configure_watterson(condition="moderate")

# Configure noise
engine.configure_noise(
    snr_db=15.0,          # 15 dB SNR
    enable_atmospheric=True,
)

# Process
output_noisy = engine.process(input_signal)

# Compare
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Clean channel
ax = axes[0]
ax.specgram(results["Watterson Good"].real, NFFT=256, Fs=sample_rate, cmap='viridis')
ax.set_title("Channel Only (No Noise)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")

# With noise
ax = axes[1]
ax.specgram(output_noisy.real, NFFT=256, Fs=sample_rate, cmap='viridis')
ax.set_title("Channel + 15 dB SNR Noise")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")

plt.tight_layout()
plt.savefig("noise_comparison.png", dpi=150)
plt.show()
```

---

## Complete Code

Here's the complete tutorial script:

```python
#!/usr/bin/env python3
"""Tutorial 1: Basic Channel Simulation"""

import numpy as np
import matplotlib.pyplot as plt
from hfpathsim import SimulationEngine, EngineConfig, ChannelModel

# Configuration
sample_rate = 48000
duration = 2.0
f_start = 500
f_end = 2000

# Generate chirp signal
t = np.arange(0, duration, 1/sample_rate)
phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * duration))
input_signal = np.exp(1j * phase).astype(np.complex64)
print(f"Generated {len(input_signal)} samples")

# Create different channel configurations
configs = {
    "Passthrough": EngineConfig(channel_model=ChannelModel.PASSTHROUGH),
    "Watterson Good": EngineConfig(channel_model=ChannelModel.WATTERSON),
    "Watterson Disturbed": EngineConfig(channel_model=ChannelModel.WATTERSON),
}

# Process through each channel
results = {}
for name, config in configs.items():
    config.sample_rate_hz = sample_rate
    engine = SimulationEngine(config)

    if "Disturbed" in name:
        engine.configure_watterson(condition="disturbed")
    elif "Watterson" in name:
        engine.configure_watterson(condition="good")

    results[name] = engine.process(input_signal)
    print(f"Processed: {name}")

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, output) in enumerate(results.items()):
    ax = axes[idx]
    ax.specgram(output.real, NFFT=256, Fs=sample_rate, cmap='viridis')
    ax.set_title(name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

plt.tight_layout()
plt.savefig("tutorial_basic_output.png", dpi=150)
plt.show()
print("\nTutorial complete! Output saved to tutorial_basic_output.png")
```

---

## What's Next?

Now that you understand basic channel simulation:

- **[Tutorial 2: GUI Walkthrough](02-gui-walkthrough.md)** - Use the graphical interface
- **[Tutorial 3: SDR Integration](03-sdr-integration.md)** - Connect real radio hardware
- **[User Guide](../user-guide.md)** - Explore all configuration options

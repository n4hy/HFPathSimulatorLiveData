# Tutorial 5: Custom Channel Models

**Time to complete:** 30 minutes
**Prerequisites:** Python experience, understanding of HF propagation basics

In this tutorial, you'll learn how to:
- Understand the channel model architecture
- Create custom channel models
- Implement specific propagation scenarios
- Test and validate your models

---

## Overview

HF Path Simulator's channel models are modular and extensible. You can:

1. **Subclass** the base channel class
2. **Implement** the required methods
3. **Register** your model with the engine

---

## Step 1: Understanding Channel Models

Every channel model implements this interface:

```python
from abc import ABC, abstractmethod
import numpy as np

class ChannelModel(ABC):
    """Base class for all channel models."""

    @abstractmethod
    def process(self, samples: np.ndarray) -> np.ndarray:
        """Process samples through the channel.

        Args:
            samples: Complex input samples (np.complex64)

        Returns:
            Complex output samples (np.complex64)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset channel state (clear delays, reset fading)."""
        pass

    @abstractmethod
    def get_state(self) -> dict:
        """Return current channel state for monitoring."""
        pass
```

---

## Step 2: Simple Custom Channel

Let's create a simple multipath channel with fixed delays:

```python
import numpy as np
from hfpathsim.core.channel import BaseChannel

class SimpleMultipathChannel(BaseChannel):
    """Simple multipath channel with fixed delays and gains."""

    def __init__(
        self,
        sample_rate_hz: float,
        delays_samples: list[int],
        gains_linear: list[float],
        phases_rad: list[float] = None,
    ):
        """
        Args:
            sample_rate_hz: Sample rate in Hz
            delays_samples: Delay for each path in samples
            gains_linear: Linear gain for each path
            phases_rad: Phase shift for each path (optional)
        """
        super().__init__(sample_rate_hz)

        self.delays = np.array(delays_samples, dtype=np.int32)
        self.gains = np.array(gains_linear, dtype=np.float32)

        if phases_rad is None:
            self.phases = np.zeros(len(delays_samples), dtype=np.float32)
        else:
            self.phases = np.array(phases_rad, dtype=np.float32)

        # Buffer for delay line
        max_delay = max(self.delays) if len(self.delays) > 0 else 0
        self.delay_buffer = np.zeros(max_delay, dtype=np.complex64)

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Apply multipath to samples."""
        # Extend buffer with new samples
        extended = np.concatenate([self.delay_buffer, samples])

        # Sum all paths
        output = np.zeros(len(samples), dtype=np.complex64)

        for delay, gain, phase in zip(self.delays, self.gains, self.phases):
            # Extract delayed samples
            start = len(self.delay_buffer) - delay
            end = start + len(samples)
            delayed = extended[start:end]

            # Apply gain and phase
            output += gain * np.exp(1j * phase) * delayed

        # Update buffer for next call
        if len(self.delay_buffer) > 0:
            self.delay_buffer = extended[-len(self.delay_buffer):]

        return output.astype(np.complex64)

    def reset(self) -> None:
        """Clear delay buffer."""
        self.delay_buffer.fill(0)

    def get_state(self) -> dict:
        """Return channel state."""
        return {
            "num_paths": len(self.delays),
            "delays_samples": self.delays.tolist(),
            "gains_db": (20 * np.log10(self.gains + 1e-10)).tolist(),
        }
```

### Using Your Custom Channel

```python
# Create the channel
channel = SimpleMultipathChannel(
    sample_rate_hz=48000,
    delays_samples=[0, 48, 144],     # 0, 1ms, 3ms delays
    gains_linear=[1.0, 0.7, 0.3],    # Relative amplitudes
    phases_rad=[0, 0.5, 1.2],        # Phase offsets
)

# Process samples
input_signal = np.exp(1j * np.linspace(0, 100, 4096)).astype(np.complex64)
output_signal = channel.process(input_signal)
```

---

## Step 3: Adding Fading

Let's enhance our channel with Rayleigh fading:

```python
import numpy as np
from scipy import signal as scipy_signal
from hfpathsim.core.channel import BaseChannel

class FadingMultipathChannel(BaseChannel):
    """Multipath channel with Rayleigh fading on each path."""

    def __init__(
        self,
        sample_rate_hz: float,
        delays_samples: list[int],
        avg_gains_linear: list[float],
        doppler_hz: float = 1.0,
    ):
        super().__init__(sample_rate_hz)

        self.delays = np.array(delays_samples, dtype=np.int32)
        self.avg_gains = np.array(avg_gains_linear, dtype=np.float32)
        self.doppler_hz = doppler_hz
        self.num_paths = len(delays_samples)

        # Fading state for each path
        self.fading_phase = np.zeros(self.num_paths, dtype=np.float32)
        self.fading_rate = 2 * np.pi * doppler_hz / sample_rate_hz

        # Delay buffer
        max_delay = max(self.delays) if len(self.delays) > 0 else 0
        self.delay_buffer = np.zeros(max_delay, dtype=np.complex64)

        # Lowpass filter for fading (Jakes spectrum approximation)
        self._design_fading_filter()

    def _design_fading_filter(self):
        """Design lowpass filter for Rayleigh fading generation."""
        # Cutoff at Doppler frequency
        normalized_cutoff = self.doppler_hz / (self.sample_rate_hz / 2)
        normalized_cutoff = min(normalized_cutoff, 0.99)  # Ensure valid

        self.fading_b, self.fading_a = scipy_signal.butter(
            4, normalized_cutoff, btype='low'
        )

        # Filter state for each path (I and Q)
        self.fading_zi = [
            scipy_signal.lfilter_zi(self.fading_b, self.fading_a)
            for _ in range(self.num_paths * 2)
        ]

    def _generate_fading(self, num_samples: int) -> np.ndarray:
        """Generate Rayleigh fading coefficients for each path."""
        fading = np.zeros((self.num_paths, num_samples), dtype=np.complex64)

        for i in range(self.num_paths):
            # Generate filtered Gaussian noise (I and Q)
            noise_i = np.random.randn(num_samples)
            noise_q = np.random.randn(num_samples)

            filtered_i, self.fading_zi[i*2] = scipy_signal.lfilter(
                self.fading_b, self.fading_a, noise_i, zi=self.fading_zi[i*2]
            )
            filtered_q, self.fading_zi[i*2+1] = scipy_signal.lfilter(
                self.fading_b, self.fading_a, noise_q, zi=self.fading_zi[i*2+1]
            )

            # Complex fading coefficient (Rayleigh distributed amplitude)
            fading[i] = (filtered_i + 1j * filtered_q) / np.sqrt(2)

        return fading

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Apply fading multipath channel."""
        extended = np.concatenate([self.delay_buffer, samples])

        # Generate fading for this block
        fading = self._generate_fading(len(samples))

        # Sum all paths with fading
        output = np.zeros(len(samples), dtype=np.complex64)

        for i, (delay, avg_gain) in enumerate(zip(self.delays, self.avg_gains)):
            start = len(self.delay_buffer) - delay
            end = start + len(samples)
            delayed = extended[start:end]

            # Apply average gain and fading
            output += avg_gain * fading[i] * delayed

        # Update buffer
        if len(self.delay_buffer) > 0:
            self.delay_buffer = extended[-len(self.delay_buffer):]

        return output.astype(np.complex64)

    def reset(self) -> None:
        """Reset channel state."""
        self.delay_buffer.fill(0)
        self.fading_phase.fill(0)
        self._design_fading_filter()  # Reset filter state

    def get_state(self) -> dict:
        """Return channel state."""
        return {
            "num_paths": self.num_paths,
            "doppler_hz": self.doppler_hz,
            "delays_ms": (self.delays / self.sample_rate_hz * 1000).tolist(),
        }
```

---

## Step 4: Registering Your Model

Register your custom channel with the engine:

```python
from hfpathsim import SimulationEngine, EngineConfig
from hfpathsim.core.channel import register_channel_model

# Register your model
register_channel_model("fading_multipath", FadingMultipathChannel)

# Now you can use it
config = EngineConfig(
    channel_model="fading_multipath",  # Your custom model
    sample_rate_hz=48000,
)

engine = SimulationEngine(config)

# Configure your model's parameters
engine.configure_channel(
    delays_samples=[0, 48, 96],
    avg_gains_linear=[1.0, 0.5, 0.25],
    doppler_hz=2.0,
)
```

---

## Step 5: Implementing Specific Scenarios

### Near-Vertical Incidence Skywave (NVIS)

```python
class NVISChannel(BaseChannel):
    """NVIS propagation channel (short range, high angle)."""

    def __init__(self, sample_rate_hz: float, time_of_day: str = "day"):
        super().__init__(sample_rate_hz)

        # NVIS typically has short delays but can have fading
        if time_of_day == "day":
            # Daytime: D-layer absorption, single F2 hop
            self.delays_samples = [0]
            self.gains_linear = [0.3]  # D-layer loss
            self.doppler_hz = 0.2
        else:
            # Nighttime: No D-layer, possible multi-hop
            self.delays_samples = [0, int(0.5e-3 * sample_rate_hz)]
            self.gains_linear = [0.8, 0.3]
            self.doppler_hz = 0.5

        # ... implement process(), reset(), get_state()
```

### Long-Path Propagation

```python
class LongPathChannel(BaseChannel):
    """Long-path (antipodal) propagation channel."""

    def __init__(self, sample_rate_hz: float, path_km: float = 20000):
        super().__init__(sample_rate_hz)

        # Calculate delays based on path length
        c = 299792.458  # km/s
        num_hops = int(path_km / 3000)  # Approximate hops

        # Each hop adds delay and loss
        self.delays_samples = []
        self.gains_linear = []

        base_delay_s = path_km / c
        for i in range(num_hops):
            delay_s = base_delay_s * (i + 1) / num_hops
            self.delays_samples.append(int(delay_s * sample_rate_hz))
            self.gains_linear.append(0.8 ** (i + 1))  # 2dB loss per hop

        self.doppler_hz = 5.0  # Higher Doppler for long path

        # ... implement remaining methods
```

### Auroral Flutter

```python
class AuroralChannel(BaseChannel):
    """Auroral propagation with characteristic flutter."""

    def __init__(self, sample_rate_hz: float, flutter_rate_hz: float = 10.0):
        super().__init__(sample_rate_hz)

        self.flutter_rate = flutter_rate_hz
        self.flutter_depth = 0.8  # Deep fading
        self.phase = 0.0

        # Auroral paths are often scattered
        self.num_scatterers = 20
        self.scatter_delays = np.random.exponential(
            0.5e-3 * sample_rate_hz,
            self.num_scatterers
        ).astype(int)

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Apply auroral flutter."""
        # Generate flutter (amplitude modulation)
        t = np.arange(len(samples)) / self.sample_rate_hz
        flutter = 1.0 - self.flutter_depth * (
            0.5 + 0.5 * np.sin(2 * np.pi * self.flutter_rate * t + self.phase)
        )
        self.phase += 2 * np.pi * self.flutter_rate * len(samples) / self.sample_rate_hz

        # Apply scattering and flutter
        output = samples * flutter.astype(np.float32)

        return output.astype(np.complex64)
```

---

## Step 6: Testing Your Model

### Unit Tests

```python
import pytest
import numpy as np

class TestFadingMultipathChannel:

    def test_initialization(self):
        """Test channel initializes correctly."""
        channel = FadingMultipathChannel(
            sample_rate_hz=48000,
            delays_samples=[0, 48],
            avg_gains_linear=[1.0, 0.5],
            doppler_hz=1.0,
        )

        assert channel.num_paths == 2
        assert len(channel.delays) == 2

    def test_process_preserves_shape(self):
        """Test output has same length as input."""
        channel = FadingMultipathChannel(48000, [0, 48], [1.0, 0.5])

        input_samples = np.random.randn(4096) + 1j * np.random.randn(4096)
        input_samples = input_samples.astype(np.complex64)

        output = channel.process(input_samples)

        assert output.shape == input_samples.shape
        assert output.dtype == np.complex64

    def test_reset_clears_state(self):
        """Test reset clears delay buffer."""
        channel = FadingMultipathChannel(48000, [0, 48], [1.0, 0.5])

        # Process some samples
        samples = np.ones(1000, dtype=np.complex64)
        channel.process(samples)

        # Reset
        channel.reset()

        # Buffer should be cleared
        assert np.allclose(channel.delay_buffer, 0)

    def test_fading_statistics(self):
        """Test fading has expected Rayleigh statistics."""
        channel = FadingMultipathChannel(
            sample_rate_hz=48000,
            delays_samples=[0],
            avg_gains_linear=[1.0],
            doppler_hz=10.0,
        )

        # Process constant signal to see fading
        input_samples = np.ones(100000, dtype=np.complex64)
        output = channel.process(input_samples)

        # Check amplitude follows Rayleigh distribution
        amplitudes = np.abs(output)
        # Rayleigh mean ≈ σ√(π/2) ≈ 1.25σ for unit variance
        # Allow some tolerance
        assert 0.5 < np.mean(amplitudes) < 2.0
```

### Visual Validation

```python
import matplotlib.pyplot as plt

def visualize_channel(channel, duration_s=5.0):
    """Visualize channel impulse response and fading."""
    sample_rate = channel.sample_rate_hz
    num_samples = int(duration_s * sample_rate)

    # Generate impulse train
    impulse_period = int(0.1 * sample_rate)  # 100ms
    impulses = np.zeros(num_samples, dtype=np.complex64)
    impulses[::impulse_period] = 1.0

    # Process through channel
    output = channel.process(impulses)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Impulse response
    ax = axes[0, 0]
    ax.plot(np.abs(output[:impulse_period*2]))
    ax.set_title("Impulse Response")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")

    # Fading over time
    ax = axes[0, 1]
    envelope = np.abs(output)
    ax.plot(np.arange(len(envelope)) / sample_rate, envelope)
    ax.set_title("Fading Envelope")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    # Amplitude histogram
    ax = axes[1, 0]
    ax.hist(envelope, bins=50, density=True)
    ax.set_title("Amplitude Distribution")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Density")

    # Spectrogram
    ax = axes[1, 1]
    ax.specgram(output, NFFT=256, Fs=sample_rate)
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig("channel_visualization.png")
    plt.show()

# Visualize
channel = FadingMultipathChannel(48000, [0, 48, 144], [1.0, 0.7, 0.3], 2.0)
visualize_channel(channel)
```

---

## Step 7: Performance Optimization

### NumPy Vectorization

```python
# Slow (loop-based)
def process_slow(self, samples):
    output = np.zeros_like(samples)
    for i, sample in enumerate(samples):
        for delay, gain in zip(self.delays, self.gains):
            if i >= delay:
                output[i] += gain * samples[i - delay]
    return output

# Fast (vectorized)
def process_fast(self, samples):
    output = np.zeros_like(samples)
    for delay, gain in zip(self.delays, self.gains):
        output += gain * np.roll(samples, delay)
    return output
```

### GPU Acceleration

```python
import cupy as cp  # GPU arrays

class GPUFadingChannel(BaseChannel):
    """GPU-accelerated fading channel."""

    def process(self, samples: np.ndarray) -> np.ndarray:
        # Transfer to GPU
        samples_gpu = cp.asarray(samples)

        # Process on GPU
        output_gpu = cp.zeros_like(samples_gpu)
        for delay, gain in zip(self.delays, self.gains):
            output_gpu += gain * cp.roll(samples_gpu, delay)

        # Transfer back to CPU
        return cp.asnumpy(output_gpu)
```

---

## What's Next?

You now know how to create custom channel models. Explore:

- **[Python API Reference](../api/python-api.md)** - Full API documentation
- **[User Guide](../user-guide.md)** - All configuration options
- **[REST API Reference](../api/rest-api.md)** - Remote control via HTTP

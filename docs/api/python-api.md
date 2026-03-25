# Python API Reference

This reference covers the main Python classes and functions in HF Path Simulator.

## Quick Start

```python
from hfpathsim import SimulationEngine, EngineConfig, ChannelModel
import numpy as np

# Create engine
config = EngineConfig(channel_model=ChannelModel.WATTERSON)
engine = SimulationEngine(config)

# Process samples
samples = np.random.randn(4096).astype(np.complex64)
output = engine.process(samples)
```

---

## Core Classes

### SimulationEngine

The main interface for channel simulation.

```python
class SimulationEngine:
    """HF channel simulation engine."""

    def __init__(self, config: EngineConfig):
        """Initialize simulation engine.

        Args:
            config: Engine configuration
        """

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Process samples through the channel.

        Args:
            samples: Input samples (complex64 or complex128)

        Returns:
            Processed samples (same type as input)
        """

    def configure_watterson(
        self,
        condition: str = "moderate",
        num_paths: int = 2,
        doppler_spread_hz: float = None,
        delay_spread_ms: float = None,
    ) -> None:
        """Configure Watterson channel model.

        Args:
            condition: "good", "moderate", or "disturbed"
            num_paths: Number of propagation paths (1-4)
            doppler_spread_hz: Override doppler spread
            delay_spread_ms: Override delay spread
        """

    def configure_vogler(
        self,
        foF2: float = 8.0,
        hmF2: float = 300.0,
        foE: float = 3.0,
        hmE: float = 110.0,
        doppler_spread_hz: float = 1.0,
        delay_spread_ms: float = 2.0,
    ) -> None:
        """Configure Vogler channel model.

        Args:
            foF2: F2 layer critical frequency (MHz)
            hmF2: F2 layer peak height (km)
            foE: E layer critical frequency (MHz)
            hmE: E layer peak height (km)
            doppler_spread_hz: Doppler spread (Hz)
            delay_spread_ms: Delay spread (ms)
        """

    def configure_vogler_hoffmeyer(
        self,
        condition: str = "moderate",
        sporadic_e_enabled: bool = False,
        spread_f_enabled: bool = False,
        magnetic_storm: bool = False,
    ) -> None:
        """Configure Vogler-Hoffmeyer channel model."""

    def configure_noise(
        self,
        snr_db: float,
        enable_atmospheric: bool = False,
        enable_galactic: bool = False,
        enable_man_made: bool = False,
    ) -> None:
        """Configure noise addition.

        Args:
            snr_db: Target signal-to-noise ratio
            enable_atmospheric: Use ITU-R P.372 atmospheric model
            enable_galactic: Include galactic background
            enable_man_made: Include man-made interference
        """

    def disable_noise(self) -> None:
        """Disable noise addition."""

    def configure_agc(
        self,
        enabled: bool,
        target_level_db: float = -20.0,
        attack_time_ms: float = 10.0,
        release_time_ms: float = 100.0,
        max_gain_db: float = 60.0,
    ) -> None:
        """Configure automatic gain control."""

    def configure_limiter(
        self,
        enabled: bool,
        threshold_db: float = -3.0,
    ) -> None:
        """Configure signal limiter."""

    def configure_frequency_offset(
        self,
        enabled: bool,
        offset_hz: float = 0.0,
        drift_hz_per_sec: float = 0.0,
    ) -> None:
        """Configure frequency offset simulation."""

    def reset(self) -> None:
        """Reset all channel state."""

    def get_state(self) -> dict:
        """Get current engine state.

        Returns:
            Dictionary with running state, sample counts, etc.
        """

    def start_streaming(
        self,
        input_source: InputSource,
        output_sink: OutputSink = None,
    ) -> None:
        """Start continuous streaming processing.

        Args:
            input_source: Source of input samples
            output_sink: Optional destination for output
        """

    def stop_streaming(self) -> None:
        """Stop streaming processing."""

    @property
    def config(self) -> EngineConfig:
        """Get engine configuration."""
```

---

### EngineConfig

Configuration for SimulationEngine.

```python
from dataclasses import dataclass
from enum import Enum

class ChannelModel(Enum):
    """Available channel models."""
    PASSTHROUGH = "passthrough"
    WATTERSON = "watterson"
    VOGLER = "vogler"
    VOGLER_HOFFMEYER = "vogler_hoffmeyer"

@dataclass
class EngineConfig:
    """Simulation engine configuration.

    Attributes:
        channel_model: Which channel model to use
        sample_rate_hz: Sample rate in Hz
        block_size: Processing block size
        use_gpu: Enable GPU acceleration
        num_threads: CPU threads (when GPU disabled)
    """
    channel_model: ChannelModel = ChannelModel.WATTERSON
    sample_rate_hz: int = 48000
    block_size: int = 4096
    use_gpu: bool = True
    num_threads: int = 4
```

---

## Input Sources

### Base Class

```python
from abc import ABC, abstractmethod

class InputSource(ABC):
    """Base class for input sources."""

    @abstractmethod
    def start(self) -> None:
        """Start the input source."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the input source."""

    @abstractmethod
    def read(self, num_samples: int) -> np.ndarray:
        """Read samples from the source.

        Args:
            num_samples: Number of samples to read

        Returns:
            Complex samples, or None if not available
        """

    @property
    @abstractmethod
    def sample_rate_hz(self) -> int:
        """Get sample rate."""
```

### FileSource

```python
from hfpathsim.input import FileSource

source = FileSource(
    path="recording.wav",       # File path
    format="auto",              # "auto", "wav", "raw", "sigmf"
    dtype=np.complex64,         # Data type for raw files
    sample_rate_hz=48000,       # Sample rate for raw files
    loop=False,                 # Loop when reaching end
)

source.start()
samples = source.read(4096)
source.stop()
```

### AudioSource

```python
from hfpathsim.input import AudioSource, list_audio_devices

# List available devices
devices = list_audio_devices()
for dev in devices:
    print(f"{dev['index']}: {dev['name']}")

# Create source
source = AudioSource(
    device_name="USB Audio",    # Or device_index=0
    sample_rate_hz=48000,
    channels=2,                 # 1 for mono, 2 for stereo→IQ
    buffer_size=4096,
)
```

### SDRSource

```python
from hfpathsim.input import SDRSource, list_sdr_devices

# List available SDRs
devices = list_sdr_devices()

# Create source
source = SDRSource(
    driver="rtlsdr",            # "rtlsdr", "hackrf", "lime", "uhd"
    center_freq_hz=7_200_000,
    sample_rate_hz=2_400_000,
    gain_db=30,
    bandwidth_hz=None,          # Auto
    antenna=None,               # Default antenna
)
```

### ZMQSource

```python
from hfpathsim.input import ZMQSource

source = ZMQSource(
    address="tcp://192.168.1.100:5555",
    socket_type="PULL",         # "PULL" or "SUB"
    topic="",                   # For SUB sockets
    dtype=np.complex64,
)
```

---

## Output Sinks

### Base Class

```python
class OutputSink(ABC):
    """Base class for output sinks."""

    @abstractmethod
    def start(self) -> None:
        """Start the output sink."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the output sink."""

    @abstractmethod
    def write(self, samples: np.ndarray) -> None:
        """Write samples to the sink."""
```

### FileOutputSink

```python
from hfpathsim.output import FileOutputSink

sink = FileOutputSink(
    path="output.wav",
    format="wav",               # "wav", "raw", "sigmf"
    sample_rate_hz=48000,
    metadata=None,              # Optional SigMF metadata
)

sink.start()
sink.write(samples)
sink.stop()
```

### AudioOutputSink

```python
from hfpathsim.output import AudioOutputSink

sink = AudioOutputSink(
    device_name="Speakers",
    sample_rate_hz=48000,
    channels=2,
    buffer_size=4096,
)
```

### SDROutputSink

```python
from hfpathsim.output import SDROutputSink

sink = SDROutputSink(
    driver="hackrf",
    center_freq_hz=7_200_000,
    sample_rate_hz=2_000_000,
    tx_gain_db=0,
    bandwidth_hz=None,
)
```

### ZMQOutputSink

```python
from hfpathsim.output import ZMQOutputSink

sink = ZMQOutputSink(
    address="tcp://*:5556",
    socket_type="PUSH",         # "PUSH" or "PUB"
)
```

### MultiplexSink

Send to multiple outputs simultaneously:

```python
from hfpathsim.output import MultiplexSink

sink = MultiplexSink([
    FileOutputSink("recording.wav"),
    ZMQOutputSink("tcp://*:5556"),
])
```

---

## GPU Functions

```python
from hfpathsim.gpu import (
    is_gpu_available,
    get_gpu_info,
    get_gpu_memory_info,
    clear_gpu_cache,
)

# Check availability
if is_gpu_available():
    info = get_gpu_info()
    print(f"GPU: {info['name']}")
    print(f"Memory: {info['total_memory_gb']:.1f} GB")

# Get memory usage
mem = get_gpu_memory_info()
print(f"Used: {mem['used_gb']:.1f} / {mem['total_gb']:.1f} GB")

# Clear cache
clear_gpu_cache()
```

---

## Channel Models

### Watterson

```python
from hfpathsim.core.watterson import WattersonChannel

channel = WattersonChannel(
    sample_rate_hz=48000,
    condition="moderate",       # "good", "moderate", "disturbed"
    num_paths=2,
)

output = channel.process(input_samples)
state = channel.get_state()
channel.reset()
```

### Vogler

```python
from hfpathsim.core.vogler import VoglerChannel

channel = VoglerChannel(
    sample_rate_hz=48000,
    foF2=8.0,
    hmF2=300.0,
    doppler_spread_hz=1.0,
)

output = channel.process(input_samples)
```

### Vogler-Hoffmeyer

```python
from hfpathsim.core.vogler_hoffmeyer import VoglerHoffmeyerChannel

channel = VoglerHoffmeyerChannel(
    sample_rate_hz=48000,
    condition="moderate",
    sporadic_e_enabled=True,
)

output = channel.process(input_samples)
```

---

## Integration

### ZMQ Bridge

```python
from hfpathsim.integration import ZMQBridge

bridge = ZMQBridge(
    config=EngineConfig(channel_model=ChannelModel.WATTERSON),
    input_address="tcp://*:5555",
    output_address="tcp://*:5556",
)

bridge.start()
# Bridge runs in background
bridge.stop()
```

### GNU Radio Integration

```python
from hfpathsim.integration import generate_gnuradio_snippet

# Generate GNU Radio Python code
code = generate_gnuradio_snippet(
    input_port=5555,
    output_port=5556,
    sample_rate=48000,
)
print(code)
```

---

## Utilities

### Sample Rate Conversion

```python
from hfpathsim.utils import resample

output = resample(
    samples,
    input_rate=2_400_000,
    output_rate=48_000,
    method="polyphase",         # "polyphase" or "fft"
)
```

### Format Conversion

```python
from hfpathsim.utils import convert_samples

# Real to complex
complex_samples = convert_samples(real_samples, "real", "complex64")

# Complex128 to complex64
float_samples = convert_samples(complex128_samples, "complex128", "complex64")
```

### Signal Generation

```python
from hfpathsim.utils import generate_test_signal

# Generate various test signals
tone = generate_test_signal("tone", sample_rate=48000, duration=1.0, freq=1000)
noise = generate_test_signal("noise", sample_rate=48000, duration=1.0)
chirp = generate_test_signal("chirp", sample_rate=48000, duration=1.0,
                             f_start=500, f_end=2000)
```

---

## Type Hints

HF Path Simulator uses type hints throughout. Key types:

```python
from typing import Optional, Dict, List, Union
import numpy as np
from numpy.typing import NDArray

# Sample arrays
ComplexSamples = NDArray[np.complex64]
RealSamples = NDArray[np.float32]

# Configuration types
ChannelCondition = Literal["good", "moderate", "disturbed"]
```

---

## Exceptions

```python
from hfpathsim.exceptions import (
    HFPathSimError,        # Base exception
    ConfigurationError,    # Invalid configuration
    ProcessingError,       # Processing failed
    GPUError,              # GPU-related error
    DeviceError,           # Hardware device error
)

try:
    engine.process(samples)
except ProcessingError as e:
    print(f"Processing failed: {e}")
except GPUError as e:
    print(f"GPU error: {e}")
```

---

## Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific modules
logging.getLogger("hfpathsim.engine").setLevel(logging.DEBUG)
logging.getLogger("hfpathsim.gpu").setLevel(logging.INFO)
```

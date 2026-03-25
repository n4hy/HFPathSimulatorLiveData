# User Guide

This comprehensive guide covers all features of HF Path Simulator. If you're new, start with the [Getting Started](getting-started.md) guide first.

## Table of Contents

1. [Overview](#overview)
2. [Channel Models](#channel-models)
3. [Configuration](#configuration)
4. [Input Sources](#input-sources)
5. [Output Destinations](#output-destinations)
6. [Signal Impairments](#signal-impairments)
7. [GPU Acceleration](#gpu-acceleration)
8. [Using the GUI](#using-the-gui)
9. [Command Line Interface](#command-line-interface)
10. [Performance Tuning](#performance-tuning)

---

## Overview

HF Path Simulator processes complex baseband IQ samples through a realistic ionospheric channel model. The processing pipeline is:

```
Input Source → Channel Model → Impairments → Output Destination
```

- **Input Source**: Where samples come from (file, SDR, network, audio)
- **Channel Model**: How the ionosphere affects the signal
- **Impairments**: Additional effects (noise, AGC, frequency offset)
- **Output Destination**: Where processed samples go

---

## Channel Models

### Watterson Model

The classic ITU-R F.520 HF channel model, widely used for modem testing.

```python
from hfpathsim import SimulationEngine, EngineConfig, ChannelModel

config = EngineConfig(channel_model=ChannelModel.WATTERSON)
engine = SimulationEngine(config)

# Configure for specific conditions
engine.configure_watterson(
    condition="moderate",  # "good", "moderate", "disturbed"
    num_paths=2,           # Number of propagation paths
)
```

**Conditions:**

| Condition | Doppler Spread | Delay Spread | Use Case |
|-----------|---------------|--------------|----------|
| Good | 0.1 Hz | 0.5 ms | Daytime, quiet conditions |
| Moderate | 0.5 Hz | 1.0 ms | Typical HF propagation |
| Disturbed | 1.0 Hz | 2.0 ms | Auroral, storm conditions |

### Vogler Model

Physics-based model using actual ionospheric parameters.

```python
config = EngineConfig(channel_model=ChannelModel.VOGLER)
engine = SimulationEngine(config)

# Configure with ionospheric parameters
engine.configure_vogler(
    foF2=8.0,           # F2 layer critical frequency (MHz)
    hmF2=300.0,         # F2 layer peak height (km)
    foE=3.0,            # E layer critical frequency (MHz)
    hmE=110.0,          # E layer peak height (km)
    doppler_spread_hz=1.0,
    delay_spread_ms=2.0,
)
```

**Getting Real Ionospheric Data:**

You can use real-time ionospheric data from ionosondes or models:

```python
from hfpathsim.iono import get_iri_parameters

# Get IRI-2020 model parameters for a location and time
params = get_iri_parameters(
    latitude=40.0,
    longitude=-105.0,
    datetime="2024-06-15T12:00:00Z"
)

engine.configure_vogler(
    foF2=params["foF2"],
    hmF2=params["hmF2"],
)
```

### Vogler-Hoffmeyer Model

Extended model including additional ionospheric phenomena.

```python
config = EngineConfig(channel_model=ChannelModel.VOGLER_HOFFMEYER)
engine = SimulationEngine(config)

engine.configure_vogler_hoffmeyer(
    condition="moderate",
    sporadic_e_enabled=True,   # Sporadic E layer effects
    spread_f_enabled=False,     # Spread-F irregularities
    magnetic_storm=False,       # Geomagnetic storm conditions
)
```

**Special Conditions:**

- **Sporadic E**: Summer daytime enhancement, can create unexpected propagation
- **Spread-F**: Nighttime equatorial irregularities causing signal scattering
- **Magnetic Storm**: Severe ionospheric disturbance during solar events

---

## Configuration

### EngineConfig Options

```python
from hfpathsim import EngineConfig, ChannelModel

config = EngineConfig(
    # Channel model selection
    channel_model=ChannelModel.WATTERSON,

    # Sample rate (must match your input)
    sample_rate_hz=48000,

    # Processing block size (affects latency vs efficiency)
    block_size=4096,

    # GPU acceleration
    use_gpu=True,

    # Multi-threading for CPU fallback
    num_threads=4,
)
```

### Block Size Trade-offs

| Block Size | Latency | Efficiency | Best For |
|------------|---------|------------|----------|
| 256 | ~5 ms | Low | Ultra-low latency |
| 1024 | ~21 ms | Medium | Real-time voice |
| 4096 | ~85 ms | High | Data modes |
| 16384 | ~341 ms | Very High | Batch processing |

### Sample Rate Considerations

The simulator works best when the sample rate matches your signal bandwidth:

| Application | Recommended Sample Rate |
|-------------|------------------------|
| SSB Voice (3 kHz BW) | 8000-16000 Hz |
| HF Data (6 kHz BW) | 12000-24000 Hz |
| Wideband HF (24 kHz BW) | 48000-96000 Hz |
| Full HF band testing | 100000+ Hz |

---

## Input Sources

HF Path Simulator can read samples from multiple sources.

### File Input

```python
from hfpathsim.input import FileSource

# WAV file
source = FileSource("recording.wav")

# Raw IQ file
source = FileSource(
    "recording.iq",
    format="complex64",
    sample_rate_hz=48000
)

# SigMF file (metadata included)
source = FileSource("recording.sigmf-data")
```

### Audio Device

```python
from hfpathsim.input import AudioSource

# Default device
source = AudioSource()

# Specific device
source = AudioSource(device_name="USB Audio CODEC")

# List available devices
from hfpathsim.input import list_audio_devices
for dev in list_audio_devices():
    print(f"{dev['index']}: {dev['name']}")
```

### SDR Hardware

```python
from hfpathsim.input import SDRSource

# Auto-detect SDR
source = SDRSource()

# Specific device
source = SDRSource(
    driver="rtlsdr",
    center_freq_hz=7_200_000,  # 7.2 MHz
    sample_rate_hz=2_400_000,
    gain_db=30,
)

# Supported drivers: rtlsdr, hackrf, lime, uhd (USRP), soapy
```

### Network Streaming

```python
from hfpathsim.input import ZMQSource, TCPSource, UDPSource

# ZMQ subscriber
source = ZMQSource("tcp://192.168.1.100:5555")

# TCP client
source = TCPSource("192.168.1.100", 5555)

# UDP receiver
source = UDPSource(port=5555)
```

---

## Output Destinations

### File Output

```python
from hfpathsim.output import FileOutputSink

# WAV file
sink = FileOutputSink("output.wav", format="wav")

# Raw IQ
sink = FileOutputSink("output.iq", format="raw")

# SigMF with metadata
sink = FileOutputSink(
    "output.sigmf-data",
    format="sigmf",
    metadata={"description": "HF simulation output"}
)
```

### Audio Output

```python
from hfpathsim.output import AudioOutputSink

# Default device
sink = AudioOutputSink()

# Specific device
sink = AudioOutputSink(device_name="Line Out")
```

### SDR Transmission

```python
from hfpathsim.output import SDROutputSink

sink = SDROutputSink(
    driver="hackrf",
    center_freq_hz=7_200_000,
    sample_rate_hz=2_000_000,
    tx_gain_db=10,
)
```

### Network Streaming

```python
from hfpathsim.output import ZMQOutputSink, TCPServerSink

# ZMQ publisher
sink = ZMQOutputSink("tcp://*:5556")

# TCP server
sink = TCPServerSink(port=5556)
```

### Multiple Outputs

Send to multiple destinations simultaneously:

```python
from hfpathsim.output import MultiplexSink

sink = MultiplexSink([
    FileOutputSink("recording.wav"),
    ZMQOutputSink("tcp://*:5556"),
    AudioOutputSink(),
])
```

---

## Signal Impairments

Add realistic RF impairments beyond the channel model.

### Noise

```python
# Add AWGN at specific SNR
engine.configure_noise(
    snr_db=20.0,
    enable_atmospheric=True,  # ITU-R P.372 atmospheric noise
    enable_galactic=True,     # Cosmic background
    enable_man_made=True,     # Electrical interference
)

# Disable noise
engine.disable_noise()
```

### AGC (Automatic Gain Control)

```python
engine.configure_agc(
    enabled=True,
    target_level_db=-20.0,    # Target output level
    attack_time_ms=10.0,      # Attack time constant
    release_time_ms=100.0,    # Release time constant
    max_gain_db=60.0,         # Maximum gain
)
```

### Limiter/Clipper

```python
engine.configure_limiter(
    enabled=True,
    threshold_db=-3.0,  # Limiting threshold
)
```

### Frequency Offset

```python
engine.configure_frequency_offset(
    enabled=True,
    offset_hz=50.0,           # Static offset
    drift_hz_per_sec=0.1,     # Linear drift rate
)
```

---

## GPU Acceleration

### Checking GPU Availability

```python
from hfpathsim.gpu import (
    is_gpu_available,
    get_gpu_info,
    get_gpu_memory_info,
)

if is_gpu_available():
    info = get_gpu_info()
    print(f"GPU: {info['name']}")
    print(f"CUDA Compute: {info['compute_capability']}")
    print(f"Memory: {info['total_memory_gb']:.1f} GB")
else:
    print("No GPU available, using CPU")
```

### GPU Performance Tips

1. **Use larger block sizes** - GPUs are most efficient with large batches
2. **Keep data on GPU** - Minimize CPU-GPU transfers
3. **Use float32** - Native GPU precision, fastest processing

```python
# Optimal GPU configuration
config = EngineConfig(
    use_gpu=True,
    block_size=16384,  # Large blocks for GPU efficiency
)
```

### Memory Management

For long-running simulations:

```python
from hfpathsim.gpu import clear_gpu_cache

# Periodically clear unused GPU memory
clear_gpu_cache()
```

---

## Using the GUI

Launch the graphical interface:

```bash
python -m hfpathsim
```

### Main Window Sections

1. **Spectrum Display** - Real-time FFT of input and output signals
2. **Channel Controls** - Model selection and parameter adjustment
3. **Input Tab** - Configure signal source
4. **Output Tab** - Configure signal destination
5. **Status Bar** - Processing statistics and GPU status

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Start/Stop processing |
| `R` | Reset channel state |
| `Ctrl+S` | Save configuration |
| `Ctrl+O` | Load configuration |
| `F11` | Toggle fullscreen |
| `Esc` | Exit fullscreen |

### Saving Configurations

Save your setup for later:

```
File → Save Configuration → my_setup.json
```

Load it back:

```
File → Load Configuration → my_setup.json
```

Or from command line:

```bash
python -m hfpathsim --config my_setup.json
```

---

## Command Line Interface

### Basic Usage

```bash
# Launch GUI
python -m hfpathsim

# Start API server
python -m hfpathsim.api

# Process a file
python -m hfpathsim.cli process input.wav output.wav --model watterson
```

### CLI Options

```bash
python -m hfpathsim.cli process --help

Options:
  --model TEXT          Channel model (watterson, vogler, vh)
  --condition TEXT      Channel condition (good, moderate, disturbed)
  --sample-rate INT     Sample rate in Hz
  --snr FLOAT          Add noise at this SNR (dB)
  --gpu / --no-gpu     Enable/disable GPU
  --verbose            Show processing details
```

### Examples

```bash
# Process with disturbed conditions
python -m hfpathsim.cli process input.wav output.wav \
  --model watterson --condition disturbed

# Add 15 dB SNR noise
python -m hfpathsim.cli process input.wav output.wav \
  --model vogler --snr 15

# High-performance batch processing
python -m hfpathsim.cli process input.wav output.wav \
  --model vh --gpu --block-size 65536
```

---

## Performance Tuning

### Measuring Performance

```python
import time

samples = np.random.randn(1_000_000).astype(np.complex64)

start = time.perf_counter()
output = engine.process(samples)
elapsed = time.perf_counter() - start

samples_per_sec = len(samples) / elapsed
print(f"Throughput: {samples_per_sec/1e6:.1f} Msamples/sec")
```

### Optimization Checklist

1. **Enable GPU** if available
2. **Increase block size** for batch processing
3. **Match sample rate** to actual signal bandwidth
4. **Use float32/complex64** data types
5. **Minimize Python overhead** by processing large chunks

### Typical Performance

| Configuration | Throughput |
|--------------|------------|
| CPU, small blocks | 0.5-2 Msamples/sec |
| CPU, large blocks | 2-10 Msamples/sec |
| GPU (GTX 1060) | 50-100 Msamples/sec |
| GPU (RTX 3080) | 200-500 Msamples/sec |

### Real-time Processing

For real-time applications, ensure:

```
Throughput > Sample Rate × Safety Margin (2x recommended)
```

Example: 48 kHz audio requires at least 96 ksamples/sec throughput.

---

## Troubleshooting

### Common Issues

**Audio glitches or dropouts:**
- Increase block size
- Reduce sample rate
- Check CPU/GPU utilization

**High latency:**
- Decrease block size
- Use GPU acceleration
- Reduce number of processing stages

**Memory errors:**
- Reduce block size
- Clear GPU cache periodically
- Use streaming processing for large files

**Incorrect channel behavior:**
- Verify sample rate matches input
- Check channel model parameters
- Enable logging for diagnostics

### Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific modules
logging.getLogger("hfpathsim.engine").setLevel(logging.DEBUG)
logging.getLogger("hfpathsim.gpu").setLevel(logging.DEBUG)
```

### Getting Support

If you encounter issues:

1. Check the [FAQ](https://github.com/n4hy/HFPathSimulatorLiveData/wiki/FAQ)
2. Search [existing issues](https://github.com/n4hy/HFPathSimulatorLiveData/issues)
3. Open a [new issue](https://github.com/n4hy/HFPathSimulatorLiveData/issues/new) with:
   - HF Path Simulator version
   - Python version
   - Operating system
   - GPU model (if applicable)
   - Minimal code to reproduce the problem

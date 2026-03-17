# HF Path Simulator

A real-time HF (High Frequency) ionospheric channel simulator implementing the Vogler-Hoffmeyer Ionospheric Propagation Model (IPM) and Watterson TDL model with GPU acceleration for RTX 5090.

## Overview

This project provides a physically-accurate simulation of HF radio propagation through the ionosphere, suitable for:

- **HF modem development and testing** - Test modems against realistic channel conditions without over-the-air transmission
- **Algorithm research** - Evaluate equalization, synchronization, and coding schemes
- **Training and education** - Visualize ionospheric propagation effects in real-time
- **Live signal processing** - Apply channel effects to SDR input for hardware-in-the-loop testing

The simulator implements the Vogler-Hoffmeyer reflection coefficient model from NTIA Technical Report TR-88-240, combined with ITU-R F.1487 channel statistics for realistic time-varying fading.

## Features

### Channel Models

- **Vogler-Hoffmeyer IPM Core**
  - Complex gamma function computation for reflection coefficient R(ω)
  - Frequency-dependent amplitude and phase response
  - Group delay variation across bandwidth
  - Multi-mode propagation (1F2, 2F2, E-layer, Es)

- **Watterson Tapped Delay Line Model**
  - Classic Watterson HF channel model
  - Multiple taps with configurable delays and amplitudes
  - Independent Rayleigh/Rician fading per tap
  - Gaussian, flat, and Jakes Doppler spectrum shapes
  - CCIR Good/Moderate/Poor presets

- **Vogler-Hoffmeyer Wideband Stochastic Model (NTIA TR-90-255)**
  - Full stochastic channel model for wideband HF (up to 1 MHz+)
  - Gaussian and exponential Doppler spectrum correlation
  - Delay-dependent Doppler shift (ionospheric dispersion)
  - Spread-F random multiplication for auroral conditions
  - Multi-mode propagation (E-layer, F-layer low/high rays)
  - Presets: equatorial, polar, mid-latitude, auroral spread-F
  - Optional Rician K-factor for specular component
  - Frequency-dependent group delay (dispersion) modeling

- **ITU-R F.1487 Channel Conditions**
  - Quiet: τ=0.5ms, ν=0.1Hz (benign mid-latitude)
  - Moderate: τ=2ms, ν=1Hz (typical daytime)
  - Disturbed: τ=4ms, ν=2Hz (magnetic storm)
  - Flutter: τ=7ms, ν=10Hz (high-latitude)

### Signal Processing

- **Real-Time Processing**
  - Overlap-save convolution for continuous streaming
  - 4096-point FFT blocks with configurable overlap
  - Support for up to 2 Msps complex sample rates
  - Full processing chain: Input → Channel → Noise → AGC → Limiter → Freq Offset → Output

- **Noise Injection**
  - AWGN (Additive White Gaussian Noise)
  - Atmospheric noise per ITU-R P.372
  - Man-made noise (city, residential, rural environments)
  - Impulse noise (lightning, switching transients)
  - Configurable SNR and noise bandwidth

- **Signal Impairments**
  - AGC (Automatic Gain Control) with slow/medium/fast/manual modes
  - Attack/release dynamics with hang AGC
  - Signal limiting (hard, soft, cubic modes)
  - Frequency offset and drift simulation
  - Phase noise modeling

- **Channel Recording & Playback**
  - Record time-varying channel states
  - Save in NPZ, HDF5, or JSON formats
  - Playback with interpolation
  - Reproducible testing scenarios

### Ionospheric Modeling

- **Physics-Based Ray Tracing**
  - Spherical Earth geometry with computed sec(φ)
  - 2D Haselgrove ray equation integration
  - Multi-hop propagation mode discovery (1F2, 2F2, 3F2, 1E, Es)
  - IonosphereProfile with electron density Ne(h) arrays
  - Refractive index and plasma frequency calculations
  - Launch angle and group delay computation

- **Sporadic-E Layer Modeling**
  - Time-varying Es layer with configurable foEs and hmEs
  - Layer injection into ionosphere profiles
  - Occurrence probability estimation (seasonal, diurnal, latitude)
  - Presets: weak, moderate, strong, intense

- **Geomagnetic Effects**
  - F10.7 solar flux scaling of foF2
  - Kp index modulation of Doppler/delay spread
  - Dst storm-time depression of critical frequencies
  - Polar blackout detection
  - Storm phase classification (initial, main, recovery)

- **Ionospheric Data Sources**
  - Manual parameter entry
  - GIRO/DIDBase real-time ionosonde data
  - IRI-2020 model integration (optional)

### Input Sources

- File playback: WAV, SigMF, raw binary (complex64, int16, int8)
- Network streams: TCP, UDP, ZeroMQ
- SDR support via SoapySDR (RTL-SDR, HackRF, USRP, etc.)
- **Flex Radio DAX IQ**: Native SmartSDR integration with VITA-49 streaming

### GPU Acceleration

- Native CUDA module with cuFFT for maximum performance
- Batched overlap-save convolution using `cufftPlanMany` (68.9 Msps throughput)
- GPU-accelerated Doppler fading generation with cuRAND
- Real-time spectrum computation for GUI
- CuPy fallback with automatic kernel compilation
- NumPy CPU fallback for compatibility
- Designed for RTX 5090 (Blackwell), supports sm_80-90 architectures

### PyQt6 Dashboard GUI

Full-featured graphical interface with tabbed controls:

```
┌─────────────────────────────────────────────────────────────────────┐
│ [Menu: File | View | Tools | Help]                                  │
├─────────────────────────────────────────────────────────────────────┤
│ [Toolbar: Start/Stop | Preset Dropdown | GPU Status]                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────┐  ┌──────────────────────────────────────┐ │
│  │ Channel Display      │  │ Scattering Function S(τ,ν)          │ │
│  │ (Freq/Impulse/Phase) │  │ (Delay-Doppler 2D intensity)        │ │
│  ├──────────────────────┤  ├──────────────────────────────────────┤ │
│  │ Input Spectrum       │  │ Output Spectrum                      │ │
│  │ (Real-time FFT)      │  │ (Real-time FFT)                      │ │
│  └──────────────────────┘  └──────────────────────────────────────┘ │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ [Input] [Channel] [Noise] [Impairments] [Ionosphere] [Recording]    │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ (Tab content - varies by selected tab)                          │ │
│ └─────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│ [Status Bar: GPU | Rate | SNR | Mode Count]                         │
└─────────────────────────────────────────────────────────────────────┘
```

**Dashboard Tabs:**

| Tab | Features |
|-----|----------|
| **Input** | File/Network/SDR/Flex Radio source selection, sample rate, format |
| **Channel** | Vogler/Watterson model selection, ITU presets, ionospheric params, tap config |
| **Noise** | AWGN, atmospheric, man-made, impulse noise with SNR control |
| **Impairments** | AGC with gain meter, limiter with modes, frequency offset/drift |
| **Ionosphere** | Ray tracing, Sporadic-E, geomagnetic indices, GIRO/IRI data sources |
| **Recording** | Channel state recording/playback with metadata |

---

## Installation

### Prerequisites

- Python 3.11+
- PyQt6 and pyqtgraph
- NumPy, SciPy
- NVIDIA GPU with CUDA 12.x (optional, for GPU acceleration)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/HFPathSimulatorLiveData.git
cd HFPathSimulatorLiveData

# Create and activate virtual environment
python -m venv .hfpathsim
source .hfpathsim/bin/activate

# Install the package with all dependencies
pip install -e .

# Launch the dashboard
python -m hfpathsim
```

### Dependencies

The package automatically installs these dependencies:

```
numpy>=1.24.0
scipy>=1.10.0
PyQt6>=6.5.0
pyqtgraph>=0.13.0
```

### Optional Dependencies

```bash
# For SDR support
pip install soapysdr

# For IRI-2020 ionospheric model
pip install iri2016

# For HDF5 recording format
pip install h5py

# For ZeroMQ network input
pip install pyzmq

# For GPU acceleration (CuPy fallback)
pip install cupy-cuda12x

# For development/testing
pip install pytest pytest-qt pytest-cov
```

### Building Native CUDA Module (Maximum Performance)

```bash
# Requires CUDA Toolkit 12.x
./scripts/build_gpu.sh

# Verify installation
python -c "from hfpathsim.gpu import get_backend_info; print(get_backend_info())"
# Expected: {'backend': 'cuda', 'native_module': True, ...}
```

---

## Usage

### Launch Dashboard GUI

```bash
python -m hfpathsim
```

The dashboard opens with the full tabbed interface. Use the toolbar preset selector to quickly switch between ITU channel conditions.

### GUI Tab Guide

#### Input Tab
Configure the signal source:
- **File**: Browse for IQ files (WAV, SigMF, raw), set sample rate and format
- **Network**: TCP/UDP/ZMQ streaming with host:port configuration
- **SDR**: Scan for SoapySDR devices, set center frequency and gain
- **Flex Radio**: Connect to SmartSDR radios via DAX IQ streaming (VITA-49)

#### Channel Tab
Configure the channel model:
- **Model Selection**: Choose Vogler IPM, Watterson TDL, or Vogler-Hoffmeyer
- **ITU Preset**: Quick select Quiet/Moderate/Disturbed/Flutter
- **Ionospheric**: Set foF2, hmF2, foE, hmE
- **Channel Stats**: Delay spread, Doppler spread, frequency, path length
- **Modes**: Enable/disable propagation modes (1F2, 2F2, 1E, Es)
- **Watterson Taps**: Add/remove taps, configure delay/amplitude/Doppler per tap
- **Vogler-Hoffmeyer**: Presets (equatorial/polar/midlatitude/auroral), delay spread, Doppler spread, correlation type, spread-F enable

#### Noise Tab
Configure noise injection:
- **Master SNR**: Slider control for overall signal-to-noise ratio
- **AWGN**: Enable white Gaussian noise
- **Atmospheric**: ITU-R P.372 model with season/time/latitude
- **Man-Made**: Environment selection (city/residential/rural)
- **Impulse**: Rate, amplitude, and duration controls

#### Impairments Tab
Configure receiver impairments:
- **AGC**: Mode (slow/medium/fast/manual), target level, max gain, attack/release
- **Limiter**: Threshold, mode (hard/soft/cubic), attack/release
- **Frequency Offset**: Static offset, drift rate, phase noise level
- **Real-time meters**: AGC gain and limiter gain reduction displays

#### Ionosphere Tab
Configure ionospheric modeling:
- **Data Source**: Manual entry, GIRO real-time data, or IRI-2020 model
- **Path Geometry**: TX/RX coordinates with distance/bearing calculation
- **Ray Tracing**: Enable/disable, max hops, discovered modes table
- **Sporadic-E**: Enable Es layer with presets and foEs/hmEs controls
- **Geomagnetic**: F10.7, Kp, Dst indices with presets and storm phase

#### Recording Tab
Record and playback channel states:
- **Record**: Start/stop, snapshot rate, max duration, format selection
- **Playback**: Load file, play/pause/stop, playback rate, loop option
- **Metadata**: Description and tags for recordings

---

## Programmatic API

### Basic Channel Processing

```python
from hfpathsim.core.parameters import VoglerParameters, ITUCondition
from hfpathsim.core.channel import HFChannel
import numpy as np

# Create channel with ITU Moderate conditions
params = VoglerParameters.from_itu_condition(ITUCondition.MODERATE)
channel = HFChannel(params, use_gpu=True)

# Process signal
input_signal = np.exp(2j * np.pi * 1000 * np.arange(4096) / 2e6).astype(np.complex64)
output_signal = channel.process(input_signal)

# Get channel state for visualization
state = channel.get_state()
print(f"Transfer function shape: {state.transfer_function.shape}")
print(f"Delay spread: {params.delay_spread_ms} ms")
print(f"Doppler spread: {params.doppler_spread_hz} Hz")
```

### Watterson Channel Model

```python
from hfpathsim.core.watterson import WattersonChannel, WattersonConfig, WattersonTap, DopplerSpectrum
from hfpathsim.core.parameters import ITUCondition
import numpy as np

# Create from ITU preset
config = WattersonConfig.from_itu_condition(ITUCondition.MODERATE)
channel = WattersonChannel(config, seed=42)

# Or create custom configuration
config = WattersonConfig(
    taps=[
        WattersonTap(delay_ms=0.0, amplitude=1.0, doppler_spread_hz=1.0),
        WattersonTap(delay_ms=2.0, amplitude=0.7, doppler_spread_hz=1.0),
        WattersonTap(delay_ms=4.0, amplitude=0.3, doppler_spread_hz=2.0,
                     is_specular=True, k_factor_db=6.0),  # Rician fading
    ],
    sample_rate_hz=2e6,
)
channel = WattersonChannel(config)

# Process signal
output = channel.process_block(np.random.randn(4096).astype(np.complex64))
```

### Vogler-Hoffmeyer Wideband Stochastic Model

```python
from hfpathsim.core.vogler_hoffmeyer import (
    VoglerHoffmeyerChannel, VoglerHoffmeyerConfig,
    ModeParameters, CorrelationType,
    get_vogler_hoffmeyer_preset, list_vogler_hoffmeyer_presets,
)
from hfpathsim.core.parameters import ITUCondition
import numpy as np

# Create from ITU preset
config = VoglerHoffmeyerConfig.from_itu_condition(ITUCondition.MODERATE)
channel = VoglerHoffmeyerChannel(config)

# Or use geographic presets
print(f"Available presets: {list_vogler_hoffmeyer_presets()}")
# ['equatorial', 'polar', 'midlatitude', 'auroral_spread_f']

# Create polar path (high Doppler spread, auroral effects)
polar_config = get_vogler_hoffmeyer_preset('polar', sample_rate=2e6)
polar_channel = VoglerHoffmeyerChannel(polar_config)

# Create custom configuration with multiple modes
custom_config = VoglerHoffmeyerConfig(
    sample_rate=2e6,
    modes=[
        ModeParameters(
            name="F-layer low-ray",
            amplitude=1.0,
            sigma_tau=100.0,      # Delay spread in microseconds
            sigma_c=50.0,         # Carrier delay subinterval
            sigma_D=2.0,          # Doppler spread in Hz
            doppler_shift=0.5,    # Mean Doppler shift in Hz
            correlation_type=CorrelationType.GAUSSIAN,
        ),
        ModeParameters(
            name="F-layer high-ray",
            amplitude=0.6,
            tau_L=150.0,          # Minimum delay in microseconds
            sigma_tau=200.0,
            sigma_c=75.0,
            sigma_D=3.0,
            correlation_type=CorrelationType.EXPONENTIAL,  # For flutter fading
        ),
    ],
    spread_f_enabled=True,     # Enable spread-F random amplitude modulation
    k_factor=6.0,              # Rician K-factor for first tap (dB)
)
channel = VoglerHoffmeyerChannel(custom_config)

# Process signal
test_signal = (np.random.randn(4096) + 1j * np.random.randn(4096)).astype(np.complex64)
output = channel.process(test_signal)

# Get theoretical scattering function
delay_axis, doppler_axis, S = channel.compute_scattering_function(
    num_delay_bins=64,
    num_doppler_bins=64
)
print(f"Scattering function shape: {S.shape}")  # (64, 64)
```

### Full Processing Chain

```python
from hfpathsim.core.channel import HFChannel
from hfpathsim.core.watterson import WattersonChannel, WattersonConfig
from hfpathsim.core.noise import NoiseGenerator, NoiseConfig, ManMadeEnvironment
from hfpathsim.core.impairments import AGC, AGCConfig, AGCMode, Limiter, LimiterConfig
from hfpathsim.core.impairments import FrequencyOffset, FrequencyOffsetConfig
from hfpathsim.core.parameters import VoglerParameters, ITUCondition
import numpy as np

# Setup components
params = VoglerParameters.from_itu_condition(ITUCondition.MODERATE)
channel = HFChannel(params, use_gpu=True)

noise = NoiseGenerator(NoiseConfig(
    snr_db=20.0,
    enable_atmospheric=True,
    frequency_mhz=14.0,
    enable_manmade=True,
    environment=ManMadeEnvironment.RESIDENTIAL,
))

agc = AGC(AGCConfig(mode=AGCMode.MEDIUM, target_level_db=-10.0))
limiter = Limiter(LimiterConfig(threshold_db=-3.0, mode="soft"))
freq_offset = FrequencyOffset(FrequencyOffsetConfig(offset_hz=25.0, drift_rate_hz_per_sec=0.1))

# Process signal through chain
input_signal = np.random.randn(4096).astype(np.complex64) + 1j * np.random.randn(4096).astype(np.complex64)

output = channel.process(input_signal)
output = noise.add_noise(output)
output = agc.process_block(output)
output = limiter.process(output)
output = freq_offset.process(output)

print(f"AGC gain: {agc.current_gain_db:.1f} dB")
print(f"Limiter GR: {limiter.gain_reduction_db:.1f} dB")
```

### Ray Tracing

```python
from hfpathsim.core.raytracing import (
    create_simple_profile,
    find_propagation_modes,
    sec_phi_spherical,
    great_circle_distance,
)

# Create ionosphere profile
profile = create_simple_profile(foF2=7.5, hmF2=300.0, foE=3.0, hmE=110.0)

# Find viable propagation modes for a path (Washington DC to London)
modes = find_propagation_modes(
    profile,
    tx_lat=38.9, tx_lon=-77.0,
    rx_lat=51.5, rx_lon=-0.1,
    frequency_mhz=14.0,
    max_hops=3,
)

for mode in modes:
    print(f"{mode.name}: delay={mode.group_delay_ms:.1f}ms, amplitude={mode.relative_amplitude:.2f}")

# Get physically-computed sec(φ) for MUF calculation
path_km = great_circle_distance(38.9, -77.0, 51.5, -0.1)
sec_phi = sec_phi_spherical(path_km, hm_km=300.0)
muf = 7.5 * sec_phi  # foF2 * sec(φ)
print(f"Path: {path_km:.0f} km, sec(φ): {sec_phi:.2f}, MUF: {muf:.1f} MHz")
```

### Sporadic-E Layer

```python
from hfpathsim.iono.sporadic_e import (
    SporadicELayer, SporadicEConfig,
    estimate_es_occurrence, create_es_from_preset,
)
from hfpathsim.core.raytracing import create_simple_profile

# Estimate Es occurrence probability
prob = estimate_es_occurrence(latitude=45.0, month=6, hour_utc=14)
print(f"Es occurrence probability: {prob:.1%}")

# Create Es layer from preset
config = create_es_from_preset("strong")  # foEs=10 MHz
es_layer = SporadicELayer(config)

# Inject into ionosphere profile
profile = create_simple_profile(foF2=7.5, hmF2=300.0)
profile_with_es = es_layer.inject(profile)

# Es MUF for 1000 km path
es_muf = es_layer.get_muf(path_km=1000.0)
print(f"Es MUF: {es_muf:.1f} MHz")
```

### Geomagnetic Effects

```python
from hfpathsim.iono.geomagnetic import GeomagneticIndices, GeomagneticModulator, classify_storm_phase

# Create storm conditions
indices = GeomagneticIndices.disturbed()  # Kp=5, Dst=-80
modulator = GeomagneticModulator(indices)

# Scale ionospheric parameters
foF2_base = 7.5
foF2_storm = modulator.scale_foF2(foF2_base, latitude=45.0)
print(f"foF2: {foF2_base} MHz -> {foF2_storm:.1f} MHz (storm)")

# Enhanced fading during storm
doppler_storm = modulator.scale_doppler_spread(1.0, latitude=60.0)
print(f"Doppler spread enhanced to {doppler_storm:.1f} Hz")

# Check for polar blackout
if modulator.is_blackout(frequency_mhz=7.0, latitude=70.0):
    print("WARNING: Polar blackout conditions!")

# Classify storm phase
phase = classify_storm_phase(dst=-80, dst_rate=-15)
print(f"Storm phase: {phase}")
```

### GIRO Real-Time Data

```python
from hfpathsim.iono.giro import GIROClient

# Connect to Boulder ionosonde
client = GIROClient("BC840")
ionogram = client.fetch_latest()

if ionogram:
    print(f"foF2: {ionogram.foF2:.1f} MHz")
    print(f"hmF2: {ionogram.hmF2:.0f} km")
    print(f"Confidence: {ionogram.confidence:.2f}")

# Convert to Vogler parameters
params = client.to_vogler_params(frequency_mhz=14.0, path_km=2000.0)
```

### Channel Recording and Playback

```python
from hfpathsim.core.recording import ChannelRecorder, ChannelPlayer, create_test_recording
from hfpathsim.core.channel import HFChannel
from hfpathsim.core.parameters import VoglerParameters, ITUCondition

# Record channel states
channel = HFChannel(VoglerParameters.from_itu_condition(ITUCondition.MODERATE))
recorder = ChannelRecorder(channel, snapshot_rate_hz=10.0, max_duration_sec=60.0)

recorder.start()
# ... channel processing happens, states are captured ...
recorder.stop()
recorder.save("recording.npz")

# Playback recorded states
player = ChannelPlayer()
player.load("recording.npz")

print(f"Duration: {player.duration:.1f} sec")
print(f"Snapshots: {player.num_snapshots}")

for H in player.iterate(rate_hz=20.0, loop=False):
    # Apply transfer function H to your signal
    pass
```

### GPU Acceleration

```python
from hfpathsim.gpu import (
    get_device_info,
    get_backend_info,
    apply_channel_batched,
    generate_doppler_fading,
    compute_spectrum_db,
)
import numpy as np

# Check GPU status
info = get_backend_info()
print(f"Backend: {info['backend']}")
print(f"Native CUDA module: {info['native_module']}")
print(f"Device: {info['device_info']['name']}")

# High-throughput batched processing (68.9 Msps achieved)
input_signal = (np.random.randn(2_000_000) + 1j * np.random.randn(2_000_000)).astype(np.complex64)
H = np.ones(4096, dtype=np.complex64)

output = apply_channel_batched(
    input_signal, H,
    block_size=4096,
    overlap=1024,
    batch_size=8  # Process 8 blocks in parallel
)

# GPU-accelerated Doppler fading generation
fading = generate_doppler_fading(
    doppler_spread_hz=1.5,
    sample_rate=2e6,
    n_samples=4096,
    seed=42,
)

# Fast spectrum computation for real-time GUI
power_db = compute_spectrum_db(input_signal[:4096], reference=1.0)
```

### File Input

```python
from hfpathsim.input.file import FileInputSource
from hfpathsim.input.base import InputFormat

# Load various formats
source = FileInputSource(
    "recording.wav",  # or .sigmf-data, .raw, .bin, .cf32
    sample_rate_hz=2e6,  # Required for raw files
    input_format=InputFormat.COMPLEX64,  # Auto-detected for WAV/SigMF
    loop=True,
)

source.open()
while True:
    samples = source.read(4096)
    if samples is None:
        break
    # Process samples...
source.close()
```

### Network Input

```python
from hfpathsim.input.network import NetworkInputSource, NetworkProtocol

# TCP client
source = NetworkInputSource(
    host="192.168.1.100",
    port=5000,
    protocol=NetworkProtocol.TCP,
    sample_rate_hz=2e6,
)

# UDP receiver
source = NetworkInputSource(
    host="0.0.0.0",
    port=5000,
    protocol=NetworkProtocol.UDP,
    sample_rate_hz=2e6,
)

# ZeroMQ subscriber
source = NetworkInputSource(
    host="localhost",
    port=5555,
    protocol=NetworkProtocol.ZMQ_SUB,
    sample_rate_hz=2e6,
)
```

### Flex Radio DAX IQ Input

```python
from hfpathsim.input.flexradio import FlexRadioInputSource

# Discover Flex Radios on the network
radios = FlexRadioInputSource.discover_radios(timeout=3.0)
for radio in radios:
    print(f"Found: {radio['nickname']} at {radio['ip']}")

# Connect to a Flex Radio
source = FlexRadioInputSource(
    host="192.168.1.100",       # Radio IP address
    dax_channel=1,              # DAX IQ channel (1-8)
    sample_rate_hz=48000,       # 24000, 48000, 96000, or 192000
    center_freq_hz=14.074e6,    # Center frequency
)

source.open()

# Read samples (returns complex64 numpy array)
while True:
    samples = source.read(4096)
    if samples is None or len(samples) == 0:
        break
    # Process samples through HF channel simulator...

# Get streaming statistics
stats = source.get_statistics()
print(f"Packets received: {stats['packets_received']}")
print(f"Samples read: {stats['samples_read']}")
print(f"Buffer overflow count: {stats['buffer_overflows']}")

source.close()
```

**Flex Radio Features:**
- Automatic radio discovery via UDP broadcast
- VITA-49 packet format parsing (DAX IQ stream packets)
- Sample rates: 24 kHz, 48 kHz, 96 kHz, 192 kHz
- DAX channels 1-8 support
- Frequency control via SmartSDR TCP API
- Buffer management with overflow detection
- Thread-safe streaming with configurable buffer size

**Requirements:**
- Flex Radio running SmartSDR v3.x or later
- DAX IQ enabled in SmartSDR
- Network connectivity to radio (same subnet recommended)

---

## Project Structure

```
HFPathSimulatorLiveData/
├── pyproject.toml                 # Package configuration
├── README.md                      # This file
├── LICENSE                        # MIT License
│
├── src/hfpathsim/
│   ├── __init__.py
│   ├── __main__.py                # Entry point (python -m hfpathsim)
│   │
│   ├── core/                      # Core simulation
│   │   ├── parameters.py          # VoglerParameters, ITUCondition, PropagationMode
│   │   ├── channel.py             # HFChannel with processing chain
│   │   ├── vogler_ipm.py          # Vogler reflection coefficient model
│   │   ├── watterson.py           # Watterson TDL model
│   │   ├── vogler_hoffmeyer.py    # Vogler-Hoffmeyer wideband stochastic model (NTIA 90-255)
│   │   ├── dispersion.py          # Frequency-dependent group delay modeling
│   │   ├── noise.py               # Noise generators (AWGN, atmospheric, etc.)
│   │   ├── impairments.py         # AGC, limiter, frequency offset
│   │   ├── recording.py           # Channel state recording/playback
│   │   │
│   │   └── raytracing/            # Physics-based ray tracing
│   │       ├── __init__.py        # Module exports
│   │       ├── geometry.py        # Spherical Earth, sec_phi, great circle
│   │       ├── ionosphere.py      # IonosphereProfile, Ne(h), refractive index
│   │       ├── ray_engine.py      # 2D Haselgrove equations, RayPath
│   │       └── path_finder.py     # Multi-hop mode discovery
│   │
│   ├── gpu/                       # GPU acceleration
│   │   ├── __init__.py            # Python interface with fallback chain
│   │   ├── bindings.cpp           # pybind11 CUDA bindings
│   │   ├── CMakeLists.txt         # CUDA build configuration
│   │   └── kernels/
│   │       ├── vogler_transfer.cu
│   │       ├── fading.cu
│   │       └── signal_proc.cu
│   │
│   ├── input/                     # Input sources
│   │   ├── base.py                # InputSource ABC, InputFormat
│   │   ├── file.py                # File playback (WAV, SigMF, raw)
│   │   ├── network.py             # TCP/UDP/ZMQ streaming
│   │   ├── sdr.py                 # SoapySDR interface
│   │   └── flexradio.py           # Flex Radio SmartSDR DAX IQ streaming
│   │
│   ├── iono/                      # Ionospheric data and effects
│   │   ├── manual.py              # Manual parameter entry
│   │   ├── giro.py                # GIRO/DIDBase client
│   │   ├── iri.py                 # IRI-2020 model wrapper
│   │   ├── sporadic_e.py          # Sporadic-E layer modeling
│   │   └── geomagnetic.py         # Geomagnetic effects (F10.7, Kp, Dst)
│   │
│   └── gui/                       # PyQt6 dashboard
│       ├── __init__.py            # Exports MainWindow
│       ├── main_window.py         # Main application window
│       │
│       └── widgets/               # GUI components
│           ├── __init__.py        # Widget exports
│           ├── control_tabs.py    # QTabWidget container for all controls
│           ├── channel_display.py # H(f), h(t), phase, group delay plots
│           ├── scattering.py      # S(τ,ν) 2D intensity display
│           ├── spectrum.py        # Real-time FFT spectrum analyzer
│           ├── input_config.py    # Input source configuration
│           ├── channel_panel.py   # Vogler/Watterson channel controls
│           ├── noise_panel.py     # Noise injection controls
│           ├── impairments_panel.py # AGC/limiter/freq offset controls
│           ├── ionosphere_panel.py  # Ray tracing, Es, geomagnetic controls
│           ├── recording_panel.py # Record/playback controls
│           └── parameters.py      # Legacy parameter panel (deprecated)
│
├── tests/                         # Unit tests (224 tests)
│   ├── test_vogler.py             # Vogler model tests
│   ├── test_channel_models.py     # Watterson, noise, impairments
│   ├── test_raytracing.py         # Ray tracing geometry & engine
│   ├── test_sporadic_e.py         # Sporadic-E layer
│   ├── test_geomagnetic.py        # Geomagnetic effects
│   ├── test_input.py              # Input sources
│   ├── test_gpu.py                # GPU acceleration
│   └── test_spectrum.py           # Spectrum widget tests
│
└── scripts/
    ├── build_gpu.sh               # Build native CUDA module
    ├── install_gpu.sh             # Install to site-packages
    └── run_dashboard.py           # Alternative launcher
```

---

## Technical Background

### Vogler-Hoffmeyer Model

The reflection coefficient R(ω) is computed using the formula from NTIA TR-88-240:

```
R(ω) = Γ(1-iσω)Γ(1/2-χ+iσω)Γ(1/2+χ+iσω)e^(-iωt₀)
       ─────────────────────────────────────────────
       Γ(1+iσω)Γ(1/2-χ)Γ(1/2+χ)
```

Where:
- **σ** - Layer thickness parameter (controls frequency selectivity)
- **χ** - Penetration parameter (function of foF2 and operating frequency)
- **t₀** - Base propagation delay
- **Γ** - Complex gamma function

### Watterson Tapped Delay Line

The Watterson model represents the channel as a sum of independently fading taps:

```
h(t,τ) = Σᵢ aᵢ(t) · δ(τ - τᵢ)
```

Each tap aᵢ(t) is a complex Gaussian process with specified Doppler spectrum shape (Gaussian, flat, or Jakes). For Rician fading, a specular component is added:

```
aᵢ(t) = √(K/(K+1)) · e^(jφ) + √(1/(K+1)) · g(t)
```

### Oblique Incidence Geometry

The secant of the angle of incidence is computed using spherical Earth geometry:

```
sec(φ) = 1 / cos(φ)

where sin(φ) = R·sin(β) / slant_range
      β = path_km / (2·R)  (half angular path)
      slant_range = √(R² + (R+h)² - 2R(R+h)cos(β))
```

This enables accurate MUF calculations: `MUF = foF2 × sec(φ)`

### Channel Statistics

The scattering function S(τ,ν) describes the power distribution in delay-Doppler space:

```
S(τ,ν) = exp(-τ/τ_rms) × exp(-(ν/ν_rms)²)
```

- **Delay spread (τ)**: Multipath delay dispersion, causing frequency-selective fading
- **Doppler spread (ν)**: Time-variation rate, causing temporal fading

### Spread-F Ionospheric Conditions

Spread-F is an ionospheric irregularity phenomenon that causes severe signal degradation on HF paths. It occurs primarily at night in equatorial regions (equatorial spread-F) and in high-latitude auroral zones (auroral spread-F). The phenomenon is caused by plasma instabilities that create irregularities in the F-layer electron density.

**Physical characteristics:**
- **Plasma bubbles**: Depleted density regions that scatter radio waves
- **Scintillation**: Rapid amplitude and phase fluctuations
- **Range spreading**: Ionogram echoes spread over extended delay ranges
- **Frequency spreading**: Diffuse reflections across frequency

**Model implementation** (in Vogler-Hoffmeyer channel):

The spread-F effect is modeled by applying random amplitude scaling to each tap gain during processing:

```
tap_gain = tap_gain × spread_factor    where spread_factor ∈ [0.1, 1.0]
```

This per-sample random multiplication captures the scattering and amplitude fluctuations characteristic of propagation through ionospheric irregularities. The model is enabled via `spread_f_enabled=True` in `VoglerHoffmeyerConfig` or the GUI "Spread-F" checkbox.

**Typical spread-F parameters:**
| Parameter | Equatorial | Auroral |
|-----------|------------|---------|
| Delay spread | 500-1000 μs | 1500-2500 μs |
| Doppler spread | 2-5 Hz | 5-15 Hz |
| Dispersion | 150-240 μs/MHz | 200-400 μs/MHz |
| Correlation type | Gaussian | Exponential |

**Usage example:**

```python
from hfpathsim.core.vogler_hoffmeyer import (
    VoglerHoffmeyerChannel,
    get_vogler_hoffmeyer_preset,
)

# Use auroral spread-F preset
config = get_vogler_hoffmeyer_preset('auroral_spread_f', sample_rate=2e6)
channel = VoglerHoffmeyerChannel(config)

# Or enable manually on any config
config.spread_f_enabled = True
```

### Geomagnetic Effects

Space weather impacts on HF propagation:

- **F10.7 scaling**: `foF2 ∝ √(1 + 0.014·(F10.7 - 100))`
- **Storm depression**: `Δ foF2 = 0.05·Dst·cos²(lat)` (Dst negative during storms)
- **Doppler enhancement**: `ν_enhanced = ν_base · (1 + 0.1·Kp)`

### Key Parameters

| Parameter | Symbol | Typical Range | Description |
|-----------|--------|---------------|-------------|
| foF2 | foF2 | 3-15 MHz | F2 layer critical frequency |
| hmF2 | hmF2 | 200-400 km | F2 layer peak height |
| foE | foE | 1-4 MHz | E layer critical frequency |
| hmE | hmE | 90-130 km | E layer peak height |
| foEs | foEs | 2-15 MHz | Sporadic-E critical frequency |
| Delay spread | τ | 0.5-7 ms | RMS multipath delay |
| Doppler spread | ν | 0.1-10 Hz | Two-sided Doppler bandwidth |
| Path length | d | 100-10000 km | Great circle distance |
| F10.7 | F10.7 | 65-300 sfu | Solar radio flux index |
| Kp | Kp | 0-9 | Geomagnetic activity index |
| Dst | Dst | -500 to +50 nT | Storm-time index |

---

## Testing

### Running Tests

```bash
# Run all tests
PYTHONPATH=src pytest tests/ -v

# Run with coverage
PYTHONPATH=src pytest tests/ --cov=hfpathsim --cov-report=html

# Run specific modules
PYTHONPATH=src pytest tests/test_vogler.py -v
PYTHONPATH=src pytest tests/test_raytracing.py -v
PYTHONPATH=src pytest tests/test_channel_models.py -v

# Run tests matching a pattern
PYTHONPATH=src pytest tests/ -k "test_sec_phi" -v
```

### Latest Test Results

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.2
collected 224 items

tests/test_channel_models.py ........................................     [ 21%]
tests/test_geomagnetic.py ................................                [ 35%]
tests/test_gpu.py ..............................                          [ 48%]
tests/test_input.py .............                                         [ 54%]
tests/test_raytracing.py .................................                [ 69%]
tests/test_spectrum.py .....................                              [ 78%]
tests/test_sporadic_e.py ........................                         [ 89%]
tests/test_vogler.py ......................                               [100%]

============================= 224 passed in 1.99s ==============================
```

### Test Summary

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_channel_models.py` | 47 | Watterson, noise, AGC, limiter, impairments, recording |
| `test_geomagnetic.py` | 32 | Indices, foF2/hmF2 scaling, storm effects, Kp/Ap conversion |
| `test_gpu.py` | 30 | Native CUDA, batched FFT, Doppler fading, spectrum, benchmarks |
| `test_input.py` | 13 | File sources, network sources, format conversion |
| `test_raytracing.py` | 33 | Geometry, ionosphere profiles, ray engine, path finder |
| `test_spectrum.py` | 21 | FFT computation, windowing, averaging, peak hold, GUI widget |
| `test_sporadic_e.py` | 24 | Es config, layer injection, occurrence estimation |
| `test_vogler.py` | 22 | Vogler parameters, HFChannel, reflection coefficients |
| **Total** | **224** | **All passing** |

### Test Categories

**Unit Tests** - Test individual components in isolation:
- Parameter validation and defaults
- Mathematical computations (gamma functions, geometry)
- Signal processing correctness
- Configuration handling

**Integration Tests** - Test component interactions:
- Channel processing pipeline
- Ray tracing with ionosphere profiles
- Geomagnetic modulation of channel parameters
- GPU/CPU fallback behavior

**Performance Tests** - Verify throughput requirements:
- GPU batched FFT throughput (68.9 Msps achieved)
- Streaming processing latency
- Memory usage patterns

---

## Performance

### Benchmarks (RTX 5090 with Native CUDA Module)

| Operation | Performance | Notes |
|-----------|-------------|-------|
| Batched overlap-save | **68.9 Msps** | 34.5x real-time, 8-block batches |
| Transfer function | ~0.5 ms / 4096 pts | Vogler reflection coefficient |
| Single-block overlap-save | ~0.2 ms / 4096 samples | Non-batched processing |
| Doppler fading generation | 0.39 ms / 4096 samples | cuRAND + cuFFT |
| Spectrum computation | ~2.6 ms / 65k samples | GPU compute_spectrum_db |
| Scattering function | ~1 ms / 64x32 grid | 2D power distribution |
| Ray tracing (single ray) | <10 ms | CPU Haselgrove integration |
| Mode discovery (3 hops) | <100 ms | CPU path finder |

### Targets
- **Throughput**: 2 Msps sustained real-time
- **Latency**: <50 ms input-to-output
- **GPU memory**: <2 GB for 1 MHz bandwidth

---

## References

1. **NTIA TR-88-240**: Vogler, L.E. and Hoffmeyer, J.A., "A full-wave calculation of ionospheric Doppler spread and its application to HF channel modeling," 1988.

2. **NTIA TR-90-255**: Vogler, L.E. and Hoffmeyer, J.A., "A Model for Wideband HF Propagation Channels," 1990. (Stochastic wideband channel model with dispersion)

2. **ITU-R F.1487**: "Testing of HF modems with bandwidths of up to about 12 kHz using ionospheric channel simulators," 2000.

3. **Watterson Model**: Watterson, C.C., Juroshek, J.R., and Bensema, W.D., "Experimental confirmation of an HF channel model," IEEE Trans. Comm. Tech., 1970.

4. **IRI-2020**: International Reference Ionosphere model, https://irimodel.org/

5. **ITU-R P.372**: "Radio noise," for atmospheric and man-made noise modeling.

6. **Haselgrove Equations**: Haselgrove, J., "Ray theory and a new method for ray tracing," Physics of the Ionosphere, 1955.

---

## Development Roadmap

### Phase 1: Foundation ✓
- [x] Project structure and packaging
- [x] PyQt6 dashboard skeleton
- [x] Input source abstraction
- [x] GPU detection and pybind11 setup

### Phase 2: Vogler-Hoffmeyer Core ✓
- [x] Complex gamma function implementation
- [x] Reflection coefficient R(ω)
- [x] ITU-R F.1487 presets
- [x] Gaussian scatter fading
- [x] Overlap-save convolution
- [x] Real-time visualization

### Phase 3: Enhanced Fidelity ✓
- [x] Watterson tapped delay line model
- [x] AWGN and atmospheric noise injection (ITU-R P.372)
- [x] Man-made noise modeling
- [x] Impulse noise generation
- [x] AGC with attack/release dynamics
- [x] Signal limiting (hard/soft/cubic)
- [x] Frequency offset and drift simulation
- [x] Phase noise modeling
- [x] Impairment chain for combined effects
- [x] Channel state recording and playback

### Phase 4: Advanced Propagation ✓
- [x] Spherical Earth geometry (replaces hardcoded sec_phi)
- [x] 2D Haselgrove ray equation integration
- [x] Multi-hop propagation mode discovery (1F, 2F, 3F, E, Es)
- [x] IonosphereProfile with Ne(h) electron density
- [x] Sporadic-E layer modeling with time variation
- [x] Geomagnetic effects (F10.7, Kp, Dst modulation)
- [x] HFChannel integration with ray tracing

### Phase 5: GPU Acceleration ✓
- [x] Native CUDA module with pybind11 bindings
- [x] Batched cuFFT overlap-save using `cufftPlanMany` (68.9 Msps)
- [x] GPU-accelerated Doppler fading generation with cuRAND
- [x] Real-time spectrum computation for GUI
- [x] Build scripts (`build_gpu.sh`, `install_gpu.sh`)
- [x] Fallback chain: Native CUDA → CuPy → NumPy

### Phase 6: Full Dashboard Integration ✓
- [x] Tabbed control interface (6 tabs)
- [x] Channel panel with Vogler/Watterson model switching
- [x] Noise panel with AWGN/atmospheric/man-made/impulse controls
- [x] Impairments panel with AGC/limiter/freq offset and meters
- [x] Ionosphere panel with ray tracing, Es, geomagnetic, GIRO/IRI
- [x] Recording panel with record/playback controls
- [x] Full processing chain integration in main window
- [x] Real-time meter updates
- [x] Flex Radio SmartSDR DAX IQ streaming support
- [x] VITA-49 packet parsing for DAX IQ streams
- [x] Radio discovery and GUI integration

### Phase 7: Output & Integration (Planned)
- [ ] Real-time IQ output (sound card, network)
- [ ] GNU Radio source/sink blocks
- [ ] MATLAB/Simulink interface
- [ ] Docker containerization
- [ ] Cloud deployment option

---

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Development Setup

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Check code style
flake8 src/hfpathsim

# Run GUI
python -m hfpathsim
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- NTIA/ITS for the Vogler-Hoffmeyer model documentation
- ITU-R for HF channel characterization standards
- The pyqtgraph team for excellent real-time plotting
- NVIDIA for CUDA and GPU acceleration support
- The PyQt team for the Qt6 Python bindings
- FlexRadio Systems for SmartSDR DAX IQ streaming specifications

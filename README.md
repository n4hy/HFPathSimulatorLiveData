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

- **RF Processing Chain (Vogler & Vogler-Hoffmeyer)**
  - Full RF-rate processing for physically accurate ionospheric simulation
  - Upsample from baseband (8 kHz) to 1 MHz RF rate
  - Mix to 100 kHz RF carrier frequency
  - Apply ionospheric channel model at RF rate
  - Mix back to baseband
  - 8th-order Butterworth anti-aliasing filter
  - Downsample back to baseband rate
  - Watterson model operates at baseband (no RF processing required)

- **ITU-R Standardized Channel Models (NEW)**
  - CCIR 520 / ITU-R F.520-2: Classic HF channel simulator presets
  - ITU-R F.1289: Wideband HF channel (up to 24 kHz bandwidth)
  - ITU-R F.1487: HF modem testing conditions
  - Latitude-specific presets (equatorial, mid-latitude, high-latitude)
  - Frequency-selective fading and group delay dispersion

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

### Output Destinations

- **File Output**
  - Raw binary (complex64, int16, int8)
  - WAV audio files (mono/stereo IQ)
  - SigMF with auto-generated metadata

- **Network Streaming**
  - ZeroMQ PUB socket for GNU Radio integration
  - TCP server for remote clients
  - UDP broadcast for low-latency distribution

- **Audio Output**
  - Real-time playback via sounddevice
  - Device enumeration and selection
  - Configurable buffer size and latency

- **SDR Transmission**
  - SoapySDR interface for TX-capable devices
  - HackRF, USRP, LimeSDR, PlutoSDR support
  - Center frequency and gain control

- **Multiplex & Tee**
  - Multiplex sink for parallel output to multiple destinations
  - Tee sink for splitting streams (e.g., file + network)
  - Independent enable/disable per sink

### External Integrations

- **GNU Radio**
  - ZMQ bridge for bidirectional IQ streaming
  - Auto-generated Python source/sink snippets
  - Compatible with GNU Radio Companion flowgraphs

- **MATLAB/Simulink**
  - `.mat` file export of channel states (requires scipy)
  - IQ recording with metadata
  - Channel evolution time series export
  - MATLAB Engine interface for live computation (optional)

### GPU Acceleration

- Native CUDA module with cuFFT for maximum performance
- Batched overlap-save convolution using `cufftPlanMany` (68.9 Msps throughput)
- GPU-accelerated Doppler fading generation with cuRAND
- Real-time spectrum computation for GUI
- **DisplayProcessor**: Optimized GUI computations (dB conversion, FFT shift, smoothing, peak hold)
- VH RF Chain with CPU fallback for systems without GPU
- CuPy fallback with automatic kernel compilation
- NumPy CPU fallback for compatibility
- Designed for RTX 5090 (Blackwell), supports sm_80-90 architectures

### Profiling and Benchmarking (NEW)

- **CPU Timing**: Decorators, context managers, and statistics collection
- **GPU Profiling**: CUDA event-based timing for accurate kernel measurement
- **Memory Tracking**: CPU and GPU memory usage monitoring
- **Throughput Benchmarks**: Measure samples/second across configurations
- **Latency Analysis**: Percentile-based latency measurement
- **Report Generation**: Export to JSON and HTML formats

### Real-World Validation (NEW)

Compare simulated channels against measured data from real ionospheric campaigns:

- **Reference Datasets**
  - NTIA TR-90-255 measurements (quiet, disturbed, auroral, spread-F)
  - ITU-R F.1487 standardized test conditions
  - Watterson 1970 original IEEE validation data

- **Statistical Analysis**
  - Delay spread (RMS, mean, window)
  - Doppler spread (RMS, bandwidth)
  - Coherence bandwidth and time
  - Scattering function S(τ,ν) computation and comparison

- **Fading Statistics**
  - Rayleigh distribution fit test (K-S test)
  - Level crossing rate
  - Average fade duration
  - Fade depth measurement

- **Validation Reports**
  - Automated pass/fail tests with tolerances
  - JSON and text report generation
  - Scattering function correlation analysis

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
│ [Input][Output][Channel][Noise][Impairments][Ionosphere][Recording] │
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
| **Output** | File/Network/Audio/SDR sink selection, multiplex config, streaming status |
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
# For SDR support (input and output)
pip install soapysdr

# For IRI-2020 ionospheric model
pip install iri2016

# For HDF5 recording format
pip install h5py

# For ZeroMQ network streaming
pip install pyzmq

# For audio output
pip install sounddevice

# For MATLAB .mat file export
pip install scipy

# For GPU acceleration (CuPy fallback)
pip install cupy-cuda12x

# For development/testing
pip install pytest pytest-qt pytest-cov

# Install all optional features
pip install -e ".[all]"
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

#### Output Tab
Configure output destinations:
- **File**: Save processed IQ to raw, WAV, or SigMF files
- **Network**: Stream via ZMQ PUB, TCP server, or UDP broadcast
- **Audio**: Real-time playback through sound card with device selection
- **SDR**: Transmit via SoapySDR-compatible devices (HackRF, USRP, etc.)
- **Multiplex**: Enable multiple simultaneous outputs
- **Status**: Real-time sample count, buffer fill, and streaming indicators

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

### ITU-R Standardized Channel Models

```python
from hfpathsim.core.itu_channels import (
    # CCIR 520 / ITU-R F.520-2 model
    CCIR520Channel, CCIR520Condition,
    # ITU-R F.1289 wideband model
    ITURF1289Channel, ITURF1289Condition,
    # ITU-R F.1487 modem testing model
    ITURF1487Channel, ITURF1487Condition,
    # Utilities
    list_ccir520_presets, list_iturf1289_presets,
    get_preset_info, create_channel,
)
import numpy as np

# CCIR 520 / ITU-R F.520-2 - Classic standardized HF channel
# Available presets: good_low_latency, good_high_latency, moderate,
#                    moderate_multipath, poor, poor_multipath, flutter
channel = CCIR520Channel.good()  # Good conditions: 0.5ms delay, 0.1Hz Doppler
channel = CCIR520Channel.moderate()  # Moderate: 1ms delay, 0.5Hz Doppler
channel = CCIR520Channel.poor()  # Poor: 2ms delay, 1Hz Doppler
channel = CCIR520Channel.flutter()  # Flutter: 1ms delay, 10Hz Doppler

# Or create from specific preset
channel = CCIR520Channel.from_preset(CCIR520Condition.POOR_MULTIPATH)

# ITU-R F.1289 - Wideband HF channel (up to 24 kHz bandwidth)
# Includes frequency-selective fading and group delay dispersion
wideband = ITURF1289Channel.from_preset(
    ITURF1289Condition.MID_LATITUDE_MODERATE,
    sample_rate_hz=48000,
    bandwidth_khz=12.0,
)

# Check frequency selectivity
coherence_bw = wideband.get_coherence_bandwidth()
is_selective = wideband.is_frequency_selective()
print(f"Coherence BW: {coherence_bw:.0f} kHz, Freq-selective: {is_selective}")

# ITU-R F.1487 - Modem testing (bandwidths up to 12 kHz)
# Table 1 presets: quiet, moderate, disturbed, flutter
channel = ITURF1487Channel.quiet()     # τ=0.5ms, ν=0.1Hz
channel = ITURF1487Channel.moderate()  # τ=2ms, ν=1Hz
channel = ITURF1487Channel.disturbed() # τ=4ms, ν=2Hz
channel = ITURF1487Channel.flutter()   # τ=7ms, ν=10Hz

# List all available presets
print("CCIR 520 presets:", list_ccir520_presets())
print("F.1289 presets:", list_iturf1289_presets())

# Get preset details
info = get_preset_info("moderate")
print(f"Moderate: delay={info['delay_spread_ms']}ms, doppler={info['doppler_spread_hz']}Hz")

# Create channel by preset name (auto-selects model type)
channel = create_channel("mid_latitude_disturbed")  # Returns ITURF1289Channel

# Process signal
input_signal = (np.random.randn(4096) + 1j * np.random.randn(4096)).astype(np.complex64)
output = channel.process_block(input_signal)
```

**Available ITU-R Presets:**

| Standard | Preset | Delay Spread | Doppler Spread | Use Case |
|----------|--------|--------------|----------------|----------|
| CCIR 520 | good_low_latency | 0.5 ms | 0.1 Hz | Stable mid-latitude daytime |
| CCIR 520 | moderate | 1.0 ms | 0.5 Hz | Typical conditions |
| CCIR 520 | poor | 2.0 ms | 1.0 Hz | Disturbed ionosphere |
| CCIR 520 | flutter | 1.0 ms | 10.0 Hz | High-latitude auroral |
| F.1289 | low_latitude_quiet | 0.3 ms | 0.05 Hz | Equatorial stable |
| F.1289 | mid_latitude_moderate | 2.0 ms | 1.0 Hz | Typical wideband |
| F.1289 | high_latitude_flutter | 2.0 ms | 10.0 Hz | Auroral wideband |
| F.1487 | quiet | 0.5 ms | 0.1 Hz | ITU modem test - good |
| F.1487 | moderate | 2.0 ms | 1.0 Hz | ITU modem test - typical |
| F.1487 | disturbed | 4.0 ms | 2.0 Hz | ITU modem test - poor |
| F.1487 | flutter | 7.0 ms | 10.0 Hz | ITU modem test - severe |

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

# VH RF Chain processor (GPU-accelerated with CPU fallback)
from hfpathsim.gpu import VHRFChainProcessor

proc = VHRFChainProcessor(
    input_rate=8000,           # Baseband sample rate
    rf_rate=1000000,           # RF processing rate (125x upsample)
    max_input_samples=4096,
    carrier_freq_hz=10000.0,   # RF carrier frequency
    coherence_time_sec=0.5,    # AR(1) fading coherence time
    k_factor=0.0,              # Rician K-factor (0 = Rayleigh)
)

# Configure tapped delay line
proc.configure_taps(
    delays=[0, 10, 25],           # Delay in RF samples
    amplitudes=[1.0, 0.7, 0.3],   # Tap amplitudes
    doppler_hz=[0.0, 0.5, -0.3],  # Doppler shift per tap
)

# Process through full RF chain
output = proc.process(input_signal)
print(f"Using GPU: {proc.is_using_gpu()}")

# DisplayProcessor for GUI computations (auto-selects GPU/CPU)
from hfpathsim.gpu import DisplayProcessor

disp = DisplayProcessor(max_spectrum_size=4096)
print(f"Display using GPU: {disp.is_using_gpu()}")

# Fast dB conversion for spectrum display
magnitude = np.abs(np.fft.fft(input_signal[:4096]))
spectrum_db = disp.magnitude_to_db(magnitude.astype(np.float32))

# FFT shift for centered spectrum display
spectrum_shifted = disp.fftshift(spectrum_db)

# Moving average smoothing
spectrum_smooth = disp.moving_average(spectrum_shifted, window_size=5)

# Peak hold with decay for waterfall display
peak_buffer = np.full(4096, -120.0, dtype=np.float32)
disp.peak_hold(spectrum_smooth, peak_buffer, decay_rate=0.5)

# Exponential smoothing for smooth updates
smoothed = np.zeros(4096, dtype=np.float32)
disp.exponential_smooth(spectrum_smooth, smoothed, alpha=0.3)

# Normalize scattering function for 2D display
S = np.random.rand(64, 64).astype(np.float32)
S_normalized = disp.normalize_scattering(S.flatten(), rows=64, cols=64, min_clip_db=-60.0)
```

### Profiling and Benchmarking

```python
from hfpathsim.profiling import (
    # CPU timing
    Timer, timer, profile_function, get_timing_stats, print_timing_report,
    # GPU profiling
    GPUProfiler, gpu_timer, CUDATimer, get_gpu_memory_info,
    # Memory profiling
    MemoryProfiler, get_memory_usage, track_memory,
    # Benchmarking
    Benchmark, BenchmarkSuite, run_throughput_benchmark, run_latency_benchmark,
    # Reports
    generate_report, export_report_json, export_report_html,
)
import numpy as np

# Simple timing with context manager
with timer("fft_processing") as t:
    result = np.fft.fft(data)
print(f"FFT took {t.elapsed_ms:.3f}ms")

# Function profiling decorator
@profile_function(print_result=True)
def process_channel(signal, H):
    return np.fft.ifft(np.fft.fft(signal) * H)

# Get timing statistics after multiple calls
for _ in range(100):
    process_channel(data, H)
print_timing_report()

# GPU kernel timing
with gpu_timer("cuda_fft") as t:
    import cupy as cp
    result = cp.fft.fft(cp.asarray(data))
print(f"GPU FFT took {t.elapsed_ms:.3f}ms")

# Comprehensive GPU profiling session
profiler = GPUProfiler()
profiler.start_session("channel_processing")

with profiler.profile("fft", n_samples=4096):
    X = np.fft.fft(data)

with profiler.profile("multiply", n_samples=4096):
    Y = X * H

with profiler.profile("ifft", n_samples=4096):
    output = np.fft.ifft(Y)

report = profiler.end_session()
profiler.print_session_report("channel_processing")

# Memory profiling
with track_memory("data_processing", print_result=True) as tracker:
    large_data = np.zeros((1000000,), dtype=np.complex64)
    processed = np.fft.fft(large_data)

# Check GPU memory
gpu_mem = get_gpu_memory_info()
print(f"GPU: {gpu_mem.used_gb:.2f}/{gpu_mem.total_gb:.2f} GB ({gpu_mem.utilization_pct:.1f}%)")

# Throughput benchmark
def my_process(data):
    return np.fft.fft(data)

results = run_throughput_benchmark(
    func=my_process,
    sample_sizes=[1024, 4096, 16384, 65536],
    iterations=50,
)

for size, result in results.items():
    print(f"n={size}: {result.throughput_msps:.2f} Msps")

# Latency benchmark with percentiles
latency = run_latency_benchmark(
    func=my_process,
    n_samples=4096,
    iterations=1000,
    percentiles=[50, 90, 99],
)
print(f"Latency: mean={latency['mean_us']:.1f}us, p99={latency['p99_us']:.1f}us")

# Full benchmark suite
suite = BenchmarkSuite("hf_channel")
suite.add("fft_4096", lambda d: np.fft.fft(d),
          setup=lambda: np.random.randn(4096).astype(np.complex64), n_samples=4096)
suite.add("overlap_save", overlap_save_func,
          setup=setup_overlap_save, n_samples=65536)
results = suite.run_all(iterations=100)
suite.print_report()

# Generate comprehensive performance report
report = generate_report(benchmark_results=results)
export_report_json(report, "performance_report.json")
export_report_html(report, "performance_report.html")
```

### Real-World Validation

```python
from hfpathsim.validation import (
    # Reference datasets
    NTIA_MIDLATITUDE_QUIET,
    NTIA_MIDLATITUDE_DISTURBED,
    ITU_F1487_MODERATE,
    WATTERSON_1970_GOOD,
    get_reference_dataset,
    list_reference_datasets,
    # Statistics
    compute_delay_spread,
    compute_doppler_spread,
    compute_coherence_bandwidth,
    compute_coherence_time,
    compute_fading_statistics,
    rayleigh_fit_test,
    compute_scattering_function,
    # Validation
    ChannelValidator,
    validate_channel,
)
import numpy as np

# List available reference datasets
print(list_reference_datasets())
# ['ntia_midlatitude_quiet', 'itu_f1487_moderate', ...]

# Get reference by name
ref = get_reference_dataset("ntia_midlatitude_quiet")
print(f"Delay spread: {ref.delay_spread_ms} ms, Doppler spread: {ref.doppler_spread_hz} Hz")

# Compute delay spread from impulse response
h = np.exp(-np.arange(100) / 50).astype(complex)  # Exponential decay
result = compute_delay_spread(h, sample_rate_hz=48000)
print(f"RMS delay spread: {result.rms_delay_spread_ms:.3f} ms")
print(f"Coherence bandwidth: {compute_coherence_bandwidth(result.rms_delay_spread_ms):.2f} kHz")

# Compute Doppler spread from fading coefficients
fading = (np.random.randn(1000) + 1j * np.random.randn(1000)) / np.sqrt(2)
result = compute_doppler_spread(fading, sample_rate_hz=100)
print(f"RMS Doppler spread: {result.rms_doppler_spread_hz:.3f} Hz")
print(f"Coherence time: {compute_coherence_time(result.rms_doppler_spread_hz):.1f} ms")

# Test if envelope follows Rayleigh distribution
envelope = np.abs(fading)
pvalue = rayleigh_fit_test(envelope)
print(f"Rayleigh fit p-value: {pvalue:.4f} (>0.05 = good fit)")

# Full fading statistics
stats = compute_fading_statistics(envelope, sample_rate_hz=100)
print(f"Fade depth: {stats.fade_depth_db:.1f} dB")
print(f"Level crossing rate: {stats.level_crossing_rate_hz:.3f} Hz")

# Validate channel against reference
report = validate_channel(
    fading_coefficients=fading,
    reference="ntia_midlatitude_quiet",
)

# Print validation summary
report.print_summary()

# Check pass rate and failed tests
print(f"Pass rate: {report.get_pass_rate():.1f}%")
for test in report.get_failed_tests():
    print(f"FAILED: {test.name} - {test.details}")

# Export report to JSON
with open("validation_report.json", "w") as f:
    f.write(report.to_json())

# Full validation with impulse responses
validator = ChannelValidator(
    reference=NTIA_MIDLATITUDE_QUIET,
    delay_tolerance_pct=50.0,
    doppler_tolerance_pct=50.0,
)

# Create synthetic channel data
n_snapshots, n_taps = 100, 50
impulse_responses = np.random.randn(n_snapshots, n_taps) + 1j * np.random.randn(n_snapshots, n_taps)

report = validator.validate(
    impulse_responses=impulse_responses,
    fading_coefficients=fading,
    sample_rate_hz=48000,
    snapshot_rate_hz=100,
)
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

### File Output

```python
from hfpathsim.output.file import FileOutputSink
from hfpathsim.output.base import OutputFormat
import numpy as np

# Write raw binary IQ
sink = FileOutputSink(
    "output.cf32",
    sample_rate_hz=2e6,
    output_format=OutputFormat.COMPLEX64,
)

sink.open()
samples = (np.random.randn(4096) + 1j * np.random.randn(4096)).astype(np.complex64)
sink.write(samples)
sink.close()

print(f"Duration: {sink.duration:.3f} sec")
print(f"Samples written: {sink.total_samples_written}")

# Write WAV file (stereo I/Q)
wav_sink = FileOutputSink(
    "output.wav",
    sample_rate_hz=48000,
    output_format=OutputFormat.WAV_STEREO,
)

# Write SigMF with metadata
sigmf_sink = FileOutputSink(
    "output.sigmf-data",
    sample_rate_hz=2e6,
    output_format=OutputFormat.SIGMF,
    center_freq_hz=14.074e6,  # Optional metadata
)
```

### Network Output

```python
from hfpathsim.output.network import NetworkOutputSink, NetworkProtocol

# ZMQ PUB socket for GNU Radio
zmq_sink = NetworkOutputSink(
    host="*",
    port=5555,
    protocol=NetworkProtocol.ZMQ_PUB,
    sample_rate_hz=2e6,
)

# TCP server
tcp_sink = NetworkOutputSink(
    host="0.0.0.0",
    port=5000,
    protocol=NetworkProtocol.TCP,
    sample_rate_hz=2e6,
)

# UDP broadcast
udp_sink = NetworkOutputSink(
    host="192.168.1.255",
    port=5000,
    protocol=NetworkProtocol.UDP,
    sample_rate_hz=2e6,
)

zmq_sink.open()
zmq_sink.write(samples)
zmq_sink.close()
```

### Audio Output

```python
from hfpathsim.output.audio import AudioOutputSink

# List available devices
devices = AudioOutputSink.list_devices()
for dev in devices:
    print(f"{dev['index']}: {dev['name']} ({dev['max_output_channels']} ch)")

# Create audio sink
audio_sink = AudioOutputSink(
    sample_rate_hz=48000,
    device_index=None,  # Use default device
    buffer_size=2048,
)

audio_sink.open()
audio_sink.write(samples)  # Plays through speakers
audio_sink.close()
```

### SDR Output (Transmission)

```python
from hfpathsim.output.sdr import SDROutputSink

# List available TX devices
devices = SDROutputSink.enumerate_devices()
for dev in devices:
    print(f"{dev['driver']}: {dev['label']}")

# Create SDR sink
sdr_sink = SDROutputSink(
    device_args="driver=hackrf",
    sample_rate_hz=2e6,
    center_freq_hz=14.074e6,
    gain_db=40.0,
)

sdr_sink.open()
sdr_sink.write(samples)  # Transmits over the air
sdr_sink.close()
```

**Warning:** Transmitting requires proper licensing and authorization. Ensure compliance with local regulations.

### Multiplex Output (Multiple Destinations)

```python
from hfpathsim.output.multiplex import MultiplexOutputSink, TeeOutputSink
from hfpathsim.output.file import FileOutputSink
from hfpathsim.output.network import NetworkOutputSink, NetworkProtocol

# Create multiplex sink for parallel output
multiplex = MultiplexOutputSink(sample_rate_hz=2e6)

# Add multiple destinations
file_sink = FileOutputSink("output.cf32", sample_rate_hz=2e6)
zmq_sink = NetworkOutputSink(host="*", port=5555, protocol=NetworkProtocol.ZMQ_PUB, sample_rate_hz=2e6)

multiplex.add_sink("file", file_sink)
multiplex.add_sink("network", zmq_sink)

multiplex.open()  # Opens all sinks
multiplex.write(samples)  # Writes to all sinks
multiplex.close()  # Closes all sinks

# Get status of each sink
status = multiplex.get_sink_status()
for name, info in status.items():
    print(f"{name}: {info['samples_written']} samples, open={info['is_open']}")

# Alternative: Tee sink (simpler two-way split)
tee = TeeOutputSink(file_sink, zmq_sink)
tee.open()
tee.write(samples)
tee.close()
```

### GNU Radio Integration

```python
from hfpathsim.integration.gnuradio_zmq import GNURadioZMQBridge

# Create bridge for bidirectional streaming
bridge = GNURadioZMQBridge(
    pub_port=5555,   # HFPathSim -> GNU Radio
    sub_port=5556,   # GNU Radio -> HFPathSim
    sample_rate_hz=2e6,
)

# Get GNU Radio Python snippets
source_code = bridge.create_source_snippet()
print("GNU Radio Source Block:")
print(source_code)
# Output:
# from gnuradio import zeromq
# zmq_source = zeromq.sub_source(gr.sizeof_gr_complex, 1, 'tcp://localhost:5555', 100, False, -1, '')

sink_code = bridge.create_sink_snippet()
print("\nGNU Radio Sink Block:")
print(sink_code)
# Output:
# from gnuradio import zeromq
# zmq_sink = zeromq.pub_sink(gr.sizeof_gr_complex, 1, 'tcp://*:5556', 100, False, -1, '')

# Open and stream
with bridge:
    bridge.send_samples(samples)
    received = bridge.receive_samples(timeout_ms=100)

# Get connection info for GRC
info = bridge.get_connection_info()
print(f"Source address: {info['source_address']}")
print(f"Sink address: {info['sink_address']}")
```

### MATLAB Integration

```python
from hfpathsim.integration.matlab_interface import MATFileInterface, ChannelSnapshot
from hfpathsim.core.channel import HFChannel
from hfpathsim.core.parameters import VoglerParameters, ITUCondition
import numpy as np

# Create interface
mat = MATFileInterface()

# Save channel state snapshot
channel = HFChannel(VoglerParameters.from_itu_condition(ITUCondition.MODERATE))
state = channel.get_state()

snapshot = ChannelSnapshot(
    transfer_function=state.transfer_function,
    impulse_response=state.impulse_response,
    delay_spread_ms=2.0,
    doppler_spread_hz=1.0,
    sample_rate_hz=2e6,
)

mat.save_channel_state(snapshot, "channel_state.mat")

# Load in MATLAB:
# >> data = load('channel_state.mat');
# >> plot(abs(data.transfer_function));
# >> title(sprintf('Delay spread: %.1f ms', data.delay_spread_ms));

# Save IQ recording with metadata
iq_data = (np.random.randn(100000) + 1j * np.random.randn(100000)).astype(np.complex64)
mat.save_iq_recording(
    iq_data,
    "recording.mat",
    sample_rate_hz=2e6,
    center_freq_hz=14.074e6,
    description="Test recording with moderate fading",
)

# Save channel evolution (time series of states)
snapshots = []
for i in range(100):
    channel.process(np.random.randn(4096).astype(np.complex64))
    state = channel.get_state()
    snapshots.append(ChannelSnapshot(
        transfer_function=state.transfer_function,
        impulse_response=state.impulse_response,
        timestamp=i * 0.1,
    ))

mat.save_channel_evolution(snapshots, "evolution.mat")

# Load in MATLAB:
# >> data = load('evolution.mat');
# >> imagesc(abs(data.transfer_functions));
# >> xlabel('Frequency bin'); ylabel('Time index');
```

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
│   │   ├── itu_channels.py        # ITU-R F.520, F.1289, F.1487 standardized channels
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
│   │       ├── signal_proc.cu
│   │       ├── vh_rf_chain.cu     # VH RF chain GPU implementation
│   │       ├── vh_rf_chain_cpu.cpp # VH RF chain CPU fallback
│   │       ├── watterson.cu       # Watterson channel GPU
│   │       ├── watterson_cpu.cpp  # Watterson CPU fallback
│   │       ├── agc.cu             # AGC GPU implementation
│   │       ├── agc_cpu.cpp        # AGC CPU fallback
│   │       ├── noise.cu           # Noise generator GPU
│   │       ├── noise_cpu.cpp      # Noise CPU fallback
│   │       ├── resampler.cu       # Polyphase resampler GPU
│   │       ├── resampler_cpu.cpp  # Resampler CPU fallback
│   │       ├── dispersion.cu      # Ionospheric dispersion GPU
│   │       ├── dispersion_cpu.cpp # Dispersion CPU fallback
│   │       ├── display.cu         # GUI display computations GPU
│   │       └── display_cpu.cpp    # Display CPU fallback (OpenMP)
│   │
│   ├── input/                     # Input sources
│   │   ├── base.py                # InputSource ABC, InputFormat
│   │   ├── file.py                # File playback (WAV, SigMF, raw)
│   │   ├── network.py             # TCP/UDP/ZMQ streaming
│   │   ├── sdr.py                 # SoapySDR interface
│   │   └── flexradio.py           # Flex Radio SmartSDR DAX IQ streaming
│   │
│   ├── output/                    # Output sinks
│   │   ├── __init__.py            # Module exports
│   │   ├── base.py                # OutputSink ABC, OutputFormat
│   │   ├── file.py                # File output (raw, WAV, SigMF)
│   │   ├── network.py             # Network streaming (ZMQ, TCP, UDP)
│   │   ├── audio.py               # Audio playback via sounddevice
│   │   ├── sdr.py                 # SDR transmission via SoapySDR
│   │   └── multiplex.py           # Multiplex and Tee sinks
│   │
│   ├── integration/               # External tool integrations
│   │   ├── __init__.py            # Module exports
│   │   ├── gnuradio_zmq.py        # GNU Radio ZMQ bridge
│   │   └── matlab_interface.py    # MATLAB/scipy .mat interface
│   │
│   ├── profiling/                 # Performance profiling and benchmarking
│   │   ├── __init__.py            # Module exports
│   │   ├── timing.py              # CPU timing utilities
│   │   ├── gpu_profiler.py        # CUDA kernel profiling
│   │   ├── memory.py              # Memory usage tracking
│   │   ├── benchmarks.py          # Throughput/latency benchmarks
│   │   └── reports.py             # Report generation (JSON, HTML)
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
│           ├── output_config.py   # Output sink configuration
│           ├── channel_panel.py   # Vogler/Watterson channel controls
│           ├── noise_panel.py     # Noise injection controls
│           ├── impairments_panel.py # AGC/limiter/freq offset controls
│           ├── ionosphere_panel.py  # Ray tracing, Es, geomagnetic controls
│           ├── recording_panel.py # Record/playback controls
│           └── parameters.py      # Legacy parameter panel (deprecated)
│
├── tests/                         # Unit tests (290+ tests)
│   ├── test_vogler.py             # Vogler model tests
│   ├── test_channel_models.py     # Watterson, noise, impairments
│   ├── test_itu_channels.py       # ITU-R F.520, F.1289, F.1487 models
│   ├── test_profiling.py          # Profiling and benchmarking
│   ├── test_raytracing.py         # Ray tracing geometry & engine
│   ├── test_sporadic_e.py         # Sporadic-E layer
│   ├── test_geomagnetic.py        # Geomagnetic effects
│   ├── test_input.py              # Input sources
│   ├── test_output.py             # Output sinks (file, network, audio, SDR, multiplex)
│   ├── test_integration.py        # External integrations (GNU Radio, MATLAB)
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
collected 397 items

tests/test_api.py ....                                                    [  1%]
tests/test_channel_models.py ...................................................[ 14%]
tests/test_engine.py ........................................             [ 24%]
tests/test_geomagnetic.py ......................................          [ 34%]
tests/test_gpu.py ...................................                     [ 43%]
tests/test_input.py .................                                     [ 47%]
tests/test_integration.py .....................                           [ 52%]
tests/test_itu_channels.py ..................................             [ 61%]
tests/test_output.py ............................                         [ 68%]
tests/test_profiling.py ..........................                        [ 75%]
tests/test_raytracing.py .....................................            [ 84%]
tests/test_spectrum.py .........................                          [ 90%]
tests/test_sporadic_e.py ............................                     [ 97%]
tests/test_validation.py ................................................ [ 97%]
tests/test_vogler.py .........................                            [100%]

================== 397 passed, 1 skipped, 1 warning in 7.17s ===================
```

### Test Summary

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_api.py` | 4 | REST API endpoints, health checks |
| `test_channel_models.py` | 51 | Watterson, noise, AGC, limiter, impairments, recording |
| `test_engine.py` | 40 | SimulationEngine, SessionManager, streaming |
| `test_geomagnetic.py` | 38 | Indices, foF2/hmF2 scaling, storm effects, Kp/Ap conversion |
| `test_gpu.py` | 35 | Native CUDA, batched FFT, Doppler fading, spectrum, benchmarks |
| `test_input.py` | 17 | File sources, network sources, format conversion |
| `test_integration.py` | 21 | GNU Radio ZMQ bridge, MATLAB .mat interface, channel snapshots |
| `test_itu_channels.py` | 34 | CCIR 520, ITU-R F.1289, ITU-R F.1487 standardized channels |
| `test_output.py` | 28 | File/network/audio/SDR sinks, multiplex, tee, format conversion |
| `test_profiling.py` | 26 | CPU timing, GPU profiling, memory tracking, benchmarks |
| `test_raytracing.py` | 37 | Geometry, ionosphere profiles, ray engine, path finder |
| `test_spectrum.py` | 25 | FFT computation, windowing, averaging, peak hold, GUI widget |
| `test_sporadic_e.py` | 28 | Es config, layer injection, occurrence estimation |
| `test_validation.py` | 48 | Reference datasets, channel statistics, fading validation |
| `test_vogler.py` | 25 | Vogler parameters, HFChannel, reflection coefficients |
| **Total** | **397** | **All passing** |

### Test Categories

**Unit Tests** - Test individual components in isolation:
- Parameter validation and defaults
- Mathematical computations (gamma functions, geometry)
- Signal processing correctness
- Configuration handling
- Output sink format conversion
- ITU-R channel model presets

**Integration Tests** - Test component interactions:
- Channel processing pipeline
- Ray tracing with ionosphere profiles
- Geomagnetic modulation of channel parameters
- GPU/CPU fallback behavior
- GNU Radio ZMQ bridge streaming
- MATLAB .mat file round-trip
- SimulationEngine with all channel models

**Output Tests** - Test output sink functionality:
- File writing (raw, WAV, SigMF formats)
- Network streaming (ZMQ, TCP, UDP protocols)
- Audio device enumeration and playback
- SDR device enumeration
- Multiplex sink parallel output
- Tee sink stream splitting

**Validation Tests** - Verify channel accuracy:
- Reference dataset comparison (NTIA, ITU-R, Watterson)
- Delay spread and Doppler spread statistics
- Rayleigh fading distribution fit
- Scattering function analysis
- Coherence bandwidth and time calculations

**Performance Tests** - Verify throughput requirements:
- GPU batched FFT throughput (68.9 Msps achieved)
- Streaming processing latency
- Memory usage patterns
- CPU and GPU profiling infrastructure

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
| DisplayProcessor dB conversion | <0.1 ms / 4096 pts | magnitude_to_db, power_to_db |
| DisplayProcessor smoothing | <0.1 ms / 4096 pts | moving_average, peak_hold |
| Scattering normalization | <0.5 ms / 64x64 grid | 2D dB + normalize to [0,1] |
| Scattering function | ~1 ms / 64x32 grid | 2D power distribution |
| Ray tracing (single ray) | <10 ms | CPU Haselgrove integration |
| Mode discovery (3 hops) | <100 ms | CPU path finder |
| ZMQ output streaming | >10 Msps | PUB socket, zero-copy |
| File output (raw) | >50 Msps | Memory-mapped I/O |
| Audio output | 192 kHz max | sounddevice limitation |

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

### Phase 7: Output & Integration ✓
- [x] Output sink architecture with base class and format conversion
- [x] File output sinks (raw binary, WAV, SigMF with metadata)
- [x] Network streaming sinks (ZMQ PUB, TCP server, UDP broadcast)
- [x] Audio output via sounddevice with device enumeration
- [x] SDR transmission via SoapySDR (HackRF, USRP, LimeSDR, etc.)
- [x] Multiplex sink for parallel output to multiple destinations
- [x] Tee sink for simple two-way stream splitting
- [x] GNU Radio ZMQ bridge with auto-generated Python snippets
- [x] MATLAB/scipy .mat file interface for channel state export
- [x] MATLAB Engine interface for live computation (optional)
- [x] GUI Output tab with sink configuration and streaming status
- [x] Main window integration with output sink processing
- [x] Comprehensive test suite (48 new tests for output and integration)

### Phase 8: Deployment & Packaging ✓
- [x] Docker containerization with GPU support
- [x] Docker Compose for multi-service deployment
- [x] PyPI package publication
- [x] Conda-forge recipe
- [x] Cloud deployment option (AWS/GCP/Azure)
- [x] REST API for remote control
- [x] Web-based dashboard alternative

### Phase 9: RF Processing Chain & Audio Improvements ✓
- [x] RF processing chain for Vogler and Vogler-Hoffmeyer models
  - Upsample from baseband (8 kHz) to 1 MHz RF rate
  - Mix to 100 kHz RF carrier frequency
  - Apply ionospheric channel model at RF rate
  - Mix back to baseband with anti-aliasing filter
  - Downsample back to baseband rate
- [x] Real voice samples for SSB signal generator (CMU Arctic corpus)
- [x] Fixed Watterson Doppler filter design (proper fade depth 20-27 dB)
- [x] Proper Rayleigh fading with audible ionospheric effects
- [x] Signal generator with RTTY, SSB Voice, and PSK31 modes

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
- The ZeroMQ team for high-performance messaging
- The sounddevice team for cross-platform audio I/O
- The SoapySDR project for unified SDR hardware abstraction
- The GNU Radio project for DSP ecosystem integration

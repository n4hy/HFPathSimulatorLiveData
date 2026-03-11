# HF Path Simulator

A real-time HF (High Frequency) ionospheric channel simulator implementing the Vogler-Hoffmeyer Ionospheric Propagation Model (IPM) with GPU acceleration for RTX 5090.

## Overview

This project provides a physically-accurate simulation of HF radio propagation through the ionosphere, suitable for:

- **HF modem development and testing** - Test modems against realistic channel conditions without over-the-air transmission
- **Algorithm research** - Evaluate equalization, synchronization, and coding schemes
- **Training and education** - Visualize ionospheric propagation effects in real-time
- **Live signal processing** - Apply channel effects to SDR input for hardware-in-the-loop testing

The simulator implements the Vogler-Hoffmeyer reflection coefficient model from NTIA Technical Report TR-88-240, combined with ITU-R F.1487 channel statistics for realistic time-varying fading.

## Features

### Implemented (Phases 1-5)

- **Vogler-Hoffmeyer IPM Core**
  - Complex gamma function computation for reflection coefficient R(ω)
  - Frequency-dependent amplitude and phase response
  - Group delay variation across bandwidth
  - Multi-mode propagation (1F2, 2F2, E-layer)

- **ITU-R F.1487 Channel Conditions**
  - Quiet: τ=0.5ms, ν=0.1Hz (benign mid-latitude)
  - Moderate: τ=2ms, ν=1Hz (typical daytime)
  - Disturbed: τ=4ms, ν=2Hz (magnetic storm)
  - Flutter: τ=7ms, ν=10Hz (high-latitude)

- **Real-Time Signal Processing**
  - Overlap-save convolution for continuous streaming
  - 4096-point FFT blocks with configurable overlap
  - Support for up to 2 Msps complex sample rates

- **Watterson Tapped Delay Line Model**
  - Classic Watterson HF channel model
  - Multiple taps with configurable delays and amplitudes
  - Independent Rayleigh/Rician fading per tap
  - Gaussian, flat, and Jakes Doppler spectrum shapes
  - CCIR Good/Moderate/Poor presets

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
  - Impairment chain for combined effects

- **Channel Recording & Playback**
  - Record time-varying channel states
  - Save in NPZ, HDF5, or JSON formats
  - Playback with interpolation
  - Reproducible testing scenarios

- **Physics-Based Ray Tracing (Phase 4)**
  - Spherical Earth geometry with computed sec(φ) replacing hardcoded approximations
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

- **PyQt6 Dashboard**
  - Channel frequency response |H(f)| display
  - Impulse response |h(t)| visualization
  - Phase response and group delay plots
  - Scattering function S(τ,ν) 2D intensity display
  - Input/output spectrum analyzers
  - Real-time parameter controls

- **Input Sources**
  - File playback: WAV, SigMF, raw binary (complex64, int16, int8)
  - Network streams: TCP, UDP, ZeroMQ
  - SDR support via SoapySDR (RTL-SDR, HackRF, USRP, etc.)

- **Ionospheric Data Sources**
  - Manual parameter entry
  - GIRO/DIDBase real-time ionosonde data
  - IRI-2020 model integration (optional)

- **GPU Acceleration (Phase 5)**
  - Native CUDA module with cuFFT for maximum performance
  - Batched overlap-save convolution using `cufftPlanMany` (68.9 Msps throughput)
  - GPU-accelerated Doppler fading generation with cuRAND
  - Real-time spectrum computation for GUI
  - CuPy fallback with automatic kernel compilation
  - NumPy CPU fallback for compatibility
  - Build scripts for easy compilation (`scripts/build_gpu.sh`)
  - Designed for RTX 5090 (Blackwell), supports sm_80-90 architectures

### Planned (Future Phases)

- **Phase 6**: Real-time IQ output, GNU Radio integration

## Installation

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.x (optional, for GPU acceleration)
- Qt6 libraries

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/HFPathSimulatorLiveData.git
cd HFPathSimulatorLiveData

# Create and activate virtual environment
python -m venv .hfpathsim
source .hfpathsim/bin/activate

# Install the package
pip install -e .

# Launch the dashboard
python -m hfpathsim
```

### Optional Dependencies

```bash
# For SDR support
pip install soapysdr

# For IRI-2020 ionospheric model
pip install iri2016

# For development/testing
pip install pytest pytest-qt
```

## Usage

### Launch Dashboard

```bash
python -m hfpathsim
```

### Programmatic API

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

### Ray Tracing with Computed Geometry

```python
from hfpathsim.core.raytracing import (
    create_simple_profile,
    find_propagation_modes,
    sec_phi_spherical,
    great_circle_distance,
)

# Create ionosphere profile
profile = create_simple_profile(foF2=7.5, hmF2=300.0, foE=3.0, hmE=110.0)

# Find viable propagation modes for a path
# Washington DC to London
modes = find_propagation_modes(
    profile,
    tx_lat=38.9, tx_lon=-77.0,   # Washington DC
    rx_lat=51.5, rx_lon=-0.1,    # London
    frequency_mhz=14.0,
    max_hops=3,
)

for mode in modes:
    print(f"{mode.name}: delay={mode.group_delay_ms:.1f}ms, "
          f"amplitude={mode.relative_amplitude:.2f}")

# Get physically-computed sec(φ) for MUF calculation
path_km = great_circle_distance(38.9, -77.0, 51.5, -0.1)
sec_phi = sec_phi_spherical(path_km, hm_km=300.0)
muf = 7.5 * sec_phi  # foF2 * sec(φ)
print(f"Path: {path_km:.0f} km, sec(φ): {sec_phi:.2f}, MUF: {muf:.1f} MHz")
```

### Channel with Ray Tracing Integration

```python
from hfpathsim.core.channel import HFChannel, RayTracingConfig
from hfpathsim.core.parameters import VoglerParameters

# Enable physics-based ray tracing
ray_config = RayTracingConfig(
    enabled=True,
    tx_lat=38.9, tx_lon=-77.0,
    rx_lat=51.5, rx_lon=-0.1,
    max_hops=3,
    use_sporadic_e=True,
    use_geomagnetic=True,
)

channel = HFChannel(
    params=VoglerParameters(frequency_mhz=14.0),
    use_ray_tracing=True,
    ray_config=ray_config,
)

# Enable sporadic-E layer
channel.enable_sporadic_e(foEs=8.0, hmEs=105.0)

# Apply geomagnetic storm conditions
channel.set_geomagnetic_indices(f10_7=150, kp=5, dst=-80)

# Get MUF for current conditions
muf = channel.get_muf("F2")
print(f"MUF: {muf:.1f} MHz")
```

### Sporadic-E Simulation

```python
from hfpathsim.iono.sporadic_e import (
    SporadicELayer, SporadicEConfig,
    estimate_es_occurrence, estimate_foEs,
    create_es_from_preset,
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

### Geomagnetic Storm Effects

```python
from hfpathsim.iono.geomagnetic import (
    GeomagneticIndices, GeomagneticModulator,
    classify_storm_phase,
)
from hfpathsim.core.raytracing import create_simple_profile

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

### File Playback

```python
from hfpathsim.input.file import FileInputSource
from hfpathsim.core.channel import HFChannel

# Load IQ file
source = FileInputSource("recording.wav", loop=True)
source.open()

channel = HFChannel()

while True:
    samples = source.read(4096)
    if samples is None:
        break
    output = channel.process(samples)
    # ... do something with output
```

### Watterson Channel Model

```python
from hfpathsim.core.watterson import WattersonChannel, WattersonConfig
from hfpathsim.core.parameters import ITUCondition
import numpy as np

# Create channel from ITU condition preset
config = WattersonConfig.from_itu_condition(ITUCondition.MODERATE)
channel = WattersonChannel(config, seed=42)

# Or use CCIR presets
config = WattersonConfig.ccir_poor()  # 2ms delay, 1Hz Doppler
channel = WattersonChannel(config)

# Process signal
input_signal = np.random.randn(4096).astype(np.complex64)
output = channel.process_block(input_signal)

# Get impulse response for visualization
h = channel.get_impulse_response(length=256)
```

### Noise Injection

```python
from hfpathsim.core.noise import NoiseGenerator, NoiseConfig, ManMadeEnvironment
import numpy as np

# Configure noise sources
config = NoiseConfig(
    snr_db=15.0,
    enable_atmospheric=True,
    frequency_mhz=7.0,  # 40m band
    season="summer",
    time_of_day="night",
    enable_manmade=True,
    environment=ManMadeEnvironment.RESIDENTIAL,
    enable_impulse=True,
    impulse_rate_hz=5.0,
)

noise_gen = NoiseGenerator(config, seed=42)

# Add noise to signal
signal = np.ones(10000, dtype=np.complex64)
noisy_signal = noise_gen.add_noise(signal, normalize=True)
```

### AGC and Impairments

```python
from hfpathsim.core.impairments import (
    AGC, AGCConfig, AGCMode,
    Limiter, LimiterConfig,
    FrequencyOffset, FrequencyOffsetConfig,
    ImpairmentChain,
)
from hfpathsim.core.noise import NoiseGenerator, NoiseConfig

# Create AGC
agc = AGC(AGCConfig(
    mode=AGCMode.FAST,
    target_level_db=-10.0,
    max_gain_db=40.0,
))

# Create limiter
limiter = Limiter(LimiterConfig(
    threshold_db=-3.0,
    mode="soft",
))

# Create frequency offset
freq_offset = FrequencyOffset(FrequencyOffsetConfig(
    offset_hz=25.0,
    drift_rate_hz_per_sec=0.5,
))

# Combine in impairment chain
chain = ImpairmentChain(
    agc=agc,
    limiter=limiter,
    freq_offset=freq_offset,
    noise_generator=NoiseGenerator(NoiseConfig(snr_db=20.0)),
)

# Process signal through entire chain
output = chain.process(input_signal)
print(chain.get_status())  # {'agc_gain_db': 15.2, 'limiter_gr_db': -1.5, ...}
```

### Channel Recording and Playback

```python
from hfpathsim.core.recording import ChannelPlayer, create_test_recording

# Create synthetic test recording
player = create_test_recording(
    duration_sec=10.0,
    snapshot_rate_hz=10.0,
    condition="moderate",
)

# Iterate through recorded channel states
for H in player.iterate(rate_hz=20.0, loop=False):
    # Apply H to your signal: Y = X * H in frequency domain
    pass

# Get channel state at specific time
H = player.get_at_time(5.0, interpolate=True)
```

### GPU Acceleration (Phase 5)

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

### Building the Native CUDA Module

```bash
# Build the GPU module (requires CUDA Toolkit 12.x)
./scripts/build_gpu.sh

# Verify installation
python3 -c "from hfpathsim.gpu import get_backend_info; print(get_backend_info())"

# Expected output with native module:
# {'backend': 'cuda', 'native_module': True, ...}
```

## Project Structure

```
hfpathsim/
├── pyproject.toml              # Package configuration
├── README.md                   # This file
│
├── src/hfpathsim/
│   ├── __init__.py
│   ├── __main__.py             # Entry point
│   │
│   ├── core/                   # Core simulation
│   │   ├── parameters.py       # VoglerParameters, ITUCondition, PropagationMode
│   │   ├── channel.py          # HFChannel with ray tracing integration
│   │   ├── vogler_ipm.py       # Vogler model interface
│   │   ├── watterson.py        # Watterson TDL model
│   │   ├── noise.py            # Noise generators (AWGN, atmospheric, etc.)
│   │   ├── impairments.py      # AGC, limiter, frequency offset
│   │   ├── recording.py        # Channel state recording/playback
│   │   │
│   │   └── raytracing/         # Physics-based ray tracing (Phase 4)
│   │       ├── __init__.py     # Module exports
│   │       ├── geometry.py     # Spherical Earth, sec_phi, great circle
│   │       ├── ionosphere.py   # IonosphereProfile, Ne(h), refractive index
│   │       ├── ray_engine.py   # 2D Haselgrove equations, RayPath
│   │       └── path_finder.py  # Multi-hop mode discovery
│   │
│   ├── gpu/                    # GPU acceleration
│   │   ├── __init__.py         # Python interface
│   │   ├── bindings.cpp        # pybind11 bindings
│   │   ├── CMakeLists.txt      # CUDA build
│   │   └── kernels/
│   │       ├── vogler_transfer.cu
│   │       ├── fading.cu
│   │       └── signal_proc.cu
│   │
│   ├── input/                  # Input sources
│   │   ├── base.py             # InputSource ABC
│   │   ├── file.py             # File playback
│   │   ├── network.py          # TCP/UDP/ZMQ
│   │   └── sdr.py              # SoapySDR
│   │
│   ├── iono/                   # Ionospheric data and effects
│   │   ├── manual.py           # Manual entry
│   │   ├── giro.py             # GIRO client
│   │   ├── iri.py              # IRI-2020 with to_ionosphere_profile()
│   │   ├── sporadic_e.py       # Sporadic-E layer modeling (Phase 4)
│   │   └── geomagnetic.py      # Geomagnetic effects (Phase 4)
│   │
│   └── gui/                    # PyQt6 interface
│       ├── main_window.py
│       ├── resources/style.qss
│       └── widgets/
│           ├── channel_display.py
│           ├── scattering.py
│           ├── spectrum.py          # GPU-accelerated spectrum display
│           ├── parameters.py
│           └── input_config.py
│
├── tests/                      # 201 unit tests
│   ├── test_vogler.py          # Vogler model tests (22 tests)
│   ├── test_input.py           # Input sources (13 tests)
│   ├── test_gpu.py             # GPU acceleration (31 tests)
│   ├── test_channel_models.py  # Watterson, noise, impairments (47 tests)
│   ├── test_raytracing.py      # Ray tracing geometry & engine (33 tests)
│   ├── test_sporadic_e.py      # Sporadic-E layer (24 tests)
│   └── test_geomagnetic.py     # Geomagnetic effects (34 tests)
│
└── scripts/
    ├── build_gpu.sh            # Build native CUDA module
    ├── install_gpu.sh          # Install to site-packages
    └── run_dashboard.py
```

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

### Oblique Incidence Geometry (Phase 4)

The simulator now computes the secant of the angle of incidence using spherical Earth geometry:

```
sec(φ) = 1 / cos(φ)

where sin(φ) = R·sin(β) / slant_range
      β = path_km / (2·R)  (half angular path)
      slant_range = √(R² + (R+h)² - 2R(R+h)cos(β))
```

This replaces the previous hardcoded `sec_phi = 3.0` with physically accurate values based on actual path length and layer height.

### Channel Statistics

The simulator implements the Watterson/Gaussian scatter model for time-varying fading:

- **Delay spread (τ)**: Multipath delay dispersion, causing frequency-selective fading
- **Doppler spread (ν)**: Time-variation rate, causing temporal fading

The scattering function S(τ,ν) describes the power distribution:
```
S(τ,ν) = exp(-τ/τ_rms) × exp(-(ν/ν_rms)²)
```

### Geomagnetic Effects

The simulator models space weather impacts on HF propagation:

- **F10.7 scaling**: `foF2 ∝ √(1 + 0.014·(F10.7 - 100))`
- **Storm depression**: `Δ foF2 = 0.05·Dst·cos²(lat)` (Dst negative during storms)
- **Doppler enhancement**: `ν_enhanced = ν_base · (1 + 0.1·Kp)`

### Key Parameters

| Parameter | Symbol | Typical Range | Description |
|-----------|--------|---------------|-------------|
| foF2 | foF2 | 3-15 MHz | F2 layer critical frequency |
| hmF2 | hmF2 | 200-400 km | F2 layer peak height |
| foEs | foEs | 2-15 MHz | Sporadic-E critical frequency |
| Delay spread | τ | 0.5-7 ms | RMS multipath delay |
| Doppler spread | ν | 0.1-10 Hz | Two-sided Doppler bandwidth |
| Path length | d | 100-10000 km | Great circle distance |
| F10.7 | F10.7 | 65-300 sfu | Solar radio flux index |
| Kp | Kp | 0-9 | Geomagnetic activity index |
| Dst | Dst | -500 to +50 nT | Storm-time index |

## Testing

The project includes comprehensive unit tests covering all modules.

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with summary
pytest tests/ -v --tb=short

# Run specific test modules
pytest tests/test_raytracing.py -v      # Ray tracing tests
pytest tests/test_sporadic_e.py -v      # Sporadic-E tests
pytest tests/test_geomagnetic.py -v     # Geomagnetic tests
pytest tests/test_vogler.py -v          # Vogler model tests
pytest tests/test_channel_models.py -v  # Channel model tests

# Run with coverage report
pytest tests/ --cov=hfpathsim --cov-report=html

# Run tests matching a pattern
pytest tests/ -k "test_sec_phi" -v
```

### Test Categories

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_vogler.py` | 22 | Vogler parameters, HFChannel, reflection coefficients |
| `test_channel_models.py` | 47 | Watterson, noise, AGC, limiter, impairments, recording |
| `test_raytracing.py` | 33 | Geometry, ionosphere profiles, ray engine, path finder |
| `test_sporadic_e.py` | 24 | Es config, layer injection, occurrence estimation |
| `test_geomagnetic.py` | 34 | Indices, foF2/hmF2 scaling, storm effects, Kp/Ap conversion |
| `test_input.py` | 13 | File sources, network sources, format conversion |
| `test_gpu.py` | 31 | Native CUDA, batched FFT, Doppler fading, spectrum, benchmarks |
| **Total** | **204** | |

### Test Structure

Each test module follows a consistent structure:

```python
class TestGeometry:
    """Tests for geometry module."""

    def test_great_circle_distance_known_path(self):
        """Test known great circle distance: NYC to London ~5570 km."""
        d = great_circle_distance(40.7128, -74.0060, 51.5074, -0.1278)
        assert 5500 < d < 5700

    def test_sec_phi_typical_values(self):
        """Check sec_phi is reasonable for typical paths."""
        sec_phi = sec_phi_spherical(1000.0, 300.0)
        assert 1.5 < sec_phi < 3.0
```

### Integration Tests

Integration tests verify end-to-end functionality:

```python
class TestIntegration:
    def test_washington_london_path(self):
        """Test transatlantic path with ray tracing."""
        profile = create_simple_profile(foF2=7.5, hmF2=300.0)
        modes = find_propagation_modes(
            profile,
            tx_lat=38.9, tx_lon=-77.0,
            rx_lat=51.5, rx_lon=-0.1,
            frequency_mhz=14.0,
        )
        assert len(modes) >= 1

    def test_geomagnetic_in_channel(self):
        """Channel should accept geomagnetic modulation."""
        channel = HFChannel(use_ray_tracing=True)
        channel.set_geomagnetic_indices(f10_7=150, kp=4, dst=-50)
        muf = channel.get_muf()
        assert muf > 0
```

### Current Test Status

```
========================= 204 passed in 1.87s =========================
```

All tests pass. The test suite validates:
- Mathematical correctness (gamma functions, geometry)
- Physical consistency (MUF calculations, propagation modes)
- Edge cases (poles, division by zero, out-of-range inputs)
- Integration between modules

## Performance

### Targets
- **Throughput**: 2 Msps sustained real-time
- **Latency**: <50 ms input-to-output
- **GPU memory**: <2 GB for 1 MHz bandwidth

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

### Building for Maximum Performance

```bash
# Build native CUDA module
./scripts/build_gpu.sh

# Verify native module is active
python3 -c "from hfpathsim.gpu import get_backend_info; print(get_backend_info()['backend'])"
# Expected: 'cuda'
```

*Note: The native CUDA module provides maximum performance. Without it, the system falls back to CuPy (if available) or NumPy. RTX 5090 (Blackwell, compute 12.0) requires CUDA Toolkit 12.8+ for optimal sm_120 code generation; CUDA 12.0 compiles for sm_90 which still runs correctly.*

## References

1. **NTIA TR-88-240**: Vogler, L.E. and Hoffmeyer, J.A., "A full-wave calculation of ionospheric Doppler spread and its application to HF channel modeling," 1988.

2. **ITU-R F.1487**: "Testing of HF modems with bandwidths of up to about 12 kHz using ionospheric channel simulators," 2000.

3. **Watterson Model**: Watterson, C.C., Juroshek, J.R., and Bensema, W.D., "Experimental confirmation of an HF channel model," IEEE Trans. Comm. Tech., 1970.

4. **IRI-2020**: International Reference Ionosphere model, https://irimodel.org/

5. **ITU-R P.372**: "Radio noise," for atmospheric and man-made noise modeling.

6. **Haselgrove Equations**: Haselgrove, J., "Ray theory and a new method for ray tracing," Physics of the Ionosphere, 1955.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Development Setup

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Check GPU functionality
python -c "from hfpathsim.gpu import get_device_info; print(get_device_info())"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NTIA/ITS for the Vogler-Hoffmeyer model documentation
- ITU-R for HF channel characterization standards
- The pyqtgraph team for excellent real-time plotting
- NVIDIA for CUDA and CuPy GPU acceleration

---

## Development Roadmap

### Phase 1: Foundation (Complete)
- [x] Project structure and packaging
- [x] PyQt6 dashboard skeleton
- [x] Input source abstraction
- [x] GPU detection and pybind11 setup

### Phase 2: Vogler-Hoffmeyer Core (Complete)
- [x] Complex gamma function implementation
- [x] Reflection coefficient R(ω)
- [x] ITU-R F.1487 presets
- [x] Gaussian scatter fading
- [x] Overlap-save convolution
- [x] Real-time visualization

### Phase 3: Enhanced Fidelity (Complete)
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

### Phase 4: Advanced Propagation (Complete)
- [x] Spherical Earth geometry (replaces hardcoded sec_phi)
- [x] 2D Haselgrove ray equation integration
- [x] Multi-hop propagation mode discovery (1F, 2F, 3F, E, Es)
- [x] IonosphereProfile with Ne(h) electron density
- [x] Sporadic-E layer modeling with time variation
- [x] Geomagnetic effects (F10.7, Kp, Dst modulation)
- [x] HFChannel integration with ray tracing
- [x] 91 new unit tests (185 total)

### Phase 5: GPU Acceleration (Complete)
- [x] Native CUDA module with pybind11 bindings
- [x] Batched cuFFT overlap-save using `cufftPlanMany` (68.9 Msps)
- [x] GPU-accelerated Doppler fading generation with cuRAND
- [x] Real-time spectrum computation for GUI
- [x] Build scripts (`build_gpu.sh`, `install_gpu.sh`)
- [x] CUDA architecture detection (sm_80-90, sm_100-120 with CUDA 12.8+)
- [x] Fallback chain: Native CUDA → CuPy → NumPy
- [x] 19 new GPU tests (204 total)

### Phase 6: Integration (Planned)
- [ ] Real-time IQ output (sound card, network)
- [ ] GNU Radio source/sink blocks
- [ ] MATLAB/Simulink interface
- [ ] Docker containerization
- [ ] Cloud deployment option

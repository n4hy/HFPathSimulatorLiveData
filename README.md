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

### Implemented (Phases 1-3)

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

- **GPU Acceleration**
  - CUDA kernels for Vogler transfer function
  - CuPy fallback with automatic kernel compilation
  - NumPy CPU fallback for compatibility
  - Designed for RTX 5090 (Blackwell, compute 12.0)

### Planned (Future Phases)

- **Phase 4**: Full ray tracing, oblique incidence geometry
- **Phase 5**: Real-time IQ output, GNU Radio integration

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
│   │   ├── parameters.py       # VoglerParameters, ITUCondition
│   │   ├── channel.py          # HFChannel class
│   │   ├── vogler_ipm.py       # Vogler model interface
│   │   ├── watterson.py        # Watterson TDL model
│   │   ├── noise.py            # Noise generators (AWGN, atmospheric, etc.)
│   │   ├── impairments.py      # AGC, limiter, frequency offset
│   │   └── recording.py        # Channel state recording/playback
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
│   ├── iono/                   # Ionospheric data
│   │   ├── manual.py           # Manual entry
│   │   ├── giro.py             # GIRO client
│   │   └── iri.py              # IRI-2020
│   │
│   └── gui/                    # PyQt6 interface
│       ├── main_window.py
│       ├── resources/style.qss
│       └── widgets/
│           ├── channel_display.py
│           ├── scattering.py
│           ├── spectrum.py
│           ├── parameters.py
│           └── input_config.py
│
├── tests/
│   ├── test_vogler.py          # 22 tests
│   ├── test_input.py           # 13 tests
│   ├── test_gpu.py             # 12 tests
│   └── test_channel_models.py  # 47 tests (Watterson, noise, impairments, recording)
│
└── scripts/
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

### Channel Statistics

The simulator implements the Watterson/Gaussian scatter model for time-varying fading:

- **Delay spread (τ)**: Multipath delay dispersion, causing frequency-selective fading
- **Doppler spread (ν)**: Time-variation rate, causing temporal fading

The scattering function S(τ,ν) describes the power distribution:
```
S(τ,ν) = exp(-τ/τ_rms) × exp(-(ν/ν_rms)²)
```

### Key Parameters

| Parameter | Symbol | Typical Range | Description |
|-----------|--------|---------------|-------------|
| foF2 | foF2 | 3-15 MHz | F2 layer critical frequency |
| hmF2 | hmF2 | 200-400 km | F2 layer peak height |
| Delay spread | τ | 0.5-7 ms | RMS multipath delay |
| Doppler spread | ν | 0.1-10 Hz | Two-sided Doppler bandwidth |
| Path length | d | 100-10000 km | Great circle distance |

## Performance

### Targets
- **Throughput**: 2 Msps sustained real-time
- **Latency**: <50 ms input-to-output
- **GPU memory**: <2 GB for 1 MHz bandwidth

### Benchmarks (RTX 5090)
- Transfer function computation: ~0.5 ms for 4096 points
- Overlap-save block: ~0.2 ms per 4096 samples
- Scattering function: ~1 ms for 64x32 grid

*Note: GPU acceleration requires building the native CUDA module. CuPy provides automatic fallback but may need kernel recompilation for new GPU architectures.*

## References

1. **NTIA TR-88-240**: Vogler, L.E. and Hoffmeyer, J.A., "A full-wave calculation of ionospheric Doppler spread and its application to HF channel modeling," 1988.

2. **ITU-R F.1487**: "Testing of HF modems with bandwidths of up to about 12 kHz using ionospheric channel simulators," 2000.

3. **Watterson Model**: Watterson, C.C., Juroshek, J.R., and Bensema, W.D., "Experimental confirmation of an HF channel model," IEEE Trans. Comm. Tech., 1970.

4. **IRI-2020**: International Reference Ionosphere model, https://irimodel.org/

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_vogler.py -v

# Run with coverage
pytest tests/ --cov=hfpathsim
```

Current test status: **94 tests passing**

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
- [x] 94 unit tests

### Phase 4: Advanced Propagation (Planned)
- [ ] Full ray tracing engine
- [ ] Oblique incidence geometry
- [ ] Multi-hop path support
- [ ] Sporadic-E layer modeling
- [ ] Geomagnetic storm effects

### Phase 5: Integration (Planned)
- [ ] Real-time IQ output (sound card, network)
- [ ] GNU Radio source/sink blocks
- [ ] MATLAB/Simulink interface
- [ ] Docker containerization
- [ ] Cloud deployment option

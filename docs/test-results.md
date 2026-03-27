# Test Results

**Date:** 2026-03-26
**Commit:** bcd2a81
**Python:** 3.12.3
**Platform:** Linux 6.17.0-19-generic

## Summary

| Metric | Value |
|--------|-------|
| **Total Tests** | 438 |
| **Passed** | 438 |
| **Failed** | 0 |
| **Warnings** | 1 |
| **Duration** | 6.70s |

## Test Coverage by Module

| Test File | Tests | Description |
|-----------|-------|-------------|
| `test_channel_models.py` | 47 | Watterson TDL, noise, AGC, limiter, frequency offset, recording |
| `test_validation.py` | 44 | Reference datasets, statistics, fading analysis, channel validation |
| `test_engine.py` | 36 | Simulation engine, processing chain, configuration |
| `test_geomagnetic.py` | 34 | Solar indices, magnetic storms, polar blackout detection |
| `test_raytracing.py` | 33 | Ionospheric ray tracing, propagation modes, geometry |
| `test_gpu.py` | 31 | CUDA acceleration, memory management, kernel operations |
| `test_api.py` | 31 | REST API endpoints, WebSocket streaming, sessions |
| `test_itu_channels.py` | 30 | CCIR 520, ITU-R F.1289, ITU-R F.1487 channel models |
| `test_sporadic_e.py` | 24 | Sporadic-E layer modeling, occurrence estimation |
| `test_output.py` | 24 | File, network, audio, SDR output sinks |
| `test_profiling.py` | 22 | CPU timing, GPU profiling, benchmarks, reports |
| `test_vogler.py` | 21 | Vogler-Hoffmeyer IPM, reflection coefficients |
| `test_spectrum.py` | 21 | FFT analysis, spectrum computation, windowing |
| `test_integration.py` | 17 | GNU Radio, MATLAB interfaces, external tools |
| `test_input.py` | 13 | File, network, SDR input sources |
| `test_vogler_hoffmeyer_gpu.py` | 10 | GPU acceleration for Vogler-Hoffmeyer channel model |

## Test Categories

### Core Channel Models (77 tests)
- Watterson tapped delay line model
- Vogler-Hoffmeyer ionospheric propagation model
- ITU-R standardized channel presets (F.520, F.1289, F.1487)
- Fading generation (Rayleigh, Rician)
- Doppler spectrum shapes (Gaussian, flat, Jakes)

### Signal Processing (68 tests)
- Noise injection (AWGN, atmospheric, man-made, impulse)
- AGC with multiple modes (slow, medium, fast, manual)
- Signal limiting (hard, soft, cubic)
- Frequency offset and drift
- Spectrum analysis and FFT

### Ionospheric Modeling (91 tests)
- Ray tracing with spherical Earth geometry
- Propagation mode discovery (E, F, Es layers)
- Sporadic-E layer effects
- Geomagnetic indices (F10.7, Kp, Dst)
- Storm-time ionospheric depression

### Validation Module (44 tests)
- NTIA TR-90-255 reference datasets
- ITU-R F.1487 test parameters
- Watterson 1970 validation data
- Delay/Doppler spread computation
- Scattering function analysis
- Fading statistics (Rayleigh fit, LCR, AFD)

### Infrastructure (85 tests)
- REST API and WebSocket endpoints
- Input/output sources and sinks
- GPU acceleration and memory
- Profiling and benchmarking
- External integrations (GNU Radio, MATLAB)

### Performance (22 tests)
- CPU timing decorators and context managers
- GPU kernel profiling with CUDA events
- Throughput and latency benchmarks
- Memory usage tracking
- Report generation (JSON, HTML)

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_validation.py -v

# Run with coverage
python -m pytest tests/ --cov=hfpathsim --cov-report=html

# Run specific test class
python -m pytest tests/test_channel_models.py::TestWattersonChannel -v

# Run tests matching pattern
python -m pytest tests/ -k "rayleigh" -v
```

## Test Dependencies

```
pytest>=9.0.0
pytest-qt>=4.5.0
pytest-cov>=7.0.0
pytest-asyncio>=1.3.0
numpy>=1.24.0
scipy>=1.10.0
```

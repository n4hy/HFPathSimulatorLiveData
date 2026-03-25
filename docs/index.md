# HF Path Simulator Documentation

Welcome to the HF Path Simulator documentation. This software simulates high-frequency (HF) radio wave propagation through the ionosphere, enabling realistic testing of HF communication systems without requiring actual radio equipment.

## What is HF Path Simulator?

HF Path Simulator is a GPU-accelerated tool that models how radio signals in the 3-30 MHz range are affected when they travel through Earth's ionosphere. It accurately simulates:

- **Multipath propagation** - Signals arriving via multiple ionospheric reflections
- **Fading** - Signal strength variations caused by ionospheric dynamics
- **Doppler spread** - Frequency shifts from ionospheric motion
- **Delay spread** - Time dispersion of signal paths
- **Atmospheric noise** - Natural interference from lightning and other sources

## Who Should Use This?

- **Radio engineers** testing HF modem and codec designs
- **Amateur radio operators** understanding propagation conditions
- **Researchers** studying ionospheric effects on communications
- **Software developers** building HF communication applications
- **Students** learning about radio wave propagation

## Documentation Sections

### Getting Started

- **[Quick Start Guide](getting-started.md)** - Get up and running in 5 minutes
- **[User Guide](user-guide.md)** - Complete guide to all features

### Tutorials

Step-by-step guides for common tasks:

1. [Basic Channel Simulation](tutorials/01-basic-simulation.md) - Your first simulation
2. [GUI Walkthrough](tutorials/02-gui-walkthrough.md) - Using the graphical interface
3. [SDR Integration](tutorials/03-sdr-integration.md) - Connect real radio hardware
4. [GNU Radio Bridge](tutorials/04-gnuradio-bridge.md) - Integrate with GNU Radio
5. [Custom Channel Models](tutorials/05-custom-channels.md) - Create your own propagation models

### API Reference

- [REST API Reference](api/rest-api.md) - HTTP endpoints for remote control
- [Python API Reference](api/python-api.md) - Direct Python integration
- [WebSocket Streaming](api/websocket-api.md) - Real-time data streaming

### Deployment

- [AWS Deployment](deployment/aws.md) - Run on Amazon Web Services
- [GCP Deployment](deployment/gcp.md) - Run on Google Cloud Platform
- [Azure Deployment](deployment/azure.md) - Run on Microsoft Azure

## Key Features

| Feature | Description |
|---------|-------------|
| GPU Acceleration | Process millions of samples per second using CUDA |
| Multiple Channel Models | Watterson, Vogler, Vogler-Hoffmeyer models |
| ITU-R Standardized Models | CCIR 520, ITU-R F.520, F.1289, F.1487 presets |
| Real-time Processing | Sub-millisecond latency for live applications |
| SDR Support | Works with HackRF, USRP, LimeSDR, RTL-SDR |
| GNU Radio Integration | ZMQ bridge for seamless GNU Radio workflows |
| REST API | Remote control from any programming language |
| Web Dashboard | Browser-based monitoring and control |
| Performance Profiling | CPU timing, GPU profiling, memory tracking, benchmarks |

## System Requirements

**Minimum:**
- Python 3.10+
- 4 GB RAM
- Any modern CPU

**Recommended (for real-time processing):**
- NVIDIA GPU with CUDA support (GTX 1060 or better)
- 16 GB RAM
- SSD storage

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/n4hy/HFPathSimulatorLiveData/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/n4hy/HFPathSimulatorLiveData/discussions)

## License

HF Path Simulator is open source software released under the MIT License.

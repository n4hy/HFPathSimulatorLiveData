# Getting Started

This guide will have you running your first HF channel simulation in under 5 minutes.

## Installation

### Option 1: pip (Recommended)

```bash
pip install hfpathsim
```

### Option 2: From Source

```bash
git clone https://github.com/n4hy/HFPathSimulatorLiveData.git
cd HFPathSimulatorLiveData
pip install -e .
```

### Option 3: Docker

```bash
# With GPU support
docker run -it --gpus all hfpathsim:latest

# CPU only
docker run -it hfpathsim:cpu
```

## Verify Installation

Check that everything is working:

```bash
python -c "import hfpathsim; print(f'HF Path Simulator v{hfpathsim.__version__}')"
```

You should see the version number printed.

## Your First Simulation

### Using Python

Create a file called `my_first_simulation.py`:

```python
import numpy as np
from hfpathsim import SimulationEngine, EngineConfig, ChannelModel

# Create engine with Watterson channel model
config = EngineConfig(
    channel_model=ChannelModel.WATTERSON,
    sample_rate_hz=48000,  # 48 kHz sample rate
)
engine = SimulationEngine(config)

# Generate a test signal (1 kHz tone)
duration = 1.0  # 1 second
t = np.arange(0, duration, 1/48000)
input_signal = np.exp(1j * 2 * np.pi * 1000 * t).astype(np.complex64)

# Process through the channel
output_signal = engine.process(input_signal)

# Compare input and output
print(f"Input samples: {len(input_signal)}")
print(f"Output samples: {len(output_signal)}")
print(f"Input power: {np.mean(np.abs(input_signal)**2):.3f}")
print(f"Output power: {np.mean(np.abs(output_signal)**2):.3f}")
```

Run it:

```bash
python my_first_simulation.py
```

**What happened?** Your 1 kHz tone passed through a simulated HF ionospheric channel. The output signal now has realistic fading, multipath distortion, and Doppler effects.

### Using the GUI

Launch the graphical interface:

```bash
python -m hfpathsim
```

The GUI provides:
- Real-time spectrum display
- Channel parameter controls
- Input/output source configuration
- Visual monitoring of signal quality

### Using the REST API

Start the API server:

```bash
python -m hfpathsim.api
```

Then from another terminal or any HTTP client:

```bash
# Check server health
curl http://localhost:8000/api/v1/health

# Configure the channel
curl -X POST http://localhost:8000/api/v1/channel/watterson \
  -H "Content-Type: application/json" \
  -d '{"condition": "moderate"}'

# Get current state
curl http://localhost:8000/api/v1/channel/state
```

## Understanding the Output

When your signal passes through the HF channel simulator, it experiences:

1. **Multipath** - The signal arrives via multiple paths with different delays
2. **Fading** - Signal amplitude varies over time (Rayleigh/Rician fading)
3. **Doppler spread** - Small frequency shifts from ionospheric motion
4. **Delay spread** - Time smearing of the signal

These effects are what make real HF communication challenging, and why testing with realistic simulation is valuable.

## Choosing a Channel Model

HF Path Simulator includes three channel models:

| Model | Best For | Characteristics |
|-------|----------|-----------------|
| **Watterson** | General HF testing | Classic ITU-R F.520 model, three propagation conditions |
| **Vogler** | Research applications | Physics-based, uses real ionospheric parameters |
| **Vogler-Hoffmeyer** | Advanced simulation | Full model with sporadic-E, spread-F, magnetic storms |

### Quick Model Selection

```python
from hfpathsim import ChannelModel

# For standard compliance testing
config = EngineConfig(channel_model=ChannelModel.WATTERSON)

# For realistic propagation research
config = EngineConfig(channel_model=ChannelModel.VOGLER)

# For worst-case or edge-case testing
config = EngineConfig(channel_model=ChannelModel.VOGLER_HOFFMEYER)
```

## Next Steps

Now that you have the basics working:

1. **[User Guide](user-guide.md)** - Learn all the configuration options
2. **[Tutorial: Basic Simulation](tutorials/01-basic-simulation.md)** - Deeper dive into channel simulation
3. **[Tutorial: GUI Walkthrough](tutorials/02-gui-walkthrough.md)** - Master the graphical interface

## Troubleshooting

### "ModuleNotFoundError: No module named 'hfpathsim'"

The package isn't installed. Run:
```bash
pip install -e .
```

### "CUDA not available"

GPU acceleration requires an NVIDIA GPU with CUDA. The simulator will fall back to CPU automatically. To verify GPU status:

```python
from hfpathsim.gpu import is_gpu_available
print(f"GPU available: {is_gpu_available()}")
```

### GUI doesn't start

Install PyQt6:
```bash
pip install PyQt6 pyqtgraph
```

### Permission denied on Linux audio

Add your user to the audio group:
```bash
sudo usermod -a -G audio $USER
# Log out and back in
```

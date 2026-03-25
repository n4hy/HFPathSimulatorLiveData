# Tutorial 3: SDR Integration

**Time to complete:** 30 minutes
**Prerequisites:** SDR hardware (RTL-SDR, HackRF, LimeSDR, or USRP), drivers installed

In this tutorial, you'll learn how to:
- Connect SDR hardware to HF Path Simulator
- Receive live HF signals and process them through the channel simulator
- Transmit processed signals through an SDR
- Build a complete hardware-in-the-loop test system

---

## Overview

HF Path Simulator can work with Software Defined Radio (SDR) hardware to:

1. **Receive** real signals and add simulated channel effects
2. **Transmit** signals that have been processed through the simulator
3. **Loop back** for end-to-end system testing

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│   SDR RX    │────▶│  HF Path Sim    │────▶│   SDR TX    │
│  (HackRF)   │     │  (Channel)      │     │  (HackRF)   │
└─────────────┘     └─────────────────┘     └─────────────┘
```

---

## Step 1: Check SDR Support

First, verify your SDR is detected:

```python
from hfpathsim.input import list_sdr_devices

devices = list_sdr_devices()
for dev in devices:
    print(f"Found: {dev['driver']} - {dev['label']}")
```

Expected output:

```
Found: rtlsdr - Generic RTL2832U
Found: hackrf - HackRF One
Found: lime - LimeSDR-USB
```

If your SDR isn't listed:
1. Check USB connection
2. Verify drivers are installed
3. Check permissions (Linux: udev rules)

### Installing SDR Drivers

**RTL-SDR:**
```bash
# Ubuntu/Debian
sudo apt install rtl-sdr librtlsdr-dev

# macOS
brew install librtlsdr
```

**HackRF:**
```bash
# Ubuntu/Debian
sudo apt install hackrf libhackrf-dev

# macOS
brew install hackrf
```

**LimeSDR:**
```bash
# Ubuntu/Debian
sudo apt install limesuite
```

**USRP:**
```bash
# Ubuntu/Debian
sudo apt install uhd-host libuhd-dev
```

---

## Step 2: Basic SDR Receive

Let's receive signals from an SDR and process them through the channel simulator:

```python
import numpy as np
from hfpathsim import SimulationEngine, EngineConfig, ChannelModel
from hfpathsim.input import SDRSource
from hfpathsim.output import FileOutputSink

# Configure the channel
config = EngineConfig(
    channel_model=ChannelModel.WATTERSON,
    sample_rate_hz=2_400_000,  # 2.4 MHz - must match SDR
)
engine = SimulationEngine(config)
engine.configure_watterson(condition="moderate")

# Configure SDR input
sdr = SDRSource(
    driver="rtlsdr",           # or "hackrf", "lime", "uhd"
    center_freq_hz=7_200_000,  # 7.2 MHz (40m amateur band)
    sample_rate_hz=2_400_000,
    gain_db=30,
)

# Configure file output
output = FileOutputSink(
    "received_with_channel.wav",
    format="wav",
    sample_rate_hz=2_400_000,
)

# Process for 10 seconds
print("Receiving and processing...")
duration_samples = 10 * 2_400_000  # 10 seconds
samples_processed = 0

try:
    sdr.start()
    output.start()

    while samples_processed < duration_samples:
        # Read from SDR
        samples = sdr.read(65536)
        if samples is None:
            continue

        # Process through channel
        processed = engine.process(samples)

        # Write to file
        output.write(processed)
        samples_processed += len(samples)

        # Progress
        progress = samples_processed / duration_samples * 100
        print(f"\rProgress: {progress:.1f}%", end="", flush=True)

finally:
    sdr.stop()
    output.stop()

print(f"\nDone! Saved to received_with_channel.wav")
```

---

## Step 3: Real-Time SDR Processing with GUI

Use the GUI for real-time visualization:

```bash
python -m hfpathsim
```

1. Go to **Input** tab
2. Select **SDR** as source type
3. Click **Scan Devices** to find your SDR
4. Configure:
   - **Center Frequency**: Your target frequency (e.g., 7200000 Hz)
   - **Sample Rate**: Match your SDR capability
   - **Gain**: Adjust for best signal level
5. Click **Start**

You'll see live signals from your SDR processed through the channel simulator.

---

## Step 4: SDR Transmit

If you have a transmit-capable SDR (HackRF, LimeSDR, USRP), you can transmit processed signals:

```python
import numpy as np
from hfpathsim import SimulationEngine, EngineConfig, ChannelModel
from hfpathsim.input import FileSource
from hfpathsim.output import SDROutputSink

# Load a test signal
source = FileSource("test_signal.wav")

# Configure channel
config = EngineConfig(
    channel_model=ChannelModel.WATTERSON,
    sample_rate_hz=source.sample_rate_hz,
)
engine = SimulationEngine(config)
engine.configure_watterson(condition="disturbed")

# Configure SDR output
sdr_tx = SDROutputSink(
    driver="hackrf",
    center_freq_hz=7_200_000,
    sample_rate_hz=source.sample_rate_hz,
    tx_gain_db=0,  # Start low for safety!
)

print("Transmitting...")
try:
    source.start()
    sdr_tx.start()

    while True:
        samples = source.read(65536)
        if samples is None:
            break

        # Process through channel
        processed = engine.process(samples)

        # Transmit
        sdr_tx.write(processed)

finally:
    source.stop()
    sdr_tx.stop()

print("Transmission complete")
```

**WARNING**: Only transmit on frequencies you're licensed to use! Verify local regulations before transmitting.

---

## Step 5: Loopback Testing

Test your entire radio system with simulated propagation:

```
┌─────────┐     ┌─────────┐     ┌─────────────┐     ┌─────────┐     ┌─────────┐
│  Your   │────▶│ SDR TX  │────▶│  HF Path    │────▶│ SDR RX  │────▶│  Your   │
│ Modem   │     │(HackRF) │     │  Simulator  │     │(RTL-SDR)│     │ Modem   │
└─────────┘     └─────────┘     └─────────────┘     └─────────┘     └─────────┘
```

This setup requires two SDRs (one TX, one RX) or a full-duplex SDR.

```python
import threading
import queue
from hfpathsim import SimulationEngine, EngineConfig, ChannelModel
from hfpathsim.input import SDRSource
from hfpathsim.output import SDROutputSink

# Shared queue between TX and RX
sample_queue = queue.Queue(maxsize=100)

# Channel simulator
config = EngineConfig(
    channel_model=ChannelModel.WATTERSON,
    sample_rate_hz=1_000_000,
)
engine = SimulationEngine(config)
engine.configure_watterson(condition="moderate")

# SDR configurations
sdr_rx = SDRSource(
    driver="hackrf",  # Receiving SDR
    center_freq_hz=7_200_000,
    sample_rate_hz=1_000_000,
)

sdr_tx = SDROutputSink(
    driver="lime",  # Transmitting SDR (different device)
    center_freq_hz=7_200_000,
    sample_rate_hz=1_000_000,
    tx_gain_db=0,
)

def receive_thread():
    """Receive samples and queue them."""
    while running:
        samples = sdr_rx.read(65536)
        if samples is not None:
            try:
                sample_queue.put(samples, timeout=0.1)
            except queue.Full:
                pass  # Drop samples if queue is full

def process_and_transmit_thread():
    """Process queued samples and transmit."""
    while running:
        try:
            samples = sample_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # Apply channel effects
        processed = engine.process(samples)

        # Transmit
        sdr_tx.write(processed)

# Start threads
running = True
rx_thread = threading.Thread(target=receive_thread)
tx_thread = threading.Thread(target=process_and_transmit_thread)

try:
    sdr_rx.start()
    sdr_tx.start()
    rx_thread.start()
    tx_thread.start()

    print("Loopback running. Press Ctrl+C to stop.")
    while True:
        import time
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopping...")
    running = False
    rx_thread.join()
    tx_thread.join()
    sdr_rx.stop()
    sdr_tx.stop()
```

---

## Step 6: Hardware Configuration Tips

### RTL-SDR Settings

```python
sdr = SDRSource(
    driver="rtlsdr",
    center_freq_hz=7_200_000,
    sample_rate_hz=2_400_000,  # Max stable rate
    gain_db=40,                # 0-50 dB typical
    freq_correction_ppm=0,     # Calibrate your specific dongle
)
```

### HackRF Settings

```python
# Receive
sdr_rx = SDRSource(
    driver="hackrf",
    center_freq_hz=7_200_000,
    sample_rate_hz=8_000_000,  # Up to 20 MHz
    gain_db=30,                # LNA + VGA
    bandwidth_hz=4_000_000,    # Filter bandwidth
)

# Transmit
sdr_tx = SDROutputSink(
    driver="hackrf",
    center_freq_hz=7_200_000,
    sample_rate_hz=8_000_000,
    tx_gain_db=0,              # 0-47 dB (be careful!)
)
```

### LimeSDR Settings

```python
sdr = SDRSource(
    driver="lime",
    center_freq_hz=7_200_000,
    sample_rate_hz=10_000_000,
    gain_db=40,
    antenna="LNAW",            # LNAW, LNAH, LNAL
    channel=0,                 # 0 or 1 for dual-channel
)
```

### USRP Settings

```python
sdr = SDRSource(
    driver="uhd",
    center_freq_hz=7_200_000,
    sample_rate_hz=10_000_000,
    gain_db=30,
    antenna="RX2",             # TX/RX or RX2
    device_args="type=b200",   # Device-specific args
)
```

---

## Step 7: Optimizing Performance

### Reduce Latency

For real-time applications:

```python
config = EngineConfig(
    sample_rate_hz=2_400_000,
    block_size=1024,           # Small blocks = low latency
    use_gpu=True,              # GPU is faster
)
```

### Handle Sample Rate Mismatch

If your SDR rate doesn't match your target:

```python
from scipy import signal

# Resample from 2.4 MHz to 48 kHz
input_rate = 2_400_000
output_rate = 48_000
ratio = output_rate / input_rate

resampled = signal.resample_poly(
    samples,
    up=output_rate,
    down=input_rate
)
```

### Buffer Management

Prevent dropouts with proper buffering:

```python
# Larger read buffer
samples = sdr.read(
    num_samples=131072,        # Larger buffer
    timeout_ms=1000,           # Wait up to 1 second
)
```

---

## Troubleshooting

### "Device not found"

1. Check USB connection
2. Verify with native tools:
   ```bash
   rtl_test        # RTL-SDR
   hackrf_info     # HackRF
   LimeUtil --find # LimeSDR
   uhd_find_devices # USRP
   ```
3. Check permissions (Linux):
   ```bash
   sudo usermod -a -G plugdev $USER
   # Log out and back in
   ```

### "Sample rate not supported"

Each SDR has specific supported rates:

| SDR | Typical Rates |
|-----|---------------|
| RTL-SDR | 225-300 kHz, 900-3200 kHz |
| HackRF | 2-20 MHz |
| LimeSDR | 100 kHz - 61.44 MHz |
| USRP B200 | 200 kHz - 56 MHz |

### "Samples being dropped"

1. Reduce sample rate
2. Increase block size
3. Close other applications
4. Use USB 3.0 port (if supported)

### "High CPU usage"

1. Enable GPU acceleration
2. Increase block size
3. Reduce FFT size in GUI

---

## What's Next?

Now that you can work with SDR hardware:

- **[Tutorial 4: GNU Radio Bridge](04-gnuradio-bridge.md)** - Integrate with GNU Radio
- **[Tutorial 5: Custom Channels](05-custom-channels.md)** - Create your own channel models
- **[API Reference](../api/python-api.md)** - Full API documentation

# Tutorial 4: GNU Radio Bridge

**Time to complete:** 25 minutes
**Prerequisites:** GNU Radio installed (3.8+), HF Path Simulator installed

In this tutorial, you'll learn how to:
- Connect HF Path Simulator to GNU Radio via ZMQ
- Create GNU Radio flowgraphs that use the channel simulator
- Build end-to-end HF modem test systems
- Generate GNU Radio Python snippets automatically

---

## Overview

HF Path Simulator connects to GNU Radio using ZeroMQ (ZMQ) sockets:

```
┌─────────────────┐      ZMQ       ┌─────────────────┐
│   GNU Radio     │◀──────────────▶│  HF Path Sim    │
│   Flowgraph     │   (samples)    │  (channel)      │
└─────────────────┘                └─────────────────┘
```

- GNU Radio sends samples to HF Path Simulator
- HF Path Simulator processes through channel model
- Processed samples return to GNU Radio

---

## Step 1: Start the ZMQ Bridge

Start HF Path Simulator in ZMQ bridge mode:

```bash
python -m hfpathsim.bridge --input-port 5555 --output-port 5556
```

Or from Python:

```python
from hfpathsim.integration import ZMQBridge
from hfpathsim import EngineConfig, ChannelModel

# Configure channel
config = EngineConfig(
    channel_model=ChannelModel.WATTERSON,
    sample_rate_hz=48000,
)

# Start bridge
bridge = ZMQBridge(
    config=config,
    input_address="tcp://*:5555",   # Receive from GNU Radio
    output_address="tcp://*:5556",  # Send back to GNU Radio
)
bridge.configure_watterson(condition="moderate")
bridge.start()

print("ZMQ Bridge running. Press Ctrl+C to stop.")
try:
    while True:
        import time
        time.sleep(1)
except KeyboardInterrupt:
    bridge.stop()
```

---

## Step 2: Create a GNU Radio Flowgraph

Open GNU Radio Companion (GRC) and create this flowgraph:

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│ Signal      │────▶│ ZMQ PUSH     │────▶│ ZMQ PULL     │────▶│ Scope/      │
│ Source      │     │ Sink         │     │ Source       │     │ Waterfall   │
└─────────────┘     └──────────────┘     └──────────────┘     └─────────────┘
                           │                    │
                           │   (via HF Path     │
                           │    Simulator)      │
                           └────────────────────┘
```

### Blocks to Use

1. **Signal Source** (or your modulator)
   - Type: Complex
   - Sample Rate: 48000
   - Waveform: Cosine
   - Frequency: 1000

2. **ZMQ PUSH Sink**
   - Address: `tcp://127.0.0.1:5555`
   - Type: Complex Float (complex64)

3. **ZMQ PULL Source**
   - Address: `tcp://127.0.0.1:5556`
   - Type: Complex Float (complex64)

4. **QT GUI Sink** or **Waterfall Sink**
   - FFT Size: 1024
   - Sample Rate: 48000

---

## Step 3: Using Auto-Generated GNU Radio Code

HF Path Simulator can generate GNU Radio Python code for you:

```python
from hfpathsim.integration import generate_gnuradio_snippet

# Generate the code
code = generate_gnuradio_snippet(
    input_port=5555,
    output_port=5556,
    sample_rate=48000,
)

print(code)
```

Output:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# HF Path Simulator GNU Radio Bridge
# Auto-generated code

from gnuradio import gr
from gnuradio import zeromq

class hf_channel_block(gr.hier_block2):
    """Hierarchical block connecting to HF Path Simulator."""

    def __init__(self, input_address="tcp://127.0.0.1:5555",
                 output_address="tcp://127.0.0.1:5556"):
        gr.hier_block2.__init__(
            self, "HF Channel",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signature(1, 1, gr.sizeof_gr_complex)
        )

        # Send samples to HF Path Simulator
        self.zmq_push = zeromq.push_sink(
            gr.sizeof_gr_complex,
            1,
            input_address,
            100,  # timeout
            True, # pass_tags
            -1    # hwm
        )

        # Receive processed samples
        self.zmq_pull = zeromq.pull_source(
            gr.sizeof_gr_complex,
            1,
            output_address,
            100,  # timeout
            True, # pass_tags
            -1    # hwm
        )

        # Connect
        self.connect(self, self.zmq_push)
        self.connect(self.zmq_pull, self)
```

---

## Step 4: Complete Modem Test Example

Here's a complete example testing an FSK modem through the HF channel:

### Part A: HF Path Simulator Setup

```python
# hf_channel_server.py
from hfpathsim.integration import ZMQBridge
from hfpathsim import EngineConfig, ChannelModel

config = EngineConfig(
    channel_model=ChannelModel.WATTERSON,
    sample_rate_hz=48000,
)

bridge = ZMQBridge(
    config=config,
    input_address="tcp://*:5555",
    output_address="tcp://*:5556",
)

# Configure for moderate HF conditions
bridge.configure_watterson(condition="moderate")

# Add some noise
bridge.configure_noise(snr_db=20.0)

print("HF Channel Server running...")
bridge.start()

try:
    while True:
        import time
        time.sleep(1)
        stats = bridge.get_stats()
        print(f"\rProcessed: {stats['samples_processed']:,} samples", end="")
except KeyboardInterrupt:
    bridge.stop()
```

### Part B: GNU Radio Flowgraph (Python)

```python
#!/usr/bin/env python3
# fsk_modem_test.py - Test FSK modem through HF channel

from gnuradio import gr, blocks, digital, zeromq, analog
from gnuradio import qtgui
import sys
import signal

class fsk_hf_test(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self, "FSK HF Test")

        # Parameters
        self.samp_rate = 48000
        self.baud_rate = 300
        self.freq_deviation = 200

        # === TRANSMITTER ===

        # Random data source
        self.random_source = blocks.vector_source_b(
            list(range(256)) * 100,  # Repeating pattern
            True,  # Repeat
            1
        )

        # Pack bits
        self.pack = blocks.pack_k_bits_bb(8)

        # FSK modulator
        self.fsk_mod = digital.gfsk_mod(
            samples_per_symbol=int(self.samp_rate/self.baud_rate),
            sensitivity=2*3.14159*self.freq_deviation/self.samp_rate,
        )

        # === HF CHANNEL (via ZMQ to HF Path Simulator) ===

        self.zmq_push = zeromq.push_sink(
            gr.sizeof_gr_complex,
            1,
            "tcp://127.0.0.1:5555",
            100, True, -1
        )

        self.zmq_pull = zeromq.pull_source(
            gr.sizeof_gr_complex,
            1,
            "tcp://127.0.0.1:5556",
            100, True, -1
        )

        # === RECEIVER ===

        # FSK demodulator
        self.fsk_demod = digital.gfsk_demod(
            samples_per_symbol=int(self.samp_rate/self.baud_rate),
        )

        # Unpack bits
        self.unpack = blocks.unpack_k_bits_bb(8)

        # BER measurement
        self.ber = digital.mpsk_snr_est_cc(
            digital.SNR_EST_SIMPLE, 10000, 0.001
        )

        # === VISUALIZATION ===

        self.qtgui_sink = qtgui.sink_c(
            1024, 5, 0, self.samp_rate, "HF Channel Output", True, True, True, True, True
        )

        # === CONNECTIONS ===

        # TX path
        self.connect(self.random_source, self.pack, self.fsk_mod)
        self.connect(self.fsk_mod, self.zmq_push)

        # RX path (from HF channel)
        self.connect(self.zmq_pull, self.ber)
        self.connect(self.ber, self.fsk_demod)
        self.connect(self.zmq_pull, self.qtgui_sink)

def main():
    tb = fsk_hf_test()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.wait()

if __name__ == '__main__':
    main()
```

---

## Step 5: Dynamic Channel Control

Control the channel parameters while the flowgraph is running:

```python
# In a separate terminal/script
import zmq

# Connect to HF Path Simulator control port
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:5557")  # Control port

# Change channel condition
socket.send_json({
    "command": "configure_watterson",
    "condition": "disturbed"
})
response = socket.recv_json()
print(f"Channel updated: {response}")

# Add more noise
socket.send_json({
    "command": "configure_noise",
    "snr_db": 10.0
})
response = socket.recv_json()
print(f"Noise updated: {response}")
```

Or use the REST API:

```bash
# Change channel to disturbed
curl -X POST http://localhost:8000/api/v1/channel/watterson \
  -H "Content-Type: application/json" \
  -d '{"condition": "disturbed"}'

# Reduce SNR to 10 dB
curl -X POST http://localhost:8000/api/v1/channel/noise \
  -H "Content-Type: application/json" \
  -d '{"snr_db": 10.0}'
```

---

## Step 6: Using OOT Blocks

For easier integration, install the HF Path Simulator GNU Radio OOT module:

```bash
cd gr-hfpathsim
mkdir build && cd build
cmake ..
make
sudo make install
sudo ldconfig
```

Then in GNU Radio Companion, you'll have a new "HF Path Simulator" category with:

- **HF Channel Block** - All-in-one channel simulator
- **ZMQ Bridge Source** - Receive from simulator
- **ZMQ Bridge Sink** - Send to simulator

---

## Step 7: Performance Tips

### Reduce Latency

For real-time applications:

```python
bridge = ZMQBridge(
    config=config,
    input_address="tcp://*:5555",
    output_address="tcp://*:5556",
    block_size=512,      # Smaller blocks
    high_water_mark=10,  # Limit buffering
)
```

### Handle Sample Rate Mismatch

If GNU Radio uses a different rate:

```python
# In GNU Radio, resample before sending
from gnuradio import filter as grfilter

self.resampler = grfilter.rational_resampler_ccc(
    interpolation=48000,
    decimation=your_sample_rate,
)
```

### Monitor Performance

```python
# Get bridge statistics
stats = bridge.get_stats()
print(f"Latency: {stats['latency_ms']:.1f} ms")
print(f"Throughput: {stats['samples_per_sec']/1000:.0f} kS/s")
print(f"Dropped: {stats['dropped_samples']}")
```

---

## Troubleshooting

### "Connection refused"

1. Verify HF Path Simulator bridge is running
2. Check port numbers match
3. Check firewall settings

### "Samples not flowing"

1. Verify ZMQ socket types match (PUSH→PULL)
2. Check data types match (complex64)
3. Verify sample rates match

### "High latency"

1. Reduce block size
2. Reduce ZMQ high water mark
3. Enable GPU acceleration
4. Use TCP instead of IPC for remote connections

### "Audio underruns"

1. Increase buffer sizes
2. Reduce sample rate
3. Simplify flowgraph

---

## What's Next?

Now that you can integrate with GNU Radio:

- **[Tutorial 5: Custom Channels](05-custom-channels.md)** - Create your own channel models
- **[Python API Reference](../api/python-api.md)** - Full API documentation
- **[WebSocket API](../api/websocket-api.md)** - Real-time streaming

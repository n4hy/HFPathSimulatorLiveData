# WebSocket API Reference

HF Path Simulator provides WebSocket endpoints for real-time streaming and monitoring.

## Connection

WebSocket endpoints are available at:

```
ws://localhost:8000/api/v1/stream/{endpoint}
```

## Endpoints

### Input Stream

**URL:** `ws://localhost:8000/api/v1/stream/input`

Stream samples to the simulator and receive processed output in real-time.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| session_id | string | null | Optional session ID |

**Client → Server Messages:**

#### Process Samples

```json
{
  "type": "samples",
  "samples_base64": "AAAA...",
  "format": "complex64"
}
```

| Field | Type | Description |
|-------|------|-------------|
| type | string | Must be "samples" |
| samples_base64 | string | Base64-encoded sample data |
| format | string | "complex64" or "complex128" |

#### Ping

```json
{
  "type": "ping"
}
```

**Server → Client Messages:**

#### Processed Samples

```json
{
  "type": "processed",
  "samples_base64": "AAAA...",
  "count": 4096,
  "timestamp": 1705312800.123
}
```

#### Pong

```json
{
  "type": "pong",
  "timestamp": 1705312800.123
}
```

#### Error

```json
{
  "type": "error",
  "error": "Error message",
  "timestamp": 1705312800.123
}
```

**Example (Python):**

```python
import asyncio
import websockets
import numpy as np
import base64
import json

async def stream_samples():
    uri = "ws://localhost:8000/api/v1/stream/input"

    async with websockets.connect(uri) as ws:
        # Generate samples
        samples = np.exp(1j * np.linspace(0, 100, 4096)).astype(np.complex64)
        samples_b64 = base64.b64encode(samples.tobytes()).decode()

        # Send samples
        await ws.send(json.dumps({
            "type": "samples",
            "samples_base64": samples_b64,
            "format": "complex64"
        }))

        # Receive processed
        response = json.loads(await ws.recv())
        print(f"Received {response['count']} processed samples")

        # Decode output
        output_bytes = base64.b64decode(response["samples_base64"])
        output = np.frombuffer(output_bytes, dtype=np.complex64)

asyncio.run(stream_samples())
```

**Example (JavaScript):**

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/stream/input');

ws.onopen = () => {
    // Send samples (you'd need to properly encode complex samples)
    ws.send(JSON.stringify({
        type: 'ping'
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data.type);

    if (data.type === 'processed') {
        console.log('Processed samples:', data.count);
    }
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};
```

---

### Output Stream

**URL:** `ws://localhost:8000/api/v1/stream/output`

Receive processed samples without sending input (for monitoring).

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| session_id | string | null | Optional session ID |

**Server → Client Messages:**

```json
{
  "type": "samples",
  "samples_base64": "AAAA...",
  "count": 4096,
  "timestamp": 1705312800.123
}
```

---

### State Stream

**URL:** `ws://localhost:8000/api/v1/stream/state`

Receive periodic state updates.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| session_id | string | null | Optional session ID |
| interval_ms | integer | 100 | Update interval (10-1000 ms) |

**Server → Client Messages:**

```json
{
  "type": "state",
  "timestamp": 1705312800.123,
  "running": true,
  "total_samples_processed": 1234567,
  "blocks_processed": 302,
  "current_sample_rate": 48000,
  "agc_gain_db": 12.5,
  "limiter_reduction_db": 0.0,
  "current_freq_offset_hz": 0.0
}
```

**Example:**

```python
import asyncio
import websockets
import json

async def monitor_state():
    uri = "ws://localhost:8000/api/v1/stream/state?interval_ms=500"

    async with websockets.connect(uri) as ws:
        while True:
            response = json.loads(await ws.recv())
            print(f"Samples: {response['total_samples_processed']:,}")
            print(f"AGC Gain: {response['agc_gain_db']:.1f} dB")
            print("---")

asyncio.run(monitor_state())
```

---

### Spectrum Stream

**URL:** `ws://localhost:8000/api/v1/stream/spectrum`

Receive real-time spectrum data (FFT).

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| session_id | string | null | Optional session ID |
| interval_ms | integer | 100 | Update interval (10-1000 ms) |

**Server → Client Messages:**

#### Spectrum Data

```json
{
  "type": "spectrum",
  "timestamp": 1705312800.123,
  "spectrum_db": [-60.0, -58.5, ...],
  "freq_axis_hz": [-24000.0, -23976.5, ...],
  "fft_size": 1024
}
```

#### Keepalive

```json
{
  "type": "keepalive",
  "timestamp": 1705312800.123
}
```

**Example (Web Dashboard):**

```html
<!DOCTYPE html>
<html>
<head>
    <title>HF Spectrum Monitor</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="spectrum" style="width:100%; height:400px;"></div>

    <script>
        // Initialize plot
        const trace = {
            x: [],
            y: [],
            type: 'scatter',
            mode: 'lines'
        };

        Plotly.newPlot('spectrum', [trace], {
            title: 'HF Path Simulator Spectrum',
            xaxis: { title: 'Frequency (Hz)' },
            yaxis: { title: 'Power (dB)', range: [-80, 0] }
        });

        // Connect to WebSocket
        const ws = new WebSocket('ws://localhost:8000/api/v1/stream/spectrum?interval_ms=100');

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'spectrum') {
                Plotly.update('spectrum', {
                    x: [data.freq_axis_hz],
                    y: [data.spectrum_db]
                });
            }
        };
    </script>
</body>
</html>
```

---

## Connection Management

### Heartbeat

Use ping/pong messages to keep connections alive:

```python
async def maintain_connection(ws):
    while True:
        await ws.send(json.dumps({"type": "ping"}))
        response = json.loads(await ws.recv())
        assert response["type"] == "pong"
        await asyncio.sleep(30)
```

### Reconnection

Handle disconnections gracefully:

```python
async def resilient_stream():
    while True:
        try:
            async with websockets.connect(uri) as ws:
                await process_stream(ws)
        except websockets.exceptions.ConnectionClosed:
            print("Connection lost, reconnecting...")
            await asyncio.sleep(1)
```

---

## Error Handling

WebSocket errors are sent as messages:

```json
{
  "type": "error",
  "error": "Invalid sample format: must be complex64 or complex128",
  "timestamp": 1705312800.123
}
```

Connection errors result in WebSocket close with reason:

| Code | Reason |
|------|--------|
| 1000 | Normal close |
| 1008 | Invalid session ID |
| 1011 | Server error |

---

## Performance Tips

### Batch Samples

Send larger batches for better efficiency:

```python
# Good: larger batches
samples = np.random.randn(16384).astype(np.complex64)

# Less efficient: small batches
samples = np.random.randn(256).astype(np.complex64)
```

### Adjust Update Intervals

For state/spectrum streams, use appropriate intervals:

```
# Low-latency monitoring
interval_ms=50

# Normal dashboard
interval_ms=100

# Low-bandwidth monitoring
interval_ms=500
```

### Binary Protocol (Advanced)

For maximum performance, use binary WebSocket frames directly:

```python
import struct

async def send_binary_samples(ws, samples):
    # Header: sample count (4 bytes)
    header = struct.pack('<I', len(samples))

    # Send as binary
    await ws.send(header + samples.tobytes())
```

---

## Session Management

### Creating a Session

First create a session via REST API:

```bash
curl -X POST http://localhost:8000/api/v1/processing/sessions \
  -H "Content-Type: application/json" \
  -d '{"channel_model": "watterson"}'
```

Response:

```json
{
  "session_id": "abc123"
}
```

### Using Session with WebSocket

```python
session_id = "abc123"
uri = f"ws://localhost:8000/api/v1/stream/input?session_id={session_id}"

async with websockets.connect(uri) as ws:
    # All processing uses this session's channel configuration
    ...
```

### Session Isolation

Each session has independent:
- Channel model configuration
- Noise settings
- Impairment settings
- State (sample counts, etc.)

---

## Example: Full Streaming Client

```python
#!/usr/bin/env python3
"""Complete WebSocket streaming example."""

import asyncio
import websockets
import numpy as np
import base64
import json
from dataclasses import dataclass

@dataclass
class StreamStats:
    samples_sent: int = 0
    samples_received: int = 0
    latency_ms: float = 0.0

async def stream_processor(uri: str, duration_s: float = 10.0):
    """Stream samples through HF Path Simulator."""

    stats = StreamStats()
    sample_rate = 48000
    block_size = 4096

    async with websockets.connect(uri) as ws:
        print(f"Connected to {uri}")

        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < duration_s:
            # Generate samples (sine wave for testing)
            t = np.arange(block_size) / sample_rate + stats.samples_sent / sample_rate
            samples = np.exp(1j * 2 * np.pi * 1000 * t).astype(np.complex64)

            # Send
            send_time = asyncio.get_event_loop().time()
            await ws.send(json.dumps({
                "type": "samples",
                "samples_base64": base64.b64encode(samples.tobytes()).decode(),
                "format": "complex64"
            }))
            stats.samples_sent += len(samples)

            # Receive
            response = json.loads(await ws.recv())

            if response["type"] == "processed":
                recv_time = asyncio.get_event_loop().time()
                stats.latency_ms = (recv_time - send_time) * 1000
                stats.samples_received += response["count"]

                # Decode output (optional)
                output = np.frombuffer(
                    base64.b64decode(response["samples_base64"]),
                    dtype=np.complex64
                )

            elif response["type"] == "error":
                print(f"Error: {response['error']}")
                break

            # Progress
            elapsed = asyncio.get_event_loop().time() - start_time
            print(f"\rProcessed: {stats.samples_sent:,} samples | "
                  f"Latency: {stats.latency_ms:.1f} ms | "
                  f"Time: {elapsed:.1f}s", end="")

        print(f"\n\nFinal stats:")
        print(f"  Samples sent: {stats.samples_sent:,}")
        print(f"  Samples received: {stats.samples_received:,}")
        print(f"  Average latency: {stats.latency_ms:.1f} ms")

if __name__ == "__main__":
    asyncio.run(stream_processor(
        "ws://localhost:8000/api/v1/stream/input",
        duration_s=10.0
    ))
```

---

## Comparison: REST vs WebSocket

| Feature | REST API | WebSocket |
|---------|----------|-----------|
| Latency | Higher (HTTP overhead) | Lower (persistent connection) |
| Throughput | Lower | Higher |
| Complexity | Simpler | More complex |
| Best for | Configuration, occasional processing | Continuous streaming |

**Recommendation:**
- Use **REST API** for configuration and one-off processing
- Use **WebSocket** for real-time streaming and monitoring

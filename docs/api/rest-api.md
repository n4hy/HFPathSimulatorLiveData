# REST API Reference

HF Path Simulator provides a comprehensive REST API for remote control and automation.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API does not require authentication. For production deployments, use a reverse proxy with authentication.

## Response Format

All responses are JSON with this structure:

```json
{
  "field1": "value1",
  "field2": "value2"
}
```

Errors return:

```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Health & Status

### GET /health

Check API server health.

**Response:**

```json
{
  "status": "ok",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "gpu_available": true
}
```

**Example:**

```bash
curl http://localhost:8000/api/v1/health
```

---

### GET /gpu

Get GPU information.

**Response:**

```json
{
  "available": true,
  "name": "NVIDIA GeForce RTX 3080",
  "compute_capability": "8.6",
  "total_memory_gb": 10.0,
  "free_memory_gb": 8.5,
  "cuda_version": "12.0"
}
```

**Example:**

```bash
curl http://localhost:8000/api/v1/gpu
```

---

## Channel Configuration

### GET /channel/state

Get current channel state.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| session_id | string | Optional session ID |

**Response:**

```json
{
  "model": "watterson",
  "running": true,
  "total_samples_processed": 1234567,
  "blocks_processed": 302,
  "current_sample_rate": 48000,
  "agc_gain_db": 12.5,
  "limiter_reduction_db": 0.0,
  "current_freq_offset_hz": 0.0,
  "watterson": {
    "condition": "moderate",
    "num_paths": 2,
    "doppler_spread_hz": 0.5,
    "delay_spread_ms": 1.0
  }
}
```

**Example:**

```bash
curl http://localhost:8000/api/v1/channel/state
```

---

### POST /channel/watterson

Configure Watterson channel model.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| condition | string | No | "good", "moderate", or "disturbed" |
| num_paths | integer | No | Number of paths (1-4) |
| doppler_spread_hz | number | No | Doppler spread (0.1-10 Hz) |
| delay_spread_ms | number | No | Delay spread (0.1-5 ms) |

**Example Request:**

```json
{
  "condition": "disturbed",
  "num_paths": 2
}
```

**Response:**

```json
{
  "model": "watterson",
  "watterson": {
    "condition": "disturbed",
    "num_paths": 2,
    "doppler_spread_hz": 1.0,
    "delay_spread_ms": 2.0
  }
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/channel/watterson \
  -H "Content-Type: application/json" \
  -d '{"condition": "disturbed"}'
```

---

### POST /channel/vogler

Configure Vogler channel model.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| foF2 | number | No | F2 critical frequency (1-20 MHz) |
| hmF2 | number | No | F2 layer height (150-500 km) |
| foE | number | No | E critical frequency (0.5-5 MHz) |
| hmE | number | No | E layer height (80-150 km) |
| doppler_spread_hz | number | No | Doppler spread (Hz) |
| delay_spread_ms | number | No | Delay spread (ms) |

**Example Request:**

```json
{
  "foF2": 8.0,
  "hmF2": 320.0,
  "doppler_spread_hz": 1.5
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/channel/vogler \
  -H "Content-Type: application/json" \
  -d '{"foF2": 8.0, "hmF2": 320.0}'
```

---

### POST /channel/vh

Configure Vogler-Hoffmeyer channel model.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| condition | string | No | "good", "moderate", or "disturbed" |
| sporadic_e_enabled | boolean | No | Enable sporadic E effects |
| spread_f_enabled | boolean | No | Enable spread-F effects |
| magnetic_storm | boolean | No | Simulate magnetic storm |

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/channel/vh \
  -H "Content-Type: application/json" \
  -d '{"condition": "moderate", "sporadic_e_enabled": true}'
```

---

### POST /channel/noise

Configure noise.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| snr_db | number | Yes | Signal-to-noise ratio (dB) |
| enable_atmospheric | boolean | No | Enable atmospheric noise |
| enable_galactic | boolean | No | Enable galactic noise |
| enable_man_made | boolean | No | Enable man-made noise |

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/channel/noise \
  -H "Content-Type: application/json" \
  -d '{"snr_db": 15.0, "enable_atmospheric": true}'
```

---

### POST /channel/noise/disable

Disable noise.

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/channel/noise/disable
```

---

### POST /channel/impairments/agc

Configure AGC.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| enabled | boolean | Yes | Enable/disable AGC |
| target_level_db | number | No | Target output level (-60 to 0 dB) |
| attack_time_ms | number | No | Attack time (1-1000 ms) |
| release_time_ms | number | No | Release time (10-10000 ms) |
| max_gain_db | number | No | Maximum gain (0-80 dB) |

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/channel/impairments/agc \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "target_level_db": -20.0}'
```

---

### POST /channel/impairments/limiter

Configure limiter.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| enabled | boolean | Yes | Enable/disable limiter |
| threshold_db | number | No | Threshold (-30 to 0 dB) |

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/channel/impairments/limiter \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "threshold_db": -3.0}'
```

---

### POST /channel/impairments/offset

Configure frequency offset.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| enabled | boolean | Yes | Enable/disable offset |
| offset_hz | number | No | Static offset (-1000 to 1000 Hz) |
| drift_hz_per_sec | number | No | Drift rate (-10 to 10 Hz/s) |

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/channel/impairments/offset \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "offset_hz": 50.0}'
```

---

### POST /channel/reset

Reset channel state.

**Response:**

```json
{
  "status": "reset",
  "blocks_processed": 0,
  "total_samples_processed": 0
}
```

---

## Sample Processing

### POST /processing/samples

Process samples through the channel.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| samples_base64 | string | Yes | Base64-encoded samples |
| format | string | Yes | "complex64" or "complex128" |

**Response:**

```json
{
  "samples_base64": "AAAA...",
  "samples_count": 4096,
  "blocks_processed": 1,
  "format": "complex64"
}
```

**Example (Python):**

```python
import requests
import numpy as np
import base64

# Create samples
samples = np.exp(1j * np.linspace(0, 10, 4096)).astype(np.complex64)
samples_b64 = base64.b64encode(samples.tobytes()).decode()

# Send request
response = requests.post(
    "http://localhost:8000/api/v1/processing/samples",
    json={
        "samples_base64": samples_b64,
        "format": "complex64"
    }
)

# Decode response
result = response.json()
output_bytes = base64.b64decode(result["samples_base64"])
output = np.frombuffer(output_bytes, dtype=np.complex64)
```

---

## Session Management

### POST /processing/sessions

Create a new processing session.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| channel_model | string | No | "watterson", "vogler", "vh", "passthrough" |
| sample_rate_hz | integer | No | Sample rate (1000-100000000) |
| metadata | object | No | Custom metadata |

**Response:**

```json
{
  "session_id": "abc123",
  "channel_model": "watterson",
  "sample_rate_hz": 48000,
  "created_at": "2024-01-15T12:00:00Z",
  "metadata": {}
}
```

---

### GET /processing/sessions

List all sessions.

**Response:**

```json
{
  "sessions": [
    {
      "session_id": "abc123",
      "channel_model": "watterson",
      "created_at": "2024-01-15T12:00:00Z",
      "samples_processed": 123456
    }
  ],
  "count": 1
}
```

---

### GET /processing/sessions/{session_id}

Get session details.

**Response:**

```json
{
  "session_id": "abc123",
  "channel_model": "watterson",
  "sample_rate_hz": 48000,
  "created_at": "2024-01-15T12:00:00Z",
  "samples_processed": 123456,
  "blocks_processed": 30,
  "metadata": {}
}
```

---

### DELETE /processing/sessions/{session_id}

Delete a session.

**Response:**

```json
{
  "status": "deleted",
  "session_id": "abc123"
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request (invalid parameters) |
| 404 | Resource not found |
| 422 | Validation error |
| 500 | Internal server error |

---

## Rate Limits

The default configuration has no rate limits. For production, configure limits via environment variables:

```bash
export HFPATHSIM_RATE_LIMIT=100  # Requests per minute
```

---

## OpenAPI Documentation

Interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

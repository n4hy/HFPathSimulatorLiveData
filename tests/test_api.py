"""Tests for REST API."""

import base64
import numpy as np
import pytest

# Skip if fastapi not installed
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from hfpathsim.api.app import create_app, get_global_engine
from hfpathsim.engine.session import shutdown_session_manager


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    with TestClient(app) as client:
        yield client
    # Cleanup
    shutdown_session_manager()


class TestHealthEndpoints:
    """Tests for system endpoints."""

    def test_health_check(self, client):
        """Test health endpoint."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "uptime_seconds" in data

    def test_gpu_info(self, client):
        """Test GPU info endpoint."""
        response = client.get("/api/v1/gpu")

        assert response.status_code == 200
        data = response.json()
        assert "available" in data


class TestChannelEndpoints:
    """Tests for channel configuration endpoints."""

    def test_get_channel_state(self, client):
        """Test getting channel state."""
        response = client.get("/api/v1/channel/state")

        assert response.status_code == 200
        data = response.json()
        assert "model" in data
        assert "running" in data
        assert "total_samples_processed" in data

    def test_configure_vogler(self, client):
        """Test Vogler channel configuration."""
        response = client.post(
            "/api/v1/channel/vogler",
            json={
                "foF2": 8.0,
                "hmF2": 320.0,
                "doppler_spread_hz": 2.0,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "vogler"
        assert data["vogler"]["foF2"] == 8.0
        assert data["vogler"]["hmF2"] == 320.0

    def test_configure_watterson(self, client):
        """Test Watterson channel configuration."""
        response = client.post(
            "/api/v1/channel/watterson",
            json={"condition": "disturbed"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "watterson"

    def test_configure_vh(self, client):
        """Test Vogler-Hoffmeyer configuration."""
        response = client.post(
            "/api/v1/channel/vh",
            json={
                "condition": "moderate",
                "spread_f_enabled": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "vogler_hoffmeyer"

    def test_configure_noise(self, client):
        """Test noise configuration."""
        response = client.post(
            "/api/v1/channel/noise",
            json={
                "snr_db": 15.0,
                "enable_atmospheric": True,
            },
        )

        assert response.status_code == 200

    def test_disable_noise(self, client):
        """Test disabling noise."""
        # First enable
        client.post("/api/v1/channel/noise", json={"snr_db": 10.0})

        # Then disable
        response = client.post("/api/v1/channel/noise/disable")

        assert response.status_code == 200

    def test_configure_agc(self, client):
        """Test AGC configuration."""
        response = client.post(
            "/api/v1/channel/impairments/agc",
            json={
                "enabled": True,
                "target_level_db": -15.0,
                "max_gain_db": 50.0,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["agc_enabled"] is True

    def test_configure_limiter(self, client):
        """Test limiter configuration."""
        response = client.post(
            "/api/v1/channel/impairments/limiter",
            json={
                "enabled": True,
                "threshold_db": -3.0,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["limiter_enabled"] is True

    def test_configure_freq_offset(self, client):
        """Test frequency offset configuration."""
        response = client.post(
            "/api/v1/channel/impairments/offset",
            json={
                "enabled": True,
                "offset_hz": 100.0,
                "drift_hz_per_sec": 0.5,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["freq_offset_enabled"] is True

    def test_reset_channel(self, client):
        """Test channel reset."""
        # Configure something
        client.post("/api/v1/channel/noise", json={"snr_db": 10.0})

        # Reset
        response = client.post("/api/v1/channel/reset")

        assert response.status_code == 200
        data = response.json()
        assert data["blocks_processed"] == 0


class TestProcessingEndpoints:
    """Tests for processing endpoints."""

    def test_process_samples(self, client):
        """Test sample processing endpoint."""
        # Generate test samples
        samples = np.exp(1j * np.linspace(0, 10 * np.pi, 4096)).astype(np.complex64)
        samples_base64 = base64.b64encode(samples.tobytes()).decode("ascii")

        response = client.post(
            "/api/v1/processing/samples",
            json={
                "samples_base64": samples_base64,
                "format": "complex64",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "samples_base64" in data
        assert data["samples_count"] == 4096
        assert data["blocks_processed"] >= 1

        # Decode output
        output_bytes = base64.b64decode(data["samples_base64"])
        output = np.frombuffer(output_bytes, dtype=np.complex64)
        assert len(output) == 4096

    def test_process_samples_invalid_format(self, client):
        """Test processing with invalid format."""
        samples = np.ones(100, dtype=np.complex64)
        samples_base64 = base64.b64encode(samples.tobytes()).decode("ascii")

        response = client.post(
            "/api/v1/processing/samples",
            json={
                "samples_base64": samples_base64,
                "format": "invalid",
            },
        )

        assert response.status_code == 400

    def test_process_samples_invalid_base64(self, client):
        """Test processing with invalid base64."""
        response = client.post(
            "/api/v1/processing/samples",
            json={
                "samples_base64": "not-valid-base64!!!",
                "format": "complex64",
            },
        )

        assert response.status_code == 400


class TestSessionEndpoints:
    """Tests for session management endpoints."""

    def test_create_session(self, client):
        """Test session creation."""
        response = client.post(
            "/api/v1/processing/sessions",
            json={
                "channel_model": "vogler",
                "sample_rate_hz": 1000000,
                "metadata": {"user": "test"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["metadata"] == {"user": "test"}

    def test_list_sessions(self, client):
        """Test session listing."""
        # Create a session
        client.post(
            "/api/v1/processing/sessions",
            json={"channel_model": "watterson"},
        )

        response = client.get("/api/v1/processing/sessions")

        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert data["count"] >= 1

    def test_get_session(self, client):
        """Test getting session details."""
        # Create session
        create_response = client.post(
            "/api/v1/processing/sessions",
            json={},
        )
        session_id = create_response.json()["session_id"]

        # Get session
        response = client.get(f"/api/v1/processing/sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id

    def test_get_nonexistent_session(self, client):
        """Test getting nonexistent session."""
        response = client.get("/api/v1/processing/sessions/nonexistent")

        assert response.status_code == 404

    def test_delete_session(self, client):
        """Test session deletion."""
        # Create session
        create_response = client.post(
            "/api/v1/processing/sessions",
            json={},
        )
        session_id = create_response.json()["session_id"]

        # Delete
        response = client.delete(f"/api/v1/processing/sessions/{session_id}")

        assert response.status_code == 200

        # Verify deleted
        get_response = client.get(f"/api/v1/processing/sessions/{session_id}")
        assert get_response.status_code == 404

    def test_process_with_session(self, client):
        """Test processing through session."""
        # Create session
        create_response = client.post(
            "/api/v1/processing/sessions",
            json={"channel_model": "passthrough"},
        )
        session_id = create_response.json()["session_id"]

        # Process samples
        samples = np.ones(1024, dtype=np.complex64)
        samples_base64 = base64.b64encode(samples.tobytes()).decode("ascii")

        response = client.post(
            f"/api/v1/processing/samples?session_id={session_id}",
            json={
                "samples_base64": samples_base64,
                "format": "complex64",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["samples_count"] == 1024

    def test_channel_state_with_session(self, client):
        """Test getting channel state for session."""
        # Create session
        create_response = client.post(
            "/api/v1/processing/sessions",
            json={"channel_model": "watterson"},
        )
        session_id = create_response.json()["session_id"]

        # Get state
        response = client.get(
            f"/api/v1/channel/state?session_id={session_id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "watterson"


class TestValidation:
    """Tests for request validation."""

    def test_vogler_foF2_range(self, client):
        """Test foF2 range validation."""
        # Too low
        response = client.post(
            "/api/v1/channel/vogler",
            json={"foF2": 0.5},  # Below 1.0
        )
        assert response.status_code == 422

        # Too high
        response = client.post(
            "/api/v1/channel/vogler",
            json={"foF2": 25.0},  # Above 20.0
        )
        assert response.status_code == 422

    def test_agc_target_range(self, client):
        """Test AGC target level range validation."""
        response = client.post(
            "/api/v1/channel/impairments/agc",
            json={"enabled": True, "target_level_db": 10.0},  # Above 0
        )
        assert response.status_code == 422

    def test_sample_rate_range(self, client):
        """Test sample rate range validation in session creation."""
        response = client.post(
            "/api/v1/processing/sessions",
            json={"sample_rate_hz": 100},  # Below 1000
        )
        assert response.status_code == 422


class TestWebSocketEndpoints:
    """Tests for WebSocket endpoints."""

    def test_input_stream_connect(self, client):
        """Test connecting to input stream."""
        with client.websocket_connect("/api/v1/stream/input") as websocket:
            # Send ping
            websocket.send_json({"type": "ping"})
            response = websocket.receive_json()

            assert response["type"] == "pong"
            assert "timestamp" in response

    def test_input_stream_process(self, client):
        """Test processing through input stream."""
        with client.websocket_connect("/api/v1/stream/input") as websocket:
            # Send samples
            samples = np.ones(1024, dtype=np.complex64)
            samples_base64 = base64.b64encode(samples.tobytes()).decode("ascii")

            websocket.send_json({
                "type": "samples",
                "samples_base64": samples_base64,
                "format": "complex64",
            })

            response = websocket.receive_json()

            assert response["type"] == "processed"
            assert response["count"] == 1024
            assert "samples_base64" in response

    def test_state_stream(self, client):
        """Test state stream."""
        with client.websocket_connect(
            "/api/v1/stream/state?interval_ms=100"
        ) as websocket:
            response = websocket.receive_json()

            assert response["type"] == "state"
            assert "running" in response
            assert "blocks_processed" in response

    def test_invalid_session_websocket(self, client):
        """Test WebSocket with invalid session."""
        # This should close immediately with error code
        with pytest.raises(Exception):
            with client.websocket_connect(
                "/api/v1/stream/input?session_id=invalid"
            ) as websocket:
                websocket.receive_json()


class TestCORS:
    """Tests for CORS handling."""

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert "access-control-allow-origin" in response.headers


class TestTimingHeader:
    """Tests for timing middleware."""

    def test_timing_header(self, client):
        """Test X-Process-Time header."""
        response = client.get("/api/v1/health")

        assert "x-process-time" in response.headers
        timing = float(response.headers["x-process-time"])
        assert timing >= 0

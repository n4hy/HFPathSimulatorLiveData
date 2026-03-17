"""Tests for integration module (GNU Radio ZMQ and MATLAB interface)."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytest

from hfpathsim.integration.gnuradio_zmq import (
    GNURadioZMQBridge,
    create_gr_flowgraph_snippet,
    create_gr_sink_snippet,
    get_zmq_connection_info,
)
from hfpathsim.integration.matlab_interface import (
    MATFileInterface,
    MATLABEngineInterface,
    ChannelSnapshot,
)


class TestGNURadioZMQBridge:
    """Tests for GNURadioZMQBridge."""

    def test_create_bridge(self):
        """Test creating ZMQ bridge."""
        bridge = GNURadioZMQBridge(
            pub_address="tcp://*:5560",
            sub_address="tcp://127.0.0.1:5561",
        )

        assert bridge._pub_address == "tcp://*:5560"
        assert bridge._sub_address == "tcp://127.0.0.1:5561"

    @pytest.mark.skipif(
        not pytest.importorskip("zmq", reason="pyzmq not installed"),
        reason="pyzmq not installed",
    )
    def test_open_publisher(self):
        """Test opening ZMQ publisher."""
        bridge = GNURadioZMQBridge(pub_address="tcp://*:5562")

        assert bridge.open_publisher()
        bridge.close()

    @pytest.mark.skipif(
        not pytest.importorskip("zmq", reason="pyzmq not installed"),
        reason="pyzmq not installed",
    )
    def test_context_manager(self):
        """Test using bridge as context manager."""
        with GNURadioZMQBridge() as bridge:
            # Context manager doesn't auto-open, but should auto-close
            pass

    @pytest.mark.skipif(
        not pytest.importorskip("zmq", reason="pyzmq not installed"),
        reason="pyzmq not installed",
    )
    def test_send_samples(self):
        """Test sending samples via ZMQ."""
        bridge = GNURadioZMQBridge(pub_address="tcp://*:5563")
        bridge.open_publisher()

        samples = np.random.randn(100) + 1j * np.random.randn(100)
        samples = samples.astype(np.complex64)

        # Send should succeed even with no subscribers
        assert bridge.send(samples)

        bridge.close()


class TestGNURadioSnippets:
    """Tests for GNU Radio code snippet generators."""

    def test_create_source_snippet(self):
        """Test creating GR source snippet."""
        snippet = create_gr_flowgraph_snippet(
            zmq_address="tcp://127.0.0.1:5556",
            sample_rate=2e6,
            center_freq=10e6,
        )

        assert "hfpathsim_source" in snippet
        assert "zeromq.sub_source" in snippet
        assert "tcp://127.0.0.1:5556" in snippet
        assert "2.000 Msps" in snippet
        assert "10.000 MHz" in snippet

    def test_create_sink_snippet(self):
        """Test creating GR sink snippet."""
        snippet = create_gr_sink_snippet(
            zmq_address="tcp://127.0.0.1:5555",
            sample_rate=2e6,
        )

        assert "hfpathsim_sink" in snippet
        assert "zeromq.pub_sink" in snippet
        assert "tcp://127.0.0.1:5555" in snippet

    def test_get_connection_info(self):
        """Test getting ZMQ connection info."""
        info = get_zmq_connection_info(port=5556)

        assert info["protocol"] == "ZeroMQ PUB/SUB"
        assert info["data_type"] == "complex64 (gr_complex)"
        assert "5556" in info["local_bind"]
        assert "5556" in info["remote_connect"]


class TestMATFileInterface:
    """Tests for MATFileInterface."""

    def test_create_interface(self):
        """Test creating MAT file interface."""
        interface = MATFileInterface(use_hdf5=True)
        assert interface is not None

    @pytest.mark.skipif(
        not pytest.importorskip("scipy", reason="scipy not installed"),
        reason="scipy not installed",
    )
    def test_save_load_channel_state(self):
        """Test saving and loading channel state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "channel_state.mat"

            # Create test snapshot
            snapshot = ChannelSnapshot(
                timestamp=datetime.now(timezone.utc),
                transfer_function=np.random.randn(256) + 1j * np.random.randn(256),
                impulse_response=np.random.randn(64) + 1j * np.random.randn(64),
                scattering_function=np.random.randn(32, 16),
                freq_axis_hz=np.linspace(-1e3, 1e3, 256),
                delay_axis_ms=np.linspace(0, 10, 64),
                doppler_axis_hz=np.linspace(-5, 5, 32),
                parameters={"snr_db": 20.0, "model": "vogler"},
            )

            interface = MATFileInterface()

            # Save
            assert interface.save_channel_state(filepath, snapshot)
            assert filepath.exists()

            # Load
            data = interface.load_mat(filepath)
            assert data is not None
            assert "transfer_function" in data
            assert "impulse_response" in data
            assert "scattering_function" in data

    @pytest.mark.skipif(
        not pytest.importorskip("scipy", reason="scipy not installed"),
        reason="scipy not installed",
    )
    def test_save_iq_recording(self):
        """Test saving IQ recording."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "iq_recording.mat"

            samples = np.random.randn(10000) + 1j * np.random.randn(10000)
            samples = samples.astype(np.complex64)

            interface = MATFileInterface()

            assert interface.save_iq_recording(
                filepath,
                samples,
                sample_rate_hz=48000.0,
                center_freq_hz=10e6,
                metadata={"description": "Test recording"},
            )

            assert filepath.exists()

            # Load and verify
            data = interface.load_mat(filepath)
            assert data is not None
            assert "iq_samples" in data
            assert "sample_rate_hz" in data

    @pytest.mark.skipif(
        not pytest.importorskip("scipy", reason="scipy not installed"),
        reason="scipy not installed",
    )
    def test_save_channel_evolution(self):
        """Test saving channel evolution time series."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "channel_evolution.mat"

            # Create multiple snapshots
            snapshots = []
            for i in range(5):
                snapshot = ChannelSnapshot(
                    timestamp=datetime.now(timezone.utc),
                    transfer_function=np.random.randn(128) + 1j * np.random.randn(128),
                    impulse_response=np.random.randn(32) + 1j * np.random.randn(32),
                    scattering_function=np.random.randn(16, 8),
                    freq_axis_hz=np.linspace(-1e3, 1e3, 128),
                    delay_axis_ms=np.linspace(0, 5, 32),
                    doppler_axis_hz=np.linspace(-3, 3, 16),
                )
                snapshots.append(snapshot)

            interface = MATFileInterface()
            time_axis = np.linspace(0, 1, 5)

            assert interface.save_channel_evolution(filepath, snapshots, time_axis)
            assert filepath.exists()

            # Load and verify
            data = interface.load_mat(filepath)
            assert data is not None
            assert "num_snapshots" in data
            assert "transfer_functions" in data

    def test_save_without_scipy(self):
        """Test handling when scipy is not available."""
        interface = MATFileInterface()
        interface._scipy_io = None
        interface._h5py = None

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.mat"

            snapshot = ChannelSnapshot(
                timestamp=datetime.now(timezone.utc),
                transfer_function=np.zeros(10),
                impulse_response=np.zeros(10),
                scattering_function=np.zeros((5, 5)),
                freq_axis_hz=np.zeros(10),
                delay_axis_ms=np.zeros(10),
                doppler_axis_hz=np.zeros(5),
            )

            # Should return False when no backend available
            assert not interface.save_channel_state(filepath, snapshot)


class TestMATLABEngineInterface:
    """Tests for MATLABEngineInterface."""

    def test_create_interface(self):
        """Test creating MATLAB engine interface."""
        interface = MATLABEngineInterface()
        assert not interface.is_running()

    def test_start_without_matlab(self):
        """Test starting engine when MATLAB not installed."""
        interface = MATLABEngineInterface()

        # Should fail gracefully when MATLAB not installed
        result = interface.start_engine()
        # Result depends on whether MATLAB is installed

    def test_context_manager(self):
        """Test using engine as context manager."""
        # This tests the interface even without MATLAB
        with MATLABEngineInterface() as engine:
            # May or may not be running depending on MATLAB availability
            pass


class TestChannelSnapshot:
    """Tests for ChannelSnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating channel snapshot."""
        snapshot = ChannelSnapshot(
            timestamp=datetime.now(timezone.utc),
            transfer_function=np.zeros(100, dtype=np.complex64),
            impulse_response=np.zeros(50, dtype=np.complex64),
            scattering_function=np.zeros((20, 10)),
            freq_axis_hz=np.linspace(-1000, 1000, 100),
            delay_axis_ms=np.linspace(0, 10, 50),
            doppler_axis_hz=np.linspace(-5, 5, 20),
            parameters={"model": "watterson"},
        )

        assert snapshot.transfer_function.shape == (100,)
        assert snapshot.impulse_response.shape == (50,)
        assert snapshot.scattering_function.shape == (20, 10)
        assert snapshot.parameters["model"] == "watterson"

    def test_default_parameters(self):
        """Test snapshot with default parameters."""
        snapshot = ChannelSnapshot(
            timestamp=datetime.now(timezone.utc),
            transfer_function=np.zeros(10),
            impulse_response=np.zeros(10),
            scattering_function=np.zeros((5, 5)),
            freq_axis_hz=np.zeros(10),
            delay_axis_ms=np.zeros(10),
            doppler_axis_hz=np.zeros(5),
        )

        assert snapshot.parameters == {}

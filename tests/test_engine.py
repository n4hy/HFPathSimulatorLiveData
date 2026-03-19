"""Tests for headless simulation engine."""

import numpy as np
import pytest
import threading
import time

from hfpathsim.engine import (
    SimulationEngine,
    EngineConfig,
    ChannelModel,
    SessionManager,
    Session,
)
from hfpathsim.core.parameters import VoglerParameters, ITUCondition


class TestSimulationEngine:
    """Tests for SimulationEngine class."""

    def test_default_initialization(self):
        """Test engine initializes with defaults."""
        engine = SimulationEngine()

        assert engine.config.channel_model == ChannelModel.VOGLER
        assert engine.config.sample_rate_hz == 2_000_000
        assert engine.config.block_size == 4096
        assert not engine._state.running

    def test_custom_config(self):
        """Test engine with custom configuration."""
        config = EngineConfig(
            channel_model=ChannelModel.WATTERSON,
            sample_rate_hz=1_000_000,
            block_size=2048,
            use_gpu=False,
        )
        engine = SimulationEngine(config=config)

        assert engine.config.channel_model == ChannelModel.WATTERSON
        assert engine.config.sample_rate_hz == 1_000_000
        assert engine._watterson is not None

    def test_process_vogler(self):
        """Test processing with Vogler channel."""
        config = EngineConfig(
            channel_model=ChannelModel.VOGLER,
            use_gpu=False,
        )
        engine = SimulationEngine(config=config)

        # Generate test signal
        samples = np.exp(1j * np.linspace(0, 10 * np.pi, 4096)).astype(np.complex64)

        output = engine.process(samples)

        assert output is not None
        assert len(output) == len(samples)
        assert output.dtype == np.complex64
        assert engine._state.blocks_processed == 1
        assert engine._state.total_samples_processed == 4096

    def test_process_watterson(self):
        """Test processing with Watterson channel."""
        config = EngineConfig(
            channel_model=ChannelModel.WATTERSON,
            use_gpu=False,
        )
        engine = SimulationEngine(config=config)

        samples = np.exp(1j * np.linspace(0, 10 * np.pi, 4096)).astype(np.complex64)
        output = engine.process(samples)

        assert output is not None
        assert len(output) == len(samples)

    def test_process_vogler_hoffmeyer(self):
        """Test processing with Vogler-Hoffmeyer channel."""
        config = EngineConfig(
            channel_model=ChannelModel.VOGLER_HOFFMEYER,
            use_gpu=False,
        )
        engine = SimulationEngine(config=config)

        samples = np.exp(1j * np.linspace(0, 10 * np.pi, 4096)).astype(np.complex64)
        output = engine.process(samples)

        assert output is not None
        assert len(output) == len(samples)

    def test_process_passthrough(self):
        """Test passthrough mode (no channel processing)."""
        config = EngineConfig(
            channel_model=ChannelModel.PASSTHROUGH,
            use_gpu=False,
        )
        engine = SimulationEngine(config=config)

        samples = np.exp(1j * np.linspace(0, 10 * np.pi, 4096)).astype(np.complex64)
        output = engine.process(samples)

        # Passthrough should return nearly identical signal (only dtype conversion)
        np.testing.assert_array_almost_equal(output, samples, decimal=5)

    def test_configure_via_dict(self):
        """Test configuration via dictionary."""
        engine = SimulationEngine()

        engine.configure({
            "noise_enabled": True,
            "agc_enabled": True,
            "vogler": {
                "foF2": 8.0,
                "doppler_spread_hz": 2.0,
            },
            "noise": {
                "snr_db": 15.0,
            },
        })

        assert engine.config.noise_enabled
        assert engine.config.agc_enabled
        assert engine._vogler_params.foF2 == 8.0
        assert engine._vogler_params.doppler_spread_hz == 2.0
        assert engine._noise.config.snr_db == 15.0

    def test_configure_vogler_shortcut(self):
        """Test Vogler configuration shortcut."""
        engine = SimulationEngine()

        engine.configure_vogler(
            foF2=9.0,
            hmF2=350.0,
            doppler_spread_hz=3.0,
        )

        assert engine._vogler_params.foF2 == 9.0
        assert engine._vogler_params.hmF2 == 350.0
        assert engine._vogler_params.doppler_spread_hz == 3.0

    def test_configure_noise_shortcut(self):
        """Test noise configuration shortcut."""
        engine = SimulationEngine()

        engine.configure_noise(
            snr_db=10.0,
            enable_atmospheric=True,
        )

        assert engine.config.noise_enabled
        assert engine._noise.config.snr_db == 10.0
        assert engine._noise.config.enable_atmospheric

    def test_configure_agc_shortcut(self):
        """Test AGC configuration shortcut."""
        engine = SimulationEngine()

        engine.configure_agc(
            enabled=True,
            target_level_db=-15.0,
            max_gain_db=50.0,
        )

        assert engine.config.agc_enabled
        assert engine._agc.config.target_level_db == -15.0
        assert engine._agc.config.max_gain_db == 50.0

    def test_process_with_noise(self):
        """Test processing with noise enabled."""
        config = EngineConfig(
            channel_model=ChannelModel.PASSTHROUGH,
            noise_enabled=True,
            use_gpu=False,
        )
        engine = SimulationEngine(config=config)

        samples = np.ones(4096, dtype=np.complex64)
        output = engine.process(samples)

        # Output should differ due to noise
        assert not np.allclose(output, samples)

    def test_process_with_agc(self):
        """Test processing with AGC enabled."""
        config = EngineConfig(
            channel_model=ChannelModel.PASSTHROUGH,
            agc_enabled=True,
            use_gpu=False,
        )
        engine = SimulationEngine(config=config)

        # Low level input
        samples = np.ones(4096, dtype=np.complex64) * 0.001
        output = engine.process(samples)

        # AGC should increase level
        assert np.mean(np.abs(output)) > np.mean(np.abs(samples))
        assert engine._state.agc_gain_db > 0

    def test_get_state(self):
        """Test state retrieval."""
        engine = SimulationEngine()

        state = engine.get_state()

        assert "running" in state
        assert "total_samples_processed" in state
        assert "blocks_processed" in state
        assert state["running"] is False
        assert state["total_samples_processed"] == 0

    def test_get_channel_state(self):
        """Test channel state retrieval."""
        config = EngineConfig(
            channel_model=ChannelModel.VOGLER,
            use_gpu=False,
        )
        engine = SimulationEngine(config=config)

        # Process something to generate state
        samples = np.exp(1j * np.linspace(0, 10 * np.pi, 4096)).astype(np.complex64)
        engine.process(samples)

        channel_state = engine.get_channel_state()

        assert channel_state is not None
        assert channel_state.transfer_function is not None

    def test_output_callback(self):
        """Test output callback mechanism."""
        engine = SimulationEngine()

        received_outputs = []

        def on_output(samples):
            received_outputs.append(samples.copy())

        engine.add_output_callback(on_output)

        samples = np.ones(4096, dtype=np.complex64)
        engine.process(samples)

        assert len(received_outputs) == 1
        assert len(received_outputs[0]) == 4096

        # Remove callback
        engine.remove_output_callback(on_output)
        engine.process(samples)
        assert len(received_outputs) == 1  # No new callback

    def test_channel_state_callback(self):
        """Test channel state callback mechanism."""
        config = EngineConfig(
            channel_model=ChannelModel.VOGLER,
            use_gpu=False,
        )
        engine = SimulationEngine(config=config)

        received_states = []

        def on_state(state):
            received_states.append(state)

        engine.add_channel_state_callback(on_state)

        samples = np.ones(4096, dtype=np.complex64)
        engine.process(samples)

        assert len(received_states) == 1
        assert received_states[0].transfer_function is not None

    def test_reset(self):
        """Test engine reset."""
        engine = SimulationEngine()

        # Process some data
        samples = np.ones(4096, dtype=np.complex64)
        engine.process(samples)
        assert engine._state.blocks_processed == 1

        # Reset
        engine.reset()
        assert engine._state.blocks_processed == 0
        assert engine._state.total_samples_processed == 0

    def test_get_gpu_info_no_gpu(self):
        """Test GPU info when GPU unavailable."""
        config = EngineConfig(use_gpu=False)
        engine = SimulationEngine(config=config)

        info = engine.get_gpu_info()

        assert isinstance(info, dict)
        # Either has GPU info or indicates unavailable
        assert "available" in info or "name" in info

    def test_thread_safety(self):
        """Test concurrent processing is thread-safe."""
        engine = SimulationEngine()

        errors = []
        results = []

        def process_thread(thread_id):
            try:
                for _ in range(10):
                    samples = np.ones(4096, dtype=np.complex64) * thread_id
                    output = engine.process(samples)
                    results.append((thread_id, len(output)))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=process_thread, args=(i,))
            for i in range(4)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 40  # 4 threads * 10 iterations


class TestSessionManager:
    """Tests for SessionManager class."""

    def test_create_session(self):
        """Test session creation."""
        manager = SessionManager(max_sessions=10)

        session = manager.create_session()

        assert session.session_id is not None
        assert session.engine is not None
        assert manager.get_session_count() == 1

        manager.shutdown()

    def test_create_session_with_config(self):
        """Test session creation with custom config."""
        manager = SessionManager(max_sessions=10)

        config = EngineConfig(
            channel_model=ChannelModel.WATTERSON,
            sample_rate_hz=1_000_000,
        )
        session = manager.create_session(config=config)

        assert session.engine.config.channel_model == ChannelModel.WATTERSON
        assert session.engine.config.sample_rate_hz == 1_000_000

        manager.shutdown()

    def test_get_session(self):
        """Test session retrieval."""
        manager = SessionManager(max_sessions=10)

        session = manager.create_session()
        session_id = session.session_id

        retrieved = manager.get_session(session_id)

        assert retrieved is not None
        assert retrieved.session_id == session_id

        manager.shutdown()

    def test_get_nonexistent_session(self):
        """Test retrieval of nonexistent session."""
        manager = SessionManager(max_sessions=10)

        retrieved = manager.get_session("nonexistent-id")

        assert retrieved is None

        manager.shutdown()

    def test_delete_session(self):
        """Test session deletion."""
        manager = SessionManager(max_sessions=10)

        session = manager.create_session()
        session_id = session.session_id

        result = manager.delete_session(session_id)

        assert result is True
        assert manager.get_session(session_id) is None
        assert manager.get_session_count() == 0

        manager.shutdown()

    def test_delete_nonexistent_session(self):
        """Test deletion of nonexistent session."""
        manager = SessionManager(max_sessions=10)

        result = manager.delete_session("nonexistent-id")

        assert result is False

        manager.shutdown()

    def test_max_sessions_limit(self):
        """Test maximum sessions limit."""
        manager = SessionManager(max_sessions=3, session_timeout_minutes=60)

        # Create max sessions
        for _ in range(3):
            manager.create_session()

        assert manager.get_session_count() == 3

        # Should raise when trying to create more
        with pytest.raises(RuntimeError, match="Maximum sessions"):
            manager.create_session()

        manager.shutdown()

    def test_list_sessions(self):
        """Test listing all sessions."""
        manager = SessionManager(max_sessions=10)

        session1 = manager.create_session(metadata={"user": "alice"})
        session2 = manager.create_session(metadata={"user": "bob"})

        sessions = manager.list_sessions()

        assert len(sessions) == 2
        session_ids = {s["session_id"] for s in sessions}
        assert session1.session_id in session_ids
        assert session2.session_id in session_ids

        manager.shutdown()

    def test_session_touch_updates_timestamp(self):
        """Test that accessing session updates timestamp."""
        manager = SessionManager(max_sessions=10)

        session = manager.create_session()
        initial_accessed = session.last_accessed

        time.sleep(0.01)
        manager.get_session(session.session_id)

        assert session.last_accessed > initial_accessed

        manager.shutdown()

    def test_session_to_dict(self):
        """Test session serialization."""
        manager = SessionManager(max_sessions=10)

        session = manager.create_session(metadata={"key": "value"})
        data = session.to_dict()

        assert "session_id" in data
        assert "created_at" in data
        assert "last_accessed" in data
        assert "age_seconds" in data
        assert "idle_seconds" in data
        assert "engine_running" in data
        assert data["metadata"] == {"key": "value"}

        manager.shutdown()

    def test_session_processing(self):
        """Test processing through session engine."""
        manager = SessionManager(max_sessions=10)

        session = manager.create_session()
        engine = session.engine

        samples = np.exp(1j * np.linspace(0, 10 * np.pi, 4096)).astype(np.complex64)
        output = engine.process(samples)

        assert output is not None
        assert len(output) == len(samples)

        manager.shutdown()

    def test_shutdown_stops_all_engines(self):
        """Test shutdown cleans up all sessions."""
        manager = SessionManager(max_sessions=10)

        sessions = [manager.create_session() for _ in range(3)]

        manager.shutdown()

        # Manager should be empty
        assert manager.get_session_count() == 0


class TestSession:
    """Tests for Session dataclass."""

    def test_session_age(self):
        """Test session age calculation."""
        from hfpathsim.engine import SimulationEngine

        engine = SimulationEngine()
        session = Session(
            session_id="test-id",
            engine=engine,
        )

        # Age should be >= 0
        assert session.age_seconds >= 0

    def test_session_idle_time(self):
        """Test session idle time calculation."""
        from hfpathsim.engine import SimulationEngine

        engine = SimulationEngine()
        session = Session(
            session_id="test-id",
            engine=engine,
        )

        initial_idle = session.idle_seconds
        time.sleep(0.05)

        # Idle time should increase
        assert session.idle_seconds > initial_idle

        # Touch resets idle - idle should be very small after touch
        session.touch()
        assert session.idle_seconds < 0.01  # Should be nearly zero after touch


# Mock input source for streaming tests
class MockInputSource:
    """Mock input source for testing."""

    def __init__(self, sample_rate: float = 2_000_000):
        self._sample_rate = sample_rate
        self._is_open = False
        self._read_count = 0

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> bool:
        self._is_open = True
        return True

    def close(self):
        self._is_open = False

    def read(self, num_samples: int):
        if self._read_count >= 10:
            return None  # Stop after 10 reads
        self._read_count += 1
        return np.ones(num_samples, dtype=np.complex64)


class MockOutputSink:
    """Mock output sink for testing."""

    def __init__(self):
        self._is_open = False
        self.written_samples = []

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> bool:
        self._is_open = True
        return True

    def close(self):
        self._is_open = False

    def write(self, samples) -> int:
        self.written_samples.append(samples.copy())
        return len(samples)


class TestStreaming:
    """Tests for streaming functionality."""

    def test_start_stop_streaming(self):
        """Test starting and stopping streaming."""
        engine = SimulationEngine()
        input_source = MockInputSource()
        output_sink = MockOutputSink()

        engine.start_streaming(input_source, output_sink)

        assert engine._state.running
        time.sleep(0.1)

        engine.stop_streaming()

        assert not engine._state.running
        assert len(output_sink.written_samples) > 0

    def test_streaming_already_running(self):
        """Test error when starting streaming twice."""
        engine = SimulationEngine()
        input_source = MockInputSource()

        engine.start_streaming(input_source)
        time.sleep(0.01)

        with pytest.raises(RuntimeError, match="already running"):
            engine.start_streaming(input_source)

        engine.stop_streaming()

    def test_streaming_state_callback(self):
        """Test state callbacks during streaming."""
        engine = SimulationEngine()
        input_source = MockInputSource()

        states = []

        def on_state(state):
            states.append(state)

        engine.add_state_callback(on_state)
        engine.start_streaming(input_source)

        time.sleep(0.2)
        engine.stop_streaming()

        assert len(states) > 0

"""Tests for output sink module."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from hfpathsim.output.base import OutputSink, OutputFormat
from hfpathsim.output.file import FileOutputSink
from hfpathsim.output.network import NetworkOutputSink, NetworkProtocol
from hfpathsim.output.audio import AudioOutputSink
from hfpathsim.output.sdr import SDROutputSink
from hfpathsim.output.multiplex import MultiplexOutputSink, TeeOutputSink


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_format_values(self):
        """Test that all expected formats exist."""
        assert OutputFormat.COMPLEX64.value == "complex64"
        assert OutputFormat.COMPLEX128.value == "complex128"
        assert OutputFormat.INT16_IQ.value == "int16_iq"
        assert OutputFormat.INT8_IQ.value == "int8_iq"
        assert OutputFormat.FLOAT32_IQ.value == "float32_iq"


class TestFileOutputSink:
    """Tests for FileOutputSink."""

    def test_create_file_sink(self):
        """Test creating a file output sink."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.raw"
            sink = FileOutputSink(filepath, sample_rate_hz=48000.0)

            assert sink.sample_rate == 48000.0
            assert sink.filepath == filepath
            assert not sink.is_open

    def test_write_raw_file(self):
        """Test writing raw binary file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.raw"
            sink = FileOutputSink(
                filepath,
                sample_rate_hz=48000.0,
                output_format=OutputFormat.COMPLEX64,
            )

            assert sink.open()
            assert sink.is_open

            # Write some samples
            samples = np.random.randn(1000) + 1j * np.random.randn(1000)
            samples = samples.astype(np.complex64)

            written = sink.write(samples)
            assert written == 1000
            assert sink.total_samples_written == 1000

            sink.close()
            assert not sink.is_open

            # Verify file exists and has correct size
            assert filepath.exists()
            assert filepath.stat().st_size == 1000 * 8  # complex64 = 8 bytes

    def test_write_wav_file(self):
        """Test writing WAV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.wav"
            sink = FileOutputSink(
                filepath,
                sample_rate_hz=48000.0,
                output_format=OutputFormat.INT16_IQ,
            )

            assert sink.open()

            # Write samples
            samples = 0.5 * (np.random.randn(1000) + 1j * np.random.randn(1000))
            samples = samples.astype(np.complex64)

            written = sink.write(samples)
            assert written == 1000

            sink.close()

            # Verify file exists
            assert filepath.exists()

    def test_write_sigmf_file(self):
        """Test writing SigMF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.sigmf-data"
            sink = FileOutputSink(
                filepath,
                sample_rate_hz=2e6,
                center_freq_hz=10e6,
                output_format=OutputFormat.COMPLEX64,
            )

            assert sink.open()

            # Write samples
            samples = np.random.randn(1000) + 1j * np.random.randn(1000)
            written = sink.write(samples.astype(np.complex64))
            assert written == 1000

            sink.close()

            # Verify both data and meta files exist
            assert filepath.exists()
            meta_path = Path(tmpdir) / "test.sigmf-meta"
            assert meta_path.exists()

    def test_context_manager(self):
        """Test using file sink as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.raw"

            with FileOutputSink(filepath, sample_rate_hz=48000.0) as sink:
                assert sink.is_open
                samples = np.random.randn(100) + 1j * np.random.randn(100)
                sink.write(samples.astype(np.complex64))

            assert not sink.is_open

    def test_duration_property(self):
        """Test duration calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.raw"
            sink = FileOutputSink(filepath, sample_rate_hz=1000.0)
            sink.open()

            # Write 500 samples at 1000 Hz = 0.5 seconds
            samples = np.zeros(500, dtype=np.complex64)
            sink.write(samples)

            assert sink.duration_seconds == 0.5

            sink.close()


class TestNetworkOutputSink:
    """Tests for NetworkOutputSink."""

    def test_create_zmq_sink(self):
        """Test creating ZMQ output sink."""
        sink = NetworkOutputSink(
            host="127.0.0.1",
            port=5556,
            protocol=NetworkProtocol.ZMQ_PUB,
            sample_rate_hz=2e6,
        )

        assert sink.host == "127.0.0.1"
        assert sink.port == 5556
        assert sink.protocol == NetworkProtocol.ZMQ_PUB
        assert sink.sample_rate == 2e6
        assert not sink.is_open

    def test_create_tcp_sink(self):
        """Test creating TCP output sink."""
        sink = NetworkOutputSink(
            host="0.0.0.0",
            port=5557,
            protocol=NetworkProtocol.TCP,
        )

        assert sink.protocol == NetworkProtocol.TCP

    def test_create_udp_sink(self):
        """Test creating UDP output sink."""
        sink = NetworkOutputSink(
            host="127.0.0.1",
            port=5558,
            protocol=NetworkProtocol.UDP,
        )

        assert sink.protocol == NetworkProtocol.UDP

    @pytest.mark.skipif(
        not pytest.importorskip("zmq", reason="pyzmq not installed"),
        reason="pyzmq not installed",
    )
    def test_open_close_zmq(self):
        """Test opening and closing ZMQ sink."""
        sink = NetworkOutputSink(
            port=5560,
            protocol=NetworkProtocol.ZMQ_PUB,
        )

        assert sink.open()
        assert sink.is_open

        sink.close()
        assert not sink.is_open

    def test_buffer_fill(self):
        """Test buffer fill property."""
        sink = NetworkOutputSink(
            port=5561,
            protocol=NetworkProtocol.ZMQ_PUB,
            buffer_size=1000,
        )

        # Without opening, buffer should be empty
        assert sink.buffer_fill == 0.0


class TestAudioOutputSink:
    """Tests for AudioOutputSink."""

    def test_create_audio_sink(self):
        """Test creating audio output sink."""
        sink = AudioOutputSink(
            sample_rate_hz=48000.0,
            buffer_size=4096,
        )

        assert sink.sample_rate == 48000.0
        assert sink.device is None  # Default device
        assert not sink.is_open

    def test_list_devices(self):
        """Test listing audio devices."""
        # This just tests the interface - may return empty list
        devices = AudioOutputSink.list_devices()
        assert isinstance(devices, list)


class TestSDROutputSink:
    """Tests for SDROutputSink."""

    def test_create_sdr_sink(self):
        """Test creating SDR output sink."""
        sink = SDROutputSink(
            sample_rate_hz=2e6,
            center_freq_hz=10e6,
            tx_gain=40.0,
        )

        assert sink.sample_rate == 2e6
        assert sink.center_frequency == 10e6
        assert sink.tx_gain == 40.0
        assert not sink.is_open

    def test_enumerate_devices(self):
        """Test enumerating SDR devices."""
        # This just tests the interface - may return empty list
        devices = SDROutputSink.enumerate_devices()
        assert isinstance(devices, list)


class TestMultiplexOutputSink:
    """Tests for MultiplexOutputSink."""

    def test_create_multiplex_sink(self):
        """Test creating multiplex output sink."""
        sink = MultiplexOutputSink(sample_rate_hz=48000.0)

        assert sink.num_sinks == 0
        assert sink.sinks == []

    def test_add_remove_sinks(self):
        """Test adding and removing sinks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mux = MultiplexOutputSink(sample_rate_hz=48000.0)

            file1 = FileOutputSink(Path(tmpdir) / "test1.raw", sample_rate_hz=48000.0)
            file2 = FileOutputSink(Path(tmpdir) / "test2.raw", sample_rate_hz=48000.0)

            mux.add_sink(file1)
            assert mux.num_sinks == 1

            mux.add_sink(file2)
            assert mux.num_sinks == 2

            mux.remove_sink(file1)
            assert mux.num_sinks == 1

            mux.clear_sinks()
            assert mux.num_sinks == 0

    def test_multiplex_write(self):
        """Test writing to multiple sinks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = FileOutputSink(Path(tmpdir) / "test1.raw", sample_rate_hz=48000.0)
            file2 = FileOutputSink(Path(tmpdir) / "test2.raw", sample_rate_hz=48000.0)

            mux = MultiplexOutputSink(sinks=[file1, file2], sample_rate_hz=48000.0)

            assert mux.open()

            samples = np.random.randn(100) + 1j * np.random.randn(100)
            written = mux.write(samples.astype(np.complex64))

            assert written == 100
            assert file1.total_samples_written == 100
            assert file2.total_samples_written == 100

            mux.close()

    def test_get_sink_status(self):
        """Test getting status of all sinks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = FileOutputSink(Path(tmpdir) / "test1.raw", sample_rate_hz=48000.0)

            mux = MultiplexOutputSink(sinks=[file1], sample_rate_hz=48000.0)
            mux.open()

            status = mux.get_sink_status()
            assert len(status) == 1
            assert status[0]["type"] == "FileOutputSink"
            assert status[0]["is_open"] == True

            mux.close()


class TestTeeOutputSink:
    """Tests for TeeOutputSink."""

    def test_create_tee_sink(self):
        """Test creating tee output sink."""
        with tempfile.TemporaryDirectory() as tmpdir:
            primary = FileOutputSink(Path(tmpdir) / "primary.raw", sample_rate_hz=48000.0)
            secondary = FileOutputSink(Path(tmpdir) / "secondary.raw", sample_rate_hz=48000.0)

            tee = TeeOutputSink(primary, secondary, sample_rate_hz=48000.0)

            assert tee.primary == primary
            assert tee.secondary == secondary
            assert tee.num_sinks == 2


class TestOutputFormatConversion:
    """Tests for format conversion in OutputSink base class."""

    def test_convert_to_int16(self):
        """Test conversion to INT16_IQ format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.raw"
            sink = FileOutputSink(
                filepath,
                sample_rate_hz=48000.0,
                output_format=OutputFormat.INT16_IQ,
            )
            sink.open()

            # Create test samples
            samples = np.array([0.5 + 0.5j, -0.5 - 0.5j], dtype=np.complex64)
            sink.write(samples)
            sink.close()

            # Read back and verify
            data = np.fromfile(filepath, dtype=np.int16)
            # Should be interleaved I/Q
            assert len(data) == 4  # 2 samples * 2 components

    def test_convert_to_int8(self):
        """Test conversion to INT8_IQ format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.raw"
            sink = FileOutputSink(
                filepath,
                sample_rate_hz=48000.0,
                output_format=OutputFormat.INT8_IQ,
            )
            sink.open()

            samples = np.array([0.0 + 0.0j], dtype=np.complex64)  # Zero = center
            sink.write(samples)
            sink.close()

            data = np.fromfile(filepath, dtype=np.uint8)
            # Zero complex should map to ~127.5
            assert len(data) == 2
            assert 126 <= data[0] <= 128
            assert 126 <= data[1] <= 128

    def test_convert_to_float32_iq(self):
        """Test conversion to FLOAT32_IQ format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.raw"
            sink = FileOutputSink(
                filepath,
                sample_rate_hz=48000.0,
                output_format=OutputFormat.FLOAT32_IQ,
            )
            sink.open()

            samples = np.array([0.25 + 0.75j], dtype=np.complex64)
            sink.write(samples)
            sink.close()

            data = np.fromfile(filepath, dtype=np.float32)
            assert len(data) == 2
            assert np.isclose(data[0], 0.25, atol=1e-6)
            assert np.isclose(data[1], 0.75, atol=1e-6)

"""Tests for input sources."""

import tempfile
import wave
import struct
from pathlib import Path

import numpy as np
import pytest

from hfpathsim.input.base import InputSource, InputFormat
from hfpathsim.input.file import FileInputSource
from hfpathsim.input.network import NetworkInputSource, NetworkProtocol


class TestInputFormat:
    """Test input format conversion."""

    def test_format_enum(self):
        """Test InputFormat enum values."""
        assert InputFormat.COMPLEX64.value == "complex64"
        assert InputFormat.INT16_IQ.value == "int16_iq"
        assert InputFormat.INT8_IQ.value == "int8_iq"


class TestFileInputSource:
    """Test file input source."""

    def test_wav_file_stereo(self):
        """Test reading stereo WAV file as I/Q."""
        # Create test WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            # Write test WAV
            sample_rate = 48000
            duration = 0.1  # 100 ms
            n_samples = int(sample_rate * duration)

            with wave.open(wav_path, "w") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)

                # Write I/Q samples
                for i in range(n_samples):
                    # Sine wave on I, cosine on Q
                    t = i / sample_rate
                    freq = 1000
                    i_sample = int(32767 * np.sin(2 * np.pi * freq * t))
                    q_sample = int(32767 * np.cos(2 * np.pi * freq * t))
                    wf.writeframes(struct.pack("<hh", i_sample, q_sample))

            # Read back
            source = FileInputSource(wav_path)
            assert source.open()

            samples = source.read(1024)
            assert samples is not None
            assert len(samples) == 1024
            assert samples.dtype == np.complex64

            source.close()

        finally:
            Path(wav_path).unlink()

    def test_raw_file_complex64(self):
        """Test reading raw complex64 file."""
        with tempfile.NamedTemporaryFile(suffix=".cf32", delete=False) as f:
            raw_path = f.name

        try:
            # Write test data
            test_data = np.random.randn(4096) + 1j * np.random.randn(4096)
            test_data = test_data.astype(np.complex64)
            test_data.tofile(raw_path)

            # Read back
            source = FileInputSource(
                raw_path,
                sample_rate_hz=1e6,
                input_format=InputFormat.COMPLEX64,
            )
            assert source.open()

            samples = source.read(2048)
            assert samples is not None
            assert len(samples) == 2048
            assert np.allclose(samples, test_data[:2048])

            source.close()

        finally:
            Path(raw_path).unlink()

    def test_raw_file_int16_iq(self):
        """Test reading raw int16 I/Q file."""
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            raw_path = f.name

        try:
            # Write test data as interleaved int16
            n_samples = 2048
            test_i = (np.random.randn(n_samples) * 16000).astype(np.int16)
            test_q = (np.random.randn(n_samples) * 16000).astype(np.int16)

            interleaved = np.zeros(n_samples * 2, dtype=np.int16)
            interleaved[0::2] = test_i
            interleaved[1::2] = test_q
            interleaved.tofile(raw_path)

            # Read back
            source = FileInputSource(
                raw_path,
                sample_rate_hz=1e6,
                input_format=InputFormat.INT16_IQ,
            )
            assert source.open()

            samples = source.read(1024)
            assert samples is not None
            assert len(samples) == 1024
            assert samples.dtype == np.complex64

            # Check conversion is correct
            expected_i = test_i[:1024].astype(np.float32) / 32768.0
            expected_q = test_q[:1024].astype(np.float32) / 32768.0
            expected = expected_i + 1j * expected_q

            assert np.allclose(samples, expected, atol=1e-4)

            source.close()

        finally:
            Path(raw_path).unlink()

    def test_file_loop(self):
        """Test file looping."""
        with tempfile.NamedTemporaryFile(suffix=".cf32", delete=False) as f:
            raw_path = f.name

        try:
            # Write small test file
            test_data = np.arange(100, dtype=np.complex64)
            test_data.tofile(raw_path)

            source = FileInputSource(
                raw_path,
                sample_rate_hz=1e6,
                input_format=InputFormat.COMPLEX64,
                loop=True,
            )
            assert source.open()

            # Read more than file contains
            samples1 = source.read(100)
            samples2 = source.read(50)  # Should loop

            assert len(samples1) == 100
            assert len(samples2) == 50
            # First samples of second read should match first samples of file
            assert np.allclose(samples2, test_data[:50])

            source.close()

        finally:
            Path(raw_path).unlink()

    def test_file_not_found(self):
        """Test handling of missing file."""
        source = FileInputSource("/nonexistent/path/file.raw")
        assert not source.open()

    def test_context_manager(self):
        """Test context manager interface."""
        with tempfile.NamedTemporaryFile(suffix=".cf32", delete=False) as f:
            raw_path = f.name
            test_data = np.zeros(100, dtype=np.complex64)
            test_data.tofile(raw_path)

        try:
            source = FileInputSource(
                raw_path,
                sample_rate_hz=1e6,
                input_format=InputFormat.COMPLEX64,
            )

            with source:
                assert source.is_open
                samples = source.read(50)
                assert len(samples) == 50

            assert not source.is_open

        finally:
            Path(raw_path).unlink()


class TestNetworkInputSource:
    """Test network input source."""

    def test_tcp_init(self):
        """Test TCP source initialization."""
        source = NetworkInputSource(
            host="127.0.0.1",
            port=12345,
            protocol=NetworkProtocol.TCP,
            sample_rate_hz=2e6,
        )

        assert source.sample_rate == 2e6
        assert source._protocol == NetworkProtocol.TCP

    def test_udp_init(self):
        """Test UDP source initialization."""
        source = NetworkInputSource(
            host="0.0.0.0",
            port=5555,
            protocol=NetworkProtocol.UDP,
        )

        assert source._protocol == NetworkProtocol.UDP

    def test_zmq_init(self):
        """Test ZMQ source initialization."""
        source = NetworkInputSource(
            host="127.0.0.1",
            port=5556,
            protocol=NetworkProtocol.ZMQ_SUB,
        )

        assert source._protocol == NetworkProtocol.ZMQ_SUB

    def test_buffer_fill(self):
        """Test buffer fill calculation."""
        source = NetworkInputSource(buffer_size=10000)

        # Initially empty
        assert source.buffer_fill == 0.0


class TestInputSourceBase:
    """Test InputSource base class."""

    def test_format_conversion_int16(self):
        """Test int16 I/Q format conversion."""
        # Create minimal concrete implementation for testing
        class TestSource(InputSource):
            def open(self):
                return True

            def close(self):
                pass

            def read(self, n):
                return None

            def available(self):
                return 0

        source = TestSource(
            sample_rate_hz=1e6,
            input_format=InputFormat.INT16_IQ,
        )

        # Test conversion
        raw = np.array([1000, 2000, -1000, -2000], dtype=np.int16)
        converted = source._convert_format(raw)

        assert len(converted) == 2
        assert converted.dtype == np.complex64

        # Check values
        expected_i = np.array([1000, -1000], dtype=np.float32) / 32768.0
        expected_q = np.array([2000, -2000], dtype=np.float32) / 32768.0

        assert np.allclose(converted.real, expected_i)
        assert np.allclose(converted.imag, expected_q)

    def test_format_conversion_int8(self):
        """Test int8 I/Q format conversion (RTL-SDR style)."""

        class TestSource(InputSource):
            def open(self):
                return True

            def close(self):
                pass

            def read(self, n):
                return None

            def available(self):
                return 0

        source = TestSource(
            sample_rate_hz=1e6,
            input_format=InputFormat.INT8_IQ,
        )

        # RTL-SDR uses unsigned 8-bit with 127.5 offset
        raw = np.array([127, 127, 255, 0], dtype=np.uint8)
        converted = source._convert_format(raw)

        assert len(converted) == 2
        # First sample: (127-127.5)/128 + j*(127-127.5)/128 ~ -0.004
        assert np.abs(converted[0].real) < 0.01
        assert np.abs(converted[0].imag) < 0.01
        # Second sample: (255-127.5)/128 + j*(0-127.5)/128 ~ 0.996 - j*0.996
        assert np.abs(converted[1].real - 0.996) < 0.01
        assert np.abs(converted[1].imag + 0.996) < 0.01

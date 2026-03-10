"""Network input sources for HF Path Simulator."""

import socket
import threading
from collections import deque
from enum import Enum
from typing import Optional
import numpy as np

from .base import InputSource, InputFormat


class NetworkProtocol(Enum):
    """Supported network protocols."""

    TCP = "tcp"
    UDP = "udp"
    ZMQ_SUB = "zmq_sub"  # ZeroMQ subscriber


class NetworkInputSource(InputSource):
    """Input source from network stream (TCP, UDP, ZMQ).

    Receives I/Q samples over the network from remote SDRs,
    GNU Radio flowgraphs, or other signal sources.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        protocol: NetworkProtocol = NetworkProtocol.TCP,
        sample_rate_hz: float = 2_000_000,
        center_freq_hz: float = 0.0,
        input_format: InputFormat = InputFormat.COMPLEX64,
        buffer_size: int = 1_000_000,
    ):
        """Initialize network input source.

        Args:
            host: Host address to connect to
            port: Port number
            protocol: Network protocol
            sample_rate_hz: Expected sample rate
            center_freq_hz: Center frequency
            input_format: Expected data format
            buffer_size: Maximum samples to buffer
        """
        super().__init__(sample_rate_hz, center_freq_hz, input_format)

        self._host = host
        self._port = port
        self._protocol = protocol
        self._buffer_size = buffer_size

        # Networking
        self._socket = None
        self._zmq_socket = None
        self._zmq_context = None

        # Threading
        self._receive_thread = None
        self._running = False
        self._lock = threading.Lock()

        # Buffer
        self._buffer = deque(maxlen=buffer_size)

    def open(self) -> bool:
        """Open network connection and start receiving."""
        try:
            if self._protocol == NetworkProtocol.TCP:
                return self._open_tcp()
            elif self._protocol == NetworkProtocol.UDP:
                return self._open_udp()
            elif self._protocol == NetworkProtocol.ZMQ_SUB:
                return self._open_zmq()
            else:
                return False

        except Exception as e:
            print(f"Error opening network connection: {e}")
            return False

    def _open_tcp(self) -> bool:
        """Open TCP connection."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(5.0)

        try:
            self._socket.connect((self._host, self._port))
        except socket.timeout:
            print(f"Connection timeout: {self._host}:{self._port}")
            return False
        except ConnectionRefusedError:
            print(f"Connection refused: {self._host}:{self._port}")
            return False

        self._socket.settimeout(0.1)  # Non-blocking reads
        self._start_receive_thread()
        self._is_open = True
        return True

    def _open_udp(self) -> bool:
        """Open UDP socket."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind(("0.0.0.0", self._port))
        self._socket.settimeout(0.1)

        self._start_receive_thread()
        self._is_open = True
        return True

    def _open_zmq(self) -> bool:
        """Open ZeroMQ subscriber socket."""
        try:
            import zmq

            self._zmq_context = zmq.Context()
            self._zmq_socket = self._zmq_context.socket(zmq.SUB)
            self._zmq_socket.connect(f"tcp://{self._host}:{self._port}")
            self._zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self._zmq_socket.setsockopt(zmq.RCVTIMEO, 100)

            self._start_receive_thread()
            self._is_open = True
            return True

        except ImportError:
            print("ZeroMQ not installed. Install with: pip install pyzmq")
            return False

    def _start_receive_thread(self):
        """Start background receive thread."""
        self._running = True
        self._receive_thread = threading.Thread(
            target=self._receive_loop, daemon=True
        )
        self._receive_thread.start()

    def _receive_loop(self):
        """Background thread to receive data."""
        # Determine bytes per sample
        fmt = self._input_format
        if fmt == InputFormat.COMPLEX64:
            dtype = np.complex64
            bytes_per_sample = 8
        elif fmt == InputFormat.INT16_IQ:
            dtype = np.int16
            bytes_per_sample = 4
        elif fmt == InputFormat.INT8_IQ:
            dtype = np.uint8
            bytes_per_sample = 2
        elif fmt == InputFormat.FLOAT32_IQ:
            dtype = np.float32
            bytes_per_sample = 8
        else:
            dtype = np.complex64
            bytes_per_sample = 8

        buffer = bytearray()
        chunk_samples = 4096
        chunk_bytes = chunk_samples * bytes_per_sample

        while self._running:
            try:
                if self._protocol == NetworkProtocol.ZMQ_SUB:
                    data = self._zmq_socket.recv()
                else:
                    data = self._socket.recv(65536)

                if not data:
                    continue

                buffer.extend(data)

                # Process complete chunks
                while len(buffer) >= chunk_bytes:
                    chunk = bytes(buffer[:chunk_bytes])
                    del buffer[:chunk_bytes]

                    raw = np.frombuffer(chunk, dtype=dtype)
                    samples = self._convert_format(raw)

                    with self._lock:
                        self._buffer.extend(samples)

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"Receive error: {e}")
                break

    def close(self):
        """Close network connection."""
        self._running = False

        if self._receive_thread:
            self._receive_thread.join(timeout=1.0)

        if self._socket:
            self._socket.close()
            self._socket = None

        if self._zmq_socket:
            self._zmq_socket.close()
            self._zmq_socket = None

        if self._zmq_context:
            self._zmq_context.term()
            self._zmq_context = None

        self._is_open = False

    def read(self, num_samples: int) -> Optional[np.ndarray]:
        """Read buffered samples."""
        if not self._is_open:
            return None

        with self._lock:
            available = len(self._buffer)
            to_read = min(num_samples, available)

            if to_read == 0:
                return np.array([], dtype=np.complex64)

            samples = np.array(
                [self._buffer.popleft() for _ in range(to_read)],
                dtype=np.complex64,
            )

        self._total_samples_read += len(samples)
        return samples

    def available(self) -> int:
        """Return samples available in buffer."""
        with self._lock:
            return len(self._buffer)

    @property
    def buffer_fill(self) -> float:
        """Return buffer fill percentage."""
        with self._lock:
            return len(self._buffer) / self._buffer_size * 100

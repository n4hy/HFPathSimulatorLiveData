"""Network output sinks for HF Path Simulator."""

import socket
import threading
from collections import deque
from enum import Enum
from typing import Optional, List
import numpy as np

from .base import OutputSink, OutputFormat


class NetworkProtocol(Enum):
    """Supported network protocols for output."""

    TCP = "tcp"  # TCP server (accepts connections)
    UDP = "udp"  # UDP sender
    ZMQ_PUB = "zmq_pub"  # ZeroMQ publisher


class NetworkOutputSink(OutputSink):
    """Output sink to network stream (TCP, UDP, ZMQ).

    Sends I/Q samples over the network to GNU Radio flowgraphs,
    remote receivers, or other signal processing tools.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5556,
        protocol: NetworkProtocol = NetworkProtocol.ZMQ_PUB,
        sample_rate_hz: float = 2_000_000,
        center_freq_hz: float = 0.0,
        output_format: OutputFormat = OutputFormat.COMPLEX64,
        buffer_size: int = 1_000_000,
    ):
        """Initialize network output sink.

        Args:
            host: Host address to bind to (for TCP/ZMQ) or send to (for UDP)
            port: Port number
            protocol: Network protocol
            sample_rate_hz: Sample rate
            center_freq_hz: Center frequency
            output_format: Output data format
            buffer_size: Maximum samples to buffer
        """
        super().__init__(sample_rate_hz, center_freq_hz, output_format, buffer_size)

        self._host = host
        self._port = port
        self._protocol = protocol

        # Networking
        self._socket: Optional[socket.socket] = None
        self._server_socket: Optional[socket.socket] = None
        self._zmq_socket = None
        self._zmq_context = None

        # TCP clients
        self._clients: List[socket.socket] = []
        self._accept_thread: Optional[threading.Thread] = None

        # Threading
        self._send_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Buffer
        self._buffer: deque = deque(maxlen=buffer_size)
        self._send_event = threading.Event()

    @property
    def host(self) -> str:
        """Return host address."""
        return self._host

    @property
    def port(self) -> int:
        """Return port number."""
        return self._port

    @property
    def protocol(self) -> NetworkProtocol:
        """Return network protocol."""
        return self._protocol

    @property
    def num_clients(self) -> int:
        """Return number of connected TCP clients."""
        with self._lock:
            return len(self._clients)

    def open(self) -> bool:
        """Open network connection and start sending."""
        try:
            if self._protocol == NetworkProtocol.TCP:
                return self._open_tcp()
            elif self._protocol == NetworkProtocol.UDP:
                return self._open_udp()
            elif self._protocol == NetworkProtocol.ZMQ_PUB:
                return self._open_zmq()
            else:
                return False

        except Exception as e:
            print(f"Error opening network output: {e}")
            return False

    def _open_tcp(self) -> bool:
        """Open TCP server."""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self._host, self._port))
        self._server_socket.listen(5)
        self._server_socket.settimeout(0.5)

        # Start accept thread
        self._running = True
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True
        )
        self._accept_thread.start()

        # Start send thread
        self._start_send_thread()

        self._is_open = True
        return True

    def _accept_loop(self):
        """Background thread to accept TCP connections."""
        while self._running:
            try:
                client, addr = self._server_socket.accept()
                client.setblocking(False)
                with self._lock:
                    self._clients.append(client)
                print(f"Client connected: {addr}")
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"Accept error: {e}")
                break

    def _open_udp(self) -> bool:
        """Open UDP sender socket."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._start_send_thread()
        self._is_open = True
        return True

    def _open_zmq(self) -> bool:
        """Open ZeroMQ publisher socket."""
        try:
            import zmq

            self._zmq_context = zmq.Context()
            self._zmq_socket = self._zmq_context.socket(zmq.PUB)
            self._zmq_socket.bind(f"tcp://{self._host}:{self._port}")
            # Set high water mark to prevent memory buildup
            self._zmq_socket.setsockopt(zmq.SNDHWM, 1000)

            self._start_send_thread()
            self._is_open = True
            return True

        except ImportError:
            print("ZeroMQ not installed. Install with: pip install pyzmq")
            return False

    def _start_send_thread(self):
        """Start background send thread."""
        self._running = True
        self._send_thread = threading.Thread(
            target=self._send_loop, daemon=True
        )
        self._send_thread.start()

    def _send_loop(self):
        """Background thread to send data."""
        chunk_size = 4096  # samples per chunk

        while self._running:
            # Wait for data or timeout
            self._send_event.wait(timeout=0.1)
            self._send_event.clear()

            # Get data from buffer
            with self._lock:
                if len(self._buffer) < chunk_size:
                    continue

                samples = np.array(
                    [self._buffer.popleft() for _ in range(min(chunk_size, len(self._buffer)))],
                    dtype=np.complex64,
                )

            if len(samples) == 0:
                continue

            # Convert to output format
            data = self._convert_to_format(samples)
            data_bytes = data.tobytes()

            try:
                if self._protocol == NetworkProtocol.ZMQ_PUB:
                    self._zmq_socket.send(data_bytes, zmq.NOBLOCK)

                elif self._protocol == NetworkProtocol.UDP:
                    # UDP has max datagram size, split if needed
                    max_size = 65000
                    for i in range(0, len(data_bytes), max_size):
                        chunk = data_bytes[i:i + max_size]
                        self._socket.sendto(chunk, (self._host, self._port))

                elif self._protocol == NetworkProtocol.TCP:
                    # Send to all connected clients
                    with self._lock:
                        dead_clients = []
                        for client in self._clients:
                            try:
                                client.sendall(data_bytes)
                            except (BrokenPipeError, ConnectionResetError):
                                dead_clients.append(client)
                            except BlockingIOError:
                                pass  # Buffer full, skip this client

                        # Remove dead clients
                        for client in dead_clients:
                            self._clients.remove(client)
                            client.close()

            except Exception as e:
                if self._running:
                    print(f"Send error: {e}")

    def close(self):
        """Close network connection."""
        self._running = False
        self._send_event.set()

        if self._send_thread:
            self._send_thread.join(timeout=1.0)

        if self._accept_thread:
            self._accept_thread.join(timeout=1.0)

        # Close TCP clients
        with self._lock:
            for client in self._clients:
                try:
                    client.close()
                except Exception:
                    pass
            self._clients.clear()

        if self._server_socket:
            self._server_socket.close()
            self._server_socket = None

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

    def write(self, samples: np.ndarray) -> int:
        """Write samples to the network buffer."""
        if not self._is_open:
            return 0

        samples = samples.astype(np.complex64)

        with self._lock:
            space = self._buffer_size - len(self._buffer)
            to_write = min(len(samples), space)

            if to_write > 0:
                self._buffer.extend(samples[:to_write])
                self._total_samples_written += to_write
                self._send_event.set()

        return to_write

    def available(self) -> int:
        """Return samples that can be written without blocking."""
        with self._lock:
            return self._buffer_size - len(self._buffer)

    @property
    def buffer_fill(self) -> float:
        """Return buffer fill percentage."""
        with self._lock:
            return len(self._buffer) / self._buffer_size * 100


# Import zmq for NOBLOCK constant
try:
    import zmq
except ImportError:
    zmq = None

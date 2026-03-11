"""Flex Radio SmartSDR DAX IQ input source.

Implements VITA-49 IQ streaming from Flex Radio 6000/8000 series radios.

References:
- FlexRadio TCP/IP API: http://wiki.flexradio.com/index.php?title=SmartSDR_TCP/IP_API
- VITA-49 Specification: https://www.vita.com/
- https://github.com/AB4EJ-1/FlexRadioIQ
"""

import socket
import struct
import threading
import queue
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import IntEnum
import numpy as np

from .base import InputSource, InputFormat


class VITA49PacketType(IntEnum):
    """VITA-49 packet types."""
    IF_DATA_WITHOUT_STREAM_ID = 0
    IF_DATA_WITH_STREAM_ID = 1
    EXTENSION_DATA_WITHOUT_STREAM_ID = 2
    EXTENSION_DATA_WITH_STREAM_ID = 3
    IF_CONTEXT = 4
    EXTENSION_CONTEXT = 5


@dataclass
class VITA49Header:
    """Parsed VITA-49 packet header."""
    packet_type: int
    has_class_id: bool
    has_trailer: bool
    tsi: int  # Timestamp integer type
    tsf: int  # Timestamp fractional type
    packet_count: int
    packet_size: int  # In 32-bit words
    stream_id: Optional[int] = None
    class_id: Optional[int] = None
    timestamp_int: Optional[int] = None
    timestamp_frac: Optional[int] = None


def parse_vita49_header(data: bytes) -> Tuple[VITA49Header, int]:
    """Parse VITA-49 packet header.

    Args:
        data: Raw packet data

    Returns:
        Tuple of (header, payload_offset)
    """
    if len(data) < 4:
        raise ValueError("Packet too short for VITA-49 header")

    # First 32-bit word
    word0 = struct.unpack(">I", data[0:4])[0]

    packet_type = (word0 >> 28) & 0x0F
    has_class_id = bool((word0 >> 27) & 0x01)
    has_trailer = bool((word0 >> 26) & 0x01)
    # Bits 25-24: reserved
    tsi = (word0 >> 22) & 0x03
    tsf = (word0 >> 20) & 0x03
    packet_count = (word0 >> 16) & 0x0F
    packet_size = word0 & 0xFFFF

    offset = 4
    stream_id = None
    class_id = None
    timestamp_int = None
    timestamp_frac = None

    # Stream ID (if packet type indicates it)
    if packet_type in [1, 3]:
        if len(data) < offset + 4:
            raise ValueError("Packet too short for stream ID")
        stream_id = struct.unpack(">I", data[offset:offset + 4])[0]
        offset += 4

    # Class ID (optional, 64-bit)
    if has_class_id:
        if len(data) < offset + 8:
            raise ValueError("Packet too short for class ID")
        class_id = struct.unpack(">Q", data[offset:offset + 8])[0]
        offset += 8

    # Integer timestamp (if present)
    if tsi != 0:
        if len(data) < offset + 4:
            raise ValueError("Packet too short for integer timestamp")
        timestamp_int = struct.unpack(">I", data[offset:offset + 4])[0]
        offset += 4

    # Fractional timestamp (if present)
    if tsf != 0:
        if len(data) < offset + 8:
            raise ValueError("Packet too short for fractional timestamp")
        timestamp_frac = struct.unpack(">Q", data[offset:offset + 8])[0]
        offset += 8

    header = VITA49Header(
        packet_type=packet_type,
        has_class_id=has_class_id,
        has_trailer=has_trailer,
        tsi=tsi,
        tsf=tsf,
        packet_count=packet_count,
        packet_size=packet_size,
        stream_id=stream_id,
        class_id=class_id,
        timestamp_int=timestamp_int,
        timestamp_frac=timestamp_frac,
    )

    return header, offset


def extract_iq_samples(data: bytes, header: VITA49Header, payload_offset: int) -> np.ndarray:
    """Extract IQ samples from VITA-49 packet payload.

    FlexRadio sends interleaved float32 I/Q samples:
    - Even indices (0, 2, 4, ...): I samples
    - Odd indices (1, 3, 5, ...): Q samples

    Args:
        data: Raw packet data
        header: Parsed VITA-49 header
        payload_offset: Byte offset to payload start

    Returns:
        Complex64 numpy array of IQ samples
    """
    # Calculate payload size in bytes
    total_bytes = header.packet_size * 4  # packet_size is in 32-bit words
    trailer_size = 4 if header.has_trailer else 0
    payload_bytes = total_bytes - payload_offset - trailer_size

    if payload_bytes <= 0:
        return np.array([], dtype=np.complex64)

    # Extract float32 samples (big-endian from network)
    num_floats = payload_bytes // 4
    payload = data[payload_offset:payload_offset + payload_bytes]

    # Unpack as big-endian float32
    floats = np.frombuffer(payload, dtype=">f4").astype(np.float32)

    if len(floats) < 2:
        return np.array([], dtype=np.complex64)

    # Interleaved I/Q: reshape and combine
    num_samples = len(floats) // 2
    i_samples = floats[0::2][:num_samples]
    q_samples = floats[1::2][:num_samples]

    return (i_samples + 1j * q_samples).astype(np.complex64)


class FlexRadioClient:
    """TCP client for Flex Radio SmartSDR API commands."""

    def __init__(self, host: str, port: int = 4992):
        """Initialize Flex Radio client.

        Args:
            host: Radio IP address
            port: API TCP port (default 4992)
        """
        self.host = host
        self.port = port
        self._socket: Optional[socket.socket] = None
        self._sequence = 0
        self._connected = False
        self._stream_id: Optional[str] = None
        self._receive_buffer = ""
        self._lock = threading.Lock()

    def connect(self) -> bool:
        """Connect to radio API."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(5.0)
            self._socket.connect((self.host, self.port))
            self._connected = True

            # Read initial banner/status messages
            self._receive_responses(timeout=1.0)

            return True
        except Exception as e:
            print(f"FlexRadio connection error: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from radio."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
        self._socket = None
        self._connected = False

    def _send_command(self, command: str) -> str:
        """Send command and get response.

        Args:
            command: Command string (without sequence prefix)

        Returns:
            Response string
        """
        if not self._connected or not self._socket:
            raise RuntimeError("Not connected to radio")

        with self._lock:
            seq = self._sequence
            self._sequence += 1

            full_command = f"C{seq}|{command}\n"
            self._socket.sendall(full_command.encode("utf-8"))

            # Wait for response with matching sequence
            return self._receive_responses(expected_seq=seq, timeout=5.0)

    def _receive_responses(
        self, expected_seq: Optional[int] = None, timeout: float = 1.0
    ) -> str:
        """Receive responses from radio.

        Args:
            expected_seq: Expected sequence number (None for any)
            timeout: Receive timeout in seconds

        Returns:
            Response for expected sequence
        """
        self._socket.settimeout(timeout)
        result = ""

        try:
            while True:
                data = self._socket.recv(4096).decode("utf-8")
                if not data:
                    break

                self._receive_buffer += data

                # Process complete lines
                while "\n" in self._receive_buffer:
                    line, self._receive_buffer = self._receive_buffer.split("\n", 1)
                    line = line.strip()

                    if not line:
                        continue

                    # Parse response: R<seq>|<code>|<message>
                    if line.startswith("R") and "|" in line:
                        parts = line.split("|", 2)
                        if len(parts) >= 2:
                            try:
                                seq = int(parts[0][1:])
                                if expected_seq is None or seq == expected_seq:
                                    result = line
                                    if expected_seq is not None:
                                        return result
                            except ValueError:
                                pass

                    # Status messages (S prefix) - just log
                    elif line.startswith("S"):
                        pass  # Status update, could log if needed

        except socket.timeout:
            pass

        return result

    def create_iq_stream(self, dax_channel: int, udp_port: int) -> Optional[str]:
        """Create DAX IQ stream.

        Args:
            dax_channel: DAX IQ channel number (1-4)
            udp_port: UDP port for IQ data

        Returns:
            Stream ID (hex string) or None on failure
        """
        # Set client UDP port
        response = self._send_command(f"client udpport {udp_port}")

        # Create DAX IQ stream
        response = self._send_command(f"stream create daxiq={dax_channel} port={udp_port}")

        # Parse stream ID from response: R<seq>|0|<stream_id>
        if "|0|" in response:
            parts = response.split("|")
            if len(parts) >= 3:
                self._stream_id = parts[2].strip()
                return self._stream_id

        return None

    def remove_iq_stream(self, stream_id: Optional[str] = None):
        """Remove DAX IQ stream.

        Args:
            stream_id: Stream ID to remove (uses last created if None)
        """
        sid = stream_id or self._stream_id
        if sid:
            try:
                self._send_command(f"stream remove {sid}")
            except Exception:
                pass
            self._stream_id = None

    def set_slice_frequency(self, slice_num: int, freq_mhz: float):
        """Set slice frequency.

        Args:
            slice_num: Slice number
            freq_mhz: Frequency in MHz
        """
        self._send_command(f"slice tune {slice_num} {freq_mhz}")

    def get_radio_info(self) -> Dict[str, Any]:
        """Get radio information."""
        response = self._send_command("radio info")
        # Parse response into dict
        info = {}
        if "|0|" in response:
            parts = response.split("|", 2)
            if len(parts) >= 3:
                for item in parts[2].split():
                    if "=" in item:
                        key, value = item.split("=", 1)
                        info[key] = value
        return info


class FlexRadioInputSource(InputSource):
    """Input source for Flex Radio DAX IQ streams.

    Supports Flex Radio 6000/8000 series via SmartSDR TCP/IP API.
    Uses VITA-49 packet format for IQ streaming over UDP.

    Example:
        source = FlexRadioInputSource(
            host="192.168.1.68",
            dax_channel=1,
            sample_rate_hz=48000,
        )
        source.open()
        samples = source.read(4096)
        source.close()
    """

    # Standard DAX IQ sample rates
    SAMPLE_RATES = [24000, 48000, 96000, 192000]

    def __init__(
        self,
        host: str,
        dax_channel: int = 1,
        sample_rate_hz: float = 48000,
        center_freq_hz: float = 14_000_000,
        udp_port: int = 7791,
        api_port: int = 4992,
        buffer_size: int = 65536,
    ):
        """Initialize Flex Radio input source.

        Args:
            host: Radio IP address
            dax_channel: DAX IQ channel (1-4)
            sample_rate_hz: Sample rate (24000, 48000, 96000, or 192000)
            center_freq_hz: Center frequency in Hz
            udp_port: UDP port for IQ data
            api_port: TCP API port (default 4992)
            buffer_size: Internal buffer size in samples
        """
        super().__init__(sample_rate_hz, center_freq_hz, InputFormat.COMPLEX64)

        self._host = host
        self._dax_channel = dax_channel
        self._udp_port = udp_port
        self._api_port = api_port
        self._buffer_size = buffer_size

        # Validate sample rate
        if int(sample_rate_hz) not in self.SAMPLE_RATES:
            print(f"Warning: Non-standard sample rate {sample_rate_hz}. "
                  f"Standard rates: {self.SAMPLE_RATES}")

        # Client and networking
        self._client: Optional[FlexRadioClient] = None
        self._udp_socket: Optional[socket.socket] = None
        self._stream_id: Optional[str] = None

        # Threading
        self._receive_thread: Optional[threading.Thread] = None
        self._running = False
        self._sample_queue: queue.Queue = queue.Queue(maxsize=100)

        # Statistics
        self._packets_received = 0
        self._packets_dropped = 0
        self._last_packet_count = -1

    def open(self) -> bool:
        """Open connection to Flex Radio and start IQ stream."""
        try:
            # Connect to radio API
            self._client = FlexRadioClient(self._host, self._api_port)
            if not self._client.connect():
                print(f"Failed to connect to Flex Radio at {self._host}:{self._api_port}")
                return False

            # Setup UDP socket for IQ data
            self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            self._udp_socket.bind(("0.0.0.0", self._udp_port))
            self._udp_socket.settimeout(0.1)

            # Create IQ stream
            self._stream_id = self._client.create_iq_stream(
                self._dax_channel, self._udp_port
            )
            if not self._stream_id:
                print("Failed to create DAX IQ stream")
                self._cleanup()
                return False

            print(f"Created DAX IQ stream: {self._stream_id}")

            # Start receive thread
            self._running = True
            self._receive_thread = threading.Thread(
                target=self._receive_loop, daemon=True
            )
            self._receive_thread.start()

            self._is_open = True
            return True

        except Exception as e:
            print(f"Error opening Flex Radio: {e}")
            self._cleanup()
            return False

    def close(self):
        """Close connection and stop stream."""
        self._running = False

        if self._receive_thread:
            self._receive_thread.join(timeout=2.0)
            self._receive_thread = None

        if self._client and self._stream_id:
            try:
                self._client.remove_iq_stream(self._stream_id)
            except Exception:
                pass

        self._cleanup()
        self._is_open = False

    def _cleanup(self):
        """Clean up resources."""
        if self._udp_socket:
            try:
                self._udp_socket.close()
            except Exception:
                pass
            self._udp_socket = None

        if self._client:
            self._client.disconnect()
            self._client = None

        self._stream_id = None

    def _receive_loop(self):
        """Background thread for receiving IQ data."""
        while self._running:
            try:
                data, addr = self._udp_socket.recvfrom(8192)

                if len(data) < 4:
                    continue

                # Parse VITA-49 header
                try:
                    header, offset = parse_vita49_header(data)
                except ValueError:
                    continue

                # Check for dropped packets
                if self._last_packet_count >= 0:
                    expected = (self._last_packet_count + 1) & 0x0F
                    if header.packet_count != expected:
                        self._packets_dropped += 1
                self._last_packet_count = header.packet_count

                # Extract IQ samples
                samples = extract_iq_samples(data, header, offset)

                if len(samples) > 0:
                    try:
                        self._sample_queue.put_nowait(samples)
                        self._packets_received += 1
                    except queue.Full:
                        self._packets_dropped += 1

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"Receive error: {e}")

    def read(self, num_samples: int) -> Optional[np.ndarray]:
        """Read IQ samples from stream.

        Args:
            num_samples: Number of samples to read

        Returns:
            Complex64 numpy array or None
        """
        if not self._is_open:
            return None

        samples_list = []
        samples_needed = num_samples

        # Collect samples from queue
        while samples_needed > 0:
            try:
                chunk = self._sample_queue.get(timeout=0.1)
                samples_list.append(chunk)
                samples_needed -= len(chunk)
            except queue.Empty:
                break

        if not samples_list:
            return np.array([], dtype=np.complex64)

        # Combine and trim
        result = np.concatenate(samples_list)
        if len(result) > num_samples:
            # Put excess back
            excess = result[num_samples:]
            result = result[:num_samples]
            try:
                self._sample_queue.put_nowait(excess)
            except queue.Full:
                pass

        self._total_samples_read += len(result)
        return result

    def available(self) -> int:
        """Return approximate samples available."""
        return self._sample_queue.qsize() * 128  # ~128 samples per packet

    def set_frequency(self, freq_hz: float, slice_num: int = 0):
        """Set center frequency via slice tuning.

        Args:
            freq_hz: Frequency in Hz
            slice_num: Slice number to tune
        """
        if self._client:
            self._client.set_slice_frequency(slice_num, freq_hz / 1e6)
            self._center_freq = freq_hz

    def get_statistics(self) -> Dict[str, Any]:
        """Get stream statistics.

        Returns:
            Dict with packets_received, packets_dropped, samples_read
        """
        return {
            "packets_received": self._packets_received,
            "packets_dropped": self._packets_dropped,
            "samples_read": self._total_samples_read,
            "queue_size": self._sample_queue.qsize(),
            "stream_id": self._stream_id,
        }

    @staticmethod
    def discover_radios(timeout: float = 2.0) -> List[Dict[str, str]]:
        """Discover Flex Radios on the network via UDP broadcast.

        Args:
            timeout: Discovery timeout in seconds

        Returns:
            List of discovered radio info dicts
        """
        radios = []

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(timeout)
            sock.bind(("", 4992))

            # Listen for discovery broadcasts
            end_time = __import__("time").time() + timeout
            while __import__("time").time() < end_time:
                try:
                    data, addr = sock.recvfrom(4096)
                    info = {"ip": addr[0], "raw": data.decode("utf-8", errors="ignore")}

                    # Parse discovery packet
                    for line in info["raw"].split():
                        if "=" in line:
                            key, value = line.split("=", 1)
                            info[key] = value

                    radios.append(info)
                except socket.timeout:
                    break

            sock.close()

        except Exception as e:
            print(f"Discovery error: {e}")

        return radios

"""Audio output sink for HF Path Simulator."""

import threading
from collections import deque
from typing import Optional, List, Dict, Any
import numpy as np

from .base import OutputSink, OutputFormat


class AudioOutputSink(OutputSink):
    """Output sink to sound card via sounddevice.

    Outputs I/Q samples as stereo audio (I=left, Q=right).
    Useful for SDR applications and audio processing.
    """

    def __init__(
        self,
        device: Optional[int] = None,
        sample_rate_hz: float = 48000.0,
        output_format: OutputFormat = OutputFormat.FLOAT32_IQ,
        buffer_size: int = 8192,
        blocksize: int = 1024,
        latency: str = "low",
    ):
        """Initialize audio output sink.

        Args:
            device: Audio device index (None for default)
            sample_rate_hz: Sample rate in Hz
            output_format: Output format (FLOAT32_IQ recommended for audio)
            buffer_size: Internal buffer size in samples
            blocksize: Audio block size
            latency: Latency setting ('low', 'high', or seconds)
        """
        super().__init__(sample_rate_hz, 0.0, output_format, buffer_size)

        self._device = device
        self._blocksize = blocksize
        self._latency = latency

        # Audio stream
        self._stream = None
        self._sd = None  # sounddevice module

        # Buffer
        self._buffer: deque = deque(maxlen=buffer_size)
        self._lock = threading.Lock()

        # Underrun tracking
        self._underruns = 0

    @property
    def device(self) -> Optional[int]:
        """Return audio device index."""
        return self._device

    @property
    def underruns(self) -> int:
        """Return number of buffer underruns."""
        return self._underruns

    @classmethod
    def list_devices(cls) -> List[Dict[str, Any]]:
        """List available audio output devices.

        Returns:
            List of device dictionaries
        """
        try:
            import sounddevice as sd

            devices = []
            for i, dev in enumerate(sd.query_devices()):
                if dev["max_output_channels"] >= 2:
                    devices.append({
                        "index": i,
                        "name": dev["name"],
                        "channels": dev["max_output_channels"],
                        "default_samplerate": dev["default_samplerate"],
                        "hostapi": sd.query_hostapis(dev["hostapi"])["name"],
                    })
            return devices

        except ImportError:
            print("sounddevice not installed. Install with: pip install sounddevice")
            return []

    def open(self) -> bool:
        """Open audio output stream."""
        try:
            import sounddevice as sd
            self._sd = sd

            # Create output stream
            self._stream = sd.OutputStream(
                device=self._device,
                samplerate=self._sample_rate,
                channels=2,  # Stereo I/Q
                dtype="float32",
                blocksize=self._blocksize,
                latency=self._latency,
                callback=self._audio_callback,
            )

            self._stream.start()
            self._is_open = True
            return True

        except ImportError:
            print("sounddevice not installed. Install with: pip install sounddevice")
            return False

        except Exception as e:
            print(f"Error opening audio output: {e}")
            return False

    def _audio_callback(self, outdata, frames, time_info, status):
        """Sounddevice callback to fill audio buffer."""
        if status:
            if status.output_underflow:
                self._underruns += 1

        with self._lock:
            available = len(self._buffer)
            to_read = min(frames, available)

            if to_read > 0:
                samples = np.array(
                    [self._buffer.popleft() for _ in range(to_read)],
                    dtype=np.complex64,
                )

                # Convert to stereo float32 (I=left, Q=right)
                outdata[:to_read, 0] = np.real(samples).astype(np.float32)
                outdata[:to_read, 1] = np.imag(samples).astype(np.float32)

            # Zero-fill any remaining frames
            if to_read < frames:
                outdata[to_read:] = 0

    def close(self):
        """Close audio stream."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._is_open = False

    def write(self, samples: np.ndarray) -> int:
        """Write samples to audio buffer."""
        if not self._is_open:
            return 0

        samples = samples.astype(np.complex64)

        # Normalize to [-1, 1] range for audio
        max_val = np.max(np.abs(samples))
        if max_val > 1.0:
            samples = samples / max_val

        with self._lock:
            space = self._buffer_size - len(self._buffer)
            to_write = min(len(samples), space)

            if to_write > 0:
                self._buffer.extend(samples[:to_write])
                self._total_samples_written += to_write

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

    @property
    def latency_seconds(self) -> float:
        """Return current output latency in seconds."""
        if self._stream and hasattr(self._stream, "latency"):
            return self._stream.latency
        return 0.0

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about current audio device.

        Returns:
            Device information dictionary
        """
        if self._sd is None:
            return {}

        try:
            if self._device is not None:
                dev = self._sd.query_devices(self._device)
            else:
                dev = self._sd.query_devices(kind="output")

            return {
                "name": dev["name"],
                "channels": dev["max_output_channels"],
                "default_samplerate": dev["default_samplerate"],
                "hostapi": self._sd.query_hostapis(dev["hostapi"])["name"],
            }

        except Exception:
            return {}

"""Audio output sink for HF Path Simulator.

Lock-free single-producer single-consumer ring buffer design.
Buffer is sized large enough to never overflow under normal use.
"""

from typing import Optional, List, Dict, Any
import numpy as np

from .base import OutputSink, OutputFormat


class AudioOutputSink(OutputSink):
    """Output sink to sound card via sounddevice.

    Uses a lock-free SPSC ring buffer:
    - Main thread is the single producer (writer)
    - Audio callback is the single consumer (reader)
    - No locks needed - pointers are updated atomically by single owner
    - Buffer sized large enough to never overflow
    """

    def __init__(
        self,
        device: Optional[int] = None,
        sample_rate_hz: float = 48000.0,
        output_format: OutputFormat = OutputFormat.FLOAT32_IQ,
        buffer_size: int = 1048576,  # 1M samples = 131 seconds at 8kHz
        blocksize: int = 256,
        latency: str = "high",
    ):
        super().__init__(sample_rate_hz, 0.0, output_format, buffer_size)

        self._device = device
        self._blocksize = blocksize
        self._latency = latency

        # Audio stream
        self._stream = None
        self._sd = None

        # Lock-free SPSC ring buffer
        # Buffer is pre-allocated, never reallocated
        self._buffer = np.zeros(buffer_size, dtype=np.complex64)

        # Single writer updates write_ptr only
        # Single reader updates read_ptr only
        # Both can read both pointers to compute available/space
        self._write_ptr = 0
        self._read_ptr = 0

        # Stats
        self._underruns = 0
        self._output_gain = 0.5

    def _available_to_read(self) -> int:
        """Samples available for reading (called by reader)."""
        wp = self._write_ptr
        rp = self._read_ptr
        if wp >= rp:
            return wp - rp
        else:
            return self._buffer_size - rp + wp

    def _available_to_write(self) -> int:
        """Space available for writing (called by writer)."""
        # Leave one slot empty to distinguish full from empty
        return self._buffer_size - 1 - self._available_to_read()

    @property
    def device(self) -> Optional[int]:
        return self._device

    @property
    def underruns(self) -> int:
        return self._underruns

    @classmethod
    def list_devices(cls) -> List[Dict[str, Any]]:
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
            return []

    def open(self) -> bool:
        try:
            import sounddevice as sd
            self._sd = sd

            # Reset buffer and pointers
            self._buffer.fill(0)
            self._write_ptr = 0
            self._read_ptr = 0
            self._underruns = 0

            self._stream = sd.OutputStream(
                device=self._device,
                samplerate=self._sample_rate,
                channels=2,
                dtype="float32",
                blocksize=self._blocksize,
                latency=self._latency,
                callback=self._audio_callback,
            )

            self._stream.start()
            self._is_open = True
            return True

        except ImportError:
            print("sounddevice not installed")
            return False
        except Exception as e:
            print(f"Error opening audio: {e}")
            return False

    def _audio_callback(self, outdata, frames, time_info, status):
        """Audio callback - single reader, no locks."""
        if status and status.output_underflow:
            self._underruns += 1

        available = self._available_to_read()
        to_read = min(frames, available)

        if to_read > 0:
            rp = self._read_ptr

            # Read from ring buffer
            end = rp + to_read
            if end <= self._buffer_size:
                # Contiguous read
                samples = self._buffer[rp:end]
            else:
                # Wrap-around read
                first = self._buffer_size - rp
                samples = np.concatenate([
                    self._buffer[rp:],
                    self._buffer[:to_read - first]
                ])

            # Update read pointer (only reader touches this)
            self._read_ptr = end % self._buffer_size

            # Output as stereo float32
            outdata[:to_read, 0] = np.real(samples).astype(np.float32)
            outdata[:to_read, 1] = np.imag(samples).astype(np.float32)

        # Zero-fill remainder if underrun
        if to_read < frames:
            outdata[to_read:] = 0

    def close(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._is_open = False

    def clear(self):
        """Clear the ring buffer immediately (flush all pending audio)."""
        self._buffer.fill(0)
        self._write_ptr = 0
        self._read_ptr = 0

    def write(self, samples: np.ndarray) -> int:
        """Write samples to ring buffer - single writer, no locks.

        Write pointer NEVER advances to meet or pass read pointer.
        If buffer is full, samples are dropped (not written).
        """
        if not self._is_open:
            return 0

        samples = (samples * self._output_gain).astype(np.complex64)

        # Simple hard clip to prevent overflow (no fancy soft clipping for now)
        mag = np.abs(samples)
        mask = mag > 1.0
        if np.any(mask):
            samples[mask] = samples[mask] / mag[mask]

        n = len(samples)

        # Snapshot read pointer (reader can only advance it, making more space)
        rp = self._read_ptr
        wp = self._write_ptr

        # Calculate space WITHOUT allowing write to catch read
        # Keep at least 1 sample gap
        if wp >= rp:
            # Write ahead of read: can write to end, then from start to read-1
            space = (self._buffer_size - wp) + (rp - 1) if rp > 0 else (self._buffer_size - wp - 1)
        else:
            # Read ahead of write: can only write up to read-1
            space = rp - wp - 1

        space = max(0, space)
        to_write = min(n, space)

        if to_write == 0:
            return 0

        # Write to ring buffer
        end = wp + to_write
        if end <= self._buffer_size:
            # Contiguous write
            self._buffer[wp:end] = samples[:to_write]
        else:
            # Wrap-around write
            first = self._buffer_size - wp
            self._buffer[wp:] = samples[:first]
            self._buffer[:to_write - first] = samples[first:to_write]

        # Update write pointer (only writer touches this)
        self._write_ptr = end % self._buffer_size

        self._total_samples_written += to_write
        return to_write

    def available(self) -> int:
        """Space available for writing."""
        return self._available_to_write()

    @property
    def buffer_fill(self) -> float:
        """Buffer fill percentage."""
        used = self._available_to_read()
        return (used / self._buffer_size) * 100

    @property
    def latency_seconds(self) -> float:
        if self._stream and hasattr(self._stream, "latency"):
            return self._stream.latency
        return 0.0

    def get_device_info(self) -> Dict[str, Any]:
        if self._sd is None:
            return {}
        try:
            dev = self._sd.query_devices(self._device if self._device else "output")
            return {
                "name": dev["name"],
                "channels": dev["max_output_channels"],
                "default_samplerate": dev["default_samplerate"],
            }
        except:
            return {}

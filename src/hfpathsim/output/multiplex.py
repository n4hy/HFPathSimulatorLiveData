"""Multiplex output sink for HF Path Simulator."""

from typing import List, Optional
import numpy as np

from .base import OutputSink, OutputFormat


class MultiplexOutputSink(OutputSink):
    """Output sink that fans out to multiple downstream sinks.

    Writes samples to multiple output sinks simultaneously.
    Useful for recording while also streaming, or sending to
    multiple network destinations.
    """

    def __init__(
        self,
        sinks: Optional[List[OutputSink]] = None,
        sample_rate_hz: float = 48000.0,
        center_freq_hz: float = 0.0,
    ):
        """Initialize multiplex output sink.

        Args:
            sinks: List of downstream output sinks
            sample_rate_hz: Sample rate (should match all sinks)
            center_freq_hz: Center frequency
        """
        super().__init__(sample_rate_hz, center_freq_hz, OutputFormat.COMPLEX64)

        self._sinks: List[OutputSink] = sinks or []

    @property
    def sinks(self) -> List[OutputSink]:
        """Return list of downstream sinks."""
        return self._sinks

    @property
    def num_sinks(self) -> int:
        """Return number of downstream sinks."""
        return len(self._sinks)

    def add_sink(self, sink: OutputSink):
        """Add a downstream sink.

        Args:
            sink: OutputSink to add
        """
        self._sinks.append(sink)

        # Open sink if we're already open
        if self._is_open and not sink.is_open:
            sink.open()

    def remove_sink(self, sink: OutputSink):
        """Remove a downstream sink.

        Args:
            sink: OutputSink to remove
        """
        if sink in self._sinks:
            if sink.is_open:
                sink.close()
            self._sinks.remove(sink)

    def clear_sinks(self):
        """Remove all downstream sinks."""
        for sink in self._sinks:
            if sink.is_open:
                sink.close()
        self._sinks.clear()

    def open(self) -> bool:
        """Open all downstream sinks."""
        success = True

        for sink in self._sinks:
            if not sink.open():
                print(f"Failed to open sink: {type(sink).__name__}")
                success = False

        self._is_open = success or len(self._sinks) == 0
        return self._is_open

    def close(self):
        """Close all downstream sinks."""
        for sink in self._sinks:
            try:
                sink.close()
            except Exception as e:
                print(f"Error closing sink {type(sink).__name__}: {e}")

        self._is_open = False

    def write(self, samples: np.ndarray) -> int:
        """Write samples to all downstream sinks.

        Returns:
            Minimum number of samples written to any sink
        """
        if not self._is_open or len(self._sinks) == 0:
            return 0

        samples = samples.astype(np.complex64)
        min_written = len(samples)

        for sink in self._sinks:
            if sink.is_open:
                try:
                    written = sink.write(samples)
                    min_written = min(min_written, written)
                except Exception as e:
                    print(f"Error writing to sink {type(sink).__name__}: {e}")
                    min_written = 0

        self._total_samples_written += min_written
        return min_written

    def available(self) -> int:
        """Return minimum samples available across all sinks."""
        if len(self._sinks) == 0:
            return self._buffer_size

        min_available = self._buffer_size

        for sink in self._sinks:
            if sink.is_open:
                min_available = min(min_available, sink.available())

        return min_available

    def flush(self):
        """Flush all downstream sinks."""
        for sink in self._sinks:
            if sink.is_open:
                try:
                    sink.flush()
                except Exception as e:
                    print(f"Error flushing sink {type(sink).__name__}: {e}")

    def get_sink_status(self) -> List[dict]:
        """Get status of all downstream sinks.

        Returns:
            List of status dictionaries for each sink
        """
        status = []

        for i, sink in enumerate(self._sinks):
            info = {
                "index": i,
                "type": type(sink).__name__,
                "is_open": sink.is_open,
                "samples_written": sink.total_samples_written,
            }

            # Add sink-specific info
            if hasattr(sink, "buffer_fill"):
                info["buffer_fill"] = sink.buffer_fill

            if hasattr(sink, "underruns"):
                info["underruns"] = sink.underruns

            if hasattr(sink, "num_clients"):
                info["num_clients"] = sink.num_clients

            status.append(info)

        return status


class TeeOutputSink(MultiplexOutputSink):
    """Output sink that tees to exactly two downstream sinks.

    Convenience wrapper for common two-way split (e.g., record + stream).
    """

    def __init__(
        self,
        primary: OutputSink,
        secondary: OutputSink,
        sample_rate_hz: float = 48000.0,
        center_freq_hz: float = 0.0,
    ):
        """Initialize tee output sink.

        Args:
            primary: Primary output sink
            secondary: Secondary output sink
            sample_rate_hz: Sample rate
            center_freq_hz: Center frequency
        """
        super().__init__([primary, secondary], sample_rate_hz, center_freq_hz)

    @property
    def primary(self) -> OutputSink:
        """Return primary sink."""
        return self._sinks[0] if len(self._sinks) > 0 else None

    @property
    def secondary(self) -> OutputSink:
        """Return secondary sink."""
        return self._sinks[1] if len(self._sinks) > 1 else None

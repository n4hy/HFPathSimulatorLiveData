"""Abstract base class for input data sources."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
import numpy as np


class InputFormat(Enum):
    """Supported input data formats."""

    COMPLEX64 = "complex64"  # numpy complex64 (float32 I/Q pairs)
    COMPLEX128 = "complex128"  # numpy complex128 (float64 I/Q pairs)
    INT16_IQ = "int16_iq"  # Interleaved 16-bit I/Q (RTL-SDR format)
    INT8_IQ = "int8_iq"  # Interleaved 8-bit I/Q
    FLOAT32_IQ = "float32_iq"  # Interleaved 32-bit float I/Q


class InputSource(ABC):
    """Abstract base class for input data sources.

    All input sources must provide complex samples at a known sample rate.
    """

    def __init__(
        self,
        sample_rate_hz: float,
        center_freq_hz: float = 0.0,
        input_format: InputFormat = InputFormat.COMPLEX64,
    ):
        """Initialize input source.

        Args:
            sample_rate_hz: Sample rate in Hz
            center_freq_hz: Center frequency in Hz (for RF sources)
            input_format: Native format of the input data
        """
        self._sample_rate = sample_rate_hz
        self._center_freq = center_freq_hz
        self._input_format = input_format
        self._is_open = False
        self._total_samples_read = 0

    @property
    def sample_rate(self) -> float:
        """Return sample rate in Hz."""
        return self._sample_rate

    @property
    def center_frequency(self) -> float:
        """Return center frequency in Hz."""
        return self._center_freq

    @property
    def input_format(self) -> InputFormat:
        """Return native input format."""
        return self._input_format

    @property
    def is_open(self) -> bool:
        """Return whether the source is open and ready."""
        return self._is_open

    @property
    def total_samples_read(self) -> int:
        """Return total number of samples read."""
        return self._total_samples_read

    @abstractmethod
    def open(self) -> bool:
        """Open the input source.

        Returns:
            True if successfully opened
        """
        pass

    @abstractmethod
    def close(self):
        """Close the input source and release resources."""
        pass

    @abstractmethod
    def read(self, num_samples: int) -> Optional[np.ndarray]:
        """Read samples from the source.

        Args:
            num_samples: Number of complex samples to read

        Returns:
            Complex64 numpy array of samples, or None if unavailable
        """
        pass

    @abstractmethod
    def available(self) -> int:
        """Return number of samples available to read without blocking."""
        pass

    def _convert_format(self, data: np.ndarray) -> np.ndarray:
        """Convert input data to complex64.

        Args:
            data: Raw input data

        Returns:
            Complex64 numpy array
        """
        fmt = self._input_format

        if fmt == InputFormat.COMPLEX64:
            return data.astype(np.complex64)

        elif fmt == InputFormat.COMPLEX128:
            return data.astype(np.complex64)

        elif fmt == InputFormat.INT16_IQ:
            # Interleaved I/Q as int16
            data = data.reshape(-1, 2)
            i = data[:, 0].astype(np.float32) / 32768.0
            q = data[:, 1].astype(np.float32) / 32768.0
            return (i + 1j * q).astype(np.complex64)

        elif fmt == InputFormat.INT8_IQ:
            # Interleaved I/Q as int8 (RTL-SDR unsigned)
            data = data.reshape(-1, 2)
            i = (data[:, 0].astype(np.float32) - 127.5) / 128.0
            q = (data[:, 1].astype(np.float32) - 127.5) / 128.0
            return (i + 1j * q).astype(np.complex64)

        elif fmt == InputFormat.FLOAT32_IQ:
            # Interleaved I/Q as float32
            data = data.reshape(-1, 2)
            return (data[:, 0] + 1j * data[:, 1]).astype(np.complex64)

        else:
            raise ValueError(f"Unknown format: {fmt}")

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

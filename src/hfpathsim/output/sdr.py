"""SDR transmit output sink for HF Path Simulator."""

import threading
from collections import deque
from typing import Optional, List, Dict, Any
import numpy as np

from .base import OutputSink, OutputFormat


class SDROutputSink(OutputSink):
    """Output sink to SDR transmitter via SoapySDR.

    Transmits I/Q samples through SDR hardware capable of transmit.
    Supports HackRF, LimeSDR, PlutoSDR, USRP, and other SoapySDR-compatible devices.
    """

    def __init__(
        self,
        device_args: str = "",
        sample_rate_hz: float = 2_000_000,
        center_freq_hz: float = 10_000_000,
        output_format: OutputFormat = OutputFormat.COMPLEX64,
        buffer_size: int = 1_000_000,
        tx_gain: float = 40.0,
        antenna: str = "",
        bandwidth: float = 0.0,
    ):
        """Initialize SDR output sink.

        Args:
            device_args: SoapySDR device arguments (e.g., "driver=hackrf")
            sample_rate_hz: Transmit sample rate
            center_freq_hz: Transmit center frequency
            output_format: Output format
            buffer_size: Internal buffer size in samples
            tx_gain: Transmit gain in dB
            antenna: Antenna selection (device-specific)
            bandwidth: Analog bandwidth (0 = auto)
        """
        super().__init__(sample_rate_hz, center_freq_hz, output_format, buffer_size)

        self._device_args = device_args
        self._tx_gain = tx_gain
        self._antenna = antenna
        self._bandwidth = bandwidth

        # SoapySDR
        self._sdr = None
        self._tx_stream = None

        # Threading
        self._tx_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Buffer
        self._buffer: deque = deque(maxlen=buffer_size)

        # Stats
        self._underruns = 0
        self._overruns = 0

    @property
    def tx_gain(self) -> float:
        """Return transmit gain in dB."""
        return self._tx_gain

    @tx_gain.setter
    def tx_gain(self, gain: float):
        """Set transmit gain."""
        self._tx_gain = gain
        if self._sdr:
            try:
                self._sdr.setGain(1, 0, gain)  # TX direction = 1
            except Exception:
                pass

    @property
    def underruns(self) -> int:
        """Return number of TX buffer underruns."""
        return self._underruns

    @classmethod
    def enumerate_devices(cls) -> List[Dict[str, Any]]:
        """Enumerate available SDR devices with TX capability.

        Returns:
            List of device dictionaries
        """
        try:
            import SoapySDR

            devices = []
            results = SoapySDR.Device.enumerate()

            for result in results:
                args = dict(result)

                # Try to check if device supports TX
                try:
                    dev = SoapySDR.Device(args)
                    tx_channels = dev.getNumChannels(SoapySDR.SOAPY_SDR_TX)

                    if tx_channels > 0:
                        info = {
                            "driver": args.get("driver", "unknown"),
                            "label": args.get("label", args.get("driver", "SDR")),
                            "serial": args.get("serial", ""),
                            "args": args,
                            "tx_channels": tx_channels,
                        }

                        # Get frequency range
                        try:
                            freq_range = dev.getFrequencyRange(SoapySDR.SOAPY_SDR_TX, 0)
                            if freq_range:
                                info["freq_min_hz"] = freq_range[0].minimum()
                                info["freq_max_hz"] = freq_range[-1].maximum()
                        except Exception:
                            pass

                        # Get sample rate range
                        try:
                            rate_range = dev.getSampleRateRange(SoapySDR.SOAPY_SDR_TX, 0)
                            if rate_range:
                                info["rate_min_hz"] = rate_range[0].minimum()
                                info["rate_max_hz"] = rate_range[-1].maximum()
                        except Exception:
                            pass

                        devices.append(info)

                    dev = None

                except Exception:
                    pass

            return devices

        except ImportError:
            print("SoapySDR not installed")
            return []

    def open(self) -> bool:
        """Open SDR device and start transmit stream."""
        try:
            import SoapySDR

            # Open device
            if self._device_args:
                self._sdr = SoapySDR.Device(self._device_args)
            else:
                # Use first available TX-capable device
                devices = self.enumerate_devices()
                if not devices:
                    print("No TX-capable SDR devices found")
                    return False
                self._sdr = SoapySDR.Device(devices[0]["args"])

            # Configure TX channel
            channel = 0

            # Set sample rate
            self._sdr.setSampleRate(SoapySDR.SOAPY_SDR_TX, channel, self._sample_rate)

            # Set frequency
            self._sdr.setFrequency(SoapySDR.SOAPY_SDR_TX, channel, self._center_freq)

            # Set gain
            self._sdr.setGain(SoapySDR.SOAPY_SDR_TX, channel, self._tx_gain)

            # Set antenna if specified
            if self._antenna:
                self._sdr.setAntenna(SoapySDR.SOAPY_SDR_TX, channel, self._antenna)

            # Set bandwidth if specified
            if self._bandwidth > 0:
                self._sdr.setBandwidth(SoapySDR.SOAPY_SDR_TX, channel, self._bandwidth)

            # Setup TX stream
            self._tx_stream = self._sdr.setupStream(
                SoapySDR.SOAPY_SDR_TX, SoapySDR.SOAPY_SDR_CF32, [channel]
            )
            self._sdr.activateStream(self._tx_stream)

            # Start TX thread
            self._running = True
            self._tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
            self._tx_thread.start()

            self._is_open = True
            return True

        except ImportError:
            print("SoapySDR not installed. Install with: pip install soapysdr")
            return False

        except Exception as e:
            print(f"Error opening SDR for TX: {e}")
            return False

    def _tx_loop(self):
        """Background thread to transmit samples."""
        import SoapySDR

        chunk_size = 4096

        while self._running:
            with self._lock:
                if len(self._buffer) < chunk_size:
                    # Wait a bit for more data
                    continue

                samples = np.array(
                    [self._buffer.popleft() for _ in range(chunk_size)],
                    dtype=np.complex64,
                )

            if len(samples) == 0:
                continue

            # Transmit
            try:
                status = self._sdr.writeStream(
                    self._tx_stream, [samples], len(samples)
                )

                if status.ret < 0:
                    if status.ret == SoapySDR.SOAPY_SDR_UNDERFLOW:
                        self._underruns += 1
                    elif status.ret == SoapySDR.SOAPY_SDR_OVERFLOW:
                        self._overruns += 1

            except Exception as e:
                if self._running:
                    print(f"TX error: {e}")

    def close(self):
        """Close SDR device and stop transmit."""
        self._running = False

        if self._tx_thread:
            self._tx_thread.join(timeout=1.0)

        if self._tx_stream and self._sdr:
            try:
                self._sdr.deactivateStream(self._tx_stream)
                self._sdr.closeStream(self._tx_stream)
            except Exception:
                pass
            self._tx_stream = None

        if self._sdr:
            self._sdr = None

        self._is_open = False

    def write(self, samples: np.ndarray) -> int:
        """Write samples to TX buffer."""
        if not self._is_open:
            return 0

        samples = samples.astype(np.complex64)

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

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about current SDR device.

        Returns:
            Device information dictionary
        """
        if self._sdr is None:
            return {}

        try:
            import SoapySDR

            info = {
                "driver": self._sdr.getDriverKey(),
                "hardware": self._sdr.getHardwareKey(),
            }

            # Get actual settings
            channel = 0
            info["sample_rate"] = self._sdr.getSampleRate(SoapySDR.SOAPY_SDR_TX, channel)
            info["frequency"] = self._sdr.getFrequency(SoapySDR.SOAPY_SDR_TX, channel)
            info["gain"] = self._sdr.getGain(SoapySDR.SOAPY_SDR_TX, channel)
            info["antenna"] = self._sdr.getAntenna(SoapySDR.SOAPY_SDR_TX, channel)
            info["bandwidth"] = self._sdr.getBandwidth(SoapySDR.SOAPY_SDR_TX, channel)

            return info

        except Exception:
            return {}

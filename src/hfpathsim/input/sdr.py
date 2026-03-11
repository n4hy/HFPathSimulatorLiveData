"""SDR input sources for HF Path Simulator."""

from typing import Optional, List, Dict, Any
import numpy as np

from .base import InputSource, InputFormat


class SDRInputSource(InputSource):
    """Input source from Software Defined Radio via SoapySDR.

    Supports any SDR with SoapySDR driver:
    - RTL-SDR, HackRF, AirSpy, SDRPlay, USRP, etc.
    """

    def __init__(
        self,
        driver: str = "",
        device_args: str = "",
        sample_rate_hz: float = 2_000_000,
        center_freq_hz: float = 10_000_000,
        gain: float = 40.0,
        bandwidth_hz: float = 0.0,  # 0 = auto
        antenna: str = "",
        channel: int = 0,
    ):
        """Initialize SDR input source.

        Args:
            driver: SoapySDR driver name (e.g., "rtlsdr", "hackrf")
            device_args: Additional device arguments
            sample_rate_hz: Sample rate in Hz
            center_freq_hz: Center frequency in Hz
            gain: RF gain in dB
            bandwidth_hz: Analog bandwidth (0 for auto)
            antenna: Antenna port name
            channel: Channel number for multi-channel devices
        """
        super().__init__(sample_rate_hz, center_freq_hz, InputFormat.COMPLEX64)

        self._driver = driver
        self._device_args = device_args
        self._gain = gain
        self._bandwidth = bandwidth_hz
        self._antenna = antenna
        self._channel = channel

        # SoapySDR objects
        self._sdr = None
        self._stream = None

        # Check SoapySDR availability
        self._soapy_available = self._check_soapy()

    def _check_soapy(self) -> bool:
        """Check if SoapySDR is available."""
        try:
            import SoapySDR

            return True
        except ImportError:
            return False

    @staticmethod
    def enumerate_devices() -> List[Dict[str, Any]]:
        """List available SDR devices.

        Returns:
            List of device info dictionaries
        """
        try:
            import SoapySDR

            devices = []
            results = SoapySDR.Device.enumerate()

            for result in results:
                info = {
                    "driver": result.get("driver", "unknown"),
                    "label": result.get("label", "Unknown Device"),
                    "serial": result.get("serial", ""),
                    "args": dict(result),
                }
                devices.append(info)

            return devices

        except ImportError:
            return []

    def open(self) -> bool:
        """Open SDR device and configure stream."""
        if not self._soapy_available:
            print("SoapySDR not installed. Install with: sudo apt install python3-soapysdr")
            return False

        try:
            import SoapySDR

            # Build device args
            args = {}
            if self._driver:
                args["driver"] = self._driver
            if self._device_args:
                for pair in self._device_args.split(","):
                    if "=" in pair:
                        key, val = pair.split("=", 1)
                        args[key.strip()] = val.strip()

            # Open device
            self._sdr = SoapySDR.Device(args)

            # Configure
            self._sdr.setSampleRate(
                SoapySDR.SOAPY_SDR_RX, self._channel, self._sample_rate
            )
            self._sdr.setFrequency(
                SoapySDR.SOAPY_SDR_RX, self._channel, self._center_freq
            )
            self._sdr.setGain(
                SoapySDR.SOAPY_SDR_RX, self._channel, self._gain
            )

            if self._bandwidth > 0:
                self._sdr.setBandwidth(
                    SoapySDR.SOAPY_SDR_RX, self._channel, self._bandwidth
                )

            if self._antenna:
                self._sdr.setAntenna(
                    SoapySDR.SOAPY_SDR_RX, self._channel, self._antenna
                )

            # Setup stream
            self._stream = self._sdr.setupStream(
                SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, [self._channel]
            )
            self._sdr.activateStream(self._stream)

            self._is_open = True
            return True

        except Exception as e:
            print(f"Error opening SDR: {e}")
            return False

    def close(self):
        """Close SDR device."""
        try:
            import SoapySDR

            if self._stream:
                self._sdr.deactivateStream(self._stream)
                self._sdr.closeStream(self._stream)
                self._stream = None

            if self._sdr:
                self._sdr = None

        except Exception as e:
            print(f"Error closing SDR: {e}")

        self._is_open = False

    def read(self, num_samples: int) -> Optional[np.ndarray]:
        """Read samples from SDR."""
        if not self._is_open or self._stream is None:
            return None

        try:
            import SoapySDR

            buff = np.zeros(num_samples, dtype=np.complex64)
            sr = self._sdr.readStream(
                self._stream, [buff], num_samples, timeoutUs=100000
            )

            if sr.ret > 0:
                self._total_samples_read += sr.ret
                return buff[: sr.ret]
            else:
                return np.array([], dtype=np.complex64)

        except Exception as e:
            print(f"SDR read error: {e}")
            return None

    def available(self) -> int:
        """Return samples available (always returns block size for SDRs)."""
        if self._is_open:
            return 4096  # Standard block size
        return 0

    def set_frequency(self, freq_hz: float):
        """Change center frequency."""
        if self._sdr:
            try:
                import SoapySDR

                self._sdr.setFrequency(
                    SoapySDR.SOAPY_SDR_RX, self._channel, freq_hz
                )
                self._center_freq = freq_hz
            except Exception as e:
                print(f"Error setting frequency: {e}")

    def set_gain(self, gain_db: float):
        """Change RF gain."""
        if self._sdr:
            try:
                import SoapySDR

                self._sdr.setGain(
                    SoapySDR.SOAPY_SDR_RX, self._channel, gain_db
                )
                self._gain = gain_db
            except Exception as e:
                print(f"Error setting gain: {e}")

    def get_gain_range(self) -> tuple:
        """Get valid gain range."""
        if self._sdr:
            try:
                import SoapySDR

                gain_range = self._sdr.getGainRange(
                    SoapySDR.SOAPY_SDR_RX, self._channel
                )
                return (gain_range.minimum(), gain_range.maximum())
            except Exception:
                pass
        return (0.0, 50.0)

    def get_frequency_range(self) -> tuple:
        """Get valid frequency range."""
        if self._sdr:
            try:
                import SoapySDR

                freq_range = self._sdr.getFrequencyRange(
                    SoapySDR.SOAPY_SDR_RX, self._channel
                )[0]
                return (freq_range.minimum(), freq_range.maximum())
            except Exception:
                pass
        return (0.0, 6_000_000_000.0)

    def get_sample_rate_range(self) -> List[float]:
        """Get valid sample rates."""
        if self._sdr:
            try:
                import SoapySDR

                rates = self._sdr.listSampleRates(
                    SoapySDR.SOAPY_SDR_RX, self._channel
                )
                return list(rates)
            except Exception:
                pass
        return [250000, 1000000, 2000000, 2400000]

    def get_antennas(self) -> List[str]:
        """Get available antenna ports."""
        if self._sdr:
            try:
                import SoapySDR

                return list(
                    self._sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, self._channel)
                )
            except Exception:
                pass
        return []

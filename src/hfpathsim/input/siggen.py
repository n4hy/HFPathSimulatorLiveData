"""Standard HF waveform signal generators for testing.

Generates one minute of continuous test signal for:
- RTTY: 45.45 baud FSK with 170 Hz shift
- SSB Voice: Real voice samples from CMU Arctic corpus (male/female)
- PSK31: 31.25 baud BPSK with raised cosine shaping
"""

from enum import Enum
from pathlib import Path
from typing import Optional
import numpy as np
import wave

from .base import InputSource, InputFormat


# Path to voice samples directory
VOICE_SAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "data" / "voice_samples"


class WaveformType(Enum):
    """Standard HF waveform types."""

    RTTY = "rtty"
    SSB_VOICE = "ssb_voice"
    PSK31 = "psk31"


class SignalGenerator(InputSource):
    """Generate standard HF test waveforms.

    Generates one minute of signal that loops continuously.
    """

    def __init__(
        self,
        waveform: WaveformType,
        sample_rate_hz: float = 8000.0,
        center_freq_hz: float = 1500.0,
        duration_sec: float = 60.0,
    ):
        """Initialize signal generator.

        Args:
            waveform: Type of waveform to generate
            sample_rate_hz: Sample rate in Hz (default 8000)
            center_freq_hz: Center/carrier frequency in Hz (default 1500)
            duration_sec: Duration of signal to generate (default 60s)
        """
        super().__init__(
            sample_rate_hz=sample_rate_hz,
            center_freq_hz=center_freq_hz,
            input_format=InputFormat.COMPLEX64,
        )
        self._waveform_type = waveform
        self._duration = duration_sec
        self._signal: Optional[np.ndarray] = None
        self._position = 0

    @property
    def waveform_type(self) -> WaveformType:
        """Return the waveform type."""
        return self._waveform_type

    def open(self) -> bool:
        """Generate the signal buffer."""
        try:
            if self._waveform_type == WaveformType.RTTY:
                self._signal = self._generate_rtty()
            elif self._waveform_type == WaveformType.SSB_VOICE:
                self._signal = self._generate_ssb_voice()
            elif self._waveform_type == WaveformType.PSK31:
                self._signal = self._generate_psk31()
            else:
                raise ValueError(f"Unknown waveform type: {self._waveform_type}")

            self._position = 0
            self._is_open = True
            return True

        except Exception as e:
            print(f"SignalGenerator.open() error: {e}")
            return False

    def close(self):
        """Release signal buffer."""
        self._signal = None
        self._position = 0
        self._is_open = False

    def read(self, num_samples: int) -> Optional[np.ndarray]:
        """Read samples from the generated signal (loops continuously)."""
        if self._signal is None:
            return None

        total_len = len(self._signal)
        output = np.zeros(num_samples, dtype=np.complex64)

        remaining = num_samples
        out_pos = 0

        while remaining > 0:
            available = total_len - self._position
            to_copy = min(remaining, available)

            output[out_pos : out_pos + to_copy] = self._signal[
                self._position : self._position + to_copy
            ]

            self._position += to_copy
            out_pos += to_copy
            remaining -= to_copy

            # Loop back to start
            if self._position >= total_len:
                self._position = 0

        self._total_samples_read += num_samples
        return output

    def available(self) -> int:
        """Signal generator always has samples available."""
        return int(self._sample_rate * 10) if self._is_open else 0

    def _generate_rtty(self) -> np.ndarray:
        """Generate RTTY signal (45.45 baud, 170 Hz shift).

        Standard amateur RTTY uses:
        - 45.45 baud (22ms per bit)
        - 170 Hz shift (mark/space frequencies)
        - Mark = 1 (higher freq), Space = 0 (lower freq)
        - Baudot encoding with start/stop bits
        """
        num_samples = int(self._duration * self._sample_rate)
        t = np.arange(num_samples) / self._sample_rate

        # RTTY parameters
        baud_rate = 45.45
        shift_hz = 170.0
        mark_freq = self._center_freq + shift_hz / 2
        space_freq = self._center_freq - shift_hz / 2

        samples_per_bit = int(self._sample_rate / baud_rate)

        # Generate random Baudot-like data
        # 5-bit Baudot with start (0) and stop (1, 1.5 bits) = ~7.5 bits per char
        num_bits = int(num_samples / samples_per_bit) + 1
        bits = np.random.randint(0, 2, num_bits)

        # Add start/stop bit structure (simplified)
        # Insert start bit (0) and stop bits (1) every 5 data bits
        structured_bits = []
        for i in range(0, len(bits), 5):
            structured_bits.append(0)  # Start bit (space)
            structured_bits.extend(bits[i : i + 5].tolist())
            structured_bits.append(1)  # Stop bit (mark)
            structured_bits.append(1)  # 1.5 stop bits (add extra)

        bits = np.array(structured_bits[:num_bits])

        # Expand bits to samples
        bit_samples = np.repeat(bits, samples_per_bit)[:num_samples]

        # Generate FSK signal
        freq = np.where(bit_samples == 1, mark_freq, space_freq)
        phase = 2 * np.pi * np.cumsum(freq) / self._sample_rate
        signal = np.exp(1j * phase).astype(np.complex64)

        # Apply slight amplitude variation for realism
        signal *= 0.8 + 0.1 * np.random.randn(num_samples).astype(np.float32)

        return signal

    def _generate_ssb_voice(self) -> np.ndarray:
        """Generate SSB voice signal using real voice samples.

        Uses CMU Arctic corpus samples (male and female voices) to create
        a realistic SSB voice signal. Falls back to synthetic voice if
        samples are not available.
        """
        num_samples = int(self._duration * self._sample_rate)

        # Try to load real voice samples
        voice_data = self._load_voice_samples()

        if voice_data is None:
            print("WARNING: Voice samples not found, using synthetic voice")
            return self._generate_synthetic_ssb_voice()

        # voice_data is a concatenated audio buffer at original sample rate
        voice, orig_rate = voice_data

        # Resample to target sample rate if needed
        if orig_rate != self._sample_rate:
            from scipy import signal as scipy_signal
            num_resampled = int(len(voice) * self._sample_rate / orig_rate)
            voice = scipy_signal.resample(voice, num_resampled).astype(np.float32)

        # Loop or truncate to fill duration
        if len(voice) < num_samples:
            # Loop the voice data to fill duration
            repeats = (num_samples // len(voice)) + 1
            voice = np.tile(voice, repeats)[:num_samples]
        else:
            voice = voice[:num_samples]

        # Apply SSB bandpass filter (300-3000 Hz for voice)
        from scipy import signal as scipy_signal
        nyq = self._sample_rate / 2
        low = 300 / nyq
        high = min(3000 / nyq, 0.99)  # Don't exceed Nyquist
        b, a = scipy_signal.butter(4, [low, high], btype="band")
        voice = scipy_signal.lfilter(b, a, voice)

        # Normalize
        voice = voice / (np.max(np.abs(voice)) + 1e-6) * 0.9

        # SSB modulation (upper sideband) - shift to center frequency
        t = np.arange(num_samples) / self._sample_rate
        carrier = np.exp(1j * 2 * np.pi * self._center_freq * t)

        # Create analytic signal (Hilbert transform for USB)
        analytic = scipy_signal.hilbert(voice)

        # USB: multiply by carrier (complex mixing)
        signal = (analytic * carrier).astype(np.complex64)

        return signal

    def _load_voice_samples(self) -> Optional[tuple]:
        """Load and concatenate voice samples from data directory.

        Returns:
            Tuple of (audio_data, sample_rate) or None if not found
        """
        if not VOICE_SAMPLES_DIR.exists():
            return None

        # Find all WAV files
        wav_files = sorted(VOICE_SAMPLES_DIR.glob("*.wav"))
        if not wav_files:
            return None

        all_audio = []
        sample_rate = None

        # Separate male and female files for alternating
        male_files = [f for f in wav_files if "male" in f.name.lower() and "female" not in f.name.lower()]
        female_files = [f for f in wav_files if "female" in f.name.lower()]

        # Interleave male and female voices with pauses
        max_files = max(len(male_files), len(female_files))

        for i in range(max_files):
            # Add male voice sample
            if i < len(male_files):
                audio, sr = self._read_wav(male_files[i])
                if audio is not None:
                    all_audio.append(audio)
                    sample_rate = sr
                    # Add pause after male voice (0.5-1.5 sec)
                    pause_samples = int(sr * (0.5 + np.random.rand()))
                    all_audio.append(np.zeros(pause_samples, dtype=np.float32))

            # Add female voice sample
            if i < len(female_files):
                audio, sr = self._read_wav(female_files[i])
                if audio is not None:
                    all_audio.append(audio)
                    sample_rate = sr
                    # Add pause after female voice (0.5-1.5 sec)
                    pause_samples = int(sr * (0.5 + np.random.rand()))
                    all_audio.append(np.zeros(pause_samples, dtype=np.float32))

        if not all_audio or sample_rate is None:
            return None

        # Concatenate all audio
        combined = np.concatenate(all_audio)
        return combined, sample_rate

    def _read_wav(self, filepath: Path) -> Optional[tuple]:
        """Read a WAV file and return audio data.

        Returns:
            Tuple of (audio_data as float32, sample_rate) or (None, None)
        """
        try:
            with wave.open(str(filepath), 'rb') as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                sample_width = wf.getsampwidth()
                n_channels = wf.getnchannels()

                raw_data = wf.readframes(n_frames)

                # Convert to numpy array
                if sample_width == 2:
                    audio = np.frombuffer(raw_data, dtype=np.int16)
                elif sample_width == 1:
                    audio = np.frombuffer(raw_data, dtype=np.uint8).astype(np.int16) - 128
                else:
                    return None, None

                # Convert to mono if stereo
                if n_channels == 2:
                    audio = audio.reshape(-1, 2).mean(axis=1)

                # Normalize to float32 [-1, 1]
                audio = audio.astype(np.float32) / 32768.0

                return audio, sample_rate

        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None, None

    def _generate_synthetic_ssb_voice(self) -> np.ndarray:
        """Generate synthetic SSB voice signal (fallback).

        Uses filtered noise with formant-like structure to simulate
        voice characteristics in the 300-3000 Hz band.
        """
        num_samples = int(self._duration * self._sample_rate)
        t = np.arange(num_samples) / self._sample_rate

        # Generate base noise
        noise = np.random.randn(num_samples).astype(np.float32)

        # Design formant-like bandpass filters
        from scipy import signal as scipy_signal

        # Create composite filter for voice-like spectrum
        b1, a1 = scipy_signal.butter(2, [280 / (self._sample_rate / 2), 600 / (self._sample_rate / 2)], btype="band")
        b2, a2 = scipy_signal.butter(2, [800 / (self._sample_rate / 2), 2200 / (self._sample_rate / 2)], btype="band")
        b3, a3 = scipy_signal.butter(2, [2000 / (self._sample_rate / 2), 3200 / (self._sample_rate / 2)], btype="band")

        voice1 = scipy_signal.lfilter(b1, a1, noise) * 0.5
        voice2 = scipy_signal.lfilter(b2, a2, noise) * 0.8
        voice3 = scipy_signal.lfilter(b3, a3, noise) * 0.3

        voice = voice1 + voice2 + voice3

        # Add syllable-like amplitude modulation
        syllable_rate = 4.0
        syllable_mod = 0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * t)

        # Add random pauses
        envelope = syllable_mod.copy()
        pause_interval = int(self._sample_rate * 3)
        for i in range(0, num_samples, pause_interval):
            if np.random.rand() > 0.6:
                pause_len = int(self._sample_rate * (0.3 + 0.5 * np.random.rand()))
                end_idx = min(i + pause_len, num_samples)
                envelope[i:end_idx] *= 0.1

        voice = voice * envelope
        voice = voice / (np.max(np.abs(voice)) + 1e-6) * 0.9

        # SSB modulation
        carrier = np.exp(1j * 2 * np.pi * self._center_freq * t)
        analytic = scipy_signal.hilbert(voice)
        signal = (analytic * carrier).astype(np.complex64)

        return signal

    def _generate_psk31(self) -> np.ndarray:
        """Generate PSK31 signal.

        PSK31 uses:
        - 31.25 baud BPSK
        - Raised cosine shaping (no key clicks)
        - Varicode encoding (not implemented here, using random bits)
        """
        num_samples = int(self._duration * self._sample_rate)
        t = np.arange(num_samples) / self._sample_rate

        # PSK31 parameters
        baud_rate = 31.25
        samples_per_symbol = int(self._sample_rate / baud_rate)

        # Generate random BPSK symbols (-1 or +1)
        num_symbols = int(num_samples / samples_per_symbol) + 1
        symbols = 2 * np.random.randint(0, 2, num_symbols) - 1

        # Create raised cosine pulse shape
        # Full raised cosine over one symbol period
        pulse = np.sin(np.pi * np.arange(samples_per_symbol) / samples_per_symbol) ** 2

        # Generate baseband signal with smooth phase transitions
        baseband = np.zeros(num_samples, dtype=np.float32)

        for i, sym in enumerate(symbols):
            start = i * samples_per_symbol
            end = start + samples_per_symbol
            if end > num_samples:
                end = num_samples
                pulse_end = end - start
            else:
                pulse_end = samples_per_symbol

            # For BPSK, we shape the amplitude, not just switch phase
            if i > 0 and symbols[i] != symbols[i - 1]:
                # Phase reversal - use cosine shaping through zero
                shaped_pulse = np.cos(np.pi * np.arange(pulse_end) / samples_per_symbol)
                baseband[start:end] = shaped_pulse * sym
            else:
                # Same phase - maintain amplitude
                baseband[start:end] = sym

        # Smooth the entire signal with a low-pass filter
        from scipy import signal as scipy_signal

        nyq = self._sample_rate / 2
        cutoff = baud_rate * 1.5  # 1.5x symbol rate bandwidth
        b, a = scipy_signal.butter(4, cutoff / nyq, btype="low")
        baseband = scipy_signal.lfilter(b, a, baseband)

        # Normalize
        baseband = baseband / (np.max(np.abs(baseband)) + 1e-6) * 0.9

        # Modulate to center frequency
        carrier = np.exp(1j * 2 * np.pi * self._center_freq * t)
        signal = (baseband * carrier).astype(np.complex64)

        return signal


def create_signal_generator(
    waveform_name: str,
    sample_rate_hz: float = 8000.0,
    center_freq_hz: float = 1500.0,
    duration_sec: float = 60.0,
) -> SignalGenerator:
    """Create a signal generator by waveform name.

    Args:
        waveform_name: One of "rtty", "ssb", "ssb_voice", "psk31", "psk"
        sample_rate_hz: Sample rate (default 8000)
        center_freq_hz: Center frequency (default 1500)
        duration_sec: Signal duration (default 60s)

    Returns:
        SignalGenerator instance
    """
    waveform_map = {
        "rtty": WaveformType.RTTY,
        "ssb": WaveformType.SSB_VOICE,
        "ssb_voice": WaveformType.SSB_VOICE,
        "voice": WaveformType.SSB_VOICE,
        "psk31": WaveformType.PSK31,
        "psk": WaveformType.PSK31,
    }

    name_lower = waveform_name.lower()
    if name_lower not in waveform_map:
        valid = ", ".join(sorted(set(waveform_map.keys())))
        raise ValueError(f"Unknown waveform '{waveform_name}'. Valid: {valid}")

    return SignalGenerator(
        waveform=waveform_map[name_lower],
        sample_rate_hz=sample_rate_hz,
        center_freq_hz=center_freq_hz,
        duration_sec=duration_sec,
    )

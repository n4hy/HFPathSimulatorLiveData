"""Channel state recording and playback.

Enables recording of time-varying channel states for:
- Reproducible testing
- Regression testing
- Offline analysis
- Channel replay for modem development

Supports multiple file formats:
- NumPy (.npz) - Simple, fast
- HDF5 (.h5) - Structured, metadata-rich (optional dependency)
- JSON (.json) - Human-readable parameters only
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator
import json
import numpy as np


@dataclass
class ChannelSnapshot:
    """Single snapshot of channel state."""

    timestamp: float  # Time in seconds from start
    transfer_function: np.ndarray  # H(f) complex array
    impulse_response: Optional[np.ndarray] = None  # h(t) if computed
    parameters: Optional[Dict[str, Any]] = None  # Parameter snapshot


@dataclass
class RecordingMetadata:
    """Metadata for channel recording."""

    # Recording info
    created: str  # ISO timestamp
    duration_sec: float
    num_snapshots: int
    snapshot_rate_hz: float

    # Channel configuration
    sample_rate_hz: float
    fft_size: int
    channel_model: str  # "vogler" or "watterson"

    # Ionospheric parameters (if applicable)
    itu_condition: Optional[str] = None
    foF2_mhz: Optional[float] = None
    hmF2_km: Optional[float] = None
    delay_spread_ms: Optional[float] = None
    doppler_spread_hz: Optional[float] = None

    # Notes
    description: str = ""
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ChannelRecorder:
    """Records channel state snapshots over time.

    Usage:
        recorder = ChannelRecorder(channel)
        recorder.start()
        # ... process signals ...
        recorder.stop()
        recorder.save("channel_recording.npz")
    """

    def __init__(
        self,
        channel,
        snapshot_rate_hz: float = 10.0,
        max_duration_sec: float = 3600.0,
    ):
        """Initialize recorder.

        Args:
            channel: HFChannel or WattersonChannel instance
            snapshot_rate_hz: Rate to capture snapshots
            max_duration_sec: Maximum recording duration
        """
        self._channel = channel
        self._snapshot_rate = snapshot_rate_hz
        self._max_duration = max_duration_sec

        self._recording = False
        self._snapshots: List[ChannelSnapshot] = []
        self._start_time = 0.0
        self._last_snapshot_time = 0.0

    def start(self):
        """Start recording."""
        self._recording = True
        self._snapshots = []
        self._start_time = 0.0
        self._last_snapshot_time = 0.0

    def stop(self):
        """Stop recording."""
        self._recording = False

    def capture(self, time: float):
        """Capture a channel snapshot.

        Called automatically if connected to channel callbacks,
        or manually for explicit capture.

        Args:
            time: Current time in seconds
        """
        if not self._recording:
            return

        # Check if enough time has passed for next snapshot
        if time - self._last_snapshot_time < 1.0 / self._snapshot_rate:
            return

        # Check duration limit
        if time - self._start_time > self._max_duration:
            self.stop()
            return

        # Get channel state
        state = self._channel.get_state()

        snapshot = ChannelSnapshot(
            timestamp=time - self._start_time,
            transfer_function=state.transfer_function.copy()
            if state.transfer_function is not None
            else None,
            impulse_response=state.impulse_response.copy()
            if hasattr(state, "impulse_response") and state.impulse_response is not None
            else None,
            parameters=self._extract_parameters(),
        )

        self._snapshots.append(snapshot)
        self._last_snapshot_time = time

    def _extract_parameters(self) -> Dict[str, Any]:
        """Extract current channel parameters."""
        params = {}

        if hasattr(self._channel, "params"):
            p = self._channel.params
            params["foF2_mhz"] = p.foF2
            params["hmF2_km"] = p.hmF2
            params["delay_spread_ms"] = p.delay_spread_ms
            params["doppler_spread_hz"] = p.doppler_spread_hz

        return params

    @property
    def num_snapshots(self) -> int:
        """Get number of recorded snapshots."""
        return len(self._snapshots)

    @property
    def duration(self) -> float:
        """Get recording duration in seconds."""
        if not self._snapshots:
            return 0.0
        return self._snapshots[-1].timestamp

    def save(self, filepath: str, format: str = "auto"):
        """Save recording to file.

        Args:
            filepath: Output file path
            format: "npz", "h5", "json", or "auto" (from extension)
        """
        path = Path(filepath)

        if format == "auto":
            format = path.suffix.lower().lstrip(".")
            if format not in ["npz", "h5", "json"]:
                format = "npz"

        if format == "npz":
            self._save_npz(path)
        elif format == "h5":
            self._save_h5(path)
        elif format == "json":
            self._save_json(path)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _create_metadata(self) -> RecordingMetadata:
        """Create metadata for recording."""
        config = getattr(self._channel, "config", None)
        params = getattr(self._channel, "params", None)

        return RecordingMetadata(
            created=datetime.now().isoformat(),
            duration_sec=self.duration,
            num_snapshots=self.num_snapshots,
            snapshot_rate_hz=self._snapshot_rate,
            sample_rate_hz=config.sample_rate_hz if config else 2_000_000,
            fft_size=config.block_size if config else 4096,
            channel_model=type(self._channel).__name__,
            itu_condition=None,
            foF2_mhz=params.foF2 if params else None,
            hmF2_km=params.hmF2 if params else None,
            delay_spread_ms=params.delay_spread_ms if params else None,
            doppler_spread_hz=params.doppler_spread_hz if params else None,
        )

    def _save_npz(self, path: Path):
        """Save as NumPy compressed archive."""
        # Prepare arrays
        timestamps = np.array([s.timestamp for s in self._snapshots])

        # Stack transfer functions
        H_array = np.stack([s.transfer_function for s in self._snapshots])

        # Stack impulse responses if available
        if self._snapshots[0].impulse_response is not None:
            h_array = np.stack([s.impulse_response for s in self._snapshots])
        else:
            h_array = None

        # Metadata as JSON string
        metadata = self._create_metadata()
        metadata_json = json.dumps(asdict(metadata))

        # Save
        save_dict = {
            "timestamps": timestamps,
            "transfer_functions": H_array,
            "metadata": np.array([metadata_json]),
        }
        if h_array is not None:
            save_dict["impulse_responses"] = h_array

        np.savez_compressed(path, **save_dict)

    def _save_h5(self, path: Path):
        """Save as HDF5 file."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 format: pip install h5py")

        with h5py.File(path, "w") as f:
            # Metadata
            metadata = self._create_metadata()
            meta_grp = f.create_group("metadata")
            for key, value in asdict(metadata).items():
                if value is not None:
                    if isinstance(value, list):
                        value = json.dumps(value)
                    meta_grp.attrs[key] = value

            # Data
            timestamps = np.array([s.timestamp for s in self._snapshots])
            f.create_dataset("timestamps", data=timestamps)

            H_array = np.stack([s.transfer_function for s in self._snapshots])
            f.create_dataset(
                "transfer_functions",
                data=H_array,
                compression="gzip",
            )

            if self._snapshots[0].impulse_response is not None:
                h_array = np.stack([s.impulse_response for s in self._snapshots])
                f.create_dataset(
                    "impulse_responses",
                    data=h_array,
                    compression="gzip",
                )

    def _save_json(self, path: Path):
        """Save metadata only as JSON (no transfer functions)."""
        metadata = self._create_metadata()

        # Add snapshot timestamps
        data = asdict(metadata)
        data["snapshot_times"] = [s.timestamp for s in self._snapshots]
        data["parameters_over_time"] = [s.parameters for s in self._snapshots]

        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class ChannelPlayer:
    """Plays back recorded channel states.

    Usage:
        player = ChannelPlayer.load("recording.npz")
        for H in player.iterate(sample_rate=100):
            output = apply_channel(input, H)
    """

    def __init__(
        self,
        timestamps: np.ndarray,
        transfer_functions: np.ndarray,
        impulse_responses: Optional[np.ndarray] = None,
        metadata: Optional[RecordingMetadata] = None,
    ):
        """Initialize player.

        Args:
            timestamps: Snapshot timestamps
            transfer_functions: H(f) arrays [N_snapshots x N_freq]
            impulse_responses: h(t) arrays if available
            metadata: Recording metadata
        """
        self._timestamps = timestamps
        self._H = transfer_functions
        self._h = impulse_responses
        self.metadata = metadata

        self._current_idx = 0
        self._loop = False

    @classmethod
    def load(cls, filepath: str) -> "ChannelPlayer":
        """Load recording from file.

        Args:
            filepath: Path to recording file

        Returns:
            ChannelPlayer instance
        """
        path = Path(filepath)
        ext = path.suffix.lower()

        if ext == ".npz":
            return cls._load_npz(path)
        elif ext in [".h5", ".hdf5"]:
            return cls._load_h5(path)
        else:
            raise ValueError(f"Unknown format: {ext}")

    @classmethod
    def _load_npz(cls, path: Path) -> "ChannelPlayer":
        """Load from NumPy archive."""
        data = np.load(path, allow_pickle=True)

        timestamps = data["timestamps"]
        transfer_functions = data["transfer_functions"]

        impulse_responses = None
        if "impulse_responses" in data:
            impulse_responses = data["impulse_responses"]

        metadata = None
        if "metadata" in data:
            meta_json = str(data["metadata"][0])
            meta_dict = json.loads(meta_json)
            metadata = RecordingMetadata(**meta_dict)

        return cls(timestamps, transfer_functions, impulse_responses, metadata)

    @classmethod
    def _load_h5(cls, path: Path) -> "ChannelPlayer":
        """Load from HDF5 file."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 format: pip install h5py")

        with h5py.File(path, "r") as f:
            timestamps = f["timestamps"][:]
            transfer_functions = f["transfer_functions"][:]

            impulse_responses = None
            if "impulse_responses" in f:
                impulse_responses = f["impulse_responses"][:]

            metadata = None
            if "metadata" in f:
                meta_dict = dict(f["metadata"].attrs)
                # Parse tags from JSON if present
                if "tags" in meta_dict:
                    meta_dict["tags"] = json.loads(meta_dict["tags"])
                metadata = RecordingMetadata(**meta_dict)

        return cls(timestamps, transfer_functions, impulse_responses, metadata)

    @property
    def num_snapshots(self) -> int:
        """Get number of snapshots."""
        return len(self._timestamps)

    @property
    def duration(self) -> float:
        """Get recording duration."""
        return self._timestamps[-1] - self._timestamps[0]

    @property
    def fft_size(self) -> int:
        """Get FFT size (transfer function length)."""
        return self._H.shape[1]

    def get_snapshot(self, index: int) -> ChannelSnapshot:
        """Get snapshot by index.

        Args:
            index: Snapshot index

        Returns:
            ChannelSnapshot
        """
        return ChannelSnapshot(
            timestamp=self._timestamps[index],
            transfer_function=self._H[index],
            impulse_response=self._h[index] if self._h is not None else None,
        )

    def get_at_time(self, time: float, interpolate: bool = True) -> np.ndarray:
        """Get transfer function at specified time.

        Args:
            time: Time in seconds
            interpolate: If True, interpolate between snapshots

        Returns:
            Transfer function H(f)
        """
        # Find surrounding snapshots
        idx = np.searchsorted(self._timestamps, time)

        if idx == 0:
            return self._H[0]
        if idx >= len(self._timestamps):
            return self._H[-1]

        if not interpolate:
            # Return nearest
            if time - self._timestamps[idx - 1] < self._timestamps[idx] - time:
                return self._H[idx - 1]
            else:
                return self._H[idx]

        # Linear interpolation
        t0, t1 = self._timestamps[idx - 1], self._timestamps[idx]
        alpha = (time - t0) / (t1 - t0)

        H0, H1 = self._H[idx - 1], self._H[idx]

        # Interpolate magnitude and phase separately for smoothness
        mag0, mag1 = np.abs(H0), np.abs(H1)
        phase0, phase1 = np.unwrap(np.angle(H0)), np.unwrap(np.angle(H1))

        mag = mag0 + alpha * (mag1 - mag0)
        phase = phase0 + alpha * (phase1 - phase0)

        return (mag * np.exp(1j * phase)).astype(np.complex64)

    def iterate(
        self,
        rate_hz: Optional[float] = None,
        loop: bool = False,
    ) -> Iterator[np.ndarray]:
        """Iterate through transfer functions.

        Args:
            rate_hz: Output rate (None = original rate)
            loop: If True, loop back to start when finished

        Yields:
            Transfer function H(f) arrays
        """
        self._loop = loop
        self._current_idx = 0

        if rate_hz is None:
            # Use original snapshots
            while True:
                if self._current_idx >= len(self._timestamps):
                    if loop:
                        self._current_idx = 0
                    else:
                        return

                yield self._H[self._current_idx]
                self._current_idx += 1

        else:
            # Interpolate to desired rate
            dt = 1.0 / rate_hz
            time = 0.0
            duration = self.duration

            while True:
                if time > duration:
                    if loop:
                        time = 0.0
                    else:
                        return

                yield self.get_at_time(time, interpolate=True)
                time += dt

    def reset(self):
        """Reset playback to beginning."""
        self._current_idx = 0

    def seek(self, time: float):
        """Seek to specified time.

        Args:
            time: Time in seconds
        """
        self._current_idx = np.searchsorted(self._timestamps, time)


def create_test_recording(
    duration_sec: float = 10.0,
    snapshot_rate_hz: float = 10.0,
    fft_size: int = 4096,
    condition: str = "moderate",
) -> ChannelPlayer:
    """Create a synthetic test recording.

    Useful for testing without a real channel.

    Args:
        duration_sec: Recording duration
        snapshot_rate_hz: Snapshot rate
        fft_size: FFT size
        condition: ITU condition name

    Returns:
        ChannelPlayer with synthetic data
    """
    from .parameters import ITUCondition

    # Get parameters for condition
    condition_map = {
        "quiet": ITUCondition.QUIET,
        "moderate": ITUCondition.MODERATE,
        "disturbed": ITUCondition.DISTURBED,
        "flutter": ITUCondition.FLUTTER,
    }
    itu_cond = condition_map.get(condition, ITUCondition.MODERATE)

    from .parameters import VoglerParameters

    params = VoglerParameters.from_itu_condition(itu_cond)

    # Generate snapshots
    num_snapshots = int(duration_sec * snapshot_rate_hz)
    timestamps = np.arange(num_snapshots) / snapshot_rate_hz

    # Generate transfer functions with time-varying fading
    rng = np.random.default_rng(42)
    H_array = np.zeros((num_snapshots, fft_size), dtype=np.complex64)

    freq = np.fft.fftfreq(fft_size, 1 / 2e6)

    for i, t in enumerate(timestamps):
        # Base response
        H = np.exp(-np.abs(freq) ** 2 / (2 * (100e3) ** 2))

        # Time-varying fading
        fading = (
            rng.standard_normal(fft_size)
            + 1j * rng.standard_normal(fft_size)
        ) / np.sqrt(2)

        # Filter fading by Doppler
        doppler_filter = np.exp(-np.abs(freq) / (params.doppler_spread_hz * 100))
        fading = np.fft.ifft(np.fft.fft(fading) * doppler_filter)

        H_array[i] = (H * (1 + 0.3 * fading)).astype(np.complex64)

    # Create metadata
    metadata = RecordingMetadata(
        created=datetime.now().isoformat(),
        duration_sec=duration_sec,
        num_snapshots=num_snapshots,
        snapshot_rate_hz=snapshot_rate_hz,
        sample_rate_hz=2_000_000,
        fft_size=fft_size,
        channel_model="synthetic",
        itu_condition=condition,
        delay_spread_ms=params.delay_spread_ms,
        doppler_spread_hz=params.doppler_spread_hz,
        description="Synthetic test recording",
        tags=["test", "synthetic"],
    )

    return ChannelPlayer(timestamps, H_array, None, metadata)

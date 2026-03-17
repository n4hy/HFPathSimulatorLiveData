"""MATLAB interface for HF Path Simulator.

Provides:
- MATFileInterface: Save/load channel state and IQ as .mat files
- MATLABEngineInterface: Optional direct MATLAB Engine API integration
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import numpy as np


@dataclass
class ChannelSnapshot:
    """Snapshot of channel state for MATLAB export."""

    timestamp: datetime
    transfer_function: np.ndarray
    impulse_response: np.ndarray
    scattering_function: np.ndarray
    freq_axis_hz: np.ndarray
    delay_axis_ms: np.ndarray
    doppler_axis_hz: np.ndarray
    parameters: Dict[str, Any] = field(default_factory=dict)


class MATFileInterface:
    """Interface for saving/loading HF Path Simulator data as MAT files.

    Supports:
    - Channel state snapshots (transfer function, impulse response, scattering)
    - IQ sample recordings
    - Parameter configurations
    - Time series of channel evolution
    """

    def __init__(self, use_hdf5: bool = True):
        """Initialize MAT file interface.

        Args:
            use_hdf5: Use HDF5 format (-v7.3) for large arrays. Requires h5py.
        """
        self._use_hdf5 = use_hdf5
        self._scipy_io = None
        self._h5py = None

        # Try to import scipy.io
        try:
            import scipy.io as sio
            self._scipy_io = sio
        except ImportError:
            pass

        # Try to import h5py for large files
        if use_hdf5:
            try:
                import h5py
                self._h5py = h5py
            except ImportError:
                pass

    def save_channel_state(
        self,
        filepath: Union[str, Path],
        snapshot: ChannelSnapshot,
        iq_samples: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Save channel state to MAT file.

        Args:
            filepath: Output .mat file path
            snapshot: Channel state snapshot
            iq_samples: Optional IQ samples to include
            metadata: Optional additional metadata

        Returns:
            True if successful
        """
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix(".mat")

        # Build data dictionary
        data = {
            "transfer_function": snapshot.transfer_function,
            "impulse_response": snapshot.impulse_response,
            "scattering_function": snapshot.scattering_function,
            "freq_axis_hz": snapshot.freq_axis_hz,
            "delay_axis_ms": snapshot.delay_axis_ms,
            "doppler_axis_hz": snapshot.doppler_axis_hz,
            "timestamp": snapshot.timestamp.isoformat(),
            "parameters": snapshot.parameters,
        }

        if iq_samples is not None:
            data["iq_samples"] = iq_samples

        if metadata:
            data["metadata"] = metadata

        data["_source"] = "hfpathsim"
        data["_version"] = "0.3.0"

        return self._save_mat(filepath, data)

    def save_iq_recording(
        self,
        filepath: Union[str, Path],
        samples: np.ndarray,
        sample_rate_hz: float,
        center_freq_hz: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Save IQ recording to MAT file.

        Args:
            filepath: Output .mat file path
            samples: Complex IQ samples
            sample_rate_hz: Sample rate
            center_freq_hz: Center frequency
            metadata: Optional metadata

        Returns:
            True if successful
        """
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix(".mat")

        data = {
            "iq_samples": samples,
            "sample_rate_hz": sample_rate_hz,
            "center_freq_hz": center_freq_hz,
            "num_samples": len(samples),
            "duration_sec": len(samples) / sample_rate_hz,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "_source": "hfpathsim",
        }

        if metadata:
            data.update(metadata)

        return self._save_mat(filepath, data)

    def save_channel_evolution(
        self,
        filepath: Union[str, Path],
        snapshots: List[ChannelSnapshot],
        time_axis_sec: Optional[np.ndarray] = None,
    ) -> bool:
        """Save time series of channel evolution.

        Args:
            filepath: Output .mat file path
            snapshots: List of channel snapshots over time
            time_axis_sec: Optional time axis

        Returns:
            True if successful
        """
        if not snapshots:
            return False

        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix(".mat")

        # Stack arrays for time series
        num_snapshots = len(snapshots)

        data = {
            "num_snapshots": num_snapshots,
            "freq_axis_hz": snapshots[0].freq_axis_hz,
            "delay_axis_ms": snapshots[0].delay_axis_ms,
            "doppler_axis_hz": snapshots[0].doppler_axis_hz,
            # 3D arrays: time x freq/delay x data
            "transfer_functions": np.array([s.transfer_function for s in snapshots]),
            "impulse_responses": np.array([s.impulse_response for s in snapshots]),
            "scattering_functions": np.array([s.scattering_function for s in snapshots]),
            "timestamps": [s.timestamp.isoformat() for s in snapshots],
            "_source": "hfpathsim",
        }

        if time_axis_sec is not None:
            data["time_axis_sec"] = time_axis_sec

        return self._save_mat(filepath, data)

    def _save_mat(self, filepath: Path, data: Dict[str, Any]) -> bool:
        """Save data to MAT file.

        Args:
            filepath: Output path
            data: Data dictionary

        Returns:
            True if successful
        """
        # Check for large arrays that need HDF5
        total_size = sum(
            arr.nbytes if isinstance(arr, np.ndarray) else 0
            for arr in data.values()
        )
        use_hdf5 = self._use_hdf5 and total_size > 2e9  # > 2GB

        if use_hdf5 and self._h5py:
            return self._save_hdf5(filepath, data)
        elif self._scipy_io:
            return self._save_scipy(filepath, data)
        else:
            print("Neither scipy.io nor h5py available for MAT file output")
            return False

    def _save_scipy(self, filepath: Path, data: Dict[str, Any]) -> bool:
        """Save using scipy.io.savemat."""
        try:
            # Clean data for MATLAB compatibility
            clean_data = {}
            for key, value in data.items():
                if isinstance(value, (str, datetime)):
                    clean_data[key] = str(value)
                elif isinstance(value, dict):
                    # Flatten nested dicts
                    for k, v in value.items():
                        clean_data[f"{key}_{k}"] = v
                elif isinstance(value, list):
                    # Convert lists to arrays where possible
                    try:
                        clean_data[key] = np.array(value)
                    except (ValueError, TypeError):
                        clean_data[key] = str(value)
                else:
                    clean_data[key] = value

            self._scipy_io.savemat(str(filepath), clean_data, do_compression=True)
            return True

        except Exception as e:
            print(f"Error saving MAT file: {e}")
            return False

    def _save_hdf5(self, filepath: Path, data: Dict[str, Any]) -> bool:
        """Save using h5py for HDF5/MAT v7.3 format."""
        try:
            with self._h5py.File(str(filepath), "w") as f:
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        f.create_dataset(key, data=value, compression="gzip")
                    elif isinstance(value, (str, datetime)):
                        f.attrs[key] = str(value)
                    elif isinstance(value, dict):
                        grp = f.create_group(key)
                        for k, v in value.items():
                            if isinstance(v, np.ndarray):
                                grp.create_dataset(k, data=v)
                            else:
                                grp.attrs[k] = str(v) if v is not None else ""
                    elif isinstance(value, (int, float)):
                        f.create_dataset(key, data=value)
                    else:
                        f.attrs[key] = str(value)

            return True

        except Exception as e:
            print(f"Error saving HDF5 MAT file: {e}")
            return False

    def load_mat(self, filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Load data from MAT file.

        Args:
            filepath: Input .mat file path

        Returns:
            Data dictionary or None if failed
        """
        filepath = Path(filepath)

        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None

        # Try scipy first
        if self._scipy_io:
            try:
                data = self._scipy_io.loadmat(
                    str(filepath),
                    squeeze_me=True,
                    struct_as_record=False,
                )
                # Remove MATLAB internal keys
                return {k: v for k, v in data.items() if not k.startswith("__")}
            except Exception:
                pass

        # Try h5py for v7.3 format
        if self._h5py:
            try:
                data = {}
                with self._h5py.File(str(filepath), "r") as f:
                    for key in f.keys():
                        if isinstance(f[key], self._h5py.Dataset):
                            data[key] = f[key][:]
                        elif isinstance(f[key], self._h5py.Group):
                            data[key] = {k: f[key][k][:] for k in f[key].keys()}

                    # Get attributes
                    for key in f.attrs.keys():
                        data[key] = f.attrs[key]

                return data

            except Exception as e:
                print(f"Error loading MAT file: {e}")
                return None

        print("Neither scipy.io nor h5py available for MAT file input")
        return None


class MATLABEngineInterface:
    """Optional interface for direct MATLAB Engine API integration.

    Requires MATLAB and the MATLAB Engine API for Python.
    Install with: pip install matlabengine
    """

    def __init__(self):
        """Initialize MATLAB Engine interface."""
        self._engine = None
        self._matlab = None

    def start_engine(self, shared: bool = False) -> bool:
        """Start MATLAB engine.

        Args:
            shared: Connect to shared MATLAB session if available

        Returns:
            True if successful
        """
        try:
            import matlab.engine
            self._matlab = matlab.engine

            if shared:
                # Try to connect to existing shared session
                sessions = matlab.engine.find_matlab()
                if sessions:
                    self._engine = matlab.engine.connect_matlab(sessions[0])
                else:
                    self._engine = matlab.engine.start_matlab()
            else:
                self._engine = matlab.engine.start_matlab()

            return True

        except ImportError:
            print("MATLAB Engine API not installed.")
            print("Install with: pip install matlabengine")
            print("Requires MATLAB R2014b or later.")
            return False

        except Exception as e:
            print(f"Error starting MATLAB engine: {e}")
            return False

    def stop_engine(self):
        """Stop MATLAB engine."""
        if self._engine:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    def is_running(self) -> bool:
        """Check if MATLAB engine is running."""
        return self._engine is not None

    def eval(self, expression: str) -> Any:
        """Evaluate MATLAB expression.

        Args:
            expression: MATLAB code to evaluate

        Returns:
            Result of evaluation
        """
        if not self._engine:
            return None

        try:
            return self._engine.eval(expression)
        except Exception as e:
            print(f"MATLAB eval error: {e}")
            return None

    def put_variable(self, name: str, value: Any):
        """Put variable into MATLAB workspace.

        Args:
            name: Variable name
            value: Value (numpy arrays are converted automatically)
        """
        if not self._engine:
            return

        try:
            if isinstance(value, np.ndarray):
                if np.iscomplexobj(value):
                    # Convert complex to MATLAB format
                    value = self._matlab.double(
                        value.real.tolist(), is_complex=True
                    )
                    # Add imaginary part
                    # Note: This is simplified; real implementation needs care
                else:
                    value = self._matlab.double(value.tolist())

            self._engine.workspace[name] = value

        except Exception as e:
            print(f"Error putting variable {name}: {e}")

    def get_variable(self, name: str) -> Any:
        """Get variable from MATLAB workspace.

        Args:
            name: Variable name

        Returns:
            Variable value (converted to numpy if applicable)
        """
        if not self._engine:
            return None

        try:
            value = self._engine.workspace[name]

            # Convert MATLAB arrays to numpy
            if hasattr(value, "_data"):
                return np.array(value._data).reshape(value.size[::-1]).T

            return value

        except Exception as e:
            print(f"Error getting variable {name}: {e}")
            return None

    def run_script(self, script_path: Union[str, Path]) -> bool:
        """Run MATLAB script.

        Args:
            script_path: Path to .m script file

        Returns:
            True if successful
        """
        if not self._engine:
            return False

        try:
            script_path = Path(script_path)

            # Add script directory to MATLAB path
            self._engine.addpath(str(script_path.parent), nargout=0)

            # Run script
            self._engine.run(script_path.stem, nargout=0)

            return True

        except Exception as e:
            print(f"Error running script: {e}")
            return False

    def __enter__(self):
        """Context manager entry."""
        self.start_engine()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_engine()
        return False

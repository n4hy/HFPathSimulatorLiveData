"""Core simulation engine decoupled from GUI.

This module provides the SimulationEngine class that replicates the
processing chain from gui/main_window.py without PyQt6 dependencies.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Any
import threading
import time
import numpy as np

from ..core.channel import HFChannel, ProcessingConfig, RayTracingConfig
from ..core.watterson import WattersonChannel, WattersonConfig
from ..core.vogler_hoffmeyer import VoglerHoffmeyerChannel, VoglerHoffmeyerConfig
from ..core.parameters import VoglerParameters, ITUCondition, ChannelState
from ..core.noise import NoiseGenerator, NoiseConfig
from ..core.impairments import AGC, AGCConfig, Limiter, LimiterConfig
from ..core.impairments import FrequencyOffset, FrequencyOffsetConfig
from ..input.base import InputSource
from ..output.base import OutputSink


class ChannelModel(Enum):
    """Available channel models."""

    VOGLER = "vogler"  # Vogler IPM (default)
    WATTERSON = "watterson"  # Watterson TDL
    VOGLER_HOFFMEYER = "vogler_hoffmeyer"  # Vogler-Hoffmeyer wideband
    PASSTHROUGH = "passthrough"  # No channel processing


@dataclass
class EngineConfig:
    """Configuration for the simulation engine."""

    # Channel model selection
    channel_model: ChannelModel = ChannelModel.VOGLER

    # Processing parameters
    sample_rate_hz: float = 2_000_000
    block_size: int = 4096
    overlap: int = 1024
    channel_update_rate_hz: float = 10.0

    # GPU acceleration
    use_gpu: bool = True

    # Ray tracing (for Vogler model)
    use_ray_tracing: bool = False

    # Processing chain enables
    noise_enabled: bool = False
    agc_enabled: bool = False
    limiter_enabled: bool = False
    freq_offset_enabled: bool = False

    # Processing timing
    process_interval_ms: float = 50.0  # Block processing interval


@dataclass
class EngineState:
    """Current state of the simulation engine."""

    running: bool = False
    total_samples_processed: int = 0
    blocks_processed: int = 0
    current_sample_rate: float = 0.0
    channel_state: Optional[ChannelState] = None

    # Impairment states
    agc_gain_db: float = 0.0
    limiter_reduction_db: float = 0.0
    current_freq_offset_hz: float = 0.0

    # Error state
    last_error: Optional[str] = None


class SimulationEngine:
    """Headless HF channel simulation engine.

    Replicates the processing chain from MainWindow._process_block()
    without GUI dependencies. Suitable for server deployment.

    Processing order:
        Input → Channel → Noise → AGC → Limiter → Freq Offset → Output
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        vogler_params: Optional[VoglerParameters] = None,
    ):
        """Initialize the simulation engine.

        Args:
            config: Engine configuration
            vogler_params: Vogler ionospheric parameters
        """
        self.config = config or EngineConfig()
        self._vogler_params = vogler_params or VoglerParameters.from_itu_condition(
            ITUCondition.MODERATE
        )

        # State
        self._state = EngineState()
        self._lock = threading.Lock()

        # Processing components (initialized lazily)
        self._channel: Optional[HFChannel] = None
        self._watterson: Optional[WattersonChannel] = None
        self._vh_channel: Optional[VoglerHoffmeyerChannel] = None
        self._noise: Optional[NoiseGenerator] = None
        self._agc: Optional[AGC] = None
        self._limiter: Optional[Limiter] = None
        self._freq_offset: Optional[FrequencyOffset] = None

        # Input/output
        self._input_source: Optional[InputSource] = None
        self._output_sink: Optional[OutputSink] = None

        # Streaming thread
        self._streaming_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Callbacks
        self._state_callbacks: list[Callable[[EngineState], None]] = []
        self._channel_state_callbacks: list[Callable[[ChannelState], None]] = []
        self._output_callbacks: list[Callable[[np.ndarray], None]] = []

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize processing components based on configuration."""
        proc_config = ProcessingConfig(
            sample_rate_hz=self.config.sample_rate_hz,
            block_size=self.config.block_size,
            overlap=self.config.overlap,
            channel_update_rate_hz=self.config.channel_update_rate_hz,
        )

        # Initialize channel models
        if self.config.channel_model == ChannelModel.VOGLER:
            self._channel = HFChannel(
                params=self._vogler_params,
                config=proc_config,
                use_gpu=self.config.use_gpu,
                use_ray_tracing=self.config.use_ray_tracing,
            )

        elif self.config.channel_model == ChannelModel.WATTERSON:
            self._watterson = WattersonChannel(
                WattersonConfig.from_itu_condition(
                    ITUCondition.MODERATE,
                    sample_rate_hz=self.config.sample_rate_hz,
                )
            )

        elif self.config.channel_model == ChannelModel.VOGLER_HOFFMEYER:
            self._vh_channel = VoglerHoffmeyerChannel(
                VoglerHoffmeyerConfig.from_itu_condition(
                    ITUCondition.MODERATE,
                    sample_rate=self.config.sample_rate_hz,
                )
            )

        # Initialize impairments
        self._noise = NoiseGenerator(
            NoiseConfig(),
            sample_rate_hz=self.config.sample_rate_hz,
        )
        self._agc = AGC(
            AGCConfig(),
            sample_rate_hz=self.config.sample_rate_hz,
        )
        self._limiter = Limiter(LimiterConfig())
        self._freq_offset = FrequencyOffset(
            FrequencyOffsetConfig(),
            sample_rate_hz=self.config.sample_rate_hz,
        )

    def configure(self, params: dict) -> None:
        """Configure the engine with parameter dictionary.

        Args:
            params: Dictionary of configuration parameters. Supported keys:
                - channel_model: str ("vogler", "watterson", "vogler_hoffmeyer", "passthrough")
                - sample_rate_hz: float
                - block_size: int
                - use_gpu: bool
                - noise_enabled: bool
                - agc_enabled: bool
                - limiter_enabled: bool
                - freq_offset_enabled: bool
                - vogler: dict (VoglerParameters fields)
                - noise: dict (NoiseConfig fields)
                - agc: dict (AGCConfig fields)
                - limiter: dict (LimiterConfig fields)
                - freq_offset: dict (FrequencyOffsetConfig fields)
        """
        with self._lock:
            # Update engine config
            if "channel_model" in params:
                self.config.channel_model = ChannelModel(params["channel_model"])

            if "sample_rate_hz" in params:
                self.config.sample_rate_hz = params["sample_rate_hz"]

            if "block_size" in params:
                self.config.block_size = params["block_size"]

            if "use_gpu" in params:
                self.config.use_gpu = params["use_gpu"]

            if "noise_enabled" in params:
                self.config.noise_enabled = params["noise_enabled"]

            if "agc_enabled" in params:
                self.config.agc_enabled = params["agc_enabled"]

            if "limiter_enabled" in params:
                self.config.limiter_enabled = params["limiter_enabled"]

            if "freq_offset_enabled" in params:
                self.config.freq_offset_enabled = params["freq_offset_enabled"]

            # Update Vogler parameters
            if "vogler" in params and self._channel:
                vogler = params["vogler"]
                if "foF2" in vogler:
                    self._vogler_params.foF2 = vogler["foF2"]
                if "hmF2" in vogler:
                    self._vogler_params.hmF2 = vogler["hmF2"]
                if "foE" in vogler:
                    self._vogler_params.foE = vogler["foE"]
                if "hmE" in vogler:
                    self._vogler_params.hmE = vogler["hmE"]
                if "doppler_spread_hz" in vogler:
                    self._vogler_params.doppler_spread_hz = vogler["doppler_spread_hz"]
                if "delay_spread_ms" in vogler:
                    self._vogler_params.delay_spread_ms = vogler["delay_spread_ms"]

                # Apply to channel
                self._channel.params = self._vogler_params
                self._channel.update_ionosphere()

            # Update noise config
            if "noise" in params and self._noise:
                noise = params["noise"]
                if "snr_db" in noise:
                    self._noise.config.snr_db = noise["snr_db"]
                if "enable_atmospheric" in noise:
                    self._noise.config.enable_atmospheric = noise["enable_atmospheric"]
                if "enable_manmade" in noise:
                    self._noise.config.enable_manmade = noise["enable_manmade"]
                if "enable_impulse" in noise:
                    self._noise.config.enable_impulse = noise["enable_impulse"]
                self._noise._update_noise_levels()

            # Update AGC config
            if "agc" in params and self._agc:
                agc = params["agc"]
                if "target_level_db" in agc:
                    self._agc.config.target_level_db = agc["target_level_db"]
                if "max_gain_db" in agc:
                    self._agc.config.max_gain_db = agc["max_gain_db"]
                if "min_gain_db" in agc:
                    self._agc.config.min_gain_db = agc["min_gain_db"]

            # Update limiter config
            if "limiter" in params and self._limiter:
                limiter = params["limiter"]
                if "threshold_db" in limiter:
                    self._limiter.config.threshold_db = limiter["threshold_db"]

            # Update frequency offset config
            if "freq_offset" in params and self._freq_offset:
                freq = params["freq_offset"]
                if "offset_hz" in freq:
                    self._freq_offset.config.offset_hz = freq["offset_hz"]
                if "drift_hz_per_sec" in freq:
                    self._freq_offset.config.drift_hz_per_sec = freq["drift_hz_per_sec"]

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Process a block of samples through the full chain.

        Processing order:
            Input → Channel → Noise → AGC → Limiter → Freq Offset → Output

        Args:
            samples: Complex input samples (complex64 or complex128)

        Returns:
            Processed complex samples
        """
        with self._lock:
            output = samples.astype(np.complex64)

            # Channel processing
            if self.config.channel_model == ChannelModel.VOGLER and self._channel:
                output = self._channel.process(output)
                self._state.channel_state = self._channel.get_state()

            elif self.config.channel_model == ChannelModel.WATTERSON and self._watterson:
                output = self._watterson.process_block(output)

            elif self.config.channel_model == ChannelModel.VOGLER_HOFFMEYER and self._vh_channel:
                output = self._vh_channel.process(output)

            # elif PASSTHROUGH: no channel processing

            # Noise
            if self.config.noise_enabled and self._noise:
                output = self._noise.add_noise(output)

            # AGC
            if self.config.agc_enabled and self._agc:
                output = self._agc.process_block(output)
                self._state.agc_gain_db = self._agc.current_gain_db

            # Limiter
            if self.config.limiter_enabled and self._limiter:
                output = self._limiter.process(output)
                self._state.limiter_reduction_db = self._limiter.gain_reduction_db

            # Frequency offset
            if self.config.freq_offset_enabled and self._freq_offset:
                output = self._freq_offset.process(output)
                self._state.current_freq_offset_hz = self._freq_offset.config.offset_hz

            # Update state
            self._state.total_samples_processed += len(samples)
            self._state.blocks_processed += 1

            # Notify callbacks
            for callback in self._channel_state_callbacks:
                if self._state.channel_state:
                    callback(self._state.channel_state)

            for callback in self._output_callbacks:
                callback(output)

            return output

    def get_state(self) -> dict:
        """Get current engine state as dictionary.

        Returns:
            Dictionary with engine state
        """
        with self._lock:
            state_dict = {
                "running": self._state.running,
                "total_samples_processed": self._state.total_samples_processed,
                "blocks_processed": self._state.blocks_processed,
                "current_sample_rate": self._state.current_sample_rate,
                "agc_gain_db": self._state.agc_gain_db,
                "limiter_reduction_db": self._state.limiter_reduction_db,
                "current_freq_offset_hz": self._state.current_freq_offset_hz,
                "last_error": self._state.last_error,
            }

            # Add channel state if available
            if self._state.channel_state:
                cs = self._state.channel_state
                state_dict["channel"] = {
                    "has_transfer_function": cs.transfer_function is not None,
                    "has_impulse_response": cs.impulse_response is not None,
                    "has_scattering_function": cs.scattering_function is not None,
                }

            return state_dict

    def get_channel_state(self) -> Optional[ChannelState]:
        """Get current channel state.

        Returns:
            ChannelState object or None
        """
        with self._lock:
            if self.config.channel_model == ChannelModel.VOGLER and self._channel:
                return self._channel.get_state()
            return None

    def start_streaming(
        self,
        input_source: InputSource,
        output_sink: Optional[OutputSink] = None,
    ) -> None:
        """Start continuous streaming processing.

        Args:
            input_source: Source of input samples
            output_sink: Optional destination for processed samples
        """
        if self._state.running:
            raise RuntimeError("Engine is already running")

        self._input_source = input_source
        self._output_sink = output_sink

        # Open input
        if not input_source.is_open:
            if not input_source.open():
                raise RuntimeError("Failed to open input source")

        # Open output if provided
        if output_sink and not output_sink.is_open:
            if not output_sink.open():
                raise RuntimeError("Failed to open output sink")

        self._state.running = True
        self._state.current_sample_rate = input_source.sample_rate
        self._stop_event.clear()

        # Start processing thread
        self._streaming_thread = threading.Thread(
            target=self._streaming_loop,
            daemon=True,
        )
        self._streaming_thread.start()

    def stop_streaming(self) -> None:
        """Stop continuous streaming processing."""
        if not self._state.running:
            return

        self._stop_event.set()

        if self._streaming_thread:
            self._streaming_thread.join(timeout=2.0)
            self._streaming_thread = None

        self._state.running = False

        # Close sources
        if self._input_source and self._input_source.is_open:
            self._input_source.close()

        if self._output_sink and self._output_sink.is_open:
            self._output_sink.close()

    def _streaming_loop(self):
        """Main streaming processing loop."""
        interval = self.config.process_interval_ms / 1000.0
        block_size = self.config.block_size

        while not self._stop_event.is_set():
            start_time = time.perf_counter()

            try:
                # Read input
                samples = self._input_source.read(block_size)

                if samples is None or len(samples) == 0:
                    # No data available, sleep briefly
                    time.sleep(0.001)
                    continue

                # Process
                output = self.process(samples)

                # Write output
                if self._output_sink and self._output_sink.is_open:
                    self._output_sink.write(output)

                # Notify state callbacks
                for callback in self._state_callbacks:
                    callback(self._state)

            except Exception as e:
                self._state.last_error = str(e)

            # Maintain timing
            elapsed = time.perf_counter() - start_time
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def add_state_callback(self, callback: Callable[[EngineState], None]) -> None:
        """Add callback for state updates during streaming."""
        self._state_callbacks.append(callback)

    def remove_state_callback(self, callback: Callable[[EngineState], None]) -> None:
        """Remove state callback."""
        if callback in self._state_callbacks:
            self._state_callbacks.remove(callback)

    def add_channel_state_callback(self, callback: Callable[[ChannelState], None]) -> None:
        """Add callback for channel state updates."""
        self._channel_state_callbacks.append(callback)

    def remove_channel_state_callback(self, callback: Callable[[ChannelState], None]) -> None:
        """Remove channel state callback."""
        if callback in self._channel_state_callbacks:
            self._channel_state_callbacks.remove(callback)

    def add_output_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Add callback for processed output samples."""
        self._output_callbacks.append(callback)

    def remove_output_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Remove output callback."""
        if callback in self._output_callbacks:
            self._output_callbacks.remove(callback)

    def reset(self) -> None:
        """Reset engine state and reinitialize components."""
        with self._lock:
            self._state = EngineState()
            self._init_components()

    # Channel model configuration shortcuts

    def configure_vogler(
        self,
        foF2: Optional[float] = None,
        hmF2: Optional[float] = None,
        foE: Optional[float] = None,
        hmE: Optional[float] = None,
        doppler_spread_hz: Optional[float] = None,
        delay_spread_ms: Optional[float] = None,
    ) -> None:
        """Configure Vogler channel parameters.

        Args:
            foF2: F2 critical frequency (MHz)
            hmF2: F2 peak height (km)
            foE: E critical frequency (MHz)
            hmE: E peak height (km)
            doppler_spread_hz: Two-sided Doppler spread
            delay_spread_ms: Delay spread
        """
        params = {}
        if foF2 is not None:
            params["foF2"] = foF2
        if hmF2 is not None:
            params["hmF2"] = hmF2
        if foE is not None:
            params["foE"] = foE
        if hmE is not None:
            params["hmE"] = hmE
        if doppler_spread_hz is not None:
            params["doppler_spread_hz"] = doppler_spread_hz
        if delay_spread_ms is not None:
            params["delay_spread_ms"] = delay_spread_ms

        self.configure({"vogler": params})

    def configure_watterson(
        self,
        condition: Optional[ITUCondition] = None,
    ) -> None:
        """Configure Watterson channel with ITU condition.

        Args:
            condition: ITU-R F.1487 condition (QUIET, MODERATE, DISTURBED, FLUTTER)
        """
        if condition and self._watterson:
            new_config = WattersonConfig.from_itu_condition(
                condition,
                sample_rate_hz=self.config.sample_rate_hz,
            )
            self._watterson = WattersonChannel(new_config)

    def configure_noise(
        self,
        snr_db: Optional[float] = None,
        enable_atmospheric: Optional[bool] = None,
        enable_manmade: Optional[bool] = None,
        enable_impulse: Optional[bool] = None,
    ) -> None:
        """Configure noise generator.

        Args:
            snr_db: Signal-to-noise ratio
            enable_atmospheric: Enable atmospheric noise
            enable_manmade: Enable man-made noise
            enable_impulse: Enable impulse noise
        """
        params = {}
        if snr_db is not None:
            params["snr_db"] = snr_db
        if enable_atmospheric is not None:
            params["enable_atmospheric"] = enable_atmospheric
        if enable_manmade is not None:
            params["enable_manmade"] = enable_manmade
        if enable_impulse is not None:
            params["enable_impulse"] = enable_impulse

        self.configure({"noise": params, "noise_enabled": True})

    def configure_agc(
        self,
        enabled: bool = True,
        target_level_db: Optional[float] = None,
        max_gain_db: Optional[float] = None,
        min_gain_db: Optional[float] = None,
    ) -> None:
        """Configure AGC.

        Args:
            enabled: Enable AGC
            target_level_db: Target output level
            max_gain_db: Maximum gain
            min_gain_db: Minimum gain
        """
        params = {}
        if target_level_db is not None:
            params["target_level_db"] = target_level_db
        if max_gain_db is not None:
            params["max_gain_db"] = max_gain_db
        if min_gain_db is not None:
            params["min_gain_db"] = min_gain_db

        self.configure({"agc": params, "agc_enabled": enabled})

    def configure_limiter(
        self,
        enabled: bool = True,
        threshold_db: Optional[float] = None,
    ) -> None:
        """Configure limiter.

        Args:
            enabled: Enable limiter
            threshold_db: Limiting threshold
        """
        params = {}
        if threshold_db is not None:
            params["threshold_db"] = threshold_db

        self.configure({"limiter": params, "limiter_enabled": enabled})

    def configure_freq_offset(
        self,
        enabled: bool = True,
        offset_hz: Optional[float] = None,
        drift_hz_per_sec: Optional[float] = None,
    ) -> None:
        """Configure frequency offset.

        Args:
            enabled: Enable frequency offset
            offset_hz: Fixed offset
            drift_hz_per_sec: Drift rate
        """
        params = {}
        if offset_hz is not None:
            params["offset_hz"] = offset_hz
        if drift_hz_per_sec is not None:
            params["drift_hz_per_sec"] = drift_hz_per_sec

        self.configure({"freq_offset": params, "freq_offset_enabled": enabled})

    # GPU info

    def get_gpu_info(self) -> dict:
        """Get GPU information.

        Returns:
            Dictionary with GPU details or empty if unavailable
        """
        try:
            from ..gpu import get_device_info, is_available

            if is_available():
                return get_device_info()
            return {"available": False}
        except ImportError:
            return {"available": False, "error": "GPU module not installed"}

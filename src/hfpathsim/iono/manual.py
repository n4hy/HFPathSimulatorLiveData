"""Manual ionospheric parameter entry."""

from dataclasses import dataclass
from typing import Optional, Callable
from datetime import datetime

from hfpathsim.core.parameters import VoglerParameters, ITUCondition, ITU_PRESETS


@dataclass
class IonoObservation:
    """Single ionospheric observation."""

    timestamp: datetime
    foF2: float  # MHz
    hmF2: float  # km
    foE: Optional[float] = None
    hmE: Optional[float] = None
    foF1: Optional[float] = None
    fmin: Optional[float] = None  # Minimum observed frequency
    muf_3000: Optional[float] = None  # MUF(3000)F2


class ManualIonoSource:
    """Manual ionospheric parameter source.

    Allows direct entry of ionospheric parameters or selection
    of ITU-R F.1487 condition presets.
    """

    def __init__(self):
        """Initialize manual source."""
        self._current = IonoObservation(
            timestamp=datetime.now(),
            foF2=7.5,
            hmF2=300.0,
            foE=3.0,
            hmE=110.0,
        )
        self._callbacks: list[Callable[[IonoObservation], None]] = []

    def get_current(self) -> IonoObservation:
        """Get current ionospheric observation."""
        return self._current

    def set_parameters(
        self,
        foF2: Optional[float] = None,
        hmF2: Optional[float] = None,
        foE: Optional[float] = None,
        hmE: Optional[float] = None,
        foF1: Optional[float] = None,
        fmin: Optional[float] = None,
        muf_3000: Optional[float] = None,
    ):
        """Update ionospheric parameters.

        Args:
            foF2: F2 layer critical frequency (MHz)
            hmF2: F2 layer peak height (km)
            foE: E layer critical frequency (MHz)
            hmE: E layer peak height (km)
            foF1: F1 layer critical frequency (MHz)
            fmin: Minimum observed frequency (MHz)
            muf_3000: MUF(3000)F2 (MHz)
        """
        if foF2 is not None:
            self._current.foF2 = foF2
        if hmF2 is not None:
            self._current.hmF2 = hmF2
        if foE is not None:
            self._current.foE = foE
        if hmE is not None:
            self._current.hmE = hmE
        if foF1 is not None:
            self._current.foF1 = foF1
        if fmin is not None:
            self._current.fmin = fmin
        if muf_3000 is not None:
            self._current.muf_3000 = muf_3000

        self._current.timestamp = datetime.now()

        # Notify callbacks
        for callback in self._callbacks:
            callback(self._current)

    def set_from_itu_condition(self, condition: ITUCondition):
        """Set parameters from ITU-R F.1487 condition preset.

        Args:
            condition: ITU condition enum
        """
        params = VoglerParameters.from_itu_condition(condition)
        self.set_parameters(
            foF2=params.foF2,
            hmF2=params.hmF2,
            foE=params.foE,
            hmE=params.hmE,
        )

    def to_vogler_params(
        self,
        frequency_mhz: float = 10.0,
        path_length_km: float = 1000.0,
        doppler_spread_hz: float = 1.0,
        delay_spread_ms: float = 2.0,
    ) -> VoglerParameters:
        """Convert current observation to Vogler parameters.

        Args:
            frequency_mhz: Operating frequency
            path_length_km: Path length
            doppler_spread_hz: Doppler spread
            delay_spread_ms: Delay spread

        Returns:
            VoglerParameters instance
        """
        return VoglerParameters(
            foF2=self._current.foF2,
            hmF2=self._current.hmF2,
            foE=self._current.foE or 3.0,
            hmE=self._current.hmE or 110.0,
            frequency_mhz=frequency_mhz,
            path_length_km=path_length_km,
            doppler_spread_hz=doppler_spread_hz,
            delay_spread_ms=delay_spread_ms,
        )

    def add_callback(self, callback: Callable[[IonoObservation], None]):
        """Register callback for parameter updates."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[IonoObservation], None]):
        """Remove parameter update callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    @staticmethod
    def get_typical_values(
        time_of_day: str = "day",
        solar_activity: str = "medium",
        latitude: str = "mid",
    ) -> dict:
        """Get typical ionospheric values for given conditions.

        Args:
            time_of_day: "day" or "night"
            solar_activity: "low", "medium", or "high"
            latitude: "low", "mid", or "high"

        Returns:
            Dictionary of typical parameter values
        """
        # Base values for mid-latitude, medium solar activity, daytime
        base = {
            "foF2": 7.0,
            "hmF2": 300.0,
            "foE": 3.0,
            "hmE": 110.0,
        }

        # Adjust for time of day
        if time_of_day == "night":
            base["foF2"] *= 0.5
            base["hmF2"] += 50
            base["foE"] *= 0.3

        # Adjust for solar activity
        if solar_activity == "low":
            base["foF2"] *= 0.7
        elif solar_activity == "high":
            base["foF2"] *= 1.4

        # Adjust for latitude
        if latitude == "low":
            base["foF2"] *= 1.2
            base["hmF2"] -= 30
        elif latitude == "high":
            base["foF2"] *= 0.8
            base["hmF2"] += 30

        return base

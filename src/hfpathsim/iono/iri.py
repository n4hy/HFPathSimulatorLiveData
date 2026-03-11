"""IRI-2020 ionospheric model wrapper."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple
import numpy as np


@dataclass
class IRIProfile:
    """IRI electron density profile."""

    altitude_km: np.ndarray  # Altitude array
    electron_density: np.ndarray  # Ne in m^-3
    foF2: float  # MHz
    hmF2: float  # km
    foF1: Optional[float] = None
    hmF1: Optional[float] = None
    foE: Optional[float] = None
    hmE: Optional[float] = None
    b0: Optional[float] = None  # IRI B0 parameter
    b1: Optional[float] = None  # IRI B1 parameter


class IRIModel:
    """International Reference Ionosphere (IRI-2020) model wrapper.

    Provides ionospheric parameters from the IRI empirical model,
    which is based on decades of ionosonde observations.

    Requires the iri2016 package: pip install iri2016
    """

    def __init__(self):
        """Initialize IRI model."""
        self._iri_available = self._check_iri()

    def _check_iri(self) -> bool:
        """Check if IRI model is available."""
        try:
            import iri2016

            return True
        except ImportError:
            return False

    @property
    def available(self) -> bool:
        """Check if IRI model is available."""
        return self._iri_available

    def get_profile(
        self,
        latitude: float,
        longitude: float,
        time: Optional[datetime] = None,
        alt_min_km: float = 60.0,
        alt_max_km: float = 600.0,
        alt_step_km: float = 5.0,
    ) -> Optional[IRIProfile]:
        """Get IRI electron density profile.

        Args:
            latitude: Geographic latitude (degrees)
            longitude: Geographic longitude (degrees)
            time: Date/time for model (default: now)
            alt_min_km: Minimum altitude
            alt_max_km: Maximum altitude
            alt_step_km: Altitude step

        Returns:
            IRIProfile object or None if unavailable
        """
        if not self._iri_available:
            print("IRI model not available. Install with: pip install iri2016")
            return None

        if time is None:
            time = datetime.utcnow()

        try:
            import iri2016

            # Run IRI model
            altitudes = np.arange(alt_min_km, alt_max_km, alt_step_km)
            iri = iri2016.IRI(time, altitudes, latitude, longitude)

            # Extract profile
            ne = iri["ne"].values  # Electron density

            # Extract characteristics
            # IRI provides these at specific altitudes
            profile = IRIProfile(
                altitude_km=altitudes,
                electron_density=ne,
                foF2=self._ne_to_freq(ne.max()),  # Approximate
                hmF2=altitudes[np.argmax(ne)],
            )

            # Find E layer peak (typically 100-120 km)
            e_mask = (altitudes >= 90) & (altitudes <= 130)
            if np.any(e_mask):
                e_ne = ne[e_mask]
                e_alt = altitudes[e_mask]
                profile.foE = self._ne_to_freq(e_ne.max())
                profile.hmE = e_alt[np.argmax(e_ne)]

            # Find F1 layer (if present, typically 150-200 km daytime)
            f1_mask = (altitudes >= 140) & (altitudes <= 220)
            if np.any(f1_mask):
                f1_ne = ne[f1_mask]
                f1_alt = altitudes[f1_mask]
                # Check for local maximum (F1 layer)
                f1_peak_idx = np.argmax(f1_ne)
                if 0 < f1_peak_idx < len(f1_ne) - 1:
                    profile.foF1 = self._ne_to_freq(f1_ne[f1_peak_idx])
                    profile.hmF1 = f1_alt[f1_peak_idx]

            return profile

        except Exception as e:
            print(f"IRI model error: {e}")
            return None

    def get_parameters(
        self,
        latitude: float,
        longitude: float,
        time: Optional[datetime] = None,
    ) -> Optional[dict]:
        """Get ionospheric parameters for location.

        Args:
            latitude: Geographic latitude (degrees)
            longitude: Geographic longitude (degrees)
            time: Date/time for model (default: now)

        Returns:
            Dictionary of ionospheric parameters
        """
        profile = self.get_profile(latitude, longitude, time)
        if profile is None:
            return None

        return {
            "foF2": profile.foF2,
            "hmF2": profile.hmF2,
            "foF1": profile.foF1,
            "hmF1": profile.hmF1,
            "foE": profile.foE,
            "hmE": profile.hmE,
        }

    def get_muf(
        self,
        tx_lat: float,
        tx_lon: float,
        rx_lat: float,
        rx_lon: float,
        time: Optional[datetime] = None,
    ) -> Optional[float]:
        """Calculate Maximum Usable Frequency for a path.

        Simplified calculation using midpoint ionosphere.

        Args:
            tx_lat: Transmitter latitude
            tx_lon: Transmitter longitude
            rx_lat: Receiver latitude
            rx_lon: Receiver longitude
            time: Date/time

        Returns:
            MUF in MHz or None
        """
        # Use midpoint of path
        mid_lat = (tx_lat + rx_lat) / 2
        mid_lon = (tx_lon + rx_lon) / 2

        params = self.get_parameters(mid_lat, mid_lon, time)
        if params is None or params.get("foF2") is None:
            return None

        # Calculate path length
        path_km = self._great_circle_distance(tx_lat, tx_lon, rx_lat, rx_lon)

        # MUF factor (simplified secant law)
        # More accurate calculation requires ray tracing
        hmF2 = params.get("hmF2", 300.0)
        half_path = path_km / 2
        sec_phi = np.sqrt(1 + (half_path / hmF2) ** 2)

        muf = params["foF2"] * sec_phi

        return muf

    def _ne_to_freq(self, ne: float) -> float:
        """Convert electron density to plasma frequency.

        Args:
            ne: Electron density in m^-3

        Returns:
            Plasma frequency in MHz
        """
        # fp = 9 * sqrt(Ne) Hz, where Ne is in m^-3
        return 9e-6 * np.sqrt(ne)

    def _freq_to_ne(self, freq_mhz: float) -> float:
        """Convert plasma frequency to electron density.

        Args:
            freq_mhz: Plasma frequency in MHz

        Returns:
            Electron density in m^-3
        """
        return (freq_mhz * 1e6 / 9) ** 2

    def _great_circle_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate great circle distance between points.

        Args:
            lat1, lon1: First point (degrees)
            lat2, lon2: Second point (degrees)

        Returns:
            Distance in km
        """
        R = 6371.0  # Earth radius km

        lat1_r = np.radians(lat1)
        lat2_r = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    def to_vogler_params(
        self,
        latitude: float,
        longitude: float,
        time: Optional[datetime] = None,
        frequency_mhz: float = 10.0,
        path_length_km: float = 1000.0,
        doppler_spread_hz: float = 1.0,
        delay_spread_ms: float = 2.0,
    ):
        """Get Vogler parameters from IRI model.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            time: Date/time
            frequency_mhz: Operating frequency
            path_length_km: Path length
            doppler_spread_hz: Doppler spread
            delay_spread_ms: Delay spread

        Returns:
            VoglerParameters instance
        """
        from hfpathsim.core.parameters import VoglerParameters

        params = self.get_parameters(latitude, longitude, time)

        if params is None:
            return VoglerParameters()

        return VoglerParameters(
            foF2=params.get("foF2") or 7.0,
            hmF2=params.get("hmF2") or 300.0,
            foE=params.get("foE") or 3.0,
            hmE=params.get("hmE") or 110.0,
            frequency_mhz=frequency_mhz,
            path_length_km=path_length_km,
            doppler_spread_hz=doppler_spread_hz,
            delay_spread_ms=delay_spread_ms,
        )

    def to_ionosphere_profile(
        self,
        latitude: float,
        longitude: float,
        time: Optional[datetime] = None,
    ):
        """Convert IRI data to IonosphereProfile for ray tracing.

        Args:
            latitude: Geographic latitude (degrees)
            longitude: Geographic longitude (degrees)
            time: Date/time for model

        Returns:
            IonosphereProfile instance or None if unavailable
        """
        from hfpathsim.core.raytracing.ionosphere import IonosphereProfile

        profile = self.get_profile(latitude, longitude, time)

        if profile is None:
            return None

        return IonosphereProfile(
            altitude_km=profile.altitude_km,
            electron_density=profile.electron_density,
            foF2=profile.foF2,
            hmF2=profile.hmF2,
            foE=profile.foE or 3.0,
            hmE=profile.hmE or 110.0,
            foF1=profile.foF1,
            hmF1=profile.hmF1,
        )

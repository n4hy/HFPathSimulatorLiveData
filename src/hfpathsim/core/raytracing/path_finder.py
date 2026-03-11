"""Multi-hop propagation mode finder.

Discovers all viable propagation modes (1F, 2F, 3F, 1E, Es)
for a given path and converts them to PropagationMode objects.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .geometry import (
    great_circle_distance,
    sec_phi_spherical,
    group_delay_ms,
    EARTH_RADIUS_KM,
)
from .ionosphere import IonosphereProfile, create_simple_profile
from .ray_engine import RayEngine, RayPath, trace_multihop


@dataclass
class PropagationModeResult:
    """Extended propagation mode with ray-traced parameters.

    This extends the base PropagationMode from core.parameters
    with physically-derived values from ray tracing.
    """

    name: str  # e.g., "1F2", "2F2", "1E", "Es"
    enabled: bool = True
    relative_amplitude: float = 1.0
    delay_offset_ms: float = 0.0

    # Ray-traced parameters
    launch_angle_deg: float = 0.0
    reflection_height_km: float = 0.0
    sec_phi: float = 1.0
    n_hops: int = 1
    layer: str = "F2"
    group_delay_ms: float = 0.0
    ground_range_km: float = 0.0

    def to_propagation_mode(self):
        """Convert to base PropagationMode.

        Returns:
            PropagationMode instance for channel model
        """
        # Avoid circular import
        from hfpathsim.core.parameters import PropagationMode

        return PropagationMode(
            name=self.name,
            enabled=self.enabled,
            relative_amplitude=self.relative_amplitude,
            delay_offset_ms=self.delay_offset_ms,
        )


class PathFinder:
    """Find all viable propagation modes for a given HF path.

    Uses ray tracing to discover multi-hop modes through
    different ionospheric layers.
    """

    def __init__(
        self,
        profile: IonosphereProfile,
        earth_radius: float = EARTH_RADIUS_KM,
    ):
        """Initialize path finder.

        Args:
            profile: Ionospheric electron density profile
            earth_radius: Earth radius in km
        """
        self.profile = profile
        self.earth_radius = earth_radius
        self.engine = RayEngine(profile, earth_radius)

    def find_modes(
        self,
        frequency_mhz: float,
        tx_lat: float,
        tx_lon: float,
        rx_lat: float,
        rx_lon: float,
        max_hops: int = 3,
        use_simplified: bool = True,
    ) -> List[PropagationModeResult]:
        """Find all propagation modes for a path.

        Args:
            frequency_mhz: Operating frequency in MHz
            tx_lat, tx_lon: Transmitter coordinates (degrees)
            rx_lat, rx_lon: Receiver coordinates (degrees)
            max_hops: Maximum number of hops to consider
            use_simplified: Use simplified geometry (faster) vs full ray trace

        Returns:
            List of PropagationModeResult objects
        """
        # Calculate path length
        path_km = great_circle_distance(tx_lat, tx_lon, rx_lat, rx_lon)

        if use_simplified:
            return self._find_modes_simplified(frequency_mhz, path_km, max_hops)
        else:
            return self._find_modes_raytraced(frequency_mhz, path_km, max_hops)

    def _find_modes_simplified(
        self,
        f_mhz: float,
        path_km: float,
        max_hops: int,
    ) -> List[PropagationModeResult]:
        """Find modes using simplified geometry (no full ray tracing).

        This is faster and uses the secant law approximation.

        Args:
            f_mhz: Frequency in MHz
            path_km: Path length in km
            max_hops: Maximum hops

        Returns:
            List of viable modes
        """
        modes = []

        # F2 layer modes
        for n_hops in range(1, max_hops + 1):
            mode = self._check_mode_f2(f_mhz, path_km, n_hops)
            if mode is not None:
                modes.append(mode)

        # E layer mode (usually only viable for shorter paths)
        if path_km < 2500:
            mode = self._check_mode_e(f_mhz, path_km)
            if mode is not None:
                modes.append(mode)

        # Sporadic-E mode
        if self.profile.foEs is not None and path_km < 2500:
            mode = self._check_mode_es(f_mhz, path_km)
            if mode is not None:
                modes.append(mode)

        # Calculate relative amplitudes based on path loss
        self._compute_relative_amplitudes(modes)

        # Calculate delay offsets relative to fastest mode
        self._compute_delay_offsets(modes)

        return modes

    def _check_mode_f2(
        self, f_mhz: float, path_km: float, n_hops: int
    ) -> Optional[PropagationModeResult]:
        """Check if F2 layer mode is viable.

        Args:
            f_mhz: Frequency in MHz
            path_km: Path length in km
            n_hops: Number of hops

        Returns:
            PropagationModeResult or None
        """
        # Range per hop
        range_per_hop = path_km / n_hops

        # Check if single-hop range is reasonable
        # Maximum F2 single-hop is about 4000 km
        if range_per_hop > 4000:
            return None

        # Calculate sec(phi) for this geometry
        sec_phi = sec_phi_spherical(range_per_hop, self.profile.hmF2, self.earth_radius)

        # MUF for this path
        muf = self.profile.foF2 * sec_phi

        # Check if frequency is below MUF
        if f_mhz > muf:
            return None

        # Calculate group delay
        delay = group_delay_ms(path_km, self.profile.hmF2, n_hops, self.earth_radius)

        # Estimate launch angle
        # Simple geometry: tan(elevation) = 2*h / d
        half_hop = range_per_hop / 2
        launch_angle = np.degrees(np.arctan2(2 * self.profile.hmF2, half_hop))

        return PropagationModeResult(
            name=f"{n_hops}F2",
            enabled=True,
            relative_amplitude=1.0,
            delay_offset_ms=0.0,
            launch_angle_deg=launch_angle,
            reflection_height_km=self.profile.hmF2,
            sec_phi=sec_phi,
            n_hops=n_hops,
            layer="F2",
            group_delay_ms=delay,
            ground_range_km=path_km,
        )

    def _check_mode_e(
        self, f_mhz: float, path_km: float
    ) -> Optional[PropagationModeResult]:
        """Check if E layer mode is viable.

        E layer typically supports single-hop up to ~2000 km.

        Args:
            f_mhz: Frequency in MHz
            path_km: Path length in km

        Returns:
            PropagationModeResult or None
        """
        # Maximum E layer single-hop distance
        if path_km > 2500:
            return None

        # Calculate sec(phi)
        sec_phi = sec_phi_spherical(path_km, self.profile.hmE, self.earth_radius)

        # MUF for E layer
        muf = self.profile.foE * sec_phi

        if f_mhz > muf:
            return None

        # Group delay
        delay = group_delay_ms(path_km, self.profile.hmE, 1, self.earth_radius)

        # Launch angle
        half_path = path_km / 2
        launch_angle = np.degrees(np.arctan2(2 * self.profile.hmE, half_path))

        return PropagationModeResult(
            name="1E",
            enabled=True,
            relative_amplitude=0.8,  # E layer typically weaker
            delay_offset_ms=0.0,
            launch_angle_deg=launch_angle,
            reflection_height_km=self.profile.hmE,
            sec_phi=sec_phi,
            n_hops=1,
            layer="E",
            group_delay_ms=delay,
            ground_range_km=path_km,
        )

    def _check_mode_es(
        self, f_mhz: float, path_km: float
    ) -> Optional[PropagationModeResult]:
        """Check if sporadic-E mode is viable.

        Args:
            f_mhz: Frequency in MHz
            path_km: Path length in km

        Returns:
            PropagationModeResult or None
        """
        if self.profile.foEs is None or self.profile.hmEs is None:
            return None

        # Es maximum single-hop similar to E layer
        if path_km > 2500:
            return None

        # Calculate sec(phi)
        sec_phi = sec_phi_spherical(path_km, self.profile.hmEs, self.earth_radius)

        # MUF for Es layer
        muf = self.profile.foEs * sec_phi

        if f_mhz > muf:
            return None

        # Group delay
        delay = group_delay_ms(path_km, self.profile.hmEs, 1, self.earth_radius)

        # Launch angle
        half_path = path_km / 2
        launch_angle = np.degrees(np.arctan2(2 * self.profile.hmEs, half_path))

        return PropagationModeResult(
            name="Es",
            enabled=True,
            relative_amplitude=0.9,
            delay_offset_ms=0.0,
            launch_angle_deg=launch_angle,
            reflection_height_km=self.profile.hmEs,
            sec_phi=sec_phi,
            n_hops=1,
            layer="Es",
            group_delay_ms=delay,
            ground_range_km=path_km,
        )

    def _find_modes_raytraced(
        self,
        f_mhz: float,
        path_km: float,
        max_hops: int,
    ) -> List[PropagationModeResult]:
        """Find modes using full ray tracing.

        More accurate but slower than simplified method.

        Args:
            f_mhz: Frequency in MHz
            path_km: Path length in km
            max_hops: Maximum hops

        Returns:
            List of viable modes
        """
        modes = []

        for n_hops in range(1, max_hops + 1):
            ray = trace_multihop(self.engine, f_mhz, path_km, n_hops)

            if ray is not None and ray.valid:
                mode = PropagationModeResult(
                    name=f"{n_hops}{ray.layer}",
                    enabled=True,
                    relative_amplitude=1.0,
                    delay_offset_ms=0.0,
                    launch_angle_deg=ray.launch_angle_deg,
                    reflection_height_km=ray.reflection_height_km,
                    sec_phi=ray.sec_phi(),
                    n_hops=n_hops,
                    layer=ray.layer,
                    group_delay_ms=ray.group_delay_ms,
                    ground_range_km=path_km,
                )
                modes.append(mode)

        self._compute_relative_amplitudes(modes)
        self._compute_delay_offsets(modes)

        return modes

    def _compute_relative_amplitudes(self, modes: List[PropagationModeResult]):
        """Compute relative amplitudes for modes.

        Based on path loss considerations:
        - More hops = more loss
        - Lower layers (E) typically weaker
        - Sporadic-E variable

        Args:
            modes: List of modes to update in place
        """
        if not modes:
            return

        # Find strongest mode (fewest hops through F2)
        max_amp = 0.0
        for mode in modes:
            # Base amplitude decreases with hops
            amp = 1.0 / mode.n_hops

            # Layer adjustments
            if mode.layer == "E":
                amp *= 0.7
            elif mode.layer == "Es":
                amp *= 0.8

            mode.relative_amplitude = amp
            if amp > max_amp:
                max_amp = amp

        # Normalize to strongest mode
        if max_amp > 0:
            for mode in modes:
                mode.relative_amplitude /= max_amp

    def _compute_delay_offsets(self, modes: List[PropagationModeResult]):
        """Compute delay offsets relative to fastest mode.

        Args:
            modes: List of modes to update in place
        """
        if not modes:
            return

        # Find minimum delay
        min_delay = min(m.group_delay_ms for m in modes)

        # Set offsets
        for mode in modes:
            mode.delay_offset_ms = mode.group_delay_ms - min_delay


def find_propagation_modes(
    profile: IonosphereProfile,
    tx_lat: float,
    tx_lon: float,
    rx_lat: float,
    rx_lon: float,
    frequency_mhz: float,
    max_hops: int = 3,
) -> List[PropagationModeResult]:
    """Convenience function to find propagation modes.

    This is the main entry point for mode discovery.

    Args:
        profile: Ionospheric profile
        tx_lat, tx_lon: Transmitter coordinates
        rx_lat, rx_lon: Receiver coordinates
        frequency_mhz: Operating frequency
        max_hops: Maximum hops to consider

    Returns:
        List of viable propagation modes
    """
    finder = PathFinder(profile)
    return finder.find_modes(
        frequency_mhz,
        tx_lat, tx_lon,
        rx_lat, rx_lon,
        max_hops,
    )


def modes_to_propagation_modes(modes: List[PropagationModeResult]) -> list:
    """Convert PropagationModeResults to base PropagationMode objects.

    Args:
        modes: List of PropagationModeResult

    Returns:
        List of PropagationMode for channel model
    """
    return [m.to_propagation_mode() for m in modes]


def estimate_muf(
    profile: IonosphereProfile,
    path_km: float,
    layer: str = "F2",
) -> float:
    """Estimate MUF using simplified geometry.

    Args:
        profile: Ionospheric profile
        path_km: Path length in km
        layer: Layer to compute MUF for ("F2", "E", "Es")

    Returns:
        MUF in MHz
    """
    if layer == "F2":
        fo = profile.foF2
        hm = profile.hmF2
    elif layer == "E":
        fo = profile.foE
        hm = profile.hmE
    elif layer == "Es" and profile.foEs is not None:
        fo = profile.foEs
        hm = profile.hmEs or 105.0
    else:
        return 0.0

    sec_phi = sec_phi_spherical(path_km, hm)
    return fo * sec_phi

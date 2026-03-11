"""Ionospheric profile model for ray tracing.

Provides electron density profiles, plasma frequency calculations,
and refractive index computation for HF ray tracing.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


# Physical constants
ELECTRON_CHARGE = 1.60217663e-19  # Coulombs
ELECTRON_MASS = 9.1093837e-31  # kg
PERMITTIVITY_0 = 8.8541878e-12  # F/m
C_LIGHT = 299792458.0  # m/s


def plasma_frequency_from_ne(ne: float) -> float:
    """Calculate plasma frequency from electron density.

    fp = (1/2pi) * sqrt(Ne * e^2 / (epsilon_0 * me))
       = 9 * sqrt(Ne) Hz (for Ne in m^-3)
       = 9e-6 * sqrt(Ne) MHz

    Args:
        ne: Electron density in m^-3

    Returns:
        Plasma frequency in MHz
    """
    if ne <= 0:
        return 0.0
    return 9e-6 * np.sqrt(ne)


def ne_from_plasma_frequency(fp_mhz: float) -> float:
    """Calculate electron density from plasma frequency.

    Args:
        fp_mhz: Plasma frequency in MHz

    Returns:
        Electron density in m^-3
    """
    if fp_mhz <= 0:
        return 0.0
    return (fp_mhz * 1e6 / 9) ** 2


@dataclass
class IonosphereProfile:
    """Ionospheric electron density profile for ray tracing.

    Stores altitude-dependent electron density and derived quantities
    needed for ray tracing through the ionosphere.
    """

    altitude_km: np.ndarray  # Altitude grid [km]
    electron_density: np.ndarray  # Ne at each altitude [m^-3]

    # Layer characteristics
    foF2: float  # F2 layer critical frequency [MHz]
    hmF2: float  # F2 layer peak height [km]
    foE: float = 3.0  # E layer critical frequency [MHz]
    hmE: float = 110.0  # E layer peak height [km]

    # Optional sporadic-E parameters
    foEs: Optional[float] = None  # Sporadic-E critical frequency [MHz]
    hmEs: Optional[float] = None  # Sporadic-E height [km]

    # Optional F1 layer
    foF1: Optional[float] = None
    hmF1: Optional[float] = None

    # Layer shape parameters (semi-thickness)
    ym_F2: float = 100.0  # F2 semi-thickness [km]
    ym_E: float = 20.0  # E semi-thickness [km]

    def __post_init__(self):
        """Validate inputs."""
        if len(self.altitude_km) != len(self.electron_density):
            raise ValueError("altitude_km and electron_density must have same length")

    def plasma_frequency(self, h_km: float) -> float:
        """Get plasma frequency at given altitude.

        Args:
            h_km: Altitude in km

        Returns:
            Plasma frequency in MHz
        """
        ne = self.interpolate_ne(h_km)
        return plasma_frequency_from_ne(ne)

    def interpolate_ne(self, h_km: float) -> float:
        """Interpolate electron density at given altitude.

        Args:
            h_km: Altitude in km

        Returns:
            Electron density in m^-3
        """
        if h_km < self.altitude_km[0] or h_km > self.altitude_km[-1]:
            return 0.0
        return float(np.interp(h_km, self.altitude_km, self.electron_density))

    def refractive_index(self, h_km: float, f_mhz: float) -> float:
        """Calculate refractive index at given altitude and frequency.

        Uses the Appleton-Hartree equation simplified for no magnetic field
        (ordinary ray): n^2 = 1 - (fp/f)^2

        Args:
            h_km: Altitude in km
            f_mhz: Wave frequency in MHz

        Returns:
            Refractive index (real part). Returns 0 for evanescent regions.
        """
        if f_mhz <= 0:
            return 0.0

        fp = self.plasma_frequency(h_km)

        n_squared = 1.0 - (fp / f_mhz) ** 2

        if n_squared <= 0:
            # Evanescent region - wave is reflected
            return 0.0

        return np.sqrt(n_squared)

    def refractive_index_squared(self, h_km: float, f_mhz: float) -> float:
        """Calculate n^2 at given altitude and frequency.

        Can be negative in reflection region.

        Args:
            h_km: Altitude in km
            f_mhz: Wave frequency in MHz

        Returns:
            n^2 (can be negative)
        """
        if f_mhz <= 0:
            return 0.0

        fp = self.plasma_frequency(h_km)
        return 1.0 - (fp / f_mhz) ** 2

    def dn_dh(self, h_km: float, f_mhz: float, dh: float = 0.1) -> float:
        """Calculate gradient of refractive index with altitude.

        Uses central difference approximation.

        Args:
            h_km: Altitude in km
            f_mhz: Wave frequency in MHz
            dh: Step size in km

        Returns:
            dn/dh in km^-1
        """
        n_plus = self.refractive_index(h_km + dh, f_mhz)
        n_minus = self.refractive_index(h_km - dh, f_mhz)
        return (n_plus - n_minus) / (2 * dh)

    def find_reflection_height(
        self, f_mhz: float, launch_angle_deg: float,
        h_start: float = 60.0, h_max: float = 600.0, dh: float = 1.0
    ) -> Optional[float]:
        """Find reflection height for given frequency and launch angle.

        Reflection occurs when n = sin(elevation) or equivalently
        when n^2 = sin^2(elevation).

        For oblique incidence, the condition is n^2 = sin^2(i)
        where i is the local angle of incidence.

        This is a simplified search assuming the ionosphere is
        horizontally stratified.

        Args:
            f_mhz: Wave frequency in MHz
            launch_angle_deg: Launch elevation angle in degrees
            h_start: Starting altitude for search
            h_max: Maximum altitude
            dh: Search step size

        Returns:
            Reflection height in km, or None if no reflection
        """
        sin_el = np.sin(np.radians(launch_angle_deg))
        sin_sq = sin_el ** 2

        h = h_start
        while h < h_max:
            n_sq = self.refractive_index_squared(h, f_mhz)

            # Reflection when n^2 <= sin^2(elevation)
            # For oblique incidence: n^2 = 1 - X where X = (fp/f)^2
            # Reflection when X >= cos^2(elevation)
            if n_sq <= sin_sq:
                return h

            h += dh

        return None  # No reflection - wave escapes

    def muf_vertical(self) -> float:
        """Get maximum usable frequency for vertical incidence.

        This is simply foF2 (or foEs if higher).

        Returns:
            MUF in MHz
        """
        muf = self.foF2
        if self.foEs is not None and self.foEs > muf:
            muf = self.foEs
        return muf

    def muf_oblique(self, sec_phi: float) -> float:
        """Get maximum usable frequency for oblique incidence.

        MUF = foF2 * sec(phi) where phi is angle of incidence.

        Args:
            sec_phi: Secant of angle of incidence

        Returns:
            MUF in MHz
        """
        return self.foF2 * sec_phi


@dataclass
class QuasiParabolicProfile(IonosphereProfile):
    """Ionosphere profile using quasi-parabolic layer model.

    The electron density follows a parabolic shape near the layer peak:
        Ne(h) = Ne_max * (1 - ((h - hm) / ym)^2)

    where Ne_max is peak density, hm is peak height, and ym is semi-thickness.
    """

    def __post_init__(self):
        """Generate profile from layer parameters if not provided."""
        if self.altitude_km is None or len(self.altitude_km) == 0:
            self._generate_profile()

    def _generate_profile(self):
        """Generate Ne profile from layer parameters."""
        # Create altitude grid
        self.altitude_km = np.arange(60, 601, 1.0)
        self.electron_density = np.zeros_like(self.altitude_km)

        # Convert critical frequencies to peak densities
        ne_F2 = ne_from_plasma_frequency(self.foF2)
        ne_E = ne_from_plasma_frequency(self.foE)

        # Add F2 layer contribution
        for i, h in enumerate(self.altitude_km):
            delta_h = h - self.hmF2
            if abs(delta_h) < self.ym_F2:
                self.electron_density[i] += ne_F2 * (1 - (delta_h / self.ym_F2) ** 2)

        # Add E layer contribution
        for i, h in enumerate(self.altitude_km):
            delta_h = h - self.hmE
            if abs(delta_h) < self.ym_E:
                self.electron_density[i] += ne_E * (1 - (delta_h / self.ym_E) ** 2)

        # Add sporadic-E if present
        if self.foEs is not None and self.hmEs is not None:
            ne_Es = ne_from_plasma_frequency(self.foEs)
            # Es is typically very thin (~5 km)
            ym_Es = 5.0
            for i, h in enumerate(self.altitude_km):
                delta_h = h - self.hmEs
                if abs(delta_h) < ym_Es:
                    self.electron_density[i] += ne_Es * (1 - (delta_h / ym_Es) ** 2)

        # Add F1 layer if present (daytime only)
        if self.foF1 is not None and self.hmF1 is not None:
            ne_F1 = ne_from_plasma_frequency(self.foF1)
            ym_F1 = 50.0  # Typical F1 semi-thickness
            for i, h in enumerate(self.altitude_km):
                delta_h = h - self.hmF1
                if abs(delta_h) < ym_F1:
                    self.electron_density[i] += ne_F1 * (1 - (delta_h / ym_F1) ** 2)

        # Ensure non-negative
        self.electron_density = np.maximum(self.electron_density, 0)


def create_chapman_profile(
    foF2: float, hmF2: float, foE: float = 3.0, hmE: float = 110.0,
    H_F2: float = 50.0, H_E: float = 10.0,
    alt_min: float = 60.0, alt_max: float = 600.0, alt_step: float = 1.0
) -> IonosphereProfile:
    """Create ionosphere profile using Chapman layer model.

    The Chapman layer has the form:
        Ne(h) = Ne_max * exp(0.5 * (1 - z - exp(-z)))
    where z = (h - hm) / H, and H is the scale height.

    This gives a more realistic asymmetric profile than parabolic.

    Args:
        foF2: F2 critical frequency [MHz]
        hmF2: F2 peak height [km]
        foE: E layer critical frequency [MHz]
        hmE: E layer peak height [km]
        H_F2: F2 scale height [km]
        H_E: E scale height [km]
        alt_min: Minimum altitude [km]
        alt_max: Maximum altitude [km]
        alt_step: Altitude step [km]

    Returns:
        IonosphereProfile with Chapman-shaped layers
    """
    altitudes = np.arange(alt_min, alt_max, alt_step)
    ne = np.zeros_like(altitudes)

    # Peak densities
    ne_F2 = ne_from_plasma_frequency(foF2)
    ne_E = ne_from_plasma_frequency(foE)

    # Add F2 Chapman layer
    z_F2 = (altitudes - hmF2) / H_F2
    ne += ne_F2 * np.exp(0.5 * (1 - z_F2 - np.exp(-z_F2)))

    # Add E Chapman layer
    z_E = (altitudes - hmE) / H_E
    ne += ne_E * np.exp(0.5 * (1 - z_E - np.exp(-z_E)))

    return IonosphereProfile(
        altitude_km=altitudes,
        electron_density=ne,
        foF2=foF2,
        hmF2=hmF2,
        foE=foE,
        hmE=hmE,
    )


def create_simple_profile(
    foF2: float, hmF2: float, foE: float = 3.0, hmE: float = 110.0,
    ym_F2: float = 100.0, ym_E: float = 20.0
) -> IonosphereProfile:
    """Create a simple quasi-parabolic ionosphere profile.

    This is a convenience function for creating profiles
    from just the basic layer parameters.

    Args:
        foF2: F2 critical frequency [MHz]
        hmF2: F2 peak height [km]
        foE: E layer critical frequency [MHz]
        hmE: E layer peak height [km]
        ym_F2: F2 semi-thickness [km]
        ym_E: E semi-thickness [km]

    Returns:
        IonosphereProfile with parabolic layers
    """
    # Create altitude grid
    altitude_km = np.arange(60, 601, 1.0)
    electron_density = np.zeros_like(altitude_km)

    # Peak densities
    ne_F2 = ne_from_plasma_frequency(foF2)
    ne_E = ne_from_plasma_frequency(foE)

    # Add layers
    for i, h in enumerate(altitude_km):
        # F2 layer
        delta_F2 = h - hmF2
        if abs(delta_F2) < ym_F2:
            electron_density[i] += ne_F2 * (1 - (delta_F2 / ym_F2) ** 2)

        # E layer
        delta_E = h - hmE
        if abs(delta_E) < ym_E:
            electron_density[i] += ne_E * (1 - (delta_E / ym_E) ** 2)

    electron_density = np.maximum(electron_density, 0)

    return IonosphereProfile(
        altitude_km=altitude_km,
        electron_density=electron_density,
        foF2=foF2,
        hmF2=hmF2,
        foE=foE,
        hmE=hmE,
        ym_F2=ym_F2,
        ym_E=ym_E,
    )

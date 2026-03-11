"""Spherical Earth geometry for HF ray tracing.

Provides great circle calculations, coordinate transforms, and
launch angle computations for ionospheric ray tracing.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


# Physical constants
EARTH_RADIUS_KM = 6371.0


@dataclass
class GeoPoint:
    """Geographic point on Earth's surface."""

    latitude: float  # degrees, positive North
    longitude: float  # degrees, positive East

    def to_radians(self) -> Tuple[float, float]:
        """Convert to radians."""
        return np.radians(self.latitude), np.radians(self.longitude)

    @classmethod
    def from_radians(cls, lat_rad: float, lon_rad: float) -> "GeoPoint":
        """Create from radian values."""
        return cls(np.degrees(lat_rad), np.degrees(lon_rad))


def great_circle_distance(
    lat1: float, lon1: float, lat2: float, lon2: float,
    earth_radius: float = EARTH_RADIUS_KM
) -> float:
    """Calculate great circle distance between two points.

    Uses the Haversine formula for numerical stability.

    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)
        earth_radius: Earth radius in km (default 6371)

    Returns:
        Distance in km
    """
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return earth_radius * c


def initial_bearing(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate initial bearing (forward azimuth) from point 1 to point 2.

    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)

    Returns:
        Initial bearing in degrees (0-360, clockwise from North)
    """
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlon_r = np.radians(lon2 - lon1)

    x = np.sin(dlon_r) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon_r)

    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def midpoint(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> Tuple[float, float]:
    """Calculate midpoint between two points on great circle.

    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)

    Returns:
        (latitude, longitude) of midpoint in degrees
    """
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlon_r = np.radians(lon2 - lon1)

    Bx = np.cos(lat2_r) * np.cos(dlon_r)
    By = np.cos(lat2_r) * np.sin(dlon_r)

    lat_mid = np.arctan2(
        np.sin(lat1_r) + np.sin(lat2_r),
        np.sqrt((np.cos(lat1_r) + Bx) ** 2 + By ** 2)
    )
    lon_mid = np.radians(lon1) + np.arctan2(By, np.cos(lat1_r) + Bx)

    return np.degrees(lat_mid), np.degrees(lon_mid)


def compute_launch_angle(
    path_km: float, reflection_height_km: float,
    earth_radius: float = EARTH_RADIUS_KM
) -> float:
    """Compute ray launch angle for single-hop reflection.

    Uses spherical Earth geometry to find the elevation angle
    required for a ray to reflect at given height and return
    to ground at the specified range.

    Args:
        path_km: Total ground path distance
        reflection_height_km: Reflection height above ground
        earth_radius: Earth radius in km

    Returns:
        Launch angle in degrees above horizon
    """
    # Half the angular path (one way to reflection point)
    theta = path_km / (2 * earth_radius)  # radians

    # Height of reflection point from Earth center
    R = earth_radius
    h = reflection_height_km

    # Use law of sines in the triangle:
    # TX -> Earth center -> reflection point
    # sin(launch + 90) / (R + h) = sin(theta) / slant_range

    # Distance from TX to reflection point (slant range)
    # Using law of cosines:
    slant_range = np.sqrt(R**2 + (R + h)**2 - 2 * R * (R + h) * np.cos(theta))

    # Angle at TX (from vertical to ray direction)
    # sin(angle_at_tx) / (R + h) = sin(theta) / slant_range
    sin_angle_at_tx = (R + h) * np.sin(theta) / slant_range
    angle_at_tx = np.arcsin(np.clip(sin_angle_at_tx, -1, 1))

    # Launch angle is measured from horizontal
    launch_angle = np.pi / 2 - angle_at_tx

    return np.degrees(launch_angle)


def sec_phi_spherical(
    path_km: float, hm_km: float, earth_radius: float = EARTH_RADIUS_KM
) -> float:
    """Calculate sec(phi) for oblique incidence using spherical geometry.

    This replaces the hardcoded sec_phi = 3.0 with a physically
    correct calculation based on path length and layer height.

    The secant of the angle of incidence is used in the MUF formula:
        MUF = foF2 * sec(phi)

    For HF propagation, the angle of incidence phi at the ionospheric
    reflection point determines the MUF. Using triangle geometry for
    a single-hop path on a spherical Earth:

    Consider the triangle formed by:
    - TX position on ground
    - Earth center
    - Reflection point at height h

    The angle phi (at the reflection point, measured from radial) satisfies:
        sin(phi) = R * sin(beta) / (R + h)

    where beta is the angle at Earth center from TX to the reflection point
    (half the total ground path angle).

    For the flat-Earth approximation: sec(phi) = sqrt(1 + (d/2h)^2)
    This works well for paths up to ~500 km.

    For spherical Earth, we need to account for the curvature.

    Args:
        path_km: Ground path distance in km
        hm_km: Ionospheric layer peak height in km
        earth_radius: Earth radius in km (default 6371)

    Returns:
        sec(phi) where phi is angle of incidence at reflection
    """
    if path_km <= 0:
        return 1.0  # Vertical incidence

    R = earth_radius
    h = hm_km

    # Half ground path
    d = path_km / 2

    # For spherical Earth correction:
    # The effective horizontal distance at height h is slightly less
    # due to Earth curvature. Using the arc-chord relationship:

    # Half angular path at ground
    beta = path_km / (2 * R)

    # Calculate the slant range from TX to reflection point
    # Using law of cosines in the TX-center-reflection triangle:
    slant_sq = R**2 + (R + h)**2 - 2 * R * (R + h) * np.cos(beta)
    slant = np.sqrt(slant_sq)

    # The angle phi at reflection point (from radial direction)
    # Using law of sines: sin(phi) / R = sin(beta) / slant
    sin_phi = R * np.sin(beta) / slant

    # Clamp for numerical safety
    sin_phi = np.clip(sin_phi, 0, 1)

    cos_phi = np.sqrt(1 - sin_phi**2)

    if cos_phi < 1e-6:
        return 100.0  # Grazing incidence

    return 1.0 / cos_phi


def sec_phi_flat_earth(path_km: float, hm_km: float) -> float:
    """Calculate sec(phi) using flat-Earth approximation.

    This is the simpler formula often used in literature:
        sec(phi) = sqrt(1 + (d/2h)^2)

    where d is path length and h is layer height.
    Valid for short paths (< ~500 km).

    Args:
        path_km: Ground path distance in km
        hm_km: Ionospheric layer peak height in km

    Returns:
        sec(phi) where phi is angle of incidence
    """
    if path_km <= 0 or hm_km <= 0:
        return 1.0

    half_path = path_km / 2
    return np.sqrt(1 + (half_path / hm_km) ** 2)


def hop_slant_range(
    path_km: float, reflection_height_km: float,
    earth_radius: float = EARTH_RADIUS_KM
) -> float:
    """Calculate total slant range for single-hop propagation.

    Args:
        path_km: Ground path distance
        reflection_height_km: Reflection height
        earth_radius: Earth radius

    Returns:
        Total ray path length in km (up and down)
    """
    R = earth_radius
    h = reflection_height_km
    theta = path_km / (2 * R)  # Half angular path

    # Slant range to reflection point
    slant = np.sqrt(R**2 + (R + h)**2 - 2 * R * (R + h) * np.cos(theta))

    # Total path is twice the slant (up and down)
    return 2 * slant


def multihop_path_length(
    path_km: float, reflection_height_km: float, n_hops: int,
    earth_radius: float = EARTH_RADIUS_KM
) -> float:
    """Calculate total ray path length for multi-hop propagation.

    Args:
        path_km: Total ground path distance
        reflection_height_km: Reflection height
        n_hops: Number of hops
        earth_radius: Earth radius

    Returns:
        Total ray path length in km
    """
    if n_hops <= 0:
        return 0.0

    # Path per hop
    path_per_hop = path_km / n_hops

    # Each hop contributes same slant range
    return n_hops * hop_slant_range(path_per_hop, reflection_height_km, earth_radius)


def group_delay_ms(
    path_km: float, reflection_height_km: float, n_hops: int = 1,
    earth_radius: float = EARTH_RADIUS_KM
) -> float:
    """Calculate group delay for ionospheric propagation.

    Assumes ray travels at speed of light (neglects ionospheric
    group delay retardation for now).

    Args:
        path_km: Total ground path distance
        reflection_height_km: Reflection height
        n_hops: Number of hops
        earth_radius: Earth radius

    Returns:
        Group delay in milliseconds
    """
    c_km_s = 299792.458  # Speed of light in km/s

    total_path = multihop_path_length(path_km, reflection_height_km, n_hops, earth_radius)

    return (total_path / c_km_s) * 1000  # Convert to ms


def point_at_distance(
    lat: float, lon: float, bearing: float, distance_km: float,
    earth_radius: float = EARTH_RADIUS_KM
) -> Tuple[float, float]:
    """Calculate destination point given start, bearing, and distance.

    Args:
        lat, lon: Start point (degrees)
        bearing: Initial bearing (degrees, clockwise from North)
        distance_km: Distance to travel
        earth_radius: Earth radius

    Returns:
        (latitude, longitude) of destination in degrees
    """
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    bearing_r = np.radians(bearing)

    delta = distance_km / earth_radius

    lat2 = np.arcsin(
        np.sin(lat_r) * np.cos(delta) +
        np.cos(lat_r) * np.sin(delta) * np.cos(bearing_r)
    )

    lon2 = lon_r + np.arctan2(
        np.sin(bearing_r) * np.sin(delta) * np.cos(lat_r),
        np.cos(delta) - np.sin(lat_r) * np.sin(lat2)
    )

    return np.degrees(lat2), np.degrees(lon2)


def great_circle_waypoints(
    lat1: float, lon1: float, lat2: float, lon2: float,
    n_points: int = 10
) -> np.ndarray:
    """Generate waypoints along great circle path.

    Args:
        lat1, lon1: Start point (degrees)
        lat2, lon2: End point (degrees)
        n_points: Number of waypoints

    Returns:
        Array of shape (n_points, 2) with (lat, lon) pairs
    """
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)

    # Calculate angular distance
    d = np.arccos(
        np.sin(lat1_r) * np.sin(lat2_r) +
        np.cos(lat1_r) * np.cos(lat2_r) * np.cos(lon2_r - lon1_r)
    )

    if d < 1e-10:
        # Same point
        return np.array([[lat1, lon1]] * n_points)

    waypoints = np.zeros((n_points, 2))

    for i, f in enumerate(np.linspace(0, 1, n_points)):
        a = np.sin((1 - f) * d) / np.sin(d)
        b = np.sin(f * d) / np.sin(d)

        x = a * np.cos(lat1_r) * np.cos(lon1_r) + b * np.cos(lat2_r) * np.cos(lon2_r)
        y = a * np.cos(lat1_r) * np.sin(lon1_r) + b * np.cos(lat2_r) * np.sin(lon2_r)
        z = a * np.sin(lat1_r) + b * np.sin(lat2_r)

        lat = np.arctan2(z, np.sqrt(x**2 + y**2))
        lon = np.arctan2(y, x)

        waypoints[i] = [np.degrees(lat), np.degrees(lon)]

    return waypoints

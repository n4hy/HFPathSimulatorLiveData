"""HF ionospheric ray tracing module.

This module provides physics-based ray tracing for HF propagation,
enabling accurate computation of oblique incidence angles, multi-hop
delays, and propagation mode discovery.

Main Components:
- geometry: Spherical Earth geometry and sec(phi) calculations
- ionosphere: Electron density profiles and refractive index
- ray_engine: 2D Haselgrove ray equation integration
- path_finder: Multi-hop mode discovery

Typical usage:
    from hfpathsim.core.raytracing import (
        create_simple_profile,
        find_propagation_modes,
        sec_phi_spherical,
    )

    # Create ionosphere profile
    profile = create_simple_profile(foF2=7.5, hmF2=300)

    # Find viable propagation modes
    modes = find_propagation_modes(
        profile,
        tx_lat=38.9, tx_lon=-77.0,  # Washington DC
        rx_lat=51.5, rx_lon=-0.1,    # London
        frequency_mhz=14.0,
    )

    # Get sec(phi) for MUF calculation
    sec_phi = sec_phi_spherical(path_km=5900, hm_km=300)
"""

# Geometry functions
from .geometry import (
    EARTH_RADIUS_KM,
    GeoPoint,
    great_circle_distance,
    initial_bearing,
    midpoint,
    compute_launch_angle,
    sec_phi_spherical,
    sec_phi_flat_earth,
    hop_slant_range,
    multihop_path_length,
    group_delay_ms,
    point_at_distance,
    great_circle_waypoints,
)

# Ionosphere profile classes and functions
from .ionosphere import (
    plasma_frequency_from_ne,
    ne_from_plasma_frequency,
    IonosphereProfile,
    QuasiParabolicProfile,
    create_chapman_profile,
    create_simple_profile,
)

# Ray tracing engine
from .ray_engine import (
    RayPath,
    RayEngine,
    trace_multihop,
)

# Path finder
from .path_finder import (
    PropagationModeResult,
    PathFinder,
    find_propagation_modes,
    modes_to_propagation_modes,
    estimate_muf,
)


__all__ = [
    # Constants
    "EARTH_RADIUS_KM",
    # Geometry
    "GeoPoint",
    "great_circle_distance",
    "initial_bearing",
    "midpoint",
    "compute_launch_angle",
    "sec_phi_spherical",
    "sec_phi_flat_earth",
    "hop_slant_range",
    "multihop_path_length",
    "group_delay_ms",
    "point_at_distance",
    "great_circle_waypoints",
    # Ionosphere
    "plasma_frequency_from_ne",
    "ne_from_plasma_frequency",
    "IonosphereProfile",
    "QuasiParabolicProfile",
    "create_chapman_profile",
    "create_simple_profile",
    # Ray engine
    "RayPath",
    "RayEngine",
    "trace_multihop",
    # Path finder
    "PropagationModeResult",
    "PathFinder",
    "find_propagation_modes",
    "modes_to_propagation_modes",
    "estimate_muf",
]

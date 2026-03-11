"""Ray tracing engine using Haselgrove equations.

Implements 2D ray path integration through a horizontally-stratified
ionosphere using the Haselgrove ray equations.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from scipy.integrate import solve_ivp

from .ionosphere import IonosphereProfile
from .geometry import EARTH_RADIUS_KM


# Speed of light in km/s
C_LIGHT_KM_S = 299792.458


@dataclass
class RayPath:
    """Result of ray tracing through the ionosphere.

    Contains the computed ray path parameters including geometry,
    delays, and layer information.
    """

    launch_angle_deg: float  # Initial elevation angle
    ground_range_km: float  # Total ground range covered
    group_delay_ms: float  # Total group delay
    reflection_height_km: float  # Maximum altitude reached
    layer: str  # Layer causing reflection: "F2", "E", "Es", "F1", or "escape"
    n_hops: int  # Number of ionospheric hops (0 if escaped)
    valid: bool  # True if ray returned to ground

    # Full ray path data (optional, for visualization)
    x_km: np.ndarray = field(default_factory=lambda: np.array([]))  # Ground range
    h_km: np.ndarray = field(default_factory=lambda: np.array([]))  # Altitude
    theta_rad: np.ndarray = field(default_factory=lambda: np.array([]))  # Ray angle

    def sec_phi(self) -> float:
        """Calculate secant of incidence angle at reflection.

        Uses spherical geometry based on path and reflection height.

        Returns:
            sec(phi) where phi is angle of incidence
        """
        if not self.valid or self.n_hops <= 0:
            return 1.0

        # Ground range per hop
        range_per_hop = self.ground_range_km / self.n_hops

        R = EARTH_RADIUS_KM
        h = self.reflection_height_km

        # Half angular path per hop
        theta = (range_per_hop / 2) / R

        # Angle of incidence
        sin_phi = R * np.sin(theta) / (R + h)
        sin_phi = np.clip(sin_phi, -1, 1)
        cos_phi = np.sqrt(1 - sin_phi ** 2)

        if cos_phi < 1e-6:
            return 100.0

        return 1.0 / cos_phi


class RayEngine:
    """2D ray tracing engine for ionospheric propagation.

    Uses the Haselgrove ray equations for a spherically-stratified
    ionosphere (radially-varying refractive index).

    The Haselgrove 2D equations in (x, h, theta) coordinates:
        dx/ds = sin(theta) / n
        dh/ds = cos(theta) / n
        dtheta/ds = (1/n) * (dn/dh * sin(theta) - cos(theta) / (R + h))

    where:
        x = ground range
        h = altitude above Earth
        theta = ray angle from vertical
        s = path length along ray
        n = refractive index
        R = Earth radius
    """

    def __init__(
        self,
        profile: IonosphereProfile,
        earth_radius: float = EARTH_RADIUS_KM,
    ):
        """Initialize ray engine.

        Args:
            profile: Ionospheric electron density profile
            earth_radius: Earth radius in km
        """
        self.profile = profile
        self.R = earth_radius

    def trace_ray(
        self,
        f_mhz: float,
        launch_angle_deg: float,
        max_path_km: float = 5000.0,
        ds_km: float = 1.0,
        store_path: bool = False,
    ) -> RayPath:
        """Trace a single ray through the ionosphere.

        Uses a simplified approach: at each step, check if the ray would
        enter an evanescent region. If so, reflect the ray (reverse
        vertical velocity component).

        Args:
            f_mhz: Wave frequency in MHz
            launch_angle_deg: Launch elevation angle above horizon (degrees)
            max_path_km: Maximum ray path length to trace
            ds_km: Integration step size in km
            store_path: If True, store full ray path coordinates

        Returns:
            RayPath object with traced results
        """
        # Initial conditions at ground
        # theta is angle from vertical (0 = up, 90 = horizontal)
        theta0 = np.radians(90 - launch_angle_deg)

        # Storage for path points
        x_list = [0.0]
        h_list = [0.0]
        theta_list = [theta0]

        x, h, theta = 0.0, 0.0, theta0
        s = 0.0
        h_max = 0.0
        going_up = True
        reflected = False

        while s < max_path_km:
            # Current refractive index
            n = self.profile.refractive_index(h, f_mhz)
            if n < 0.01:
                n = 0.01  # Minimum to avoid division issues

            # Get gradient
            dn_dh = self.profile.dn_dh(h, f_mhz, dh=0.5)

            # Check for reflection condition
            # Ray reflects when n^2 * cos^2(theta) <= sin^2(elevation)
            # which is the Snell's law at the turning point
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            # Check if entering evanescent region (n^2 approaching sin^2 of zenith angle)
            n_sq = n * n
            sin_zenith_sq = cos_theta * cos_theta  # sin of zenith = cos of theta from vertical

            if going_up and n_sq < sin_zenith_sq + 0.01:
                # Ray reflects
                going_up = False
                reflected = True
                theta = np.pi - theta  # Reflect: theta -> pi - theta

            # Update state using Euler step (simple but stable)
            dx = sin_theta / n * ds_km
            dh = cos_theta / n * ds_km
            dtheta = (1 / n) * (dn_dh * sin_theta - cos_theta / (self.R + h)) * ds_km

            x += dx
            h += dh
            theta += dtheta

            # Ensure theta stays reasonable
            theta = theta % (2 * np.pi)

            s += ds_km
            h_max = max(h_max, h)

            if store_path:
                x_list.append(x)
                h_list.append(h)
                theta_list.append(theta)

            # Check termination
            if h < 0 and not going_up:
                # Hit ground after reflection
                break
            elif h > 600:
                # Escaped
                break
            elif h < 0 and going_up:
                # Shouldn't happen - reset to ground level
                h = 0.1

        # Determine if ray returned to ground
        valid = h <= 0 and reflected

        # Determine reflection layer
        layer = self._identify_layer(h_max)
        if not reflected and h > 500:
            layer = "escape"

        # Calculate group delay
        group_delay = s / C_LIGHT_KM_S * 1000  # ms

        # Store path if requested
        x_array = np.array(x_list) if store_path else np.array([])
        h_array = np.array(h_list) if store_path else np.array([])
        theta_array = np.array(theta_list) if store_path else np.array([])

        return RayPath(
            launch_angle_deg=launch_angle_deg,
            ground_range_km=x,
            group_delay_ms=group_delay,
            reflection_height_km=h_max,
            layer=layer,
            n_hops=1 if valid else 0,
            valid=valid,
            x_km=x_array,
            h_km=h_array,
            theta_rad=theta_array,
        )

    def _identify_layer(self, h_max: float) -> str:
        """Identify which ionospheric layer caused reflection.

        Args:
            h_max: Maximum altitude reached

        Returns:
            Layer identifier string
        """
        hmE = self.profile.hmE
        hmF2 = self.profile.hmF2
        hmEs = self.profile.hmEs

        if h_max > 500:
            return "escape"

        # Check sporadic-E first (if present)
        if hmEs is not None:
            if abs(h_max - hmEs) < 20:
                return "Es"

        if h_max < hmE + 30:
            return "E"
        elif h_max < 200:
            return "F1"
        else:
            return "F2"

    def find_path(
        self,
        f_mhz: float,
        target_range_km: float,
        angle_min: float = 5.0,
        angle_max: float = 85.0,
        tolerance_km: float = 10.0,
        max_iterations: int = 50,
    ) -> Optional[RayPath]:
        """Find ray that reaches target ground range.

        Uses bisection search to find the launch angle that results
        in the ray landing at the target range.

        Args:
            f_mhz: Wave frequency in MHz
            target_range_km: Desired ground range
            angle_min: Minimum launch angle to try
            angle_max: Maximum launch angle to try
            tolerance_km: Acceptable range error
            max_iterations: Maximum search iterations

        Returns:
            RayPath reaching target, or None if not found
        """
        # Initial bounds
        a, b = angle_min, angle_max

        # Evaluate bounds
        ray_a = self.trace_ray(f_mhz, a)
        ray_b = self.trace_ray(f_mhz, b)

        if not ray_a.valid and not ray_b.valid:
            return None

        for _ in range(max_iterations):
            mid = (a + b) / 2
            ray_mid = self.trace_ray(f_mhz, mid)

            if not ray_mid.valid:
                # Ray escaped - try lower angle
                b = mid
                continue

            error = ray_mid.ground_range_km - target_range_km

            if abs(error) < tolerance_km:
                return ray_mid

            # Bisect based on range comparison
            if ray_mid.ground_range_km < target_range_km:
                a = mid
            else:
                b = mid

            if b - a < 0.1:
                break

        # Return best result even if not within tolerance
        ray = self.trace_ray(f_mhz, (a + b) / 2, store_path=True)
        return ray if ray.valid else None

    def find_all_paths(
        self,
        f_mhz: float,
        target_range_km: float,
        angle_step: float = 1.0,
        tolerance_km: float = 20.0,
    ) -> List[RayPath]:
        """Find all ray paths that reach target range.

        Scans launch angles to find multiple propagation modes
        (high ray, low ray, different layer reflections).

        Args:
            f_mhz: Wave frequency in MHz
            target_range_km: Desired ground range
            angle_step: Angle scan step
            tolerance_km: Range matching tolerance

        Returns:
            List of valid RayPaths reaching target
        """
        paths = []
        prev_range = 0

        for angle in np.arange(5, 85, angle_step):
            ray = self.trace_ray(f_mhz, angle)

            if not ray.valid:
                continue

            # Check if close to target
            if abs(ray.ground_range_km - target_range_km) < tolerance_km:
                # Refine the path
                refined = self.find_path(
                    f_mhz, target_range_km,
                    angle_min=angle - angle_step,
                    angle_max=angle + angle_step,
                    tolerance_km=tolerance_km / 2,
                )
                if refined is not None:
                    paths.append(refined)

            prev_range = ray.ground_range_km

        return paths

    def compute_muf(
        self,
        target_range_km: float,
        f_start: float = 2.0,
        f_max: float = 30.0,
        f_step: float = 0.5,
    ) -> float:
        """Compute Maximum Usable Frequency for a path.

        Finds the highest frequency that can reach the target range.

        Args:
            target_range_km: Desired ground range
            f_start: Starting frequency to test
            f_max: Maximum frequency to test
            f_step: Frequency search step

        Returns:
            MUF in MHz
        """
        muf = f_start

        for f in np.arange(f_start, f_max, f_step):
            path = self.find_path(f, target_range_km)
            if path is not None:
                muf = f
            else:
                break

        return muf


def trace_multihop(
    engine: RayEngine,
    f_mhz: float,
    target_range_km: float,
    n_hops: int,
) -> Optional[RayPath]:
    """Trace a multi-hop ray path.

    For n-hop propagation, each hop covers 1/n of the total range.

    Args:
        engine: Ray engine instance
        f_mhz: Wave frequency
        target_range_km: Total ground range
        n_hops: Number of hops

    Returns:
        Combined RayPath or None
    """
    range_per_hop = target_range_km / n_hops

    # Trace single hop
    single_hop = engine.find_path(f_mhz, range_per_hop)

    if single_hop is None:
        return None

    # Scale results for multi-hop
    return RayPath(
        launch_angle_deg=single_hop.launch_angle_deg,
        ground_range_km=target_range_km,
        group_delay_ms=single_hop.group_delay_ms * n_hops,
        reflection_height_km=single_hop.reflection_height_km,
        layer=single_hop.layer,
        n_hops=n_hops,
        valid=True,
    )

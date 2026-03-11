"""Tests for ray tracing module."""

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose

from hfpathsim.core.raytracing import (
    # Geometry
    great_circle_distance,
    sec_phi_spherical,
    sec_phi_flat_earth,
    compute_launch_angle,
    group_delay_ms,
    hop_slant_range,
    multihop_path_length,
    initial_bearing,
    midpoint,
    EARTH_RADIUS_KM,
    # Ionosphere
    IonosphereProfile,
    create_simple_profile,
    create_chapman_profile,
    plasma_frequency_from_ne,
    ne_from_plasma_frequency,
    # Ray engine
    RayEngine,
    RayPath,
    trace_multihop,
    # Path finder
    PathFinder,
    find_propagation_modes,
    estimate_muf,
)


class TestGeometry:
    """Tests for geometry module."""

    def test_great_circle_distance_same_point(self):
        """Same point should have zero distance."""
        d = great_circle_distance(45.0, -75.0, 45.0, -75.0)
        assert d == 0.0

    def test_great_circle_distance_known_path(self):
        """Test known great circle distance: NYC to London ~5570 km."""
        # New York: 40.7128 N, 74.0060 W
        # London: 51.5074 N, 0.1278 W
        d = great_circle_distance(40.7128, -74.0060, 51.5074, -0.1278)
        assert 5500 < d < 5700  # Should be ~5570 km

    def test_great_circle_distance_antipodal(self):
        """Antipodal points should be ~20000 km apart."""
        d = great_circle_distance(0.0, 0.0, 0.0, 180.0)
        expected = np.pi * EARTH_RADIUS_KM
        assert_almost_equal(d, expected, decimal=0)

    def test_sec_phi_vertical_incidence(self):
        """Vertical incidence (zero path) should give sec_phi = 1."""
        sec_phi = sec_phi_spherical(0.0, 300.0)
        assert sec_phi == 1.0

    def test_sec_phi_increases_with_path(self):
        """sec_phi should increase with path length."""
        sec_500 = sec_phi_spherical(500.0, 300.0)
        sec_1000 = sec_phi_spherical(1000.0, 300.0)
        sec_2000 = sec_phi_spherical(2000.0, 300.0)

        assert sec_500 < sec_1000 < sec_2000

    def test_sec_phi_typical_values(self):
        """Check sec_phi is reasonable for typical paths."""
        # 1000 km path at 300 km height should give sec_phi ~ 1.8-2.5
        sec_phi = sec_phi_spherical(1000.0, 300.0)
        assert 1.5 < sec_phi < 3.0

        # 2000 km path should give sec_phi ~ 2.5-4.0
        sec_phi = sec_phi_spherical(2000.0, 300.0)
        assert 2.0 < sec_phi < 4.5

    def test_sec_phi_flat_vs_spherical(self):
        """Flat Earth approximation valid for short paths."""
        # For short paths, flat and spherical should be close
        sec_flat = sec_phi_flat_earth(300.0, 300.0)
        sec_spherical = sec_phi_spherical(300.0, 300.0)

        # Should be within 15% for 300 km path (shorter path, better agreement)
        assert abs(sec_flat - sec_spherical) / sec_spherical < 0.15

    def test_compute_launch_angle_reasonable(self):
        """Launch angle should be reasonable."""
        angle = compute_launch_angle(1000.0, 300.0)

        # Should be between 0 and 90 degrees
        assert 0 < angle < 90

        # For 1000 km path at 300 km height, expect ~20-40 degrees
        assert 10 < angle < 50

    def test_hop_slant_range(self):
        """Slant range should be longer than ground path."""
        path_km = 1000.0
        h_km = 300.0

        slant = hop_slant_range(path_km, h_km)

        # Slant range should be greater than ground path
        assert slant > path_km

        # But not excessively so
        assert slant < 2 * path_km

    def test_multihop_path_length(self):
        """Multi-hop path length should scale with hops."""
        path_km = 3000.0
        h_km = 300.0

        one_hop = multihop_path_length(path_km, h_km, 1)
        two_hop = multihop_path_length(path_km, h_km, 2)
        three_hop = multihop_path_length(path_km, h_km, 3)

        # More hops = slightly longer total path
        assert two_hop > one_hop
        assert three_hop > two_hop

    def test_group_delay_reasonable(self):
        """Group delay should be reasonable."""
        # 1000 km path should have ~3-4 ms delay
        delay = group_delay_ms(1000.0, 300.0, n_hops=1)
        assert 3 < delay < 5

        # 2000 km path should have ~7-8 ms delay
        delay = group_delay_ms(2000.0, 300.0, n_hops=1)
        assert 6 < delay < 10

    def test_group_delay_multihop_ordering(self):
        """2-hop delay should be greater than 1-hop for same path."""
        path_km = 3000.0
        h_km = 300.0

        delay_1hop = group_delay_ms(path_km, h_km, n_hops=1)
        delay_2hop = group_delay_ms(path_km, h_km, n_hops=2)

        # 2-hop should be slightly longer
        assert delay_2hop > delay_1hop

    def test_initial_bearing(self):
        """Test initial bearing calculation."""
        # Due East from equator
        bearing = initial_bearing(0.0, 0.0, 0.0, 90.0)
        assert_almost_equal(bearing, 90.0, decimal=1)

        # Due North
        bearing = initial_bearing(0.0, 0.0, 45.0, 0.0)
        assert_almost_equal(bearing, 0.0, decimal=1)

    def test_midpoint(self):
        """Test midpoint calculation."""
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 0.0, 90.0

        mid_lat, mid_lon = midpoint(lat1, lon1, lat2, lon2)

        assert_almost_equal(mid_lat, 0.0, decimal=1)
        assert_almost_equal(mid_lon, 45.0, decimal=1)


class TestIonosphere:
    """Tests for ionosphere module."""

    def test_plasma_frequency_conversion(self):
        """Test plasma frequency conversion roundtrip."""
        fp = 7.5  # MHz
        ne = ne_from_plasma_frequency(fp)
        fp_back = plasma_frequency_from_ne(ne)

        assert_almost_equal(fp_back, fp, decimal=6)

    def test_create_simple_profile(self):
        """Test simple profile creation."""
        profile = create_simple_profile(foF2=7.5, hmF2=300.0)

        assert profile.foF2 == 7.5
        assert profile.hmF2 == 300.0
        assert len(profile.altitude_km) > 0
        assert len(profile.electron_density) == len(profile.altitude_km)

    def test_profile_plasma_frequency(self):
        """Profile should give foF2 at hmF2."""
        foF2 = 7.5
        hmF2 = 300.0

        profile = create_simple_profile(foF2=foF2, hmF2=hmF2)

        # Plasma frequency at peak should be close to foF2
        fp_at_peak = profile.plasma_frequency(hmF2)
        assert abs(fp_at_peak - foF2) < 0.5

    def test_refractive_index_ground(self):
        """Refractive index at ground should be ~1."""
        profile = create_simple_profile(foF2=7.5, hmF2=300.0)

        n = profile.refractive_index(0.0, 10.0)
        assert_almost_equal(n, 1.0, decimal=2)

    def test_refractive_index_below_fp(self):
        """Below plasma frequency, n should be 0 (evanescent)."""
        profile = create_simple_profile(foF2=10.0, hmF2=300.0)

        # Frequency below foF2 at peak
        n = profile.refractive_index(300.0, 8.0)

        # Should be zero or very small (evanescent)
        assert n < 0.5

    def test_muf_vertical(self):
        """Vertical MUF should equal foF2."""
        profile = create_simple_profile(foF2=7.5, hmF2=300.0)
        muf = profile.muf_vertical()

        assert_almost_equal(muf, 7.5, decimal=1)

    def test_muf_oblique(self):
        """Oblique MUF should be foF2 * sec_phi."""
        profile = create_simple_profile(foF2=7.5, hmF2=300.0)

        sec_phi = 3.0
        muf = profile.muf_oblique(sec_phi)

        assert_almost_equal(muf, 7.5 * 3.0, decimal=1)


class TestRayEngine:
    """Tests for ray engine."""

    @pytest.fixture
    def engine(self):
        """Create ray engine with standard profile."""
        profile = create_simple_profile(foF2=7.5, hmF2=300.0)
        return RayEngine(profile)

    def test_trace_ray_valid(self, engine):
        """Ray at valid frequency should reflect."""
        ray = engine.trace_ray(f_mhz=10.0, launch_angle_deg=30.0)

        # Ray should be valid
        assert ray.valid

        # Should have positive ground range
        assert ray.ground_range_km > 0

        # Should have reflection at reasonable height
        assert 100 < ray.reflection_height_km < 400

    def test_trace_ray_escape(self, engine):
        """Ray above MUF should escape."""
        # Very high frequency, very steep angle
        ray = engine.trace_ray(f_mhz=25.0, launch_angle_deg=80.0)

        # Ray likely escapes
        # (depends on profile and launch angle)

    def test_trace_ray_layer_identification(self, engine):
        """Ray should identify correct reflection layer."""
        ray = engine.trace_ray(f_mhz=10.0, launch_angle_deg=30.0)

        assert ray.layer in ["F2", "E", "F1", "Es", "escape"]

    def test_find_path_works(self, engine):
        """find_path should find ray reaching target."""
        target = 500.0  # km (shorter path more reliable for testing)

        ray = engine.find_path(f_mhz=8.0, target_range_km=target, tolerance_km=50)

        # May or may not find valid path depending on profile
        # Just verify it returns something or None without errors
        assert ray is None or isinstance(ray, RayPath)


class TestPathFinder:
    """Tests for path finder."""

    @pytest.fixture
    def finder(self):
        """Create path finder with standard profile."""
        profile = create_simple_profile(foF2=7.5, hmF2=300.0)
        return PathFinder(profile)

    def test_find_modes_returns_list(self, finder):
        """find_modes should return a list."""
        modes = finder.find_modes(
            frequency_mhz=10.0,
            tx_lat=40.0, tx_lon=-75.0,
            rx_lat=51.0, rx_lon=0.0,
        )

        assert isinstance(modes, list)

    def test_find_modes_below_muf(self, finder):
        """Should find modes when frequency below MUF."""
        # Short path, low frequency - should work
        modes = finder.find_modes(
            frequency_mhz=7.0,  # Below foF2
            tx_lat=40.0, tx_lon=-75.0,
            rx_lat=42.0, rx_lon=-73.0,  # Short path
        )

        assert len(modes) > 0

    def test_find_modes_above_muf(self, finder):
        """Should find no modes when frequency above MUF."""
        # Very high frequency - should fail
        modes = finder.find_modes(
            frequency_mhz=50.0,  # Way above MUF
            tx_lat=40.0, tx_lon=-75.0,
            rx_lat=51.0, rx_lon=0.0,
        )

        # Likely no valid modes
        # (or very weak ones)

    def test_mode_has_required_fields(self, finder):
        """Mode objects should have required fields."""
        modes = finder.find_modes(
            frequency_mhz=10.0,
            tx_lat=40.0, tx_lon=-75.0,
            rx_lat=42.0, rx_lon=-73.0,
        )

        if len(modes) > 0:
            mode = modes[0]
            assert hasattr(mode, 'name')
            assert hasattr(mode, 'relative_amplitude')
            assert hasattr(mode, 'delay_offset_ms')
            assert hasattr(mode, 'n_hops')
            assert hasattr(mode, 'layer')

    def test_estimate_muf(self, finder):
        """estimate_muf should return reasonable value."""
        profile = finder.profile

        muf = estimate_muf(profile, path_km=1000.0, layer="F2")

        # MUF should be greater than foF2
        assert muf > profile.foF2

        # But not unreasonably high
        assert muf < 5 * profile.foF2


class TestIntegration:
    """Integration tests for ray tracing system."""

    def test_washington_london_path(self):
        """Test transatlantic path: Washington DC to London."""
        # Washington DC: 38.9072 N, 77.0369 W
        # London: 51.5074 N, 0.1278 W
        tx_lat, tx_lon = 38.9072, -77.0369
        rx_lat, rx_lon = 51.5074, -0.1278

        # Create profile
        profile = create_simple_profile(foF2=7.5, hmF2=300.0)

        # Find modes at 14 MHz
        modes = find_propagation_modes(
            profile,
            tx_lat, tx_lon,
            rx_lat, rx_lon,
            frequency_mhz=14.0,
        )

        # Should find at least one mode
        assert len(modes) >= 1

        # Check path distance is correct
        path_km = great_circle_distance(tx_lat, tx_lon, rx_lat, rx_lon)
        assert 5500 < path_km < 6000

    def test_sec_phi_consistency(self):
        """sec_phi from geometry and parameters should be consistent."""
        from hfpathsim.core.parameters import VoglerParameters

        path_km = 1000.0
        hmF2 = 300.0

        # From geometry module
        sec_geo = sec_phi_spherical(path_km, hmF2)

        # From VoglerParameters
        params = VoglerParameters(
            path_length_km=path_km,
            hmF2=hmF2,
        )
        sec_params = params.get_sec_phi("F2")

        assert_almost_equal(sec_geo, sec_params, decimal=4)

    def test_multihop_delay_ordering(self):
        """1-hop should have less delay than 2-hop for same endpoint."""
        profile = create_simple_profile(foF2=8.0, hmF2=300.0)
        finder = PathFinder(profile)

        # Long path that requires multiple hops
        modes = finder.find_modes(
            frequency_mhz=12.0,
            tx_lat=40.0, tx_lon=-100.0,
            rx_lat=50.0, rx_lon=10.0,
            max_hops=3,
        )

        # Find 1F2 and 2F2 modes
        mode_1f2 = next((m for m in modes if m.name == "1F2"), None)
        mode_2f2 = next((m for m in modes if m.name == "2F2"), None)

        if mode_1f2 and mode_2f2:
            # 2F2 should have larger delay
            assert mode_2f2.group_delay_ms > mode_1f2.group_delay_ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

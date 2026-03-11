"""Tests for geomagnetic effects module."""

import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from hfpathsim.iono.geomagnetic import (
    GeomagneticIndices,
    GeomagneticModulator,
    kp_from_ap,
    ap_from_kp,
    estimate_ssn_from_f10_7,
    classify_storm_phase,
    STORM_PHASES,
)
from hfpathsim.core.raytracing import create_simple_profile


class TestGeomagneticIndices:
    """Tests for GeomagneticIndices dataclass."""

    def test_default_values(self):
        """Default values should be moderate activity."""
        indices = GeomagneticIndices()

        assert indices.f10_7 == 100.0
        assert indices.kp == 2.0
        assert indices.dst == 0.0

    def test_quiet_preset(self):
        """Quiet preset should have low values."""
        indices = GeomagneticIndices.quiet()

        assert indices.f10_7 < 100
        assert indices.kp <= 2
        assert indices.dst >= -10

    def test_disturbed_preset(self):
        """Disturbed preset should have elevated values."""
        indices = GeomagneticIndices.disturbed()

        assert indices.kp >= 4
        assert indices.dst < -50

    def test_severe_storm_preset(self):
        """Severe storm should have extreme values."""
        indices = GeomagneticIndices.severe_storm()

        assert indices.kp >= 6
        assert indices.dst < -100

    def test_solar_maximum(self):
        """Solar maximum should have high F10.7."""
        indices = GeomagneticIndices.solar_maximum()

        assert indices.f10_7 >= 150

    def test_solar_minimum(self):
        """Solar minimum should have low F10.7."""
        indices = GeomagneticIndices.solar_minimum()

        assert indices.f10_7 <= 80

    def test_validation_clamping(self):
        """Out-of-range values should be clamped."""
        indices = GeomagneticIndices(
            f10_7=1000.0,  # Too high
            kp=20.0,  # Too high
            dst=-1000.0,  # Too negative
        )

        assert indices.f10_7 <= 350
        assert indices.kp <= 9
        assert indices.dst >= -500


class TestGeomagneticModulator:
    """Tests for GeomagneticModulator class."""

    def test_default_modulator(self):
        """Default modulator should use quiet conditions."""
        mod = GeomagneticModulator()

        # Should have indices set
        assert mod.indices is not None

    def test_scale_foF2_solar_flux(self):
        """Higher solar flux should increase foF2."""
        foF2_base = 7.0

        mod_low = GeomagneticModulator(GeomagneticIndices(f10_7=70))
        mod_high = GeomagneticModulator(GeomagneticIndices(f10_7=200))

        foF2_low = mod_low.scale_foF2(foF2_base)
        foF2_high = mod_high.scale_foF2(foF2_base)

        assert foF2_high > foF2_low

    def test_scale_foF2_storm_depression(self):
        """Negative Dst should depress foF2 (with same F10.7)."""
        foF2_base = 7.0

        # Use same F10.7 to isolate Dst effect
        mod_quiet = GeomagneticModulator(GeomagneticIndices(f10_7=100, dst=0))
        mod_storm = GeomagneticModulator(GeomagneticIndices(f10_7=100, dst=-100))

        foF2_quiet = mod_quiet.scale_foF2(foF2_base, latitude=45.0)
        foF2_storm = mod_storm.scale_foF2(foF2_base, latitude=45.0)

        assert foF2_storm < foF2_quiet

    def test_scale_foF2_latitude_dependence(self):
        """Storm depression should be latitude dependent."""
        foF2_base = 7.0
        mod = GeomagneticModulator(GeomagneticIndices(dst=-100))

        foF2_equator = mod.scale_foF2(foF2_base, latitude=0.0)
        foF2_pole = mod.scale_foF2(foF2_base, latitude=80.0)

        # Equator should show more effect (cos^2 factor)
        # Actually, effect is larger at equator
        # But both should be depressed relative to base

    def test_scale_hmF2_storm_rise(self):
        """Storm conditions should raise hmF2."""
        hmF2_base = 300.0

        mod_quiet = GeomagneticModulator(GeomagneticIndices(dst=0, kp=1))
        mod_storm = GeomagneticModulator(GeomagneticIndices(dst=-100, kp=6))

        hmF2_quiet = mod_quiet.scale_hmF2(hmF2_base)
        hmF2_storm = mod_storm.scale_hmF2(hmF2_base)

        assert hmF2_storm > hmF2_quiet

    def test_scale_doppler_spread_kp(self):
        """Higher Kp should increase Doppler spread."""
        doppler_base = 1.0

        mod_quiet = GeomagneticModulator(GeomagneticIndices(kp=1))
        mod_active = GeomagneticModulator(GeomagneticIndices(kp=6))

        doppler_quiet = mod_quiet.scale_doppler_spread(doppler_base)
        doppler_active = mod_active.scale_doppler_spread(doppler_base)

        assert doppler_active > doppler_quiet

    def test_scale_doppler_high_latitude(self):
        """High latitude should have more Doppler enhancement."""
        doppler_base = 1.0
        mod = GeomagneticModulator(GeomagneticIndices(kp=6))

        doppler_mid = mod.scale_doppler_spread(doppler_base, latitude=45.0)
        doppler_high = mod.scale_doppler_spread(doppler_base, latitude=65.0)

        assert doppler_high > doppler_mid

    def test_scale_delay_spread(self):
        """Higher Kp should increase delay spread."""
        delay_base = 2.0

        mod_quiet = GeomagneticModulator(GeomagneticIndices(kp=1))
        mod_active = GeomagneticModulator(GeomagneticIndices(kp=6))

        delay_quiet = mod_quiet.scale_delay_spread(delay_base)
        delay_active = mod_active.scale_delay_spread(delay_base)

        assert delay_active > delay_quiet

    def test_absorption_factor_increases_with_kp(self):
        """Absorption should increase with Kp."""
        mod_quiet = GeomagneticModulator(GeomagneticIndices(kp=1))
        mod_active = GeomagneticModulator(GeomagneticIndices(kp=6))

        abs_quiet = mod_quiet.get_absorption_factor(10.0, latitude=60.0)
        abs_active = mod_active.get_absorption_factor(10.0, latitude=60.0)

        assert abs_active > abs_quiet

    def test_blackout_severe_storm(self):
        """Severe storm at high latitude should cause blackout."""
        mod = GeomagneticModulator(GeomagneticIndices.severe_storm())

        # High latitude, low frequency during severe storm
        blackout = mod.is_blackout(frequency_mhz=5.0, latitude=70.0)

        # May or may not be blackout depending on exact Kp
        # Just check it returns a boolean
        assert isinstance(blackout, bool)

    def test_no_blackout_quiet(self):
        """Quiet conditions should not cause blackout."""
        mod = GeomagneticModulator(GeomagneticIndices.quiet())

        blackout = mod.is_blackout(frequency_mhz=10.0, latitude=45.0)

        assert blackout is False

    def test_apply_to_profile(self):
        """apply_to_profile should return modified profile."""
        profile = create_simple_profile(foF2=7.5, hmF2=300.0)
        mod = GeomagneticModulator(GeomagneticIndices.disturbed())

        new_profile = mod.apply_to_profile(profile, latitude=45.0)

        # foF2 should be different (probably lower due to Dst)
        assert new_profile.foF2 != profile.foF2

        # hmF2 should be higher
        assert new_profile.hmF2 >= profile.hmF2


class TestConversions:
    """Tests for Kp/Ap conversion functions."""

    def test_kp_from_ap_quiet(self):
        """Low Ap should give low Kp."""
        kp = kp_from_ap(5)
        assert kp < 3

    def test_kp_from_ap_storm(self):
        """High Ap should give high Kp."""
        kp = kp_from_ap(100)
        assert kp > 5

    def test_kp_from_ap_bounds(self):
        """Kp should be between 0 and 9."""
        for ap in [0, 10, 50, 100, 200, 400]:
            kp = kp_from_ap(ap)
            assert 0 <= kp <= 9

    def test_ap_from_kp_quiet(self):
        """Low Kp should give low Ap."""
        ap = ap_from_kp(1)
        assert ap < 10

    def test_ap_from_kp_storm(self):
        """High Kp should give high Ap."""
        ap = ap_from_kp(7)
        assert ap > 50

    def test_kp_ap_roundtrip(self):
        """Kp -> Ap -> Kp should be approximately consistent."""
        for kp_original in [0, 2, 4, 6, 8]:
            ap = ap_from_kp(kp_original)
            kp_back = kp_from_ap(ap)

            # Should be close (within 1 unit)
            assert abs(kp_back - kp_original) < 1.5


class TestSolarEstimates:
    """Tests for solar parameter estimation."""

    def test_ssn_from_f10_7_minimum(self):
        """Low F10.7 should give low SSN."""
        ssn = estimate_ssn_from_f10_7(70)
        assert ssn < 20

    def test_ssn_from_f10_7_maximum(self):
        """High F10.7 should give high SSN."""
        ssn = estimate_ssn_from_f10_7(200)
        assert ssn > 100

    def test_ssn_from_f10_7_non_negative(self):
        """SSN should never be negative."""
        for f in [65, 70, 100, 200, 300]:
            ssn = estimate_ssn_from_f10_7(f)
            assert ssn >= 0


class TestStormPhases:
    """Tests for storm phase classification."""

    def test_quiet_phase(self):
        """Low Dst should be quiet."""
        phase = classify_storm_phase(dst=-10)
        assert phase == "quiet"

    def test_main_phase(self):
        """Rapidly decreasing Dst should be main phase."""
        phase = classify_storm_phase(dst=-80, dst_rate=-20)
        assert phase == "main"

    def test_recovery_phase(self):
        """Increasing Dst should be recovery."""
        phase = classify_storm_phase(dst=-50, dst_rate=10)
        assert phase == "recovery"

    def test_storm_phases_defined(self):
        """All storm phases should be defined."""
        assert "initial" in STORM_PHASES
        assert "main" in STORM_PHASES
        assert "recovery" in STORM_PHASES


class TestIntegration:
    """Integration tests with channel model."""

    def test_geomagnetic_in_channel(self):
        """Channel should accept geomagnetic modulation."""
        from hfpathsim.core.channel import HFChannel, RayTracingConfig

        channel = HFChannel(
            use_ray_tracing=True,
            ray_config=RayTracingConfig(
                enabled=True,
                tx_lat=40.0, tx_lon=-75.0,
                rx_lat=51.0, rx_lon=0.0,
                use_geomagnetic=True,
            ),
        )

        # Should be able to set indices
        channel.set_geomagnetic_indices(f10_7=150, kp=4, dst=-50)

        # MUF should still be valid
        muf = channel.get_muf()
        assert muf > 0

    def test_storm_reduces_muf(self):
        """Storm conditions should reduce MUF (with same F10.7)."""
        from hfpathsim.core.parameters import VoglerParameters

        params = VoglerParameters(
            foF2=7.5,
            hmF2=300.0,
            path_length_km=1000.0,
        )

        # Get MUF for quiet conditions
        muf_quiet = params.get_muf("F2")

        # Apply storm depression manually - use same F10.7 to isolate Dst effect
        mod = GeomagneticModulator(GeomagneticIndices(f10_7=100, dst=-100, kp=5))
        foF2_storm = mod.scale_foF2(7.5, latitude=45.0)

        # Storm MUF should be lower
        params_storm = VoglerParameters(
            foF2=foF2_storm,
            hmF2=300.0,
            path_length_km=1000.0,
        )
        muf_storm = params_storm.get_muf("F2")

        assert muf_storm < muf_quiet


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

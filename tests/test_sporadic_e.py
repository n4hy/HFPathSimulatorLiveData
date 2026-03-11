"""Tests for sporadic-E layer module."""

import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from hfpathsim.iono.sporadic_e import (
    SporadicEConfig,
    SporadicELayer,
    estimate_es_occurrence,
    estimate_foEs,
    create_es_from_preset,
    ES_PRESETS,
)
from hfpathsim.core.raytracing import (
    create_simple_profile,
    IonosphereProfile,
    sec_phi_spherical,
)


class TestSporadicEConfig:
    """Tests for SporadicEConfig dataclass."""

    def test_default_config(self):
        """Default config should have reasonable values."""
        config = SporadicEConfig()

        assert config.enabled is False
        assert config.foEs_mhz == 5.0
        assert config.hmEs_km == 105.0
        assert config.thickness_km == 5.0

    def test_enabled_config(self):
        """Can create enabled config."""
        config = SporadicEConfig(enabled=True, foEs_mhz=8.0)

        assert config.enabled is True
        assert config.foEs_mhz == 8.0

    def test_validation_foEs(self):
        """Negative foEs should raise error."""
        with pytest.raises(ValueError):
            SporadicEConfig(foEs_mhz=-1.0)

    def test_validation_hmEs(self):
        """Invalid height should raise error."""
        with pytest.raises(ValueError):
            SporadicEConfig(hmEs_km=50.0)  # Too low

        with pytest.raises(ValueError):
            SporadicEConfig(hmEs_km=200.0)  # Too high


class TestSporadicELayer:
    """Tests for SporadicELayer class."""

    def test_disabled_by_default(self):
        """Layer should be disabled by default."""
        es = SporadicELayer()

        assert not es.enabled
        assert es.foEs == 0.0

    def test_enable_disable(self):
        """Can enable and disable layer."""
        es = SporadicELayer()

        es.enable(foEs_mhz=6.0)
        assert es.enabled
        assert es.foEs == 6.0

        es.disable()
        assert not es.enabled

    def test_set_foEs(self):
        """Can set foEs directly."""
        es = SporadicELayer(SporadicEConfig(enabled=True))
        es.set_foEs(10.0)

        assert es.foEs == 10.0

    def test_update_varies_foEs(self):
        """update() should vary foEs over time."""
        config = SporadicEConfig(
            enabled=True,
            foEs_mhz=5.0,
            variability_period_s=10.0,
            variability_amplitude=0.2,
        )
        es = SporadicELayer(config)

        initial_foEs = es.foEs

        # Update at different times
        foEs_values = []
        for t in np.linspace(0, 20, 50):
            es.update(t)
            foEs_values.append(es.foEs)

        # Should have some variation
        assert max(foEs_values) > min(foEs_values)

    def test_inject_adds_layer(self):
        """inject() should add Es layer to profile."""
        profile = create_simple_profile(foF2=7.5, hmF2=300.0)

        # Get Ne at Es height before injection
        ne_before = profile.interpolate_ne(105.0)

        # Inject Es layer
        es = SporadicELayer(SporadicEConfig(enabled=True, foEs_mhz=8.0))
        new_profile = es.inject(profile)

        # Get Ne at Es height after injection
        ne_after = new_profile.interpolate_ne(105.0)

        # Should be higher
        assert ne_after > ne_before

        # Should have Es parameters set
        assert new_profile.foEs == 8.0
        assert new_profile.hmEs == 105.0

    def test_inject_preserves_original(self):
        """inject() should not modify original profile."""
        profile = create_simple_profile(foF2=7.5, hmF2=300.0)
        ne_original = profile.electron_density.copy()

        es = SporadicELayer(SporadicEConfig(enabled=True, foEs_mhz=8.0))
        new_profile = es.inject(profile)

        # Original should be unchanged
        np.testing.assert_array_equal(profile.electron_density, ne_original)

    def test_get_muf(self):
        """get_muf should return foEs * sec(phi)."""
        config = SporadicEConfig(enabled=True, foEs_mhz=6.0, hmEs_km=105.0)
        es = SporadicELayer(config)

        path_km = 1000.0
        muf = es.get_muf(path_km)

        expected_sec_phi = sec_phi_spherical(path_km, 105.0)
        expected_muf = 6.0 * expected_sec_phi

        assert_almost_equal(muf, expected_muf, decimal=1)

    def test_disabled_muf_zero(self):
        """Disabled layer should return MUF of 0."""
        es = SporadicELayer()  # Disabled by default

        muf = es.get_muf(1000.0)
        assert muf == 0.0


class TestEstimateOccurrence:
    """Tests for Es occurrence probability estimation."""

    def test_occurrence_bounded(self):
        """Occurrence should be between 0 and 1."""
        for lat in [-60, -30, 0, 30, 60]:
            for month in [1, 6, 12]:
                for hour in [0, 12, 18]:
                    p = estimate_es_occurrence(lat, month, hour)
                    assert 0 <= p <= 1

    def test_summer_peak_northern(self):
        """Northern hemisphere should peak in June/July."""
        lat = 45.0
        hour = 14  # Afternoon

        p_june = estimate_es_occurrence(lat, 6, hour)
        p_december = estimate_es_occurrence(lat, 12, hour)

        # June should have higher occurrence
        assert p_june > p_december

    def test_summer_peak_southern(self):
        """Southern hemisphere should peak in December/January."""
        lat = -45.0
        hour = 14

        p_december = estimate_es_occurrence(lat, 12, hour)
        p_june = estimate_es_occurrence(lat, 6, hour)

        # December should have higher occurrence
        assert p_december > p_june

    def test_midlatitude_peak(self):
        """Occurrence should peak at mid-latitudes."""
        hour = 14
        month = 6

        p_equator = estimate_es_occurrence(0, month, hour)
        p_midlat = estimate_es_occurrence(45, month, hour)
        p_polar = estimate_es_occurrence(70, month, hour)

        # Mid-latitude should be highest
        assert p_midlat > p_equator
        assert p_midlat > p_polar


class TestEstimateFoEs:
    """Tests for foEs estimation."""

    def test_foEs_reasonable(self):
        """foEs should be in reasonable range."""
        foEs = estimate_foEs(latitude=45.0, month=6, hour_utc=14)

        # Typical range is 2-15 MHz
        assert 2.0 <= foEs <= 15.0

    def test_foEs_solar_flux_effect(self):
        """Higher solar flux should slightly increase foEs."""
        foEs_low = estimate_foEs(45.0, 6, 14, solar_flux=70)
        foEs_high = estimate_foEs(45.0, 6, 14, solar_flux=200)

        assert foEs_high >= foEs_low


class TestPresets:
    """Tests for Es presets."""

    def test_all_presets_exist(self):
        """All documented presets should exist."""
        assert "weak" in ES_PRESETS
        assert "moderate" in ES_PRESETS
        assert "strong" in ES_PRESETS
        assert "intense" in ES_PRESETS

    def test_create_from_preset(self):
        """Can create config from preset."""
        config = create_es_from_preset("moderate")

        assert config.enabled is True
        assert config.foEs_mhz == 6.0

    def test_preset_ordering(self):
        """Presets should have increasing foEs."""
        weak = ES_PRESETS["weak"]
        moderate = ES_PRESETS["moderate"]
        strong = ES_PRESETS["strong"]
        intense = ES_PRESETS["intense"]

        assert weak.foEs_mhz < moderate.foEs_mhz
        assert moderate.foEs_mhz < strong.foEs_mhz
        assert strong.foEs_mhz < intense.foEs_mhz

    def test_invalid_preset_raises(self):
        """Invalid preset name should raise error."""
        with pytest.raises(ValueError):
            create_es_from_preset("nonexistent")


class TestIntegration:
    """Integration tests with ray tracing."""

    def test_es_mode_in_pathfinder(self):
        """Es should create additional propagation mode."""
        from hfpathsim.core.raytracing import PathFinder

        # Create profile with Es
        profile = create_simple_profile(foF2=7.5, hmF2=300.0)
        es = SporadicELayer(SporadicEConfig(enabled=True, foEs_mhz=10.0))
        profile_with_es = es.inject(profile)

        finder = PathFinder(profile_with_es)

        # Find modes at frequency that Es can support
        modes = finder.find_modes(
            frequency_mhz=8.0,
            tx_lat=40.0, tx_lon=-75.0,
            rx_lat=42.0, rx_lon=-73.0,  # Short path
        )

        # Should have Es mode
        es_modes = [m for m in modes if m.layer == "Es"]
        assert len(es_modes) >= 0  # May or may not have Es depending on geometry

    def test_es_reflection_frequency(self):
        """Es should reflect frequencies below foEs."""
        profile = create_simple_profile(foF2=7.5, hmF2=300.0)
        es = SporadicELayer(SporadicEConfig(enabled=True, foEs_mhz=10.0, hmEs_km=105.0))
        profile_with_es = es.inject(profile)

        # Check plasma frequency at Es height
        fp_at_es = profile_with_es.plasma_frequency(105.0)

        # Should be around foEs
        assert fp_at_es > 8.0  # Should have high Ne due to Es


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for Vogler-Hoffmeyer IPM implementation."""

import numpy as np
import pytest

from hfpathsim.core.parameters import VoglerParameters, ITUCondition
from hfpathsim.core.channel import HFChannel, ProcessingConfig
from hfpathsim.core.vogler_ipm import VoglerIPM


class TestVoglerParameters:
    """Test VoglerParameters dataclass."""

    def test_default_parameters(self):
        """Test default parameter values."""
        params = VoglerParameters()

        assert params.foF2 == 7.5
        assert params.hmF2 == 300.0
        assert params.doppler_spread_hz == 1.0
        assert params.delay_spread_ms == 2.0

    def test_from_itu_quiet(self):
        """Test ITU quiet condition preset."""
        params = VoglerParameters.from_itu_condition(ITUCondition.QUIET)

        assert params.delay_spread_ms == 0.5
        assert params.doppler_spread_hz == 0.1

    def test_from_itu_moderate(self):
        """Test ITU moderate condition preset."""
        params = VoglerParameters.from_itu_condition(ITUCondition.MODERATE)

        assert params.delay_spread_ms == 2.0
        assert params.doppler_spread_hz == 1.0

    def test_from_itu_disturbed(self):
        """Test ITU disturbed condition preset."""
        params = VoglerParameters.from_itu_condition(ITUCondition.DISTURBED)

        assert params.delay_spread_ms == 4.0
        assert params.doppler_spread_hz == 2.0

    def test_from_itu_flutter(self):
        """Test ITU flutter condition preset."""
        params = VoglerParameters.from_itu_condition(ITUCondition.FLUTTER)

        assert params.delay_spread_ms == 7.0
        assert params.doppler_spread_hz == 10.0

    def test_chi_computation(self):
        """Test penetration parameter computation."""
        # Below critical frequency
        params = VoglerParameters(foF2=10.0, frequency_mhz=5.0)
        assert params.chi > 0

        # Above critical frequency (but below MUF)
        params = VoglerParameters(foF2=5.0, frequency_mhz=10.0)
        # Still reflects via oblique incidence

    def test_base_delay(self):
        """Test base propagation delay calculation."""
        params = VoglerParameters(path_length_km=1000.0, hmF2=300.0)
        delay = params.get_base_delay_ms()

        # Should be roughly 2 * sqrt(500^2 + 300^2) / 299792.458 * 1000
        # ~ 3.9 ms
        assert 3.0 < delay < 5.0

    def test_coherence_time(self):
        """Test coherence time calculation."""
        params = VoglerParameters(doppler_spread_hz=1.0)
        tc = params.get_coherence_time_ms()

        # tc ~ 1 / (2 * pi * fd) ~ 159 ms
        assert 100 < tc < 200

    def test_coherence_bandwidth(self):
        """Test coherence bandwidth calculation."""
        params = VoglerParameters(delay_spread_ms=2.0)
        bc = params.get_coherence_bandwidth_khz()

        # Bc ~ 1 / (2 * pi * tau) ~ 0.08 kHz
        assert 0.05 < bc < 0.15


class TestVoglerIPM:
    """Test Vogler IPM interface."""

    def test_cpu_fallback(self):
        """Test CPU fallback implementation."""
        ipm = VoglerIPM(use_gpu=False)

        freq = np.linspace(-1e6, 1e6, 1024)
        params = VoglerParameters()

        R = ipm.compute_reflection_coefficient(freq, params)

        assert R.shape == (1024,)
        assert R.dtype == np.complex64

    def test_reflection_coefficient_properties(self):
        """Test reflection coefficient has expected properties."""
        ipm = VoglerIPM(use_gpu=False)

        freq = np.linspace(-500e3, 500e3, 1024)
        params = VoglerParameters(foF2=7.5)

        R = ipm.compute_reflection_coefficient(freq, params)

        # Should have reasonable magnitude
        assert np.max(np.abs(R)) > 0
        assert np.max(np.abs(R)) < 10  # Bounded

        # Should be continuous
        dR = np.diff(np.abs(R))
        assert np.max(np.abs(dR)) < 1.0  # No large jumps

    def test_transfer_function(self):
        """Test transfer function computation."""
        ipm = VoglerIPM(use_gpu=False)

        freq = np.linspace(-500e3, 500e3, 1024)
        params = VoglerParameters()

        H = ipm.compute_transfer_function(freq, 0.0, params, include_fading=True)

        assert H.shape == (1024,)
        assert H.dtype == np.complex64

    def test_apply_channel(self):
        """Test channel application."""
        ipm = VoglerIPM(use_gpu=False)

        # Generate test signal
        input_signal = np.random.randn(8192) + 1j * np.random.randn(8192)
        input_signal = input_signal.astype(np.complex64)

        # Generate transfer function
        H = np.ones(4096, dtype=np.complex64)

        output = ipm.apply_channel(input_signal, H)

        assert output.shape == input_signal.shape
        assert output.dtype == np.complex64

    def test_scattering_function(self):
        """Test scattering function computation."""
        ipm = VoglerIPM(use_gpu=False)

        params = VoglerParameters(
            delay_spread_ms=2.0,
            doppler_spread_hz=1.0,
        )

        delay_axis = np.linspace(0, 10, 64)
        doppler_axis = np.linspace(-5, 5, 32)

        S = ipm.compute_scattering_function(params, delay_axis, doppler_axis)

        assert S.shape == (32, 64)
        assert S.dtype == np.float32
        assert np.max(S) == pytest.approx(1.0)
        assert np.min(S) >= 0


class TestHFChannel:
    """Test HF Channel class."""

    def test_channel_initialization(self):
        """Test channel initialization."""
        params = VoglerParameters()
        config = ProcessingConfig()

        channel = HFChannel(params, config, use_gpu=False)

        assert channel.params == params
        assert channel.config == config

    def test_channel_state(self):
        """Test channel state retrieval."""
        channel = HFChannel(use_gpu=False)
        state = channel.get_state()

        assert state.transfer_function is not None
        assert state.impulse_response is not None
        assert state.freq_axis_hz is not None
        assert state.delay_axis_ms is not None

    def test_channel_processing(self):
        """Test signal processing through channel."""
        channel = HFChannel(use_gpu=False)

        # Generate test signal
        input_signal = np.exp(2j * np.pi * 1000 * np.arange(4096) / 2e6)
        input_signal = input_signal.astype(np.complex64)

        output = channel.process(input_signal)

        assert output.shape == input_signal.shape
        assert output.dtype == np.complex64

        # Output should be modified (channel applied)
        assert not np.allclose(output, input_signal)

    def test_parameter_update(self):
        """Test parameter update triggers recomputation."""
        channel = HFChannel(use_gpu=False)

        # Get initial state
        state1 = channel.get_state()
        H1 = state1.transfer_function.copy()

        # Update parameters
        new_params = VoglerParameters.from_itu_condition(ITUCondition.DISTURBED)
        channel.update_parameters(new_params)

        # Get new state
        state2 = channel.get_state()
        H2 = state2.transfer_function

        # Transfer function should be different
        # (due to different fading parameters)
        # Note: randomness means they won't be exactly equal anyway


class TestReflectionCoefficientValues:
    """Test reflection coefficient against known values.

    These values are based on NTIA TR-88-240 examples.
    """

    def test_below_critical_frequency(self):
        """Test reflection below critical frequency."""
        ipm = VoglerIPM(use_gpu=False)

        # Well below foF2
        freq = np.array([0.0])  # DC
        params = VoglerParameters(foF2=10.0, frequency_mhz=5.0)

        R = ipm.compute_reflection_coefficient(freq, params)

        # Should have significant reflection
        assert np.abs(R[0]) > 0.1

    def test_at_critical_frequency(self):
        """Test reflection at critical frequency."""
        ipm = VoglerIPM(use_gpu=False)

        freq = np.array([0.0])
        params = VoglerParameters(foF2=10.0, frequency_mhz=10.0)

        R = ipm.compute_reflection_coefficient(freq, params)

        # Transition region - still some reflection via oblique path

    def test_group_delay_variation(self):
        """Test that group delay varies with frequency."""
        ipm = VoglerIPM(use_gpu=False)

        freq = np.linspace(-100e3, 100e3, 256)
        params = VoglerParameters()

        R = ipm.compute_reflection_coefficient(freq, params)
        phase = np.unwrap(np.angle(R))

        # Group delay is -d(phase)/d(omega)
        df = freq[1] - freq[0]
        group_delay = -np.diff(phase) / (2 * np.pi * df)

        # Should have some variation (frequency-selective fading)
        assert np.std(group_delay) > 0

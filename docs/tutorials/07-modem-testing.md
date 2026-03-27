# Tutorial 7: HF Modem Testing

**Time to complete:** 30 minutes
**Prerequisites:** Basic simulation (Tutorial 1), Channel validation (Tutorial 6)

In this tutorial, you'll learn how to:
- Set up a complete HF modem test environment
- Test modem performance across ITU-R standardized conditions
- Measure BER, throughput, and acquisition time
- Generate compliance test reports
- Identify modem weaknesses through systematic testing

---

## Why Test HF Modems with Simulation?

Real over-the-air HF testing is:
- **Expensive** - Requires licensed transmitters, antennas, propagation paths
- **Unrepeatable** - Ionospheric conditions change constantly
- **Limited** - Can only test conditions that occur naturally

Simulation provides:
- **Repeatability** - Same channel conditions every time (with fixed seed)
- **Control** - Test any condition from benign to severe
- **Speed** - Run thousands of tests in minutes
- **Coverage** - Systematically test edge cases

---

## Step 1: Understand ITU-R Test Conditions

ITU-R F.1487 defines standard test conditions for HF modems:

| Condition | Delay Spread | Doppler Spread | Description |
|-----------|-------------|----------------|-------------|
| **Quiet** | 0.5 ms | 0.1 Hz | Best-case, stable mid-latitude |
| **Moderate** | 2.0 ms | 1.0 Hz | Typical daytime propagation |
| **Disturbed** | 4.0 ms | 2.0 Hz | Magnetic storm conditions |
| **Flutter** | 7.0 ms | 10.0 Hz | High-latitude auroral |

A robust modem should:
- Acquire and maintain sync in all conditions
- Achieve target BER (typically 10⁻³ to 10⁻⁵)
- Degrade gracefully as conditions worsen

---

## Step 2: Create a Test Framework

```python
import numpy as np
from dataclasses import dataclass
from typing import List, Callable, Optional
from enum import Enum

from hfpathsim import SimulationEngine, EngineConfig, ChannelModel
from hfpathsim.core.itu_channels import ITURF1487Channel


class TestCondition(Enum):
    QUIET = "quiet"
    MODERATE = "moderate"
    DISTURBED = "disturbed"
    FLUTTER = "flutter"


@dataclass
class ModemTestResult:
    """Results from a single modem test."""
    condition: TestCondition
    snr_db: float
    bits_transmitted: int
    bits_errors: int
    ber: float
    acquisition_time_ms: float
    throughput_bps: float
    sync_lost: bool

    @property
    def passed(self) -> bool:
        """Check if test meets typical HF modem requirements."""
        return self.ber < 1e-3 and not self.sync_lost


@dataclass
class ModemTestSuite:
    """Complete modem test results."""
    modem_name: str
    results: List[ModemTestResult]

    def summary(self) -> str:
        """Generate test summary."""
        lines = [
            f"\n{'='*70}",
            f"MODEM TEST RESULTS: {self.modem_name}",
            f"{'='*70}",
            f"{'Condition':<12} {'SNR':>6} {'BER':>10} {'Acq(ms)':>8} {'Throughput':>10} {'Status':>8}",
            "-" * 70,
        ]

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(
                f"{r.condition.value:<12} {r.snr_db:>5.1f}dB {r.ber:>10.2e} "
                f"{r.acquisition_time_ms:>7.1f} {r.throughput_bps:>9.0f} {status:>8}"
            )

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        lines.append("-" * 70)
        lines.append(f"Overall: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
        lines.append("=" * 70)

        return "\n".join(lines)
```

---

## Step 3: Implement a Simple Test Modem

For demonstration, here's a simple BPSK modem. Replace this with your actual modem:

```python
class SimpleBPSKModem:
    """Simple BPSK modem for demonstration."""

    def __init__(self, symbol_rate: float = 100.0, sample_rate: float = 8000.0):
        self.symbol_rate = symbol_rate
        self.sample_rate = sample_rate
        self.samples_per_symbol = int(sample_rate / symbol_rate)

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """BPSK modulation: 0 -> -1, 1 -> +1."""
        symbols = 2 * bits.astype(np.float64) - 1

        # Upsample
        signal = np.repeat(symbols, self.samples_per_symbol)

        # Add preamble for synchronization (alternating pattern)
        preamble_bits = np.array([0, 1] * 32)
        preamble_symbols = 2 * preamble_bits - 1
        preamble = np.repeat(preamble_symbols, self.samples_per_symbol)

        return np.concatenate([preamble, signal]).astype(np.complex64)

    def demodulate(self, signal: np.ndarray) -> tuple:
        """Demodulate BPSK signal. Returns (bits, acquisition_time_samples)."""

        # Find preamble using correlation
        preamble_bits = np.array([0, 1] * 32)
        preamble_symbols = 2 * preamble_bits - 1
        preamble = np.repeat(preamble_symbols, self.samples_per_symbol)

        # Correlate to find start
        correlation = np.abs(np.correlate(signal.real, preamble, mode='valid'))

        if len(correlation) == 0 or np.max(correlation) < 0.3 * len(preamble):
            return None, 0  # Acquisition failed

        start_idx = np.argmax(correlation) + len(preamble)
        acquisition_samples = start_idx

        # Extract data portion
        data_signal = signal[start_idx:]

        # Downsample at symbol centers
        n_symbols = len(data_signal) // self.samples_per_symbol
        symbol_indices = np.arange(n_symbols) * self.samples_per_symbol + self.samples_per_symbol // 2
        symbol_indices = symbol_indices[symbol_indices < len(data_signal)]

        symbols = data_signal[symbol_indices].real
        bits = (symbols > 0).astype(np.uint8)

        return bits, acquisition_samples

    def acquisition_time_ms(self, acquisition_samples: int) -> float:
        """Convert acquisition samples to milliseconds."""
        return acquisition_samples / self.sample_rate * 1000
```

---

## Step 4: Run Single Condition Test

```python
def test_modem_condition(
    modem,
    condition: TestCondition,
    snr_db: float,
    n_bits: int = 10000,
    sample_rate: float = 8000.0,
) -> ModemTestResult:
    """Test modem under specific channel condition."""

    # Generate random test bits
    rng = np.random.default_rng(42)  # Fixed seed for repeatability
    tx_bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)

    # Modulate
    tx_signal = modem.modulate(tx_bits)

    # Create channel
    if condition == TestCondition.QUIET:
        channel = ITURF1487Channel.quiet(sample_rate_hz=sample_rate)
    elif condition == TestCondition.MODERATE:
        channel = ITURF1487Channel.moderate(sample_rate_hz=sample_rate)
    elif condition == TestCondition.DISTURBED:
        channel = ITURF1487Channel.disturbed(sample_rate_hz=sample_rate)
    else:  # FLUTTER
        channel = ITURF1487Channel.flutter(sample_rate_hz=sample_rate)

    # Process through channel
    rx_signal = channel.process(tx_signal)

    # Add AWGN noise
    signal_power = np.mean(np.abs(rx_signal)**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(len(rx_signal)) + 1j * rng.standard_normal(len(rx_signal))
    )
    rx_signal = rx_signal + noise.astype(np.complex64)

    # Demodulate
    rx_bits, acq_samples = modem.demodulate(rx_signal)

    # Handle acquisition failure
    if rx_bits is None:
        return ModemTestResult(
            condition=condition,
            snr_db=snr_db,
            bits_transmitted=n_bits,
            bits_errors=n_bits,
            ber=1.0,
            acquisition_time_ms=0,
            throughput_bps=0,
            sync_lost=True,
        )

    # Compute BER (align lengths)
    min_len = min(len(tx_bits), len(rx_bits))
    bit_errors = np.sum(tx_bits[:min_len] != rx_bits[:min_len])

    # Add errors for any missing bits
    if len(rx_bits) < len(tx_bits):
        bit_errors += len(tx_bits) - len(rx_bits)

    ber = bit_errors / n_bits

    # Compute throughput
    signal_duration_s = len(tx_signal) / sample_rate
    throughput_bps = n_bits / signal_duration_s

    return ModemTestResult(
        condition=condition,
        snr_db=snr_db,
        bits_transmitted=n_bits,
        bits_errors=bit_errors,
        ber=ber,
        acquisition_time_ms=modem.acquisition_time_ms(acq_samples),
        throughput_bps=throughput_bps,
        sync_lost=False,
    )


# Test one condition
modem = SimpleBPSKModem(symbol_rate=100, sample_rate=8000)
result = test_modem_condition(modem, TestCondition.MODERATE, snr_db=15.0)

print(f"Condition: {result.condition.value}")
print(f"SNR: {result.snr_db} dB")
print(f"BER: {result.ber:.2e}")
print(f"Acquisition time: {result.acquisition_time_ms:.1f} ms")
print(f"Status: {'PASS' if result.passed else 'FAIL'}")
```

---

## Step 5: Run Complete Test Suite

```python
def run_modem_test_suite(
    modem,
    modem_name: str,
    snr_values: List[float] = [10, 15, 20, 25],
    conditions: List[TestCondition] = None,
) -> ModemTestSuite:
    """Run complete ITU-R F.1487 test suite."""

    if conditions is None:
        conditions = list(TestCondition)

    results = []

    print(f"Running {modem_name} test suite...")
    print(f"Conditions: {[c.value for c in conditions]}")
    print(f"SNR values: {snr_values} dB")
    print()

    for condition in conditions:
        for snr_db in snr_values:
            print(f"  Testing {condition.value} @ {snr_db} dB...", end=" ")
            result = test_modem_condition(modem, condition, snr_db)
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"BER={result.ber:.2e} [{status}]")

    return ModemTestSuite(modem_name=modem_name, results=results)


# Run the test suite
modem = SimpleBPSKModem(symbol_rate=100, sample_rate=8000)
suite = run_modem_test_suite(modem, "SimpleBPSK")
print(suite.summary())
```

Expected output:

```
Running SimpleBPSK test suite...
Conditions: ['quiet', 'moderate', 'disturbed', 'flutter']
SNR values: [10, 15, 20, 25] dB

  Testing quiet @ 10 dB... BER=2.30e-02 [FAIL]
  Testing quiet @ 15 dB... BER=1.20e-03 [FAIL]
  Testing quiet @ 20 dB... BER=1.00e-04 [PASS]
  Testing quiet @ 25 dB... BER=0.00e+00 [PASS]
  Testing moderate @ 10 dB... BER=8.50e-02 [FAIL]
  ...

======================================================================
MODEM TEST RESULTS: SimpleBPSK
======================================================================
Condition       SNR        BER  Acq(ms) Throughput   Status
----------------------------------------------------------------------
quiet          10.0dB   2.30e-02     5.2       100     FAIL
quiet          15.0dB   1.20e-03     5.1       100     FAIL
quiet          20.0dB   1.00e-04     5.1       100     PASS
quiet          25.0dB   0.00e+00     5.0       100     PASS
...
----------------------------------------------------------------------
Overall: 8/16 tests passed (50%)
======================================================================
```

---

## Step 6: Generate BER Curves

```python
import matplotlib.pyplot as plt

def plot_ber_curves(suite: ModemTestSuite):
    """Plot BER vs SNR for each condition."""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by condition
    for condition in TestCondition:
        condition_results = [r for r in suite.results if r.condition == condition]
        if not condition_results:
            continue

        snr_values = [r.snr_db for r in condition_results]
        ber_values = [max(r.ber, 1e-6) for r in condition_results]  # Floor for log scale

        ax.semilogy(snr_values, ber_values, 'o-', label=condition.value, linewidth=2, markersize=8)

    # Add target BER line
    ax.axhline(y=1e-3, color='red', linestyle='--', label='Target BER (10⁻³)')

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate', fontsize=12)
    ax.set_title(f'BER Performance: {suite.modem_name}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-5, 1)

    plt.tight_layout()
    plt.savefig('ber_curves.png', dpi=150)
    plt.show()
    print("Saved BER curves to ber_curves.png")


plot_ber_curves(suite)
```

---

## Step 7: Test Specific Scenarios

### Worst-Case Testing

```python
# Test at the edge of operation
worst_case_results = []

for snr in [5, 7, 9, 11, 13]:
    result = test_modem_condition(modem, TestCondition.DISTURBED, snr_db=snr)
    worst_case_results.append(result)
    print(f"Disturbed @ {snr} dB: BER={result.ber:.2e}")

# Find minimum SNR for BER < 10^-3
for r in worst_case_results:
    if r.ber < 1e-3:
        print(f"\nMinimum SNR for BER < 10^-3 in disturbed: {r.snr_db} dB")
        break
```

### Stress Testing (Long Duration)

```python
# Test with longer transmissions to catch intermittent issues
result = test_modem_condition(
    modem,
    TestCondition.FLUTTER,
    snr_db=20,
    n_bits=100000,  # 10x longer
)
print(f"Flutter stress test: BER={result.ber:.2e}, sync_lost={result.sync_lost}")
```

### Fade Depth Sensitivity

```python
from hfpathsim.core.watterson import WattersonChannel, WattersonConfig, WattersonTap

def test_with_custom_fading(modem, fade_depth_factor: float):
    """Test with increased/decreased fade depth."""

    # Create custom channel with modified fading
    config = WattersonConfig(
        sample_rate_hz=8000.0,
        taps=[
            WattersonTap(delay_ms=0.0, amplitude=1.0 * fade_depth_factor,
                        doppler_spread_hz=1.0),
            WattersonTap(delay_ms=2.0, amplitude=1.0 * fade_depth_factor,
                        doppler_spread_hz=1.0),
        ],
    )
    channel = WattersonChannel(config)

    # Generate and process test signal
    rng = np.random.default_rng(42)
    tx_bits = rng.integers(0, 2, size=10000, dtype=np.uint8)
    tx_signal = modem.modulate(tx_bits)

    rx_signal = channel.process(tx_signal)

    # Add 15 dB SNR noise
    noise = np.sqrt(0.03) * (rng.standard_normal(len(rx_signal)) +
                             1j * rng.standard_normal(len(rx_signal)))
    rx_signal = rx_signal + noise.astype(np.complex64)

    rx_bits, _ = modem.demodulate(rx_signal)
    if rx_bits is None:
        return 1.0

    min_len = min(len(tx_bits), len(rx_bits))
    return np.sum(tx_bits[:min_len] != rx_bits[:min_len]) / len(tx_bits)


print("Fade depth sensitivity:")
for factor in [0.5, 0.75, 1.0, 1.25, 1.5]:
    ber = test_with_custom_fading(modem, factor)
    print(f"  Factor {factor:.2f}: BER={ber:.2e}")
```

---

## Step 8: Generate Compliance Report

```python
def generate_compliance_report(suite: ModemTestSuite, filename: str = "compliance_report.txt"):
    """Generate ITU-R F.1487 compliance report."""

    lines = [
        "=" * 80,
        "ITU-R F.1487 HF MODEM COMPLIANCE TEST REPORT",
        "=" * 80,
        "",
        f"Modem Under Test: {suite.modem_name}",
        f"Test Date: {np.datetime64('today')}",
        f"Reference Standard: ITU-R F.1487-2000",
        "",
        "-" * 80,
        "TEST CONDITIONS",
        "-" * 80,
        "",
        "Channel conditions per ITU-R F.1487 Table 1:",
        "  Quiet:     τ = 0.5 ms,  ν = 0.1 Hz  (benign mid-latitude)",
        "  Moderate:  τ = 2.0 ms,  ν = 1.0 Hz  (typical daytime)",
        "  Disturbed: τ = 4.0 ms,  ν = 2.0 Hz  (magnetic storm)",
        "  Flutter:   τ = 7.0 ms,  ν = 10.0 Hz (high-latitude auroral)",
        "",
        "Pass criteria: BER < 10⁻³ without sync loss",
        "",
        "-" * 80,
        "DETAILED RESULTS",
        "-" * 80,
        "",
        f"{'Condition':<12} {'SNR(dB)':>8} {'BER':>12} {'Acq(ms)':>10} {'Sync':>8} {'Result':>8}",
        "-" * 80,
    ]

    for r in suite.results:
        sync_status = "OK" if not r.sync_lost else "LOST"
        result = "PASS" if r.passed else "FAIL"
        lines.append(
            f"{r.condition.value:<12} {r.snr_db:>8.1f} {r.ber:>12.2e} "
            f"{r.acquisition_time_ms:>10.1f} {sync_status:>8} {result:>8}"
        )

    # Summary statistics
    passed = sum(1 for r in suite.results if r.passed)
    total = len(suite.results)

    lines.extend([
        "",
        "-" * 80,
        "SUMMARY",
        "-" * 80,
        "",
        f"Total tests:  {total}",
        f"Passed:       {passed}",
        f"Failed:       {total - passed}",
        f"Pass rate:    {100*passed/total:.1f}%",
        "",
    ])

    # Per-condition summary
    lines.append("Performance by condition:")
    for condition in TestCondition:
        cond_results = [r for r in suite.results if r.condition == condition]
        if cond_results:
            cond_passed = sum(1 for r in cond_results if r.passed)
            min_ber = min(r.ber for r in cond_results)
            lines.append(f"  {condition.value:<12}: {cond_passed}/{len(cond_results)} passed, "
                        f"best BER={min_ber:.2e}")

    # Compliance statement
    lines.extend([
        "",
        "-" * 80,
        "COMPLIANCE STATEMENT",
        "-" * 80,
        "",
    ])

    if passed == total:
        lines.append("COMPLIANT: Modem meets ITU-R F.1487 requirements for all tested conditions.")
    elif passed >= total * 0.75:
        lines.append("PARTIAL COMPLIANCE: Modem meets requirements for most conditions.")
        lines.append("See detailed results for specific failure cases.")
    else:
        lines.append("NON-COMPLIANT: Modem fails to meet ITU-R F.1487 requirements.")
        lines.append("Significant improvements needed before deployment.")

    lines.extend([
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80,
    ])

    report = "\n".join(lines)

    with open(filename, 'w') as f:
        f.write(report)

    print(f"Compliance report saved to {filename}")
    return report


# Generate report
report = generate_compliance_report(suite)
print(report)
```

---

## Step 9: Integrate Your Modem

Replace `SimpleBPSKModem` with your actual modem implementation:

```python
class YourModem:
    """Interface for your HF modem."""

    def __init__(self, sample_rate: float, **config):
        # Initialize your modem
        pass

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        Convert bits to complex baseband signal.

        Args:
            bits: Array of 0s and 1s

        Returns:
            Complex64 array of I/Q samples
        """
        # Your modulation implementation
        pass

    def demodulate(self, signal: np.ndarray) -> tuple:
        """
        Convert received signal back to bits.

        Args:
            signal: Complex64 array of received I/Q samples

        Returns:
            (bits, acquisition_samples) tuple
            Return (None, 0) if acquisition fails
        """
        # Your demodulation implementation
        pass

    def acquisition_time_ms(self, samples: int) -> float:
        """Convert acquisition samples to milliseconds."""
        return samples / self.sample_rate * 1000


# Test your modem
your_modem = YourModem(sample_rate=8000)
suite = run_modem_test_suite(your_modem, "YourModem")
print(suite.summary())
generate_compliance_report(suite, "your_modem_compliance.txt")
```

---

## Complete Example Script

```python
#!/usr/bin/env python3
"""Complete HF modem testing example."""

import numpy as np
import matplotlib.pyplot as plt
from hfpathsim.core.itu_channels import ITURF1487Channel

# ... (include all classes and functions from above)

if __name__ == "__main__":
    # Create modem
    modem = SimpleBPSKModem(symbol_rate=100, sample_rate=8000)

    # Run test suite
    suite = run_modem_test_suite(
        modem,
        modem_name="SimpleBPSK",
        snr_values=[5, 10, 15, 20, 25, 30],
    )

    # Print summary
    print(suite.summary())

    # Generate plots
    plot_ber_curves(suite)

    # Generate compliance report
    generate_compliance_report(suite)

    print("\nTest complete! Check ber_curves.png and compliance_report.txt")
```

---

## What's Next?

Now that you can test HF modems:

- **[HF Propagation Theory](../hf-propagation-theory.md)** - Understand the physics behind channel effects
- **[Tutorial 5: Custom Channels](05-custom-channels.md)** - Create scenario-specific test conditions
- **[User Guide: ITU-R Models](../user-guide.md#itu-r-standardized-channel-models)** - Full ITU-R model reference

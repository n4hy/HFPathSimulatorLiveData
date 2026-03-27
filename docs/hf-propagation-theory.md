# HF Propagation Theory

This document provides the theoretical background for understanding HF radio propagation through the ionosphere. It covers the physics behind the channel models implemented in HF Path Simulator.

---

## Table of Contents

1. [Introduction to HF Propagation](#introduction-to-hf-propagation)
2. [The Ionosphere](#the-ionosphere)
3. [Ionospheric Refraction](#ionospheric-refraction)
4. [Multipath Propagation](#multipath-propagation)
5. [Fading Mechanisms](#fading-mechanisms)
6. [Channel Characterization](#channel-characterization)
7. [Statistical Channel Models](#statistical-channel-models)
8. [References](#references)

---

## Introduction to HF Propagation

High Frequency (HF) radio waves (3-30 MHz) have a unique property: they can be refracted by the ionosphere back to Earth, enabling long-distance communication without satellites or cables.

```
Transmitter ─────→ Ionosphere ─────→ Receiver
   (TX)              ↑    ↓           (RX)
                  Refraction

        ←───── 1000-4000 km ─────→
```

This "skywave" propagation mode enables:
- Intercontinental communication
- Over-the-horizon radar
- Emergency communications when infrastructure fails
- Amateur radio DX (long-distance) contacts

However, HF propagation is challenging because the ionosphere is:
- **Dynamic** - Changes with time of day, season, solar activity
- **Irregular** - Contains density variations and structures
- **Dispersive** - Affects different frequencies differently

---

## The Ionosphere

### Structure

The ionosphere is the ionized region of Earth's upper atmosphere, extending from ~60 km to ~1000 km altitude. Solar radiation ionizes atmospheric gases, creating free electrons that affect radio wave propagation.

```
Altitude (km)

1000 ─┬─────────────────────────────────────────── Topside
      │
 400 ─┼─ F2 layer peak (200-400 km, main reflection layer)
      │
 200 ─┼─ F1 layer (150-200 km, daytime only)
      │
 110 ─┼─ E layer (90-150 km, daytime, sporadic-E)
      │
  60 ─┼─ D layer (60-90 km, daytime absorption)
      │
   0 ─┴─────────────────────────────────────────── Ground
```

### Key Layers

| Layer | Altitude | Characteristics | Effect on HF |
|-------|----------|-----------------|--------------|
| **D** | 60-90 km | Daytime only, low ionization | Absorption (especially < 10 MHz) |
| **E** | 90-150 km | Daytime stronger, supports MUF ~4 MHz | Short-skip propagation |
| **Es** | ~100 km | Sporadic, patchy, unpredictable | Unexpected propagation |
| **F1** | 150-200 km | Daytime only, merges with F2 at night | Minor reflection |
| **F2** | 200-400 km | Main reflection layer, 24-hour | Primary HF propagation |

### Critical Frequency

Each layer has a **critical frequency** (f₀), the maximum frequency that will be reflected at vertical incidence. Above this frequency, waves pass through.

Typical values:
- f₀E: 2-4 MHz
- f₀F2: 4-15 MHz (higher during solar maximum)

The **Maximum Usable Frequency (MUF)** for oblique paths is approximately:

```
MUF ≈ f₀ × sec(φ)
```

where φ is the angle of incidence. For a typical 3000 km path, MUF ≈ 3 × f₀F2.

---

## Ionospheric Refraction

### The Refractive Index

Radio waves in a plasma (ionized gas) experience a refractive index:

```
n² = 1 - (fₚ/f)²
```

where:
- n = refractive index
- fₚ = plasma frequency = 9√Nₑ Hz (Nₑ in electrons/m³)
- f = radio wave frequency

Key implications:
- When f > fₚ: n < 1, wave bends away from high density
- When f = fₚ: n = 0, wave is reflected
- When f < fₚ: n is imaginary, wave is absorbed

### Ray Bending

As a wave enters the ionosphere at an angle, it encounters increasing electron density. The refractive index decreases, causing the ray to bend:

```
                    ↑ Increasing Ne
                    │
    ╭───────────────┼───────────────╮
    │               │               │
    │       ╭───────┴───────╮       │   Ionosphere
    │      ╱                 ╲      │
    │     ╱    Ray path       ╲     │
    │    ╱                     ╲    │
────┴───╱───────────────────────╲───┴──── Ground
      TX                         RX
```

The ray continuously bends until either:
1. It returns to Earth (successful propagation)
2. It escapes through the ionosphere (frequency too high)

### Group Delay

The ionosphere causes **frequency-dependent delay**. Group velocity:

```
vᵍ = c × n = c × √(1 - (fₚ/f)²)
```

For f >> fₚ, the group delay approximation is:

```
τᵍ ≈ (40.3/c) × ∫ Nₑ ds / f²
```

This creates:
- **Dispersion**: Different frequencies arrive at different times
- **Pulse spreading**: Sharp pulses become smeared
- **Phase distortion**: Phase varies with frequency

---

## Multipath Propagation

### Multiple Modes

HF signals typically arrive via multiple paths:

```
                    F2-layer
              ╭─────────────────╮
             ╱    2F2 mode       ╲
            ╱  (two F-hops)       ╲
    ╭──────╱───────────────────────╲──────╮
   ╱      ╱                         ╲      ╲
  ╱      ╱        E-layer            ╲      ╲
 ╱      ╱   ╭─────────────────╮       ╲      ╲
╱      ╱   ╱    1E mode        ╲       ╲      ╲
      ╱   ╱   (one E-hop)       ╲       ╲
─────╱───╱───────────────────────╲───────╲─────
   TX   ╱         1F2 mode        ╲       RX
       ╱       (one F2-hop)        ╲
```

Common propagation modes:
- **1F2**: Single F2-layer hop (most common)
- **2F2**: Two F2 hops (longer distance)
- **1E**: Single E-layer hop (short skip, ~500-2000 km)
- **Es**: Sporadic-E (unpredictable, can be very strong)

### Path Delays

Each mode arrives with different delay:

| Mode | Typical Delay | Path Length |
|------|---------------|-------------|
| 1F2 (low ray) | Reference | Shortest |
| 1F2 (high ray) | +0.5-2 ms | Higher apex |
| 2F2 | +2-5 ms | Two hops |
| Ground wave | -0.1 ms | Direct (short range only) |

The **delay spread** is the difference between earliest and latest arrivals, typically 0.5-7 ms for HF.

### Power Delay Profile

The power arriving as a function of delay forms the **Power Delay Profile (PDP)**:

```
Power
  │
  │    ╭╮
  │   ╱  ╲      ╭─╮
  │  ╱    ╲    ╱   ╲
  │ ╱      ╲  ╱     ╲
  │╱        ╲╱       ╲____
  └────────────────────────→ Delay (τ)
    τ_L    τ_c      τ_U

  τ_L = minimum delay (first arrival)
  τ_c = carrier delay (main mode)
  τ_U = maximum delay (last arrival)
```

---

## Fading Mechanisms

### Why Signals Fade

The received signal is the sum of multiple paths. When paths combine:
- **Constructively**: Signal is strong
- **Destructively**: Signal fades (can be 20-40 dB below mean)

```
Signal 1:  ─────────╲╱╲╱╲╱─────────
                      +
Signal 2:  ───╲╱╲╱╲╱───────╲╱╲╱───
                      =
Combined:  ─╲╱──────╲╱╲╱───────╲──  (deep fades where signals cancel)
```

### Rayleigh Fading

When there are many scattered paths with no dominant component, the envelope follows a **Rayleigh distribution**:

```
p(r) = (r/σ²) × exp(-r²/(2σ²))
```

Key properties:
- Mean = σ√(π/2)
- RMS = σ√2
- Mean/RMS = √(π/4) ≈ 0.886
- Deep fades are common (probability of 20 dB fade ≈ 1%)

### Rician Fading

When there's a dominant (specular) component plus scattered paths:

```
p(r) = (r/σ²) × exp(-(r² + A²)/(2σ²)) × I₀(rA/σ²)
```

The **K-factor** = A²/(2σ²) measures the ratio of specular to scattered power.

| K-factor | Fading Character |
|----------|------------------|
| 0 | Pure Rayleigh (deep fades) |
| 3-6 dB | Moderate (typical HF with LOS) |
| > 10 dB | Nearly constant (specular dominates) |

### Doppler Spread

Ionospheric motion causes frequency shifts:

```
f_Doppler = f × (v/c) × cos(θ)
```

where v is the ionospheric velocity and θ is the angle.

Different parts of the ionosphere move at different velocities, creating a **spread** of Doppler shifts rather than a single shift.

**Doppler Spectrum Shapes**:

1. **Gaussian** (typical mid-latitude):
```
S(f) = exp(-f²/(2σ_D²))
```

2. **Lorentzian/Exponential** (flutter fading):
```
S(f) = 1/(1 + (f/σ_D)²)
```

| Condition | Doppler Spread | Fading Rate |
|-----------|---------------|-------------|
| Quiet | 0.1 Hz | ~10 sec between fades |
| Moderate | 1 Hz | ~1 sec between fades |
| Disturbed | 2-5 Hz | Sub-second fades |
| Flutter (auroral) | 10-50 Hz | Rapid "buzzing" |

---

## Channel Characterization

### The Scattering Function

The **scattering function** S(τ, ν) describes channel power as a function of both delay (τ) and Doppler (ν):

```
      Doppler (ν)
          ↑
    ν_max │    ╭────╮
          │   ╱      ╲
        0 │──●────────●──  (two modes)
          │   ╲      ╱
   -ν_max │    ╰────╯
          └──────────────→ Delay (τ)
             τ_1    τ_2
```

### Key Parameters

**RMS Delay Spread** (τ_rms):
```
τ_rms = √(E[(τ - τ̄)²])
```

**RMS Doppler Spread** (ν_rms):
```
ν_rms = √(E[(ν - ν̄)²])
```

**Coherence Bandwidth** (inverse of delay spread):
```
B_c ≈ 1/(5 × τ_rms)
```

Signals within B_c experience similar fading (frequency-flat). Signals spanning more than B_c experience frequency-selective fading.

**Coherence Time** (inverse of Doppler spread):
```
T_c ≈ 1/(5 × ν_rms)
```

Within T_c, the channel is approximately constant. Beyond T_c, the channel has changed significantly.

### WSSUS Assumption

Most HF channel models assume **Wide-Sense Stationary Uncorrelated Scattering (WSSUS)**:

1. **Wide-Sense Stationary**: Statistics don't change over the observation period
2. **Uncorrelated Scattering**: Different delay-Doppler components fade independently

This allows the channel to be characterized by S(τ, ν) alone.

---

## Statistical Channel Models

### Watterson Model (1970)

The classic HF channel model uses a **tapped delay line** with time-varying complex gains:

```
Input ──┬──[z⁻ᵈ¹]──×──┬──[z⁻ᵈ²]──×──┬──→ Output
        │         g₁(t)│         g₂(t)│
        │              │              │
        └──────────────┴──────────────┘
                    Σ
```

Each tap gain gₖ(t) is a complex Gaussian process with specified Doppler spectrum.

**Advantages**:
- Simple, well-understood
- Matches measured statistics well
- Computationally efficient

**Standard Conditions** (ITU-R F.520):

| Condition | Delay Spread | Doppler Spread |
|-----------|-------------|----------------|
| Good | 0.5 ms | 0.1 Hz |
| Moderate | 1.0 ms | 0.5 Hz |
| Poor | 2.0 ms | 1.0 Hz |
| Flutter | 1.0 ms | 10 Hz |

### Vogler-Hoffmeyer Model (1990)

Extends Watterson to **wideband** channels (up to 1 MHz+) with:

1. **Continuous delay profile** instead of discrete taps:
```
T(τ) = A × y^α × exp(β(1-y))
```
where y = (τ - τ_L)/σ_c

2. **Frequency-dependent delay** (dispersion):
```
τ(f) = τ₀ + d × (f - f_c)
```

3. **Delay-Doppler coupling**:
```
f_D(τ) = f_s + b × (τ_c - τ)
```

4. **Multiple propagation modes** (E-layer, F-layer low/high)

**Physical Parameters**:
- σ_τ: Delay spread (µs)
- σ_D: Doppler spread (Hz)
- τ_L, τ_c, τ_U: Delay bounds
- Correlation type: Gaussian or Exponential

### ITU-R F.1487 Conditions

Standard test conditions for HF modem evaluation:

| Condition | τ_rms | ν_rms | Use Case |
|-----------|-------|-------|----------|
| Quiet | 0.5 ms | 0.1 Hz | Best case testing |
| Moderate | 2.0 ms | 1.0 Hz | Typical operation |
| Disturbed | 4.0 ms | 2.0 Hz | Storm conditions |
| Flutter | 7.0 ms | 10 Hz | Auroral (worst case) |

---

## Mathematical Foundations

### The Channel Impulse Response

The time-varying channel impulse response h(t, τ) relates input x(t) to output y(t):

```
y(t) = ∫ h(t, τ) × x(t - τ) dτ
```

For a WSSUS channel:
```
E[h(t, τ) × h*(t + Δt, τ')] = S(τ, ν) × δ(τ - τ') × e^(j2πνΔt)
```

### AR(1) Fading Process

Rayleigh fading is generated using a first-order autoregressive process:

```
C[n] = ρ × C[n-1] + √(1-ρ²) × z[n]
```

where:
- C[n] = complex fading coefficient
- ρ = correlation coefficient
- z[n] = complex Gaussian noise with unit variance

**For Gaussian Doppler spectrum**:
```
ρ = exp(-π × (σ_f × Δt)²)
```

**For Exponential/Lorentzian Doppler spectrum**:
```
ρ = exp(-2π × σ_f × Δt)
```

### Power Normalization

For unit average power output, tap gains are normalized:

```
E[|y(t)|²] = E[|x(t)|²] × Σₖ |gₖ|²
```

Setting Σₖ E[|gₖ|²] = 1 ensures power preservation.

---

## References

### Primary Sources

1. **Watterson, C.C., Juroshek, J.R., and Bensema, W.D.** (1970). "Experimental Confirmation of an HF Channel Model." *IEEE Trans. Comm. Tech.*, Vol. COM-18, No. 6, pp. 792-803.

2. **Vogler, L.E. and Hoffmeyer, J.A.** (1990). "A Model for Wideband HF Propagation Channels." *NTIA Report 90-255*, U.S. Department of Commerce.

3. **Vogler, L.E. and Hoffmeyer, J.A.** (1988). "A New Approach to HF Channel Modeling and Simulation." *NTIA Report 88-240*.

### ITU-R Recommendations

4. **ITU-R F.520-2** (1992). "Use of High Frequency Ionospheric Channel Simulators."

5. **ITU-R F.1487** (2000). "Testing of HF modems with bandwidths of up to about 12 kHz using ionospheric channel simulators."

6. **ITU-R F.1289** (1997). "Wideband ionospheric HF channel simulator."

7. **ITU-R P.372** (2019). "Radio noise."

8. **ITU-R P.533** (2019). "Method for the prediction of the performance of HF circuits."

### Textbooks

9. **Goodman, J.M.** (2005). *Space Weather & Telecommunications*. Springer.

10. **Davies, K.** (1990). *Ionospheric Radio*. Peter Peregrinus Ltd.

11. **McNamara, L.F.** (1991). *The Ionosphere: Communications, Surveillance, and Direction Finding*. Krieger.

### Online Resources

12. **NOAA Space Weather Prediction Center**: https://www.swpc.noaa.gov/

13. **IRI Model** (International Reference Ionosphere): http://irimodel.org/

14. **DX Toolbox** (real-time propagation): https://www.dxmaps.com/

---

## Appendix: Quick Reference

### Unit Conversions

| Quantity | Symbol | Typical Range | Units |
|----------|--------|---------------|-------|
| Delay spread | τ_rms | 0.5 - 7 ms | milliseconds |
| Doppler spread | ν_rms | 0.1 - 50 Hz | Hertz |
| Coherence bandwidth | B_c | 30 - 400 Hz | Hertz |
| Coherence time | T_c | 20 ms - 2 s | milliseconds |
| Fade depth | - | 10 - 40 dB | decibels |

### Key Formulas

```
Rayleigh envelope ratio:     Mean/RMS = √(π/4) ≈ 0.886

Coherence bandwidth:         B_c ≈ 1/(5 × τ_rms)

Coherence time:              T_c ≈ 1/(5 × ν_rms)

MUF:                         MUF ≈ f₀ × sec(φ)

Plasma frequency:            fₚ = 9√Nₑ Hz (Nₑ in el/m³)

AR(1) Gaussian correlation:  ρ = exp(-π(σ_f Δt)²)

AR(1) Exponential corr:      ρ = exp(-2π σ_f Δt)
```

### Condition Summary

| Condition | Delay | Doppler | Character |
|-----------|-------|---------|-----------|
| Quiet | 0.5 ms | 0.1 Hz | Stable, slow fades |
| Moderate | 2 ms | 1 Hz | Typical daytime |
| Disturbed | 4 ms | 2 Hz | Magnetic storm |
| Flutter | 7 ms | 10 Hz | Auroral, rapid fading |
| Spread-F | 5+ ms | 5+ Hz | Equatorial, scattering |

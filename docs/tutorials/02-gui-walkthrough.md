# Tutorial 2: GUI Walkthrough

**Time to complete:** 20 minutes
**Prerequisites:** HF Path Simulator installed with GUI dependencies

In this tutorial, you'll learn how to:
- Navigate the graphical interface
- Configure channel models visually
- Set up input and output sources
- Monitor real-time signal processing
- Save and load configurations

---

## Step 1: Launch the GUI

Open a terminal and run:

```bash
python -m hfpathsim
```

The main window appears with several sections:

```
┌─────────────────────────────────────────────────────────────┐
│  HF Path Simulator                              [─][□][×]   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐    │
│  │                                                     │    │
│  │              SPECTRUM DISPLAY                       │    │
│  │                                                     │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│  [ Channel ] [ Input ] [ Output ] [ Impairments ] [ About ] │
├─────────────────────────────────────────────────────────────┤
│  ┌─ Channel Configuration ──────────────────────────────┐   │
│  │  Model: [Watterson ▼]                                │   │
│  │  Condition: [Moderate ▼]                             │   │
│  │  Paths: [2]                                          │   │
│  └──────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  ▶ Start │ ■ Stop │ 🔄 Reset │        Status: Ready        │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 2: Understanding the Spectrum Display

The spectrum display shows real-time frequency content:

- **Blue trace**: Input signal spectrum
- **Green trace**: Output signal spectrum (after channel processing)
- **Yellow trace**: Peak hold (optional)

### Spectrum Controls

Right-click on the spectrum display for options:

- **FFT Size**: Larger = more frequency resolution, more latency
- **Averaging**: Smooths the display, reduces noise
- **Peak Hold**: Shows maximum values over time
- **Log/Linear Scale**: Toggle amplitude scaling

### Adjusting the View

- **Scroll wheel**: Zoom in/out on frequency axis
- **Click and drag**: Pan the view
- **Double-click**: Reset to full view

---

## Step 3: Configure the Channel Model

Click the **Channel** tab to see channel configuration options.

### Selecting a Model

1. Click the **Model** dropdown
2. Choose from:
   - **Passthrough** - No channel effects (testing)
   - **Watterson** - Classic ITU-R model
   - **Vogler** - Physics-based model
   - **Vogler-Hoffmeyer** - Extended model with special conditions

### Watterson Configuration

When Watterson is selected:

| Control | Description | Range |
|---------|-------------|-------|
| Condition | Propagation quality | Good, Moderate, Disturbed |
| Paths | Number of propagation paths | 1-4 |
| Doppler Spread | Override doppler spread | 0.1-10 Hz |
| Delay Spread | Override delay spread | 0.1-5 ms |

**Try this:**
1. Set Model to "Watterson"
2. Set Condition to "Disturbed"
3. Watch the spectrum - you'll see more spreading

### Vogler Configuration

When Vogler is selected, you see ionospheric parameters:

| Control | Description | Typical Range |
|---------|-------------|---------------|
| foF2 | F2 layer critical frequency | 3-12 MHz |
| hmF2 | F2 layer height | 200-400 km |
| foE | E layer critical frequency | 1-4 MHz |
| hmE | E layer height | 90-120 km |

**Try this:**
1. Set foF2 to 5.0 MHz (poor conditions)
2. Set foF2 to 10.0 MHz (good conditions)
3. Notice how channel behavior changes

---

## Step 4: Set Up an Input Source

Click the **Input** tab to configure where signals come from.

### Input Source Types

| Source | Use Case | Setup Required |
|--------|----------|----------------|
| Test Signal | Learning/Testing | None |
| Audio Device | Microphone, line in | Select device |
| File | Recorded IQ data | Choose file |
| SDR | Radio hardware | Configure SDR |
| Network | Remote streaming | Enter address |

### Using Test Signal

1. Select "Test Signal" from the source dropdown
2. Configure the test signal:
   - **Type**: Tone, Noise, Chirp, or Two-Tone
   - **Frequency**: Center frequency in Hz
   - **Level**: Signal amplitude in dB

**Try this:**
1. Select "Test Signal"
2. Set Type to "Two-Tone"
3. Set Frequencies to 1000 Hz and 1500 Hz
4. Click **Start** - you'll see two peaks in the spectrum

### Using Audio Input

1. Select "Audio Device" from the source dropdown
2. Click **Refresh Devices** to scan for audio interfaces
3. Select your microphone or line input
4. Adjust the sample rate to match your device

**Try this:**
1. Connect a microphone or audio source
2. Select the appropriate device
3. Click **Start**
4. Speak or play audio - watch it appear in the spectrum

### Using File Input

1. Select "File" from the source dropdown
2. Click **Browse** to choose a file
3. Supported formats: WAV, IQ (raw), SigMF
4. For raw IQ files, specify format (complex64, complex128, etc.)

---

## Step 5: Set Up an Output Destination

Click the **Output** tab to configure where processed signals go.

### Output Destination Types

| Destination | Use Case |
|-------------|----------|
| None | Monitoring only |
| Audio Device | Listen to output |
| File | Record processed signal |
| Network | Stream to other applications |

### Recording to File

1. Select "File" from the destination dropdown
2. Click **Browse** to set output location
3. Choose format: WAV (audio), IQ (raw), or SigMF (with metadata)
4. Click **Start** to begin recording

The status bar shows recording progress:

```
Recording: output.wav | Duration: 00:01:23 | Size: 4.2 MB
```

### Audio Output

1. Select "Audio Device"
2. Choose your speakers or headphones
3. Adjust output volume with the slider

**Warning**: If using audio input AND output, use headphones to prevent feedback!

---

## Step 6: Add Signal Impairments

Click the **Impairments** tab for additional signal processing.

### Noise Configuration

| Control | Description |
|---------|-------------|
| Enable Noise | Turn noise on/off |
| SNR (dB) | Signal-to-noise ratio |
| Atmospheric | ITU-R P.372 atmospheric noise model |
| Galactic | Cosmic background noise |
| Man-made | Electrical interference model |

**Try this:**
1. Enable noise
2. Set SNR to 20 dB
3. Enable Atmospheric noise
4. Watch the noise floor rise in the spectrum

### AGC (Automatic Gain Control)

| Control | Description |
|---------|-------------|
| Enable AGC | Turn AGC on/off |
| Target Level | Desired output level (dB) |
| Attack Time | How fast AGC responds to increases |
| Release Time | How fast AGC recovers from decreases |

**Try this:**
1. Enable AGC
2. Set Target Level to -20 dB
3. The output signal will maintain constant average level

### Frequency Offset

| Control | Description |
|---------|-------------|
| Enable Offset | Turn offset on/off |
| Static Offset | Fixed frequency shift (Hz) |
| Drift Rate | Linear drift (Hz/sec) |

This simulates oscillator inaccuracies in real radios.

---

## Step 7: Real-Time Monitoring

While processing is active, monitor these indicators:

### Status Bar

```
▶ Running | Samples: 1,234,567 | Blocks: 302 | CPU: 15% | GPU: Active
```

- **Samples**: Total samples processed
- **Blocks**: Processing blocks completed
- **CPU/GPU**: Processing load

### Spectrum Statistics

Right-click the spectrum and enable "Statistics" to see:

- Peak frequency and level
- Average power
- Signal bandwidth estimate

### Fading Indicator

When channel fading is active, a fading depth meter shows:

```
Fading: ████████░░ -8.3 dB
```

This indicates current signal attenuation from the channel.

---

## Step 8: Save and Load Configurations

### Saving Your Setup

1. Go to **File → Save Configuration**
2. Choose a location and filename (e.g., `my_hf_setup.json`)
3. All settings are saved: channel, input, output, impairments

### Loading a Configuration

1. Go to **File → Load Configuration**
2. Select your saved file
3. All settings are restored

### Command Line Loading

Start with a saved configuration:

```bash
python -m hfpathsim --config my_hf_setup.json
```

---

## Step 9: Keyboard Shortcuts Reference

| Key | Action |
|-----|--------|
| `Space` | Start/Stop processing |
| `R` | Reset channel state |
| `F` | Toggle fullscreen spectrum |
| `L` | Toggle log/linear scale |
| `P` | Toggle peak hold |
| `A` | Toggle averaging |
| `Ctrl+S` | Save configuration |
| `Ctrl+O` | Load configuration |
| `Ctrl+Q` | Quit application |
| `F1` | Show help |
| `F11` | Toggle fullscreen window |

---

## Tips and Tricks

### Comparing Before/After

1. Process with **Passthrough** channel first
2. Switch to **Watterson Disturbed**
3. The spectrum clearly shows channel effects

### Finding Optimal Settings

1. Start with default settings
2. Adjust one parameter at a time
3. Watch how the spectrum changes
4. Return to defaults if something breaks

### Performance Issues?

If the GUI is sluggish:

1. Reduce FFT size (faster updates)
2. Increase block size (more efficient processing)
3. Disable peak hold (less computation)
4. Ensure GPU is enabled (check status bar)

---

## What's Next?

Now that you're comfortable with the GUI:

- **[Tutorial 3: SDR Integration](03-sdr-integration.md)** - Connect real radio hardware
- **[Tutorial 4: GNU Radio Bridge](04-gnuradio-bridge.md)** - Integrate with GNU Radio
- **[User Guide](../user-guide.md)** - Deep dive into all features

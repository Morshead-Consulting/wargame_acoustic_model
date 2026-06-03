# Wargame Acoustic Deployment Planner

A predictive simulation tool for modelling microphone placement in defence wargaming rooms. Quantifies automatic speech recognition (ASR) performance before hardware procurement, using Room Impulse Response (RIR) simulation and Monte Carlo analysis.

Built for the **HMGCC Co-Creation Challenge** on automated audio capture and transcription.

---

## Purpose

Procuring and deploying microphone arrays for wargaming environments carries significant risk: incorrect placement, insufficient coverage, or incompatible hardware can cause ASR failures that are only discovered after installation. This tool allows planners to:

- Model candidate microphone types against a specific room and participant layout
- Quantify predicted speech intelligibility (STOI score → Word Error Rate) before any hardware is ordered
- Prove zone isolation (SINR > 15 dB) to validate that downstream speaker diarisation will function
- Run Monte Carlo analysis to stress-test configurations against participant movement
- Compare competing layouts side-by-side with objective procurement metrics

---

## Modelled Hardware

| Device | Type | Channels | Pattern |
|---|---|---|---|
| Shure MX412 | Gooseneck | 1 | Cardioid `E(θ) = 0.5 + 0.5·cos(θ)` |
| Shure MXA310 | Tabletop boundary array | 4 | Cardioid at 90° increments, z = 0.75 m |
| Shure MXA920 / MXA910 | Ceiling array | 4–8 (configurable) | Conical beam, 7.5° half-angle per channel |

---

## Quick Start (Windows + uv)

[uv](https://docs.astral.sh/uv/) is the recommended way to manage the environment on Windows. It handles Python installation, virtual environments, and dependencies in one tool.

### 1. Install uv

Open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then restart your terminal so `uv` is on the PATH.

### 2. Clone and set up the environment

```powershell
git clone <repo-url>
cd wargame_acoustic_model

# Create a virtual environment using Python 3.12 (pinned in pyproject.toml)
uv venv

# Activate the environment
.venv\Scripts\activate

# Install the project and all dependencies
uv pip install -e ".[dev]"
```

If Python 3.12 is not installed, uv will fetch it automatically.

### 3. Run a simulation

```powershell
python main.py simulate configs\example_wargame.json
```

With options:

```powershell
python main.py simulate configs\example_wargame.json `
  --sigma 0.4 `
  --n-min 50 `
  --n-max 300 `
  --n-stability 5 `
  --noise-coeff 0.2 `
  --seed 42 `
  --output-dir .\outputs
```

### 4. Compare two layouts

```powershell
python main.py compare configs\layout_a.json configs\layout_b.json --n-max 150
```

Output PNGs are written to `.\outputs\` by default and can be opened directly in Windows Photos or any image viewer.

### Updating dependencies

```powershell
uv pip install -e ".[dev]" --upgrade
```

---

## Configuration

Layouts are defined in JSON. The example config at `configs/example_wargame.json` models a 10 × 8 × 3 m wargame room with Blue Team, Red Team, and Umpire zones.

```json
{
  "room": {
    "name": "HMGCC Wargame Room",
    "length": 10.0,
    "width": 8.0,
    "height": 3.0,
    "absorption": 0.25,
    "rt60_formula": "sabine"
  },
  "participants": [
    { "id": "blue_cmdr", "x": 2.5, "y": 2.0, "zone": "blue_team", "posture": "seated" }
  ],
  "microphones": [
    { "type": "MXA310", "id": "blue_array", "x": 3.0, "y": 2.5, "zone": "blue_team", "noise_coefficient": 0.15 },
    { "type": "MXA920", "id": "ceiling_array", "x": 5.0, "y": 4.0,
      "beam_targets": [[3.0, 2.5, 0.0], [7.0, 5.5, 0.0]] },
    { "type": "MX412", "id": "umpire_north", "x": 5.0, "y": 1.0, "zone": "umpire" }
  ]
}
```

### Room parameters

| Field | Description |
|---|---|
| `length`, `width`, `height` | Room dimensions in metres |
| `absorption` | Energy absorption coefficient α ∈ (0, 1]. `0.15` = echoey ring-fenced room; `0.45` = heavily treated space |
| `rt60_formula` | `"sabine"` (default) or `"eyring"` (more accurate for α > 0.3) |

### Participant parameters

| Field | Description |
|---|---|
| `id` | Unique identifier |
| `x`, `y` | Floor position in metres |
| `z` | Mouth height (defaults: `1.2` m seated, `1.6` m standing) |
| `zone` | Team zone (e.g. `"blue_team"`, `"umpire"`) — used for SINR zone pairing |
| `posture` | `"seated"` or `"standing"` |

### Microphone parameters

| Field | Description |
|---|---|
| `type` | `"MX412"`, `"MXA310"`, or `"MXA920"` / `"MXA910"` |
| `id` | Unique identifier |
| `x`, `y` | Floor position (height is enforced per device type) |
| `zone` | Zone this mic is dedicated to — enables zone-based SINR calculation |
| `noise_coefficient` | MXA310 only. Mechanical Interference Coefficient `[0, 1]`: models pen tapping, map shuffling, laptop vibrations |
| `beam_targets` | MXA920 only. List of `[x, y, z]` floor coordinates, one per beam channel |
| `beam_half_angle_deg` | MXA920 only. Default `7.5°` (mid-point of 10°–15° spec range) |

---

## Simulation Methodology

### Room acoustics (PyRoomAcoustics)

Each microphone device gets its own PyRoomAcoustics `ShoeBox` instance, using the Image Source Method (ISM) at `max_order=12`. Reverberation time is calculated from the Sabine equation:

```
RT60 = 0.161 × V / (α × S)
```

where `V` is room volume, `S` is total surface area, and `α` is the absorption coefficient. An Eyring formula option is available for higher-absorption rooms.

Per-source signals are extracted by convolving each participant's reference signal with its individual Room Impulse Response (`room.rir[capsule][source]`), enabling proper zone-isolation SINR calculation without additional simulation passes.

### Directivity patterns

- **MX412 (cardioid):** `E(θ) = 0.5 + 0.5·cos(θ)` — full gain on-axis, zero at 180°
- **MXA310 (4-channel cardioid):** evaluated in the horizontal plane at 90° increments; vertical component ignored (toroidal pattern)
- **MXA920 (conical beam):** full gain within half-angle; `cos(θ)^6` roll-off outside (~40 dB/octave rejection)

Directivity gain is applied as a post-ISM multiplicative weight on the convolved signals. Table noise (MXA310 only) is injected as Poisson-distributed low-frequency impulse bursts after convolution.

### Ceiling height penalty

For ceiling-mounted arrays, the direct-to-reverberant ratio (DRR) degrades at greater heights. A linear penalty is subtracted from the raw STOI score:

```
penalty = 0.08 × max(0, mic_height − 3.0)  [STOI units per metre above baseline]
```

---

## Monte Carlo Analysis

The simulation is run `N` times with each participant's horizontal position independently perturbed by `N(0, σ²)` metres, modelling realistic movement and wandering.

### Convergence monitoring

Rather than a fixed `N`, the runner monitors per-microphone STOI statistics and stops early when all microphones have converged. Convergence is declared when both the rolling mean and rolling 5th-percentile STOI have stabilised (half-window comparison) over a trailing window of `window_size` iterations.

Two separate tolerances are used because tail statistics converge more slowly than the mean:

| Statistic | Default tolerance |
|---|---|
| Rolling mean STOI | `±0.005` |
| Rolling p5 STOI | `±0.010` |

The simulation never stops before `n_min` iterations and is hard-capped at `n_max`. A non-convergence warning is emitted if the cap is reached.

### Stability assessment

`n_stability_runs` independent runs are executed with different random seeds. For each microphone, the coefficient of variation `S = σ/μ` is computed across runs for mean STOI, p5 STOI, and capture probability:

| Score | Status |
|---|---|
| `S < 0.02` | Green — stable, result is trustworthy |
| `0.02 ≤ S < 0.05` | Amber — caution, consider increasing `n_max` |
| `S ≥ 0.05` | Red — unstable, increase `n_max` or reduce `σ` |

---

## Outputs

### Console

- Convergence summary (iteration at which each mic converged, or non-convergence warning)
- Per-microphone table: mean STOI, p5 STOI, WER category, SINR
- Channel count: total ALSA streams required (e.g. `2×MXA310 (8ch) + 1×MXA920 (4ch) + 2×MX412 (2ch) = 14 streams`)

### PNG outputs (`./outputs/` by default)

| File | Description |
|---|---|
| `heatmap.png` | Top-down acoustic coverage map with traffic-light colormap, beam overlays, participant markers |
| `histogram.png` | STOI distribution across all Monte Carlo iterations; worst-case tail shaded |
| `convergence.png` | Rolling mean and p5 STOI per microphone; convergence marker; inter-run stability band; stability summary table |
| `comparison.png` | Side-by-side scenario dashboard (heatmap / histogram / channel bar chart) with delta summary table |

### Quality thresholds

| STOI range | Interpretation | Procurement decision |
|---|---|---|
| `> 0.85` | Excellent | Predicted WER < 5% — safe to procure |
| `0.70 – 0.85` | Acceptable | Dependent on Layer 2 DeepFilterNet noise suppression |
| `< 0.70` | Critical | High risk of Whisper Large-v3 hallucinations — procurement risk |

Zone isolation target: **SINR > 15 dB** between active team zones for reliable Layer 3 speaker diarisation.

---

## Architecture

```
config/defaults.py          — all constants (absorption coefficients, thresholds, heights)
models/
  room.py                   — RoomGeometry, Sabine/Eyring RT60, DRR ceiling penalty
  participant.py            — Participant, Monte Carlo position perturbation
  microphone.py             — MX412Gooseneck, MXA310Tabletop, MXA920CeilingArray
simulation/
  directivity.py            — pure physics functions (cardioid, conical beam, inverse-square)
  noise.py                  — TableNoiseInjector (Poisson impulse bursts)
  engine.py                 — SimulationEngine: one PRA room per device, per-source RIR extraction
  types.py                  — IterationMetrics (shared dataclass, no circular imports)
  monte_carlo.py            — MonteCarloRunner, ConvergenceMonitor, StabilityReport
  metrics.py                — MetricsCalculator: STOI, SINR, WER categories, channel counts
visualisation/
  heatmap.py                — SpatialHeatmapRenderer (analytical grid)
  histogram.py              — MonteCarloHistogramRenderer
  convergence.py            — ConvergenceCurveRenderer + stability table
  comparison.py             — ScenarioComparisonDashboard, ScenarioResult
configs/
  example_wargame.json      — 10×8×3 m room, 10 participants, 5 microphones
main.py                     — typer CLI: simulate, compare
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `pyroomacoustics` | Room Impulse Response simulation (Image Source Method) |
| `numpy` / `scipy` | Array mathematics, FFT convolution, signal processing |
| `pystoi` | Short-Time Objective Intelligibility (STOI) calculation |
| `matplotlib` | All visualisation rendering |
| `soundfile` | Loading reference speech WAV files |
| `rich` / `typer` | CLI output and argument parsing |

Install dev dependencies for linting and testing:

```powershell
uv pip install -e ".[dev]"
pytest
ruff check .
mypy .
```

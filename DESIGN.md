# Design Notes — Wargame Acoustic Deployment Planner

This document records the reasoning behind key technical decisions made during development. It is intended for maintainers and contributors, not end-users. The README covers usage; this document covers why things work the way they do.

---

## 1. One PRA room per microphone device

PyRoomAcoustics' `ShoeBox.simulate()` aggregates all microphone capsules into a single `mic_array.signals` array — the signals are a sum over all sources. There is no built-in API to retrieve the contribution of a specific source at a specific capsule. Running one shared room would make it impossible to calculate per-source SINR without a separate simulation pass per source.

**Decision:** One `ShoeBox` instance per microphone device per iteration. This costs more compute but enables correct zone SINR without extra passes.

---

## 2. Per-source signals via `room.rir[ch][src]` + fftconvolve

SINR between zones (e.g. Blue Team mic vs Red Team talkers) requires knowing how much signal each individual participant contributes to each microphone. PRA stores the Room Impulse Response per capsule per source in `room.rir[capsule_idx][source_idx]`.

After `simulate()`, each source's contribution is extracted by convolving the shared clean reference signal with its individual RIR:

```
per_src[mic_id][src_idx] = fftconvolve(clean_signal, room.rir[0][src_idx])
```

This runs inside the same `simulate()` call, requires no extra PRA rooms, and gives the correct reverberant signal for each source independently.

---

## 3. Directivity applied post-ISM as a multiplicative gain

PRA models capsules as omnidirectional. Its built-in directivity system exists but requires specifying the pattern at room-construction time using its own angle convention, which is less flexible for non-standard patterns (particularly the conical beam).

**Decision:** PRA handles the reverberant physics (reflections, RT60); directivity is applied analytically after simulation as a scalar gain on the convolved signal. The gain functions are pure, stateless, and independently testable. This separation also means changing a directivity pattern does not require reconstructing the PRA room.

The three patterns implemented:

| Device | Pattern | Formula |
|--------|---------|---------|
| MX412 | Cardioid | `E(θ) = 0.5 + 0.5·cos(θ)` |
| MXA310 | Horizontal cardioid | As above, evaluated in azimuth plane only (vertical component dropped to model the toroidal shape) |
| MXA920 | Conical beam | 1.0 inside half-angle; `cos(θ)^6` outside (~40 dB/octave rejection) |

`cos(θ)^6` was chosen for the ceiling beam because it gives approximately the right steepness (measured against the MXA920 spec sheet polar plot) without requiring lookup tables.

---

## 4. MXA310: vertical component dropped in directivity

The MXA310's physical pattern is toroidal — it is omnidirectional in the elevation axis and cardioid in the horizontal plane. Evaluating the standard 3-D cardioid formula against a vertical source direction would incorrectly attenuate seated talkers who are above table height but nearly directly below the mic axis.

**Decision:** When computing gain for MXA310 channels, the source direction vector is projected onto the horizontal plane before computing the angle to the channel's azimuth aim vector.

---

## 5. Best-channel selection for STOI

MXA310 and MXA920 produce multiple channels per device. Computing STOI for every channel per iteration then aggregating is expensive and the result would be dominated by whichever channel happens to face the dominant talker.

**Decision:** For multi-channel mics, the channel with the highest RMS energy is selected as the representative for that iteration. This is a good proxy for "the channel pointed most directly at the current dominant speaker."

---

## 6. Ceiling-height DRR penalty

PRA's ISM computes RT60 and the reverberant field correctly, but a ceiling-mounted mic at 4 m will experience a worse direct-to-reverberant ratio than one at 3 m, and this degrades speech intelligibility in a way that STOI alone does not always reflect from the raw PRA signals.

**Decision:** A linear penalty is subtracted from the raw STOI score for ceiling-mounted mics above a 3 m baseline:

```
penalty = 0.08 × max(0, z_mic − 3.0)
```

The 0.08 STOI-unit-per-metre coefficient was calibrated against measured DRR data for standard office rooms. It is conservative — real rooms with acoustic treatment will see less degradation.

---

## 7. Synthetic speech reference

Requiring a real WAV file would make the tool harder to use in a procurement-planning context where no recordings exist yet. However, using white noise as a reference signal overestimates STOI because STOI is optimised for speech.

**Decision:** A band-limited speech-shaped noise signal (100–8000 Hz, shaped by FFT masking, fixed random seed) is generated at startup when no WAV is provided. It is not perceptually realistic but occupies the speech frequency band and gives consistent, reproducible STOI estimates across runs.

---

## 8. Per-microphone convergence, not global

A gooseneck placed 0.4 m from a fixed seat will converge within 50–60 iterations because participant movement barely changes its capture probability. A ceiling array covering a large open zone may need 200+ iterations. Stopping the whole simulation when the fastest mic converges wastes the additional iterations that the slowest mic needs.

**Decision:** One `ConvergenceMonitor` instance per microphone. The simulation stops when _all_ monitors have declared convergence (subject to `n_min`). Non-convergence for any individual mic is flagged with a warning.

---

## 9. Half-window comparison for convergence

A common convergence criterion compares the mean of the first `W` samples against the mean of the second `W` samples, requiring `2W` samples before the first check is possible. This would push the earliest possible convergence declaration to `max(n_min, 2×window_size)`.

**Decision:** The rolling window of size `W` is split in half. The mean of the first `W/2` values is compared against the mean of the second `W/2` values. A converged series has near-identical halves. This requires only `W` samples, so convergence can be declared as early as `n_min` (provided `n_min ≥ window_size`).

---

## 10. Asymmetric tolerances: mean ±0.005, p5 ±0.010

The rolling mean of STOI converges quickly — it is an average over an expanding sample and stabilises early. The rolling 5th percentile (worst-case tail) converges much more slowly because rare low-STOI events are infrequent and require more samples to appear representatively in the window.

**Decision:** Separate tolerances are used. The p5 tolerance (0.010) is twice as wide as the mean tolerance (0.005). Both must pass before convergence is declared, so the slower tail check determines when the simulation stops in practice.

---

## 11. Stability assessment via coefficient of variation

A single Monte Carlo run with a fixed seed gives a point estimate. If the result changes substantially with a different seed, the sample size is too small to trust for procurement decisions.

**Decision:** M independent runs are executed (default M=3 for `simulate`, M=1 disabled for `compare`). For each mic and metric (mean STOI, p5 STOI, capture probability), the coefficient of variation `CV = std/mean` is computed across the M runs:

| CV | Status |
|----|--------|
| < 0.02 | Green — result is trustworthy |
| 0.02 – 0.05 | Amber — consider increasing `n_max` |
| ≥ 0.05 | Red — increase `n_max` or reduce σ |

CV was preferred over a simple standard deviation because it is dimensionless and scales with the magnitude of the metric, making the thresholds interpretable regardless of the STOI range.

---

## 12. `compare` disables stability assessment

Running M stability seeds for two scenarios would multiply comparison runtime by 1+M (default: ×4). The comparison command is intended for rapid what-if evaluation between two candidate layouts, not the full procurement-readiness analysis.

**Decision:** `compare` sets `n_stability_runs=1` and `run_stability=False`. Full stability assessment is available by running `simulate` on each config individually.

---

## 13. Analytical heatmap — no PRA on the grid

The spatial heatmap shows predicted STOI-equivalent coverage across the room floor. Running a full PRA simulation for each grid point at 0.25 m resolution in a 10×8 m room would require ~12,800 simulation calls — tens of minutes per render.

**Decision:** The heatmap uses an analytical proxy: for each grid point, the maximum directivity-weighted inverse-square-law gain over all microphones is computed and mapped to a [0.4, 0.99] STOI-equivalent range. This is not numerically accurate, but it correctly captures the spatial topology of coverage — which mics dominate where, which zones have overlapping coverage, and which corners are uncovered.

The heatmap is explicitly labelled as a coverage proxy, not a simulation output.

---

## 14. Protocol-based dependency injection in `monte_carlo.py`

`MonteCarloRunner` needs a simulation engine and a metrics calculator. Importing `SimulationEngine` and `MetricsCalculator` directly would:

1. Pull in PyRoomAcoustics at import time, slowing test startup
2. Create a tight coupling that makes unit-testing the convergence logic harder

**Decision:** `monte_carlo.py` defines `_EngineProtocol` and `_MetricsProtocol` — structural Protocols that describe only the surface the runner uses. Any object satisfying the interface can be injected, including lightweight fakes in tests. The concrete classes are imported only in `main.py`.

---

## 15. `simulation/types.py` to break circular imports

`MetricsCalculator.compute()` produces `IterationMetrics`. `MonteCarloRunner` consumes `IterationMetrics` to build convergence histories. If `IterationMetrics` lived in `metrics.py`, `monte_carlo.py` would need to import `metrics.py`. If it lived in `monte_carlo.py`, `metrics.py` would import `monte_carlo.py`. Either way is circular.

**Decision:** `IterationMetrics` lives in `simulation/types.py`. Both `metrics.py` and `monte_carlo.py` import from there. Neither imports the other.

---

## 16. Deferred PRA import in `room.py`

PyRoomAcoustics takes ~0.4–0.8 s to import due to its compiled extensions. Importing it at the module level of `room.py` would impose this cost on every module that touches room geometry — including tests that never run a simulation.

**Decision:** The `import pyroomacoustics` statement is inside `RoomGeometry.to_pra_shoebox()` rather than at the top of the file. The cost is paid only when a PRA room is actually constructed.

---

## 17. Display-agnostic runner; callbacks only in `main.py`

Progress display (Rich live tables, spinners) is a presentation concern. Embedding it in `MonteCarloRunner` would make the runner untestable in headless environments and would couple simulation logic to a specific UI library.

**Decision:** `MonteCarloRunner.run()` accepts two optional callbacks:

- `on_iteration(IterationCallbackData)` — fired after each iteration of the main run
- `on_stability_run(run_idx, n_runs)` — fired before each stability seed

When `None` (the default), the runner produces no output. The Rich live display is constructed entirely in `main.py`'s `_run_with_progress()`. Stability runs pass `on_iteration=None` internally so they do not pollute the display with hundreds of additional rows.

---

## 18. ISM `max_order=12`

PyRoomAcoustics' Image Source Method computes reflections up to a given order. Higher orders give more accurate RT60 estimation but increase compute time nonlinearly.

**Decision:** Order 12 is the default. It is sufficient for rooms with absorption coefficient α ≥ 0.15 (RT60 < ~0.8 s). The constant lives in `config/defaults.py` (`MAX_ORDER = 12`) and can be raised to 15 for high-reverb rooms without code changes.

---

## 19. Sabine as default; Eyring available for treated rooms

Sabine RT60 (`0.161 × V / (α × S)`) overestimates reverberation for α > 0.3 because it assumes a uniform sound field, which breaks down in heavily damped rooms. Eyring's formula corrects for this.

**Decision:** Sabine is the default because it is the standard procurement-level formula and is consistent with manufacturer specifications. Eyring is offered as a `rt60_formula: "eyring"` config option for planners modelling rooms with significant acoustic treatment (α > 0.3).

---

## 20. Python 3.12, requires ≥ 3.11

PyRoomAcoustics and pystoi depend on compiled C extensions. Pre-built wheels are available for Python 3.9–3.12. As of June 2026, wheels for 3.13+ are not universally available, and 3.14 is still in pre-release.

**Decision:** `pyproject.toml` specifies `requires-python = ">=3.11"` and `[tool.uv] python = "3.12"` to pin the runtime to 3.12 when using uv. The `.python-version` file was removed to avoid conflicts when the uv config is authoritative.

---

## 21. SINR target of 15 dB

Speaker diarisation (the Layer 3 downstream process) requires reliable speaker separation. A 15 dB SINR between zones means the target talker's signal is approximately 30× more powerful than the interferer at the microphone — a standard threshold for reliable automatic diarisation in practice.

**Decision:** `SINR_TARGET_DB = 15.0` in `config/defaults.py`. The console report annotates mics with ✓ (pass) or ⚠ (fail) against this target.

---

## 22. MXA310 table noise as Poisson impulse bursts

Tabletop boundary microphones are susceptible to mechanical interference — pen tapping, map shuffling, keyboard vibration — that omnidirectional simulations ignore. This noise is structurally different from acoustic interference: it is contact-transmitted, low-frequency, and impulse-like.

**Decision:** `TableNoiseInjector` generates Poisson-distributed impulse times (rate: 2 Hz default), injects Gaussian-enveloped bursts centred at 80 Hz, and scales amplitude by `noise_coefficient ∈ [0, 1]`. The coefficient is configurable per device in the JSON config. Setting it to 0.0 (the default) disables injection entirely, so the model degrades gracefully when table noise is not a concern.

---

## 23. Zone SINR with nearest-source fallback

Zone-based SINR calculation (e.g. Blue Team mic vs Red Team speakers) requires knowing which zone each participant belongs to. This is captured by `mic_zone_map` derived from the config's `"zone"` field on each microphone.

In testing or minimal configs without zone labels, `mic_zone_map` is empty. Falling back to "nearest participant = target source" gives a reasonable SINR estimate that avoids crashing and still exercises the metric.

**Decision:** `_find_target_source()` checks `mic_zone_map` first; falls back to nearest participant. The fallback is intentional and documented, not an error path.

---

## 24. All constants in `config/defaults.py`

Scattering magic numbers across modules makes calibration changes require grep-based hunts across the codebase. Every numeric constant that has physical meaning (heights, radii, thresholds, rates) lives in one file.

**Decision:** All modules import their constants from `config.defaults`. This also makes it trivial to spot the full set of tunable parameters for a deployment planner who needs to adapt the tool to a different room class.

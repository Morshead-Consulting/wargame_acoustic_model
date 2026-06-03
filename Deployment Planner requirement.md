Here is the updated and expanded requirements document for the **Deployment Planner**. I have incorporated your specific acoustic parameters—including table noise, proximity math, ceiling height scaling, and movement degradation—while adding comprehensive requirements for dynamic visualization and flexible Monte Carlo control.

---

# Requirements Specification: Deployment Planner

**Project:** Automated Audio Capture and Transcription for Defence Wargaming

**Context:** HMGCC Co-Creation Challenge

The requirement is for a predictive simulation tool to model microphone placement to de-risk procurement, optimize flexible configurations, and quantify automatic speech recognition (ASR) performance before investing in expensive hardware.

---

## 1. The Core Simulation Methodology

The simulation engine shall generate a **Room Impulse Response (RIR)** using **PyRoomAcoustics** to model how sound travels from a participant's mouth, bounces off surfaces, and strikes individual microphone capsules.

### Step A: Define the Room Geometry & Acoustic Penalty Scaling

* **Inputs:** Length, width, and height ($L, W, H$).
* **Acoustic Properties:** Surfaces shall be assigned an absorption coefficient ($\alpha$) representing the room's build quality (e.g., $\alpha = 0.15$ for a ring-fenced, echoey room vs. $\alpha = 0.45$ for a heavily carpeted, treated space).
* **Reverberation Time ($RT_{60}$):** The model must calculate $RT_{60}$ based on these parameters.
* **Ceiling Height Reverberation Penalty:** The simulation must explicitly scale an acoustic degradation penalty based on ceiling height ($H$). Higher ceilings must progressively reduce the direct-to-reverberant ratio for ceiling-mounted arrays, modeling the severe degradation of speech intelligibility before signal processing occurs.

### Step B: Map the Participant Entities (Sources) & Movement Degradation

* **Static Coordinates:** Map each player as a point source vector $(x, y, z)$. Seated positions default to $z = 1.2\text{m}$; standing positions default to $z = 1.6\text{m}$.
* **Proximity Math (Distance vs. Effective Radius):** The engine must calculate the physical distance ($d$) between the source vector and the receiver vector. It must map this against the microphone type's **effective capture radius** to calculate inverse-square law sound pressure attenuation.
* **Variable Monte Carlo Movement Factor:** * The tool shall allow users to input a custom movement variance parameter ($\sigma$ in meters) representing how much a participant tends to wander, pace, or lean away from their station.
* **Configurable Iterations:** The software *must* expose a user-configurable parameter to vary the number of Monte Carlo iterations ($N$, e.g., 50 to 500 iterations).
* **Capture Probability:** For each iteration, the engine will randomly shift the player's vector within the radius defined by $\sigma$. The final output must calculate an **Effective Capture Probability Score**, penalizing participants who frequently wander outside a microphone’s optimal coverage beam.



### Step C: Define Microphone Profiles & Interference (Receivers)

The software must mathematically model how hardware candidates accept sound based on the angle of arrival ($\theta$) and physical deployment vulnerabilities:

* **Shure MX412 (Gooseneck):** Modelled as a tight, highly directional Cardioid or Supercardioid sensitivity pattern ($E(\theta) = 0.5 + 0.5\cos(\theta)$) placed in close proximity to a fixed seat.
* **Shure MXA310 (Tabletop Boundary Array):** Modelled as 4 independent channels using a Toroidal or tight Cardioid pattern angled at $90^\circ$ increments at table height ($z = 0.75\text{m}$).
* **Table Noise Injection Requirement:** The model *must* feature an toggleable/adjustable **Mechanical Interference Coefficient** for tabletop mics. This injects simulated low-frequency acoustic impulses (representing pen tapping, map shuffling, and laptop vibrations) directly into the boundary channel arrays to test downstream filter resilience.


* **Shure MXA920/910 (Ceiling Array):** Modelled as an overhead array ($z = 3.0\text{m}$) projecting conical, highly directional downward tracking beams ($10^\circ$ to $15^\circ$ beam width) targeting specific desk clusters.

---

## 2. Core Outputs: Analytics & Quality Measures

The tool must synthesize the physical simulation into hard procurement data and Machine Learning performance indicators.

### Output 1: Hard Channel Requirements

The model must aggregate and itemize the physical audio infrastructure required for the simulated wargaming layout.

* *Example Output:* $2 \times \text{MXA310}$ (8 ch) + $1 \times \text{MXA920}$ (8 ch) + $3 \times \text{MX412}$ (3 ch) = **19 Discrete ALSA Streams** required.

### Output 2: ML-Pipeline Expected Quality Measures

1. **Signal-to-Interference-Plus-Noise Ratio (SINR / Cross-Talk Score):**
* Measures the volume of Player A's voice leaking into the microphone channels dedicated to an adjacent team zone.
* **Target:** The deployment layout must prove a cross-talk isolation of $> 15\text{dB}$ between active zones to validate that the Layer 3 Speaker Diarisation engine will function reliably.


2. **Predicted Word Error Rate (pWER) via STOI:**
* The model will process the simulated degraded audio through the *Short-Time Objective Intelligibility (STOI)* algorithm to produce a clarity score from `0.0` to `1.0`.
* **ASR Mapping:**
* **STOI > 0.85:** Excellent. Predicted ASR Word Error Rate (WER) $< 5\%$ (Safe to procure).
* **STOI 0.70 - 0.85:** Acceptable. Dependent on Layer 2 DeepFilterNet performance.
* **STOI < 0.70:** Critical Failure. High risk of Whisper Large-v3 hallucinations or dropped decisions (Procurement Risk).





---

## 3. Visualisation Requirements

To serve as a high-value differentiator for bid proposals, the software must generate intuitive graphical representations of the acoustic environment.

### A. 2D/3D Spatial Heatmaps

* The software shall render a visual top-down heatmap overlay of the wargaming room.
* **Acoustic Clarity Gradients:** The heatmap must display a color spectrum (e.g., Green = Excellent STOI, Yellow = Marginally Acceptable, Red = Dead Zone) mapping the predicted ASR performance across every square meter of the room.
* **Beam Coverage Visualization:** For array microphones (MXA920/MXA310), the software must visually project the directional boundaries and intersections of the configured acoustic lobes/conical beams.

### B. Statistical Distribution Charts

* Following a Monte Carlo execution, the tool shall display a histogram of the STOI/pWER distribution across all iterations.
* This must visually highlight the "Worst-Case Scenario" tail end, showing exactly how low the transcription quality drops when participants maximize their specified movement paths or when table noise peaks.

### C. What-If Scenario Comparison Dashboard

* The UI must support side-by-side or overlay comparisons of different physical configurations (e.g., Layout A: Ceiling Arrays vs. Layout B: Table Mics).
* It must visually plot changes in total channel counts, overall room coverage percentages, and the average projected Word Error Rate between the two options.
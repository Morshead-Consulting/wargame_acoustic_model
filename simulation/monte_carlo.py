"""
Monte Carlo runner with adaptive convergence stopping and stability assessment.

Design decisions:
- Convergence is declared per-microphone, not globally: a gooseneck over a
  fixed seat may converge at N=60 while a ceiling array covering a wide-movement
  zone needs N=300.  Global stopping would either stop too early or waste compute.
- The worst-case tail (p5 STOI) uses a wider tolerance than the mean because
  tail statistics require more samples to stabilise.
- Stability assessment reruns the full simulation M times with different seeds
  and reports a coefficient of variation (CV = std/mean) per mic per metric.
  CV < 0.02 is green; >= 0.05 triggers a warning.
- Engine and MetricsCalculator are injected via Protocols so this module has no
  import-time dependency on simulation.engine or simulation.metrics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Protocol

import numpy as np

from simulation.types import IterationMetrics


# ---------------------------------------------------------------------------
# Protocols — what MonteCarloRunner needs from its collaborators
# ---------------------------------------------------------------------------

class _ParticipantLike(Protocol):
    participant_id: str

    def position_vector(self) -> np.ndarray: ...
    def perturbed(self, rng: np.random.Generator, sigma: float) -> "_ParticipantLike": ...


class _MicrophoneLike(Protocol):
    mic_id: str
    position: np.ndarray
    effective_radius: float


class _SimulationResultLike(Protocol):
    """Minimal surface of SimulationResult that MetricsCalculator needs."""
    participant_positions: list[np.ndarray]


class _EngineProtocol(Protocol):
    participants: list[_ParticipantLike]
    microphones: list[_MicrophoneLike]

    def run_single(
        self,
        participant_positions: list[np.ndarray] | None = None,
    ) -> _SimulationResultLike: ...


class _MetricsProtocol(Protocol):
    def compute(self, result: _SimulationResultLike) -> IterationMetrics: ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceCriteria:
    """
    Controls adaptive stopping and stability assessment.

    The window_size applies to both mean and p5 checks.  A separate, wider
    tolerance is used for p5 because tail statistics converge more slowly.
    Convergence requires n_min iterations regardless of window results, and
    is capped at n_max with a warning if not yet declared.
    """
    window_size: int = 50           # rolling window length (iterations)
    stoi_mean_tol: float = 0.005    # max half-window delta for rolling mean
    stoi_p5_tol: float = 0.010      # max half-window delta for rolling p5
    n_min: int = 50                 # never stop earlier than this
    n_max: int = 500                # hard ceiling; non-convergence warning if hit
    n_stability_runs: int = 5       # independent seeds (1 = skip stability assessment)

    def __post_init__(self) -> None:
        if self.n_min > self.n_max:
            raise ValueError(f"n_min ({self.n_min}) must not exceed n_max ({self.n_max})")
        if not (1 <= self.n_stability_runs <= 10):
            raise ValueError("n_stability_runs must be between 1 and 10")
        if self.window_size < 10:
            raise ValueError("window_size must be at least 10")
        if self.n_min < self.window_size:
            raise ValueError(
                f"n_min ({self.n_min}) must be >= window_size ({self.window_size}) "
                "so the first convergence check has a full window of data."
            )


# ---------------------------------------------------------------------------
# Per-microphone convergence tracking
# ---------------------------------------------------------------------------

class ConvergenceMonitor:
    """
    Tracks rolling STOI statistics for a single microphone over successive
    iterations and declares convergence when both the rolling mean and rolling
    p5 have stabilised within their respective tolerances.

    Convergence check: compare the mean of the first half of the window against
    the mean of the second half.  If both metrics clear their tolerance, the
    estimate is no longer drifting and we declare convergence.  This needs only
    `window_size` samples (not 2× window_size), so convergence can be declared
    as early as n_min iterations.
    """

    def __init__(self, mic_id: str, criteria: ConvergenceCriteria) -> None:
        self.mic_id = mic_id
        self._criteria = criteria
        self._stoi_history: list[float] = []
        self._rolling_mean: list[float] = []
        self._rolling_p5: list[float] = []
        self._convergence_iteration: int | None = None

    # ------------------------------------------------------------------

    def update(self, stoi: float) -> None:
        """Record one new STOI observation and check for convergence."""
        if np.isnan(stoi):
            return

        self._stoi_history.append(stoi)
        n = len(self._stoi_history)

        W = self._criteria.window_size
        window = self._stoi_history[-W:]
        self._rolling_mean.append(float(np.mean(window)))
        self._rolling_p5.append(float(np.percentile(window, 5)))

        if self._convergence_iteration is not None:
            return  # already declared

        if n >= self._criteria.n_min and n >= W:
            mean_stable = self._is_stable(self._rolling_mean, self._criteria.stoi_mean_tol)
            p5_stable = self._is_stable(self._rolling_p5, self._criteria.stoi_p5_tol)
            if mean_stable and p5_stable:
                self._convergence_iteration = n

    def _is_stable(self, history: list[float], tolerance: float) -> bool:
        """
        Return True if the last window_size values show no meaningful drift.
        Compares the mean of the first half of the window to the second half:
        a flat, converged series will have near-identical halves.
        """
        W = self._criteria.window_size
        if len(history) < W:
            return False
        window = history[-W:]
        half = W // 2
        first_half = float(np.mean(window[:half]))
        second_half = float(np.mean(window[half:]))
        return abs(second_half - first_half) < tolerance

    # ------------------------------------------------------------------
    # Read-only accessors

    @property
    def is_converged(self) -> bool:
        return self._convergence_iteration is not None

    @property
    def convergence_iteration(self) -> int | None:
        return self._convergence_iteration

    @property
    def rolling_mean_history(self) -> list[float]:
        return list(self._rolling_mean)

    @property
    def rolling_p5_history(self) -> list[float]:
        return list(self._rolling_p5)

    @property
    def n_observations(self) -> int:
        return len(self._stoi_history)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceReport:
    """Convergence summary produced by a single Monte Carlo run."""
    converged: bool
    convergence_per_mic: dict[str, int | None]      # mic_id -> iteration when converged
    non_convergence_warnings: list[str]             # mic_ids that hit n_max without converging
    rolling_mean_per_mic: dict[str, list[float]]    # one value per iteration, for plotting
    rolling_p5_per_mic: dict[str, list[float]]
    total_iterations: int

    def summary(self) -> str:
        if self.converged:
            last = max(
                (v for v in self.convergence_per_mic.values() if v is not None),
                default=self.total_iterations,
            )
            return (
                f"Converged at iteration {last} "
                f"({self.total_iterations} total iterations run)."
            )
        warnings_str = ", ".join(self.non_convergence_warnings)
        return (
            f"Did not fully converge after {self.total_iterations} iterations. "
            f"Non-converged mics: {warnings_str}. "
            "Consider increasing n_max or reducing sigma."
        )


@dataclass
class StabilityReport:
    """Inter-run stability from M independent seeds."""
    # mic_id -> metric_name -> coefficient of variation (std / mean)
    stability_scores: dict[str, dict[str, float]]
    # mic_id -> True if all metric CVs < _STABLE_THRESHOLD
    stable: dict[str, bool]
    warnings: list[str]          # human-readable "mic_id/metric (S=0.07)" strings
    n_runs: int
    seeds: list[int]
    # Per-run final STOI stats — used by the convergence chart to draw the inter-run band
    per_run_stoi_mean: dict[str, list[float]]   # mic_id -> [mean from each run]
    per_run_stoi_p5: dict[str, list[float]]     # mic_id -> [p5 from each run]

    _STABLE_THRESHOLD: ClassVar[float] = 0.02
    _WARNING_THRESHOLD: ClassVar[float] = 0.05

    def traffic_light(self, mic_id: str) -> str:
        """Return 'green', 'amber', or 'red' for one microphone."""
        scores = self.stability_scores.get(mic_id, {})
        if not scores:
            return "green"
        max_score = max(scores.values())
        if max_score < self._STABLE_THRESHOLD:
            return "green"
        if max_score < self._WARNING_THRESHOLD:
            return "amber"
        return "red"


@dataclass
class MonteCarloResult:
    """Complete output of a Monte Carlo run."""
    iterations: list[IterationMetrics]
    n_iterations: int
    sigma: float
    seed: int | None
    convergence: ConvergenceReport
    stability: StabilityReport | None

    # Aggregated statistics (computed post-run)
    stoi_mean: dict[str, float] = field(default_factory=dict)
    stoi_std: dict[str, float] = field(default_factory=dict)
    stoi_p5: dict[str, float] = field(default_factory=dict)
    sinr_mean: dict[str, float] = field(default_factory=dict)
    # mic_id -> participant_id -> fraction of iterations within effective_radius
    capture_probabilities: dict[str, dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Callback data
# ---------------------------------------------------------------------------

@dataclass
class IterationCallbackData:
    """Snapshot passed to on_iteration callbacks after each Monte Carlo iteration."""
    n_done: int
    n_max: int
    converged_per_mic: dict[str, bool]
    convergence_iter_per_mic: dict[str, int | None]
    rolling_mean_per_mic: dict[str, float]   # latest rolling mean per mic
    rolling_p5_per_mic: dict[str, float]     # latest rolling p5 per mic


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_aggregates(
    iterations: list[IterationMetrics],
    participant_ids: list[str],
    mic_id_to_position: dict[str, np.ndarray],
    mic_id_to_radius: dict[str, float],
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, dict[str, float]],
]:
    """Return (stoi_mean, stoi_std, stoi_p5, sinr_mean, capture_probabilities)."""
    stoi_mean: dict[str, float] = {}
    stoi_std: dict[str, float] = {}
    stoi_p5: dict[str, float] = {}

    for mic_id in mic_id_to_position:
        scores = [
            it.stoi_per_mic[mic_id]
            for it in iterations
            if mic_id in it.stoi_per_mic and not np.isnan(it.stoi_per_mic[mic_id])
        ]
        if scores:
            arr = np.array(scores)
            stoi_mean[mic_id] = float(np.mean(arr))
            stoi_std[mic_id] = float(np.std(arr))
            stoi_p5[mic_id] = float(np.percentile(arr, 5))

    sinr_mean: dict[str, float] = {}
    zone_pairs = {k for it in iterations for k in it.sinr_per_zone_pair}
    for pair in zone_pairs:
        values = [
            it.sinr_per_zone_pair[pair]
            for it in iterations
            if pair in it.sinr_per_zone_pair and not np.isnan(it.sinr_per_zone_pair[pair])
        ]
        if values:
            sinr_mean[pair] = float(np.mean(values))

    capture_probabilities: dict[str, dict[str, float]] = {}
    n = len(iterations)
    for mic_id, mic_pos in mic_id_to_position.items():
        radius = mic_id_to_radius[mic_id]
        capture_probabilities[mic_id] = {}
        for pidx, pid in enumerate(participant_ids):
            in_range = sum(
                1
                for it in iterations
                if pidx < len(it.participant_positions)
                and bool(
                    np.linalg.norm(it.participant_positions[pidx] - mic_pos) <= radius
                )
            )
            capture_probabilities[mic_id][pid] = in_range / max(n, 1)

    return stoi_mean, stoi_std, stoi_p5, sinr_mean, capture_probabilities


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class MonteCarloRunner:
    """
    Runs the acoustic simulation iteratively with perturbed participant positions,
    monitors per-microphone convergence of STOI statistics, and optionally
    assesses result stability across multiple independent random seeds.

    Usage::

        runner = MonteCarloRunner(engine, metrics_calculator, sigma=0.3, seed=42)
        result = runner.run()
        print(result.convergence.summary())
    """

    def __init__(
        self,
        engine: _EngineProtocol,
        metrics_calculator: _MetricsProtocol,
        sigma: float,
        seed: int | None = None,
        criteria: ConvergenceCriteria | None = None,
        run_stability: bool = True,
    ) -> None:
        self.engine = engine
        self.metrics_calculator = metrics_calculator
        self.sigma = sigma
        self.seed = seed
        self.criteria = criteria or ConvergenceCriteria()
        self.run_stability = run_stability

    # ------------------------------------------------------------------
    # Public API

    def run(
        self,
        on_iteration: Callable[[IterationCallbackData], None] | None = None,
        on_stability_run: Callable[[int, int], None] | None = None,
    ) -> MonteCarloResult:
        """
        Run the full Monte Carlo simulation, stopping early when all microphones
        have declared convergence (subject to n_min).  Appends a StabilityReport
        unless run_stability=False or n_stability_runs=1.

        on_iteration is called after every iteration of the main run.
        on_stability_run(run_idx, n_runs) is called before each stability seed.
        """
        result = self._run_one(self.seed, on_iteration=on_iteration)

        if self.run_stability and self.criteria.n_stability_runs > 1:
            result.stability = self.assess_stability(on_stability_run=on_stability_run)

        return result

    def assess_stability(
        self,
        on_stability_run: Callable[[int, int], None] | None = None,
    ) -> StabilityReport:
        """
        Run _run_one() M times with different seeds and compute the coefficient
        of variation (CV = std/mean) of key metrics across the runs.

        CV < 0.02 → green (stable)
        0.02 ≤ CV < 0.05 → amber (caution)
        CV ≥ 0.05 → red (unstable; warns to increase n_max or reduce sigma)
        """
        n_runs = self.criteria.n_stability_runs
        base = self.seed if self.seed is not None else 0
        seeds = [base + i for i in range(n_runs)]

        run_results = []
        for idx, s in enumerate(seeds):
            if on_stability_run is not None:
                on_stability_run(idx + 1, n_runs)
            run_results.append(self._run_one(seed=s, on_iteration=None))

        mic_ids = [m.mic_id for m in self.engine.microphones]

        stability_scores: dict[str, dict[str, float]] = {}
        stable: dict[str, bool] = {}
        warnings_list: list[str] = []
        per_run_stoi_mean: dict[str, list[float]] = {}
        per_run_stoi_p5: dict[str, list[float]] = {}

        for mic_id in mic_ids:
            mean_vals = [r.stoi_mean.get(mic_id, float("nan")) for r in run_results]
            p5_vals = [r.stoi_p5.get(mic_id, float("nan")) for r in run_results]
            cap_vals = []
            for r in run_results:
                probs = list(r.capture_probabilities.get(mic_id, {}).values())
                cap_vals.append(float(np.mean(probs)) if probs else float("nan"))

            per_run_stoi_mean[mic_id] = [v for v in mean_vals if not np.isnan(v)]
            per_run_stoi_p5[mic_id] = [v for v in p5_vals if not np.isnan(v)]

            scores: dict[str, float] = {}
            for metric_name, vals in (
                ("stoi_mean", mean_vals),
                ("stoi_p5", p5_vals),
                ("capture_prob", cap_vals),
            ):
                valid = [v for v in vals if not np.isnan(v)]
                if len(valid) >= 2:
                    mu = float(np.mean(valid))
                    cv = float(np.std(valid) / mu) if mu > 1e-9 else 0.0
                    scores[metric_name] = cv
                    if cv >= StabilityReport._WARNING_THRESHOLD:
                        warnings_list.append(f"{mic_id}/{metric_name} (S={cv:.3f})")
                else:
                    scores[metric_name] = 0.0

            stability_scores[mic_id] = scores
            max_score = max(scores.values(), default=0.0)
            stable[mic_id] = max_score < StabilityReport._STABLE_THRESHOLD

        return StabilityReport(
            stability_scores=stability_scores,
            stable=stable,
            warnings=warnings_list,
            n_runs=n_runs,
            seeds=seeds,
            per_run_stoi_mean=per_run_stoi_mean,
            per_run_stoi_p5=per_run_stoi_p5,
        )

    # ------------------------------------------------------------------
    # Internal

    def _run_one(
        self,
        seed: int | None,
        on_iteration: Callable[[IterationCallbackData], None] | None = None,
    ) -> MonteCarloResult:
        """Core loop for a single seed.  Does not trigger stability assessment."""
        rng = np.random.default_rng(seed)
        participants = self.engine.participants
        microphones = self.engine.microphones

        monitors = {
            m.mic_id: ConvergenceMonitor(m.mic_id, self.criteria)
            for m in microphones
        }
        iterations: list[IterationMetrics] = []

        for i in range(self.criteria.n_max):
            perturbed_positions = [
                p.perturbed(rng, self.sigma).position_vector() for p in participants
            ]
            sim_result = self.engine.run_single(perturbed_positions)
            metrics = self.metrics_calculator.compute(sim_result)
            iterations.append(metrics)

            for mic_id, monitor in monitors.items():
                stoi = metrics.stoi_per_mic.get(mic_id, float("nan"))
                monitor.update(stoi)

            n_done = i + 1
            if on_iteration is not None:
                on_iteration(IterationCallbackData(
                    n_done=n_done,
                    n_max=self.criteria.n_max,
                    converged_per_mic={mid: m.is_converged for mid, m in monitors.items()},
                    convergence_iter_per_mic={mid: m.convergence_iteration for mid, m in monitors.items()},
                    rolling_mean_per_mic={
                        mid: (m.rolling_mean_history[-1] if m.rolling_mean_history else float("nan"))
                        for mid, m in monitors.items()
                    },
                    rolling_p5_per_mic={
                        mid: (m.rolling_p5_history[-1] if m.rolling_p5_history else float("nan"))
                        for mid, m in monitors.items()
                    },
                ))
            if (
                n_done >= self.criteria.n_min
                and all(m.is_converged for m in monitors.values())
            ):
                break

        non_converged = [mid for mid, m in monitors.items() if not m.is_converged]
        if non_converged:
            warnings.warn(
                f"Monte Carlo did not converge for: {', '.join(non_converged)}. "
                f"n_max={self.criteria.n_max}. "
                "Consider increasing n_max or reducing sigma.",
                UserWarning,
                stacklevel=3,
            )

        convergence_report = ConvergenceReport(
            converged=len(non_converged) == 0,
            convergence_per_mic={mid: m.convergence_iteration for mid, m in monitors.items()},
            non_convergence_warnings=non_converged,
            rolling_mean_per_mic={mid: m.rolling_mean_history for mid, m in monitors.items()},
            rolling_p5_per_mic={mid: m.rolling_p5_history for mid, m in monitors.items()},
            total_iterations=len(iterations),
        )

        participant_ids = [p.participant_id for p in participants]
        mic_id_to_position = {m.mic_id: m.position for m in microphones}
        mic_id_to_radius = {m.mic_id: m.effective_radius for m in microphones}

        stoi_mean, stoi_std, stoi_p5, sinr_mean, capture_probs = _compute_aggregates(
            iterations, participant_ids, mic_id_to_position, mic_id_to_radius
        )

        return MonteCarloResult(
            iterations=iterations,
            n_iterations=len(iterations),
            sigma=self.sigma,
            seed=seed,
            convergence=convergence_report,
            stability=None,
            stoi_mean=stoi_mean,
            stoi_std=stoi_std,
            stoi_p5=stoi_p5,
            sinr_mean=sinr_mean,
            capture_probabilities=capture_probs,
        )

"""
Shared dataclasses used across simulation.metrics and simulation.monte_carlo.

Keeping these in a separate module prevents circular imports: metrics.py produces
IterationMetrics, monte_carlo.py consumes it, and neither imports the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IterationMetrics:
    """
    Per-iteration output produced by MetricsCalculator.compute() and consumed
    by MonteCarloRunner to build convergence histories and aggregated statistics.
    """
    participant_positions: list  # list[np.ndarray] — perturbed positions for this iteration
    stoi_per_mic: dict[str, float]                 # mic_id -> STOI score [0, 1]
    sinr_per_zone_pair: dict[str, float]           # "zoneA_vs_zoneB" -> SINR in dB
    wer_category_per_mic: dict[str, str]           # mic_id -> "excellent"|"acceptable"|"critical"
    ceiling_penalty_applied: dict[str, float] = field(default_factory=dict)  # mic_id -> penalty subtracted
    table_noise_coefficient: float = 0.0

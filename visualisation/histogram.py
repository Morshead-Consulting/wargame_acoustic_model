"""
Monte Carlo STOI distribution histogram.

Shows how the intelligibility estimate is distributed across all iterations,
highlights the worst-case tail (p5), and annotates the procurement thresholds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from config.defaults import STOI_ACCEPTABLE, STOI_EXCELLENT

if TYPE_CHECKING:
    from simulation.monte_carlo import MonteCarloResult


class MonteCarloHistogramRenderer:
    """
    Renders a STOI distribution histogram for one or all microphones.

    Parameters
    ----------
    bins:
        Number of histogram bins.
    fig_width, fig_height:
        Figure size in inches.
    """

    def __init__(
        self,
        bins: int = 25,
        fig_width: float = 10.0,
        fig_height: float = 5.0,
    ) -> None:
        self.bins = bins
        self.fig_width = fig_width
        self.fig_height = fig_height

    def render(
        self,
        mc_result: MonteCarloResult,
        mic_ids: list[str] | None = None,
        output_path: str | None = None,
    ) -> Figure:
        """
        Render one subplot per microphone (or filtered to mic_ids).

        Parameters
        ----------
        mc_result:
            Result from MonteCarloRunner.run().
        mic_ids:
            Subset of microphones to plot.  None = all.
        output_path:
            Save path (PNG, SVG, etc.).  None = display only.
        """
        all_ids = list(mc_result.stoi_mean.keys())
        ids_to_plot = mic_ids if mic_ids is not None else all_ids
        ids_to_plot = [m for m in ids_to_plot if m in mc_result.stoi_mean]

        if not ids_to_plot:
            fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
            ax.text(0.5, 0.5, "No STOI data available.", transform=ax.transAxes,
                    ha="center", va="center")
            return fig

        n = len(ids_to_plot)
        fig, axes = plt.subplots(
            1, n,
            figsize=(self.fig_width * n, self.fig_height),
            sharey=True,
            squeeze=False,
        )

        for col, mic_id in enumerate(ids_to_plot):
            ax = axes[0, col]
            scores = [
                it.stoi_per_mic.get(mic_id, float("nan"))
                for it in mc_result.iterations
                if not np.isnan(it.stoi_per_mic.get(mic_id, float("nan")))
            ]
            self._render_mic_histogram(ax, mic_id, scores, mc_result)

        fig.suptitle(
            f"Monte Carlo STOI Distribution — σ={mc_result.sigma:.2f} m, "
            f"N={mc_result.n_iterations} iterations",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------

    def _render_mic_histogram(
        self,
        ax: plt.Axes,
        mic_id: str,
        scores: list[float],
        mc_result: MonteCarloResult,
    ) -> None:
        if not scores:
            ax.set_visible(False)
            return

        arr = np.array(scores)
        p5 = float(np.percentile(arr, 5))
        mean = float(np.mean(arr))

        # Histogram bars, colour-coded by STOI zone
        n_bins = self.bins
        counts, edges = np.histogram(arr, bins=n_bins, range=(0.0, 1.0))

        for i, (lo, hi, count) in enumerate(zip(edges[:-1], edges[1:], counts)):
            mid = (lo + hi) / 2.0
            if mid < STOI_ACCEPTABLE:
                color = "#e74c3c"
            elif mid < STOI_EXCELLENT:
                color = "#f39c12"
            else:
                color = "#27ae60"
            ax.bar(lo, count, width=(hi - lo) * 0.95, align="edge",
                   color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

        # Shade the worst-case tail below p5
        ax.axvspan(0.0, p5, color="#e74c3c", alpha=0.18, label=f"Worst-case tail (p5={p5:.3f})")

        # Threshold lines
        ax.axvline(STOI_EXCELLENT, color="#27ae60", linestyle="--",
                   linewidth=1.5, label=f"Excellent ({STOI_EXCELLENT})")
        ax.axvline(STOI_ACCEPTABLE, color="#f39c12", linestyle=":",
                   linewidth=1.5, label=f"Acceptable ({STOI_ACCEPTABLE})")
        ax.axvline(mean, color="#2980b9", linestyle="-",
                   linewidth=1.8, label=f"Mean ({mean:.3f})")
        ax.axvline(p5, color="#c0392b", linestyle="-.",
                   linewidth=1.5, label=f"p5 ({p5:.3f})")

        # WER category annotation
        wer_category = (
            "excellent (pWER < 5%)" if mean > STOI_EXCELLENT
            else "acceptable" if mean > STOI_ACCEPTABLE
            else "CRITICAL"
        )
        ax.text(
            0.97, 0.95,
            f"Mean WER: {wer_category}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, fontweight="bold",
            color="#27ae60" if mean > STOI_EXCELLENT
            else "#f39c12" if mean > STOI_ACCEPTABLE
            else "#e74c3c",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                      edgecolor="#cccccc"),
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("STOI Score", fontsize=10)
        ax.set_ylabel("Iteration count", fontsize=10)
        ax.set_title(f"Microphone: {mic_id}", fontsize=10, fontweight="bold")
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.85)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)

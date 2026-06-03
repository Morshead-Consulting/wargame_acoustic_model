"""
Convergence curve chart and stability summary table.

Produces one subplot per microphone showing:
  - Rolling mean STOI (primary line)
  - Rolling p5 STOI (dashed secondary line — the worst-case tail estimate)
  - Shaded inter-run band derived from the StabilityReport
  - Vertical dashed marker at the detected convergence iteration
  - STOI threshold reference lines at 0.85 (excellent) and 0.70 (acceptable)

A stability summary table sits below the per-mic subplots, with a traffic-light
status column (green / amber / red) for each microphone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from simulation.monte_carlo import MonteCarloResult, StabilityReport

# STOI thresholds (match config.defaults when that module exists)
_STOI_EXCELLENT = 0.85
_STOI_ACCEPTABLE = 0.70

_TRAFFIC_COLORS = {
    "green": "#2ecc71",
    "amber": "#f39c12",
    "red":   "#e74c3c",
}
_TRAFFIC_LABELS = {
    "green": "Stable",
    "amber": "Caution",
    "red":   "Unstable",
}

_COLOR_MEAN = "#2980b9"
_COLOR_P5 = "#8e44ad"
_COLOR_CONV = "#e67e22"
_COLOR_BAND = "#2980b9"


class ConvergenceCurveRenderer:
    """
    Renders per-microphone convergence curves and an optional stability table.

    Parameters
    ----------
    fig_width:
        Overall figure width in inches.
    row_height:
        Height per microphone subplot row in inches.
    table_height:
        Height of the stability summary table row in inches.
    """

    def __init__(
        self,
        fig_width: float = 12.0,
        row_height: float = 3.2,
        table_height: float = 2.0,
    ) -> None:
        self.fig_width = fig_width
        self.row_height = row_height
        self.table_height = table_height

    def render(
        self,
        mc_result: MonteCarloResult,
        output_path: str | None = None,
    ) -> Figure:
        """
        Build and return the complete convergence figure.

        Parameters
        ----------
        mc_result:
            The result from MonteCarloRunner.run().
        output_path:
            If provided, save the figure to this path (PNG, SVG, etc.).
        """
        mic_ids = list(mc_result.convergence.rolling_mean_per_mic.keys())
        if not mic_ids:
            fig, ax = plt.subplots(figsize=(self.fig_width, 3))
            ax.text(0.5, 0.5, "No microphone data available.",
                    transform=ax.transAxes, ha="center", va="center")
            return fig

        has_stability = mc_result.stability is not None
        n_mic_rows = len(mic_ids)
        n_rows = n_mic_rows + (1 if has_stability else 0)

        heights = [self.row_height] * n_mic_rows
        if has_stability:
            heights.append(self.table_height)

        total_height = sum(heights) + 0.8  # top margin for suptitle
        fig = plt.figure(figsize=(self.fig_width, total_height))
        gs = gridspec.GridSpec(
            n_rows, 1, figure=fig,
            height_ratios=heights,
            hspace=0.55,
            top=0.93, bottom=0.05,
        )

        for row, mic_id in enumerate(mic_ids):
            ax = fig.add_subplot(gs[row])
            self._render_mic_curve(ax, mic_id, mc_result)

        if has_stability:
            ax_table = fig.add_subplot(gs[n_mic_rows])
            self._render_stability_table(ax_table, mc_result.stability, mic_ids)  # type: ignore[arg-type]

        status = " ✓ Converged" if mc_result.convergence.converged else " ⚠ Not fully converged"
        fig.suptitle(
            f"Monte Carlo Convergence — σ={mc_result.sigma:.2f} m, "
            f"N={mc_result.n_iterations} iterations{status}",
            fontsize=12,
            fontweight="bold",
        )

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------
    # Per-microphone subplot

    def _render_mic_curve(
        self,
        ax: plt.Axes,
        mic_id: str,
        mc_result: MonteCarloResult,
    ) -> None:
        mean_hist = mc_result.convergence.rolling_mean_per_mic.get(mic_id, [])
        p5_hist = mc_result.convergence.rolling_p5_per_mic.get(mic_id, [])
        conv_iter = mc_result.convergence.convergence_per_mic.get(mic_id)

        if not mean_hist:
            ax.set_visible(False)
            return

        x = np.arange(1, len(mean_hist) + 1)

        # --- Core curves ---
        ax.plot(x, mean_hist, color=_COLOR_MEAN, linewidth=1.8,
                label="Rolling mean STOI")
        ax.plot(x, p5_hist, color=_COLOR_P5, linewidth=1.5, linestyle="--",
                label="Rolling p5 STOI (worst-case)")

        # --- Inter-run stability band ---
        self._draw_stability_band(ax, mic_id, x, mc_result)

        # --- STOI threshold reference lines ---
        ax.axhline(_STOI_EXCELLENT, color="#27ae60", linestyle=":",
                   linewidth=1.2, alpha=0.85)
        ax.axhline(_STOI_ACCEPTABLE, color="#c0392b", linestyle=":",
                   linewidth=1.2, alpha=0.85)
        ax.text(x[-1] * 0.995, _STOI_EXCELLENT + 0.007, "Excellent (0.85)",
                ha="right", fontsize=7, color="#27ae60")
        ax.text(x[-1] * 0.995, _STOI_ACCEPTABLE + 0.007, "Acceptable (0.70)",
                ha="right", fontsize=7, color="#c0392b")

        # --- Convergence marker ---
        if conv_iter is not None and conv_iter <= len(mean_hist):
            conv_y = mean_hist[conv_iter - 1]
            ax.axvline(conv_iter, color=_COLOR_CONV, linestyle="-.",
                       linewidth=1.5, alpha=0.9)
            ax.annotate(
                f"Converged\nN={conv_iter}",
                xy=(conv_iter, conv_y),
                xytext=(
                    min(conv_iter + max(len(x) * 0.06, 5), len(x) * 0.85),
                    conv_y + 0.04,
                ),
                fontsize=7.5,
                color=_COLOR_CONV,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=_COLOR_CONV, lw=1.2),
            )
        else:
            ax.text(
                0.985, 0.06,
                "⚠ Not converged",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=8, color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#fdecea",
                          edgecolor="#c0392b", alpha=0.9),
            )

        ax.set_xlim(1, len(x))
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel("STOI", fontsize=9)
        ax.set_title(f"Microphone: {mic_id}", fontsize=10, fontweight="bold", loc="left")
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.85)
        ax.grid(True, alpha=0.25, linewidth=0.7)

    def _draw_stability_band(
        self,
        ax: plt.Axes,
        mic_id: str,
        x: np.ndarray,
        mc_result: MonteCarloResult,
    ) -> None:
        """
        Shade the inter-run spread as a horizontal band using per_run_stoi_mean
        from the StabilityReport.  The band shows min-to-max across seeds,
        giving a visual sense of how much the final estimate varies by seed.
        """
        if mc_result.stability is None:
            return

        per_run_means = mc_result.stability.per_run_stoi_mean.get(mic_id, [])
        if len(per_run_means) < 2:
            return

        band_lo = min(per_run_means)
        band_hi = max(per_run_means)
        ax.axhspan(
            band_lo, band_hi,
            color=_COLOR_BAND, alpha=0.10,
            label=f"Inter-run range ({len(per_run_means)} seeds)",
        )
        # Tick marks at the band edges
        ax.axhline(band_lo, color=_COLOR_BAND, linewidth=0.7, linestyle="-", alpha=0.35)
        ax.axhline(band_hi, color=_COLOR_BAND, linewidth=0.7, linestyle="-", alpha=0.35)

    # ------------------------------------------------------------------
    # Stability summary table

    def _render_stability_table(
        self,
        ax: plt.Axes,
        stability: StabilityReport,
        mic_ids: list[str],
    ) -> None:
        ax.axis("off")
        ax.set_title("Stability Summary  (S = CV = σ/μ across seeds; green < 0.02, red ≥ 0.05)",
                     fontsize=9, fontweight="bold", loc="left", pad=4)

        headers = [
            "Microphone",
            "S — mean STOI",
            "S — p5 STOI",
            "S — capture prob.",
            "Status",
            f"Runs (N={stability.n_runs})",
        ]

        rows = []
        traffic_per_row: list[str] = []

        for mic_id in mic_ids:
            scores = stability.stability_scores.get(mic_id, {})
            traffic = stability.traffic_light(mic_id)
            traffic_per_row.append(traffic)
            rows.append([
                mic_id,
                f"{scores.get('stoi_mean', 0.0):.4f}",
                f"{scores.get('stoi_p5', 0.0):.4f}",
                f"{scores.get('capture_prob', 0.0):.4f}",
                _TRAFFIC_LABELS[traffic],
                str(stability.n_runs),
            ])

        if not rows:
            ax.text(0.5, 0.5, "No stability data.", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
            return

        table = ax.table(
            cellText=rows,
            colLabels=headers,
            cellLoc="center",
            loc="upper center",
            bbox=[0.0, 0.0, 1.0, 0.92],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)

        # Style header row
        n_cols = len(headers)
        for col in range(n_cols):
            header_cell = table[0, col]
            header_cell.set_facecolor("#2c3e50")
            header_cell.set_text_props(color="white", fontweight="bold")

        # Colour the Status column and set text
        STATUS_COL = 4
        for row_idx, traffic in enumerate(traffic_per_row):
            cell = table[row_idx + 1, STATUS_COL]
            cell.set_facecolor(_TRAFFIC_COLORS[traffic])
            cell.set_text_props(color="white", fontweight="bold")

        # Alternate row shading for readability
        for row_idx in range(len(rows)):
            if row_idx % 2 == 0:
                for col in range(n_cols):
                    if col != STATUS_COL:
                        table[row_idx + 1, col].set_facecolor("#f5f6fa")

        # Append warning notes below if any
        if stability.warnings:
            warning_text = "Warnings: " + "; ".join(stability.warnings)
            ax.text(
                0.0, -0.04,
                warning_text,
                transform=ax.transAxes,
                fontsize=7.5, color="#c0392b",
                va="top",
            )

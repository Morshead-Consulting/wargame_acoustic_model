"""
Side-by-side what-if scenario comparison dashboard.

Renders a 2×3 subplot grid (one row per scenario):
  col 0 — spatial heatmap
  col 1 — Monte Carlo STOI histogram (all mics overlaid)
  col 2 — channel count bar chart

Plus a delta summary table beneath.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from config.defaults import STOI_ACCEPTABLE, STOI_EXCELLENT
from models.microphone import MicrophoneProfile
from models.participant import Participant
from models.room import RoomGeometry
from simulation.metrics import ChannelCountSummary
from visualisation.heatmap import SpatialHeatmapRenderer
from visualisation.histogram import MonteCarloHistogramRenderer

if TYPE_CHECKING:
    from simulation.monte_carlo import MonteCarloResult


@dataclass
class ScenarioResult:
    """Complete processed results for one deployment scenario."""
    label: str
    room: RoomGeometry
    microphones: list[MicrophoneProfile]
    participants: list[Participant]
    mc_result: MonteCarloResult
    channel_summary: ChannelCountSummary
    coverage_percent: float   # % of room floor above STOI_ACCEPTABLE threshold


class ScenarioComparisonDashboard:
    """
    Renders a side-by-side comparison of two deployment scenarios.

    Parameters
    ----------
    fig_width:
        Total figure width in inches.
    row_height:
        Height of each scenario row.
    table_height:
        Height of the delta summary table row.
    """

    def __init__(
        self,
        fig_width: float = 18.0,
        row_height: float = 5.0,
        table_height: float = 2.5,
    ) -> None:
        self.fig_width = fig_width
        self.row_height = row_height
        self.table_height = table_height
        self._heatmap = SpatialHeatmapRenderer(grid_resolution_m=0.25)
        self._histogram = MonteCarloHistogramRenderer(bins=20)

    def render(
        self,
        scenario_a: ScenarioResult,
        scenario_b: ScenarioResult,
        output_path: str | None = None,
    ) -> Figure:
        n_rows = 3   # row A, row B, summary table
        heights = [self.row_height, self.row_height, self.table_height]
        total_height = sum(heights) + 0.8

        fig = plt.figure(figsize=(self.fig_width, total_height))
        gs = gridspec.GridSpec(
            n_rows, 3, figure=fig,
            height_ratios=heights,
            hspace=0.45, wspace=0.35,
            top=0.93, bottom=0.04,
        )

        for row_idx, scenario in enumerate([scenario_a, scenario_b]):
            self._render_heatmap_cell(fig, gs[row_idx, 0], scenario)
            self._render_histogram_cell(fig, gs[row_idx, 1], scenario)
            self._render_channel_bar_cell(fig, gs[row_idx, 2], scenario)

        ax_table = fig.add_subplot(gs[2, :])
        self._render_delta_table(ax_table, scenario_a, scenario_b)

        fig.suptitle(
            f"Scenario Comparison: {scenario_a.label}  vs.  {scenario_b.label}",
            fontsize=13, fontweight="bold",
        )

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------
    # Cell renderers

    def _render_heatmap_cell(
        self,
        fig: Figure,
        gs_cell: gridspec.SubplotSpec,
        scenario: ScenarioResult,
    ) -> None:
        ax = fig.add_subplot(gs_cell)
        from visualisation.heatmap import _STOI_CMAP

        room = scenario.room
        res = 0.3
        xs = np.arange(0.0, room.length + res, res)
        ys = np.arange(0.0, room.width + res, res)
        score_grid = self._heatmap._evaluate_grid(xs, ys, scenario.microphones)

        ax.imshow(
            score_grid, origin="lower",
            extent=[0, room.length, 0, room.width],
            cmap=_STOI_CMAP, vmin=0.0, vmax=1.0,
            aspect="equal", interpolation="bilinear",
        )
        ax.set_title(f"{scenario.label}\nHeatmap", fontsize=9, fontweight="bold")
        ax.set_xlabel("Length (m)", fontsize=8)
        ax.set_ylabel("Width (m)", fontsize=8)
        ax.tick_params(labelsize=7)

        for p in scenario.participants:
            ax.plot(p.x, p.y, "w*", markersize=7, markeredgecolor="black", markeredgewidth=0.4)

    def _render_histogram_cell(
        self,
        fig: Figure,
        gs_cell: gridspec.SubplotSpec,
        scenario: ScenarioResult,
    ) -> None:
        ax = fig.add_subplot(gs_cell)
        mc = scenario.mc_result

        colors = ["#2980b9", "#8e44ad", "#e67e22", "#27ae60", "#e74c3c"]
        mic_ids = list(mc.stoi_mean.keys())

        for color_idx, mic_id in enumerate(mic_ids):
            scores = [
                it.stoi_per_mic.get(mic_id, float("nan"))
                for it in mc.iterations
                if not np.isnan(it.stoi_per_mic.get(mic_id, float("nan")))
            ]
            if not scores:
                continue
            color = colors[color_idx % len(colors)]
            ax.hist(scores, bins=20, range=(0, 1), alpha=0.55,
                    color=color, label=mic_id, edgecolor="white", linewidth=0.4)

        ax.axvline(STOI_EXCELLENT, color="#27ae60", linestyle="--", linewidth=1.2)
        ax.axvline(STOI_ACCEPTABLE, color="#f39c12", linestyle=":", linewidth=1.2)
        ax.set_xlim(0, 1)
        ax.set_xlabel("STOI", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_title(f"{scenario.label}\nSTOI Distribution", fontsize=9, fontweight="bold")
        ax.legend(fontsize=6.5, framealpha=0.8)
        ax.tick_params(labelsize=7)

    def _render_channel_bar_cell(
        self,
        fig: Figure,
        gs_cell: gridspec.SubplotSpec,
        scenario: ScenarioResult,
    ) -> None:
        ax = fig.add_subplot(gs_cell)
        cs = scenario.channel_summary

        categories = ["MX412\n(gooseneck)", "MXA310\n(tabletop)", "MXA920\n(ceiling)"]
        values = [cs.gooseneck_channels, cs.tabletop_channels, cs.ceiling_channels]
        colors = ["#e67e22", "#2980b9", "#8e44ad"]

        bars = ax.bar(categories, values, color=colors, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    str(val),
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

        ax.set_ylabel("ALSA streams", fontsize=8)
        ax.set_title(
            f"{scenario.label}\nChannel Count: {cs.total_alsa_streams} total",
            fontsize=9, fontweight="bold",
        )
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, max(values) * 1.25 + 1)
        ax.grid(True, axis="y", alpha=0.25)

    # ------------------------------------------------------------------
    # Delta summary table

    def _render_delta_table(
        self,
        ax: plt.Axes,
        scenario_a: ScenarioResult,
        scenario_b: ScenarioResult,
    ) -> None:
        ax.axis("off")
        ax.set_title(
            "Procurement Comparison Summary",
            fontsize=10, fontweight="bold", loc="left", pad=4,
        )

        def mean_stoi(mc: MonteCarloResult) -> float:
            vals = list(mc.stoi_mean.values())
            return float(np.mean(vals)) if vals else float("nan")

        def worst_p5(mc: MonteCarloResult) -> float:
            vals = list(mc.stoi_p5.values())
            return float(np.min(vals)) if vals else float("nan")

        def wer_label(stoi: float) -> str:
            if np.isnan(stoi):
                return "—"
            if stoi > STOI_EXCELLENT:
                return "< 5% (Excellent)"
            if stoi > STOI_ACCEPTABLE:
                return "Acceptable"
            return "CRITICAL"

        headers = ["Metric", scenario_a.label, scenario_b.label, "Delta (A→B)"]

        a_mean = mean_stoi(scenario_a.mc_result)
        b_mean = mean_stoi(scenario_b.mc_result)
        a_p5 = worst_p5(scenario_a.mc_result)
        b_p5 = worst_p5(scenario_b.mc_result)

        rows = [
            [
                "Total ALSA streams",
                str(scenario_a.channel_summary.total_alsa_streams),
                str(scenario_b.channel_summary.total_alsa_streams),
                f"{scenario_b.channel_summary.total_alsa_streams - scenario_a.channel_summary.total_alsa_streams:+d}",
            ],
            [
                "Floor coverage (≥ acceptable)",
                f"{scenario_a.coverage_percent:.1f}%",
                f"{scenario_b.coverage_percent:.1f}%",
                f"{scenario_b.coverage_percent - scenario_a.coverage_percent:+.1f}%",
            ],
            [
                "Mean STOI (all mics)",
                f"{a_mean:.3f}" if not np.isnan(a_mean) else "—",
                f"{b_mean:.3f}" if not np.isnan(b_mean) else "—",
                f"{b_mean - a_mean:+.3f}" if not (np.isnan(a_mean) or np.isnan(b_mean)) else "—",
            ],
            [
                "Worst-case p5 STOI",
                f"{a_p5:.3f}" if not np.isnan(a_p5) else "—",
                f"{b_p5:.3f}" if not np.isnan(b_p5) else "—",
                f"{b_p5 - a_p5:+.3f}" if not (np.isnan(a_p5) or np.isnan(b_p5)) else "—",
            ],
            [
                "Projected WER (mean)",
                wer_label(a_mean),
                wer_label(b_mean),
                "—",
            ],
        ]

        table = ax.table(
            cellText=rows,
            colLabels=headers,
            cellLoc="center",
            loc="upper center",
            bbox=[0.0, 0.0, 1.0, 0.92],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)

        n_cols = len(headers)
        for col in range(n_cols):
            table[0, col].set_facecolor("#2c3e50")
            table[0, col].set_text_props(color="white", fontweight="bold")

        for row_idx in range(len(rows)):
            if row_idx % 2 == 0:
                for col in range(n_cols):
                    table[row_idx + 1, col].set_facecolor("#f5f6fa")

        # Colour the delta column: green = improvement, red = regression
        DELTA_COL = 3
        for row_idx, row in enumerate(rows):
            delta_str = row[DELTA_COL]
            if delta_str.startswith("+") and delta_str != "—":
                table[row_idx + 1, DELTA_COL].set_facecolor("#d5f5e3")
            elif delta_str.startswith("-") and delta_str != "—":
                table[row_idx + 1, DELTA_COL].set_facecolor("#fadbd8")

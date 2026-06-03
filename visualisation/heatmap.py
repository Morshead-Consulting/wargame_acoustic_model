"""
2-D spatial acoustic quality heatmap.

Evaluates an analytical coverage score at every grid point in the room floor
plan and renders it with a traffic-light colormap (red / yellow / green keyed to
the STOI_ACCEPTABLE and STOI_EXCELLENT thresholds).

The score at each point is:
  max over all mics of (directivity_gain × inverse_square_attenuation)
normalised to [0, 1] and then linearly mapped onto the STOI range so the
colorbar and threshold lines are directly interpretable.

This is an analytical proxy, not a full PRA-based STOI.  It is designed for fast
interactive exploration before committing to the full Monte Carlo run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from config.defaults import STOI_ACCEPTABLE, STOI_EXCELLENT, Z_SEATED
from models.microphone import MicrophoneProfile, MXA310Tabletop, MXA920CeilingArray
from simulation.directivity import angle_between, cardioid_gain, inverse_square_attenuation

if TYPE_CHECKING:
    from models.participant import Participant
    from models.room import RoomGeometry


# Traffic-light colormap: red → yellow → green with fixed threshold breakpoints
_STOI_CMAP = LinearSegmentedColormap.from_list(
    "stoi_traffic",
    [
        (0.0,  "#c0392b"),   # critical  — red
        (STOI_ACCEPTABLE, "#f39c12"),  # acceptable — amber
        (STOI_EXCELLENT,  "#27ae60"),  # excellent  — green
        (1.0,  "#1a5c34"),
    ],
)


class SpatialHeatmapRenderer:
    """
    Renders a top-down acoustic quality heatmap for a wargame room layout.

    Parameters
    ----------
    grid_resolution_m:
        Spacing between grid sample points in metres (default 0.25 m).
    source_z:
        Height of the virtual source at each grid point (default Z_SEATED).
    fig_width, fig_height:
        Figure dimensions in inches.
    """

    def __init__(
        self,
        grid_resolution_m: float = 0.25,
        source_z: float = Z_SEATED,
        fig_width: float = 10.0,
        fig_height: float = 8.0,
    ) -> None:
        self.grid_resolution_m = grid_resolution_m
        self.source_z = source_z
        self.fig_width = fig_width
        self.fig_height = fig_height

    def render(
        self,
        room: RoomGeometry,
        microphones: list[MicrophoneProfile],
        participants: list[Participant],
        output_path: str | None = None,
    ) -> Figure:
        xs = np.arange(0.0, room.length + self.grid_resolution_m, self.grid_resolution_m)
        ys = np.arange(0.0, room.width + self.grid_resolution_m, self.grid_resolution_m)
        score_grid = self._evaluate_grid(xs, ys, microphones)

        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        im = ax.imshow(
            score_grid,
            origin="lower",
            extent=[0, room.length, 0, room.width],
            cmap=_STOI_CMAP,
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
            interpolation="bilinear",
        )

        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("Estimated Coverage Score (proxy STOI)", fontsize=9)
        cb.ax.axhline(STOI_EXCELLENT, color="white", linewidth=1.5, linestyle="--")
        cb.ax.axhline(STOI_ACCEPTABLE, color="white", linewidth=1.5, linestyle=":")

        # Threshold contour lines on the heatmap
        ax.contour(
            xs, ys, score_grid,
            levels=[STOI_ACCEPTABLE, STOI_EXCELLENT],
            colors=["#f39c12", "#27ae60"],
            linewidths=1.2,
            linestyles=[":", "--"],
        )

        self._draw_mic_overlays(ax, microphones)
        self._draw_participants(ax, participants)

        ax.set_xlim(0, room.length)
        ax.set_ylim(0, room.width)
        ax.set_xlabel("Room length (m)", fontsize=10)
        ax.set_ylabel("Room width (m)", fontsize=10)
        ax.set_title(
            f"{room.name}  —  Acoustic Coverage Heatmap\n"
            f"(α={room.absorption:.2f}, RT60≈{room.sabine_rt60():.2f}s)",
            fontsize=11, fontweight="bold",
        )

        legend_patches = [
            mpatches.Patch(color="#c0392b", label=f"Critical STOI < {STOI_ACCEPTABLE}"),
            mpatches.Patch(color="#f39c12", label=f"Acceptable {STOI_ACCEPTABLE}–{STOI_EXCELLENT}"),
            mpatches.Patch(color="#27ae60", label=f"Excellent STOI > {STOI_EXCELLENT}"),
        ]
        ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.85)

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------

    def _evaluate_grid(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        microphones: list[MicrophoneProfile],
    ) -> np.ndarray:
        """
        Build (ny, nx) array of coverage scores.
        Score = max over mics of (best-channel directivity gain × inverse-square att.)
        Normalised to [0, 1] then linearly mapped to STOI range.
        """
        ny, nx = len(ys), len(xs)
        raw = np.zeros((ny, nx), dtype=float)

        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                source = np.array([x, y, self.source_z])
                best = 0.0
                for mic in microphones:
                    for ch_idx in range(mic.channel_count):
                        g = mic.gain(source, channel=ch_idx)
                        d = float(np.linalg.norm(source - mic.position))
                        att = inverse_square_attenuation(d, ref_distance_m=mic.effective_radius)
                        score = g * min(att, 1.0)
                        if score > best:
                            best = score
                raw[yi, xi] = best

        # Normalise raw scores → [0, 1] then map to plausible STOI range
        peak = np.max(raw)
        if peak > 1e-9:
            raw = raw / peak
        # Map [0, 1] → [0.4, 0.99] so dead zones appear red and peak zones appear green
        return 0.4 + raw * 0.59

    def _draw_mic_overlays(self, ax: plt.Axes, microphones: list[MicrophoneProfile]) -> None:
        for mic in microphones:
            cx, cy = mic.position[0], mic.position[1]

            if isinstance(mic, MXA310Tabletop):
                # Draw 4 directional wedge sectors at 90° increments
                for deg in [0, 90, 180, 270]:
                    wedge = mpatches.Wedge(
                        (cx, cy),
                        r=mic.effective_radius,
                        theta1=deg - 45, theta2=deg + 45,
                        alpha=0.18, color="#2980b9",
                    )
                    ax.add_patch(wedge)
                ax.plot(cx, cy, "bs", markersize=8, label="MXA310" if "MXA310" not in ax.get_legend_handles_labels()[1] else "")

            elif isinstance(mic, MXA920CeilingArray):
                # Draw floor footprint circle for each beam
                for ch, target in enumerate(mic.beam_targets):
                    beam_radius = mic.effective_radius
                    circle = plt.Circle(
                        (target[0], target[1]),
                        beam_radius, fill=False,
                        edgecolor="#8e44ad", linewidth=1.2, linestyle="--", alpha=0.7,
                    )
                    ax.add_patch(circle)
                ax.plot(cx, cy, "m^", markersize=9, label="MXA920" if "MXA920" not in ax.get_legend_handles_labels()[1] else "")

            else:   # MX412 gooseneck
                circle = plt.Circle(
                    (cx, cy),
                    mic.effective_radius, fill=False,
                    edgecolor="#e67e22", linewidth=1.2, alpha=0.7,
                )
                ax.add_patch(circle)
                ax.plot(cx, cy, "ro", markersize=7, label="MX412" if "MX412" not in ax.get_legend_handles_labels()[1] else "")

            ax.text(cx, cy + 0.15, mic.mic_id, fontsize=6.5, ha="center", color="white",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="#2c3e50", alpha=0.7))

    def _draw_participants(self, ax: plt.Axes, participants: list[Participant]) -> None:
        zone_colors: dict[str, str] = {}
        palette = ["#f1c40f", "#1abc9c", "#e74c3c", "#3498db", "#9b59b6", "#e67e22"]
        for p in participants:
            if p.zone not in zone_colors:
                zone_colors[p.zone] = palette[len(zone_colors) % len(palette)]
            ax.plot(
                p.x, p.y, "*",
                color=zone_colors[p.zone], markersize=11,
                markeredgecolor="black", markeredgewidth=0.5,
            )
            ax.text(p.x + 0.12, p.y + 0.12, p.participant_id, fontsize=6, color="white",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor=zone_colors[p.zone], alpha=0.8))

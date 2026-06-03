"""
Deployment Planner CLI.

Commands:
  simulate   Run Monte Carlo simulation for one layout config.
  compare    Run two configs side-by-side and render the comparison dashboard.

Example:
  python main.py simulate configs/example_wargame.json --sigma 0.3 --iterations 100
  python main.py compare configs/layout_a.json configs/layout_b.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from config.defaults import MC_DEFAULT_SIGMA, SAMPLE_RATE, SINR_TARGET_DB
from models.microphone import MX412Gooseneck, MXA310Tabletop, MXA920CeilingArray
from models.participant import Participant
from models.room import RoomGeometry
from simulation.engine import SimulationEngine
from simulation.metrics import MetricsCalculator
from simulation.monte_carlo import ConvergenceCriteria, IterationCallbackData, MonteCarloRunner
from visualisation.comparison import ScenarioComparisonDashboard, ScenarioResult
from visualisation.convergence import ConvergenceCurveRenderer
from visualisation.heatmap import SpatialHeatmapRenderer
from visualisation.histogram import MonteCarloHistogramRenderer

app = typer.Typer(help="Acoustic deployment planner for defence wargaming rooms.")
console = Console()


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def _load_room(cfg: dict) -> RoomGeometry:
    r = cfg["room"]
    return RoomGeometry(
        length=r["length"],
        width=r["width"],
        height=r["height"],
        absorption=r.get("absorption", 0.25),
        name=r.get("name", "wargame_room"),
        rt60_formula=r.get("rt60_formula", "sabine"),
    )


def _load_participants(cfg: dict) -> list[Participant]:
    participants = []
    for p in cfg.get("participants", []):
        participants.append(Participant(
            participant_id=p["id"],
            x=p["x"],
            y=p["y"],
            z=p.get("z", 1.2),
            zone=p.get("zone", "default"),
            posture=p.get("posture", "seated"),
        ))
    return participants


def _load_microphones(cfg: dict) -> list:
    mics = []
    for m in cfg.get("microphones", []):
        pos = np.array([m["x"], m["y"], m.get("z", 0.0)])
        mic_type = m["type"]

        if mic_type == "MX412":
            aim = m.get("aim_vector")
            mics.append(MX412Gooseneck(
                mic_id=m["id"],
                position=pos,
                aim_vector=np.array(aim) if aim else None,
            ))

        elif mic_type == "MXA310":
            mics.append(MXA310Tabletop(
                mic_id=m["id"],
                position=pos,
                noise_coefficient=m.get("noise_coefficient", 0.0),
            ))

        elif mic_type in ("MXA920", "MXA910"):
            targets = [np.array(t) for t in m["beam_targets"]]
            mics.append(MXA920CeilingArray(
                mic_id=m["id"],
                position=pos,
                beam_targets=targets,
                beam_half_angle_deg=m.get("beam_half_angle_deg", 7.5),
            ))

        else:
            console.print(f"[yellow]Unknown mic type '{mic_type}' — skipping.[/yellow]")

    return mics


def _load_mic_zone_map(cfg: dict) -> dict[str, str]:
    return {m["id"]: m["zone"] for m in cfg.get("microphones", []) if "zone" in m}


def _build_engine_and_calculator(
    cfg: dict,
    sigma: float,
    noise_coefficient: float | None,
) -> tuple[SimulationEngine, MetricsCalculator, RoomGeometry, list, list]:
    room = _load_room(cfg)
    participants = _load_participants(cfg)
    microphones = _load_microphones(cfg)

    if noise_coefficient is not None:
        for mic in microphones:
            if isinstance(mic, MXA310Tabletop):
                mic.noise_coefficient = noise_coefficient

    engine = SimulationEngine(room=room, participants=participants, microphones=microphones)
    calculator = MetricsCalculator(
        clean_signal=engine._clean_signal,
        microphones=microphones,
        room=room,
        participants=participants,
        fs=SAMPLE_RATE,
        mic_zone_map=_load_mic_zone_map(cfg),
    )
    return engine, calculator, room, participants, microphones


def _compute_coverage_percent(mc_result, microphones, room) -> float:
    """Estimate % of floor area above STOI_ACCEPTABLE using heatmap grid."""
    from config.defaults import STOI_ACCEPTABLE
    from visualisation.heatmap import SpatialHeatmapRenderer
    renderer = SpatialHeatmapRenderer(grid_resolution_m=0.5)
    xs = np.arange(0.0, room.length + 0.5, 0.5)
    ys = np.arange(0.0, room.width + 0.5, 0.5)
    grid = renderer._evaluate_grid(xs, ys, microphones)
    return float(100.0 * np.mean(grid >= STOI_ACCEPTABLE))


# ---------------------------------------------------------------------------
# Rich live progress display
# ---------------------------------------------------------------------------

def _run_with_progress(runner: MonteCarloRunner, label: str):
    """Run runner.run() inside a Rich live display — progress bar + per-mic status table."""
    mic_ids = [m.mic_id for m in runner.engine.microphones]
    n_max = runner.criteria.n_max

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    main_task = progress.add_task(label, total=n_max)
    latest: list[IterationCallbackData | None] = [None]

    def _make_status_table() -> Table:
        tbl = Table(box=None, padding=(0, 2), show_header=True, header_style="bold")
        tbl.add_column("Microphone", style="cyan", no_wrap=True, min_width=18)
        tbl.add_column("Mean STOI", justify="right", min_width=10)
        tbl.add_column("p5 STOI", justify="right", min_width=9)
        tbl.add_column("Status", min_width=22)
        data = latest[0]
        for mic_id in mic_ids:
            if data is None:
                tbl.add_row(mic_id, "—", "—", "[dim]waiting…[/dim]")
            else:
                m_val = data.rolling_mean_per_mic.get(mic_id, float("nan"))
                p_val = data.rolling_p5_per_mic.get(mic_id, float("nan"))
                mean_str = f"{m_val:.3f}" if not np.isnan(m_val) else "—"
                p5_str = f"{p_val:.3f}" if not np.isnan(p_val) else "—"
                if data.converged_per_mic.get(mic_id):
                    conv_iter = data.convergence_iter_per_mic.get(mic_id)
                    status = f"[green]converged @{conv_iter}[/green]"
                else:
                    status = "[yellow]running…[/yellow]"
                tbl.add_row(mic_id, mean_str, p5_str, status)
        return tbl

    with Live(Group(progress, _make_status_table()), console=console, refresh_per_second=10) as live:
        def on_iteration(data: IterationCallbackData) -> None:
            latest[0] = data
            progress.update(main_task, completed=data.n_done)
            live.update(Group(progress, _make_status_table()))

        def on_stability_run(run_idx: int, n_runs: int) -> None:
            progress.update(
                main_task,
                description=f"Stability run {run_idx}/{n_runs}",
                completed=0,
            )
            live.update(Group(progress, _make_status_table()))

        result = runner.run(on_iteration=on_iteration, on_stability_run=on_stability_run)

    return result


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def simulate(
    config_file: Annotated[Path, typer.Argument(help="Path to layout config JSON.")],
    sigma: Annotated[float, typer.Option(help="Movement variance σ (metres).")] = MC_DEFAULT_SIGMA,
    n_min: Annotated[int, typer.Option(help="Minimum iterations before early stopping.")] = 50,
    n_max: Annotated[int, typer.Option(help="Maximum iterations.")] = 200,
    n_stability: Annotated[int, typer.Option(help="Seeds for stability assessment (1=skip).")] = 3,
    noise_coeff: Annotated[float | None, typer.Option(help="Override MXA310 noise coefficient.")] = None,
    output_dir: Annotated[Path, typer.Option(help="Directory for PNG outputs.")] = Path("./outputs"),
    seed: Annotated[int | None, typer.Option(help="Random seed for reproducibility.")] = 42,
) -> None:
    """Run Monte Carlo simulation for a single deployment layout."""

    cfg = json.loads(config_file.read_text())
    engine, calculator, room, participants, microphones = _build_engine_and_calculator(
        cfg, sigma, noise_coeff
    )

    criteria = ConvergenceCriteria(
        n_min=n_min,
        n_max=n_max,
        n_stability_runs=n_stability,
    )
    runner = MonteCarloRunner(
        engine=engine,
        metrics_calculator=calculator,
        sigma=sigma,
        seed=seed,
        criteria=criteria,
        run_stability=(n_stability > 1),
    )

    console.print(f"σ={sigma} m, N_min={n_min}, N_max={n_max}")
    mc_result = _run_with_progress(runner, "Monte Carlo simulation")
    console.print(f"[green]{mc_result.convergence.summary()}[/green]")

    # ---- Console report ----
    channel_summary = calculator.compute_channel_count()
    console.print(f"\n[bold]Channel count:[/bold] {channel_summary.description}")

    t = Table("Microphone", "Mean STOI", "p5 STOI", "WER Category", "SINR (dB)",
              title="Results per Microphone")
    for mic_id in mc_result.stoi_mean:
        sinr_val = mc_result.sinr_mean.get(mic_id, float("nan"))
        sinr_str = f"{sinr_val:.1f}" if not np.isnan(sinr_val) else "—"
        sinr_pass = "" if np.isnan(sinr_val) else (" ✓" if sinr_val >= SINR_TARGET_DB else " ⚠")
        wer = next(
            (it.wer_category_per_mic.get(mic_id, "—") for it in mc_result.iterations[:1]),
            "—",
        )
        t.add_row(
            mic_id,
            f"{mc_result.stoi_mean[mic_id]:.3f}",
            f"{mc_result.stoi_p5[mic_id]:.3f}",
            wer,
            sinr_str + sinr_pass,
        )
    console.print(t)

    if mc_result.stability:
        unstable = [mid for mid, ok in mc_result.stability.stable.items() if not ok]
        if unstable:
            console.print(f"[yellow]Stability warnings: {', '.join(unstable)}[/yellow]")

    # ---- Save visualisations ----
    output_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = output_dir / "heatmap.png"
    SpatialHeatmapRenderer().render(
        room, microphones, participants,
        output_path=str(heatmap_path),
    )
    console.print(f"Saved heatmap → {heatmap_path}")

    hist_path = output_dir / "histogram.png"
    MonteCarloHistogramRenderer().render(mc_result, output_path=str(hist_path))
    console.print(f"Saved histogram → {hist_path}")

    conv_path = output_dir / "convergence.png"
    ConvergenceCurveRenderer().render(mc_result, output_path=str(conv_path))
    console.print(f"Saved convergence chart → {conv_path}")

    console.print(f"\n[bold green]Done.[/bold green] Outputs in {output_dir}/")


@app.command()
def compare(
    config_a: Annotated[Path, typer.Argument(help="Layout A config JSON.")],
    config_b: Annotated[Path, typer.Argument(help="Layout B config JSON.")],
    sigma: Annotated[float, typer.Option()] = MC_DEFAULT_SIGMA,
    n_max: Annotated[int, typer.Option()] = 100,
    output_dir: Annotated[Path, typer.Option()] = Path("./outputs"),
    seed: Annotated[int | None, typer.Option()] = 42,
) -> None:
    """Compare two deployment layouts side-by-side."""

    def _run_scenario(cfg_path: Path, label: str) -> ScenarioResult:
        cfg = json.loads(cfg_path.read_text())
        engine, calculator, room, participants, microphones = _build_engine_and_calculator(
            cfg, sigma, None
        )
        criteria = ConvergenceCriteria(n_min=50, n_max=n_max, n_stability_runs=1)
        runner = MonteCarloRunner(
            engine=engine,
            metrics_calculator=calculator,
            sigma=sigma,
            seed=seed,
            criteria=criteria,
            run_stability=False,
        )
        mc = _run_with_progress(runner, label)
        console.print(f"  {mc.convergence.summary()}")
        ch = calculator.compute_channel_count()
        cov = _compute_coverage_percent(mc, microphones, room)
        return ScenarioResult(
            label=label,
            room=room,
            microphones=microphones,
            participants=participants,
            mc_result=mc,
            channel_summary=ch,
            coverage_percent=cov,
        )

    label_a = config_a.stem.replace("_", " ").title()
    label_b = config_b.stem.replace("_", " ").title()
    result_a = _run_scenario(config_a, label_a)
    result_b = _run_scenario(config_b, label_b)

    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = output_dir / "comparison.png"
    ScenarioComparisonDashboard().render(result_a, result_b, output_path=str(dashboard_path))
    console.print(f"\nSaved comparison dashboard → {dashboard_path}")


if __name__ == "__main__":
    app()

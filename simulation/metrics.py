"""
Acoustic quality metrics: STOI, SINR, predicted WER categories, and channel counts.

MetricsCalculator.compute() is the single entry-point called by MonteCarloRunner
per iteration.  It satisfies the _MetricsProtocol defined in monte_carlo.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from config.defaults import SINR_TARGET_DB, STOI_ACCEPTABLE, STOI_EXCELLENT
from models.microphone import MicrophoneProfile, MX412Gooseneck, MXA310Tabletop, MXA920CeilingArray
from models.participant import Participant
from models.room import RoomGeometry
from simulation.types import IterationMetrics

if TYPE_CHECKING:
    from simulation.engine import SimulationResult

try:
    from pystoi import stoi as _pystoi
    _PYSTOI_AVAILABLE = True
except ImportError:
    _PYSTOI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Channel count summary
# ---------------------------------------------------------------------------

@dataclass
class ChannelCountSummary:
    mx412_units: int
    mxa310_units: int
    mxa920_units: int
    gooseneck_channels: int
    tabletop_channels: int
    ceiling_channels: int
    total_alsa_streams: int
    description: str


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class MetricsCalculator:
    """
    Computes STOI, SINR, WER categories, and channel counts from a SimulationResult.

    Participants are needed for zone-based SINR grouping.
    mic_zone_map (optional): maps mic_id -> zone name.  When provided, zone SINR
      uses a participant-zone / mic-zone pairing instead of nearest-source heuristic.
    """

    def __init__(
        self,
        clean_signal: np.ndarray,
        microphones: list[MicrophoneProfile],
        room: RoomGeometry,
        participants: list[Participant],
        fs: int = 16000,
        mic_zone_map: dict[str, str] | None = None,
    ) -> None:
        self.clean_signal = clean_signal
        self.microphones = microphones
        self.room = room
        self.participants = participants
        self.fs = fs
        self.mic_zone_map = mic_zone_map or {}
        self._mic_by_id = {m.mic_id: m for m in microphones}

    # ------------------------------------------------------------------
    # MonteCarloRunner entry-point

    def compute(self, result: SimulationResult) -> IterationMetrics:
        stoi_per_mic: dict[str, float] = {}
        ceiling_penalty_applied: dict[str, float] = {}
        table_noise_coeff: float = 0.0

        for mic in self.microphones:
            signals = result.mic_signals.get(mic.mic_id, [])
            if not signals:
                continue

            # Pick the channel with the highest energy as the representative signal
            best_ch = max(range(len(signals)), key=lambda i: float(np.mean(signals[i] ** 2)))
            degraded = signals[best_ch]
            clean = self.clean_signal[: len(degraded)]

            raw_stoi = self.compute_stoi(degraded, clean, self.fs)

            penalty = 0.0
            if isinstance(mic, MXA920CeilingArray):
                penalty = self.room.drr_penalty(float(mic.position[2]))
                raw_stoi = max(0.0, raw_stoi - penalty)

            stoi_per_mic[mic.mic_id] = raw_stoi
            ceiling_penalty_applied[mic.mic_id] = penalty

            if isinstance(mic, MXA310Tabletop):
                table_noise_coeff = max(table_noise_coeff, mic.noise_coefficient)

        sinr_per_zone_pair = self.compute_zone_sinr(result)
        wer_category_per_mic = {
            mid: self.stoi_to_wer_category(s) for mid, s in stoi_per_mic.items()
        }

        return IterationMetrics(
            participant_positions=result.participant_positions,
            stoi_per_mic=stoi_per_mic,
            sinr_per_zone_pair=sinr_per_zone_pair,
            wer_category_per_mic=wer_category_per_mic,
            ceiling_penalty_applied=ceiling_penalty_applied,
            table_noise_coefficient=table_noise_coeff,
        )

    # ------------------------------------------------------------------
    # STOI

    def compute_stoi(self, degraded: np.ndarray, clean: np.ndarray, fs: int) -> float:
        n = min(len(degraded), len(clean))
        min_samples = fs // 4   # 250 ms minimum for a valid STOI score
        if n < min_samples:
            return float("nan")

        if _PYSTOI_AVAILABLE:
            try:
                return float(_pystoi(clean[:n], degraded[:n], fs, extended=False))
            except Exception:
                pass

        return self._approximate_stoi(degraded[:n], clean[:n])

    @staticmethod
    def _approximate_stoi(degraded: np.ndarray, clean: np.ndarray) -> float:
        """
        SNR-to-STOI approximation used when pystoi is not installed.
        Not a STOI implementation — provided only so unit tests can run without pystoi.
        """
        n = min(len(degraded), len(clean))
        p_clean = float(np.mean(clean[:n] ** 2))
        p_noise = float(np.mean((clean[:n] - degraded[:n]) ** 2))
        if p_clean < 1e-12:
            return 0.0
        snr_db = 10.0 * np.log10(p_clean / (p_noise + 1e-12))
        return float(np.clip(0.5 + snr_db / 40.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # SINR

    def compute_zone_sinr(self, result: SimulationResult) -> dict[str, float]:
        """
        Compute SINR for each microphone against all sources.

        Strategy:
          1. If mic_zone_map is provided, the target source for mic M is the
             participant whose zone matches M's mapped zone; all others are interference.
          2. Otherwise, the nearest participant is treated as the target.

        Returns a dict keyed "mic_id" -> SINR in dB.
        (Named zone-pair for readability but keyed by mic when zones are unavailable.)
        """
        per_src = result.per_source_signals
        if not per_src:
            return {}

        positions = result.participant_positions
        sinr: dict[str, float] = {}

        for mic in self.microphones:
            src_signals = per_src.get(mic.mic_id)
            if not src_signals or len(src_signals) < 2:
                continue

            # Identify target source index
            mic_zone = self.mic_zone_map.get(mic.mic_id)
            target_idx = self._find_target_source(mic, positions, mic_zone)

            target_power = float(np.mean(src_signals[target_idx] ** 2))
            interf_power = sum(
                float(np.mean(src_signals[i] ** 2))
                for i in range(len(src_signals))
                if i != target_idx
            ) or 1e-12

            sinr_db = 10.0 * np.log10(target_power / interf_power)
            sinr[mic.mic_id] = sinr_db

        return sinr

    def _find_target_source(
        self,
        mic: MicrophoneProfile,
        positions: list[np.ndarray],
        mic_zone: str | None,
    ) -> int:
        """Return the index of the participant treated as the 'target' for this mic."""
        if mic_zone:
            for idx, p in enumerate(self.participants):
                if p.zone == mic_zone and idx < len(positions):
                    return idx
        # Fallback: nearest participant
        distances = [float(np.linalg.norm(pos - mic.position)) for pos in positions]
        return int(np.argmin(distances))

    @staticmethod
    def sinr_passes(sinr_db: float) -> bool:
        return sinr_db >= SINR_TARGET_DB

    # ------------------------------------------------------------------
    # WER category

    @staticmethod
    def stoi_to_wer_category(stoi: float) -> str:
        if np.isnan(stoi):
            return "unknown"
        if stoi > STOI_EXCELLENT:
            return "excellent"   # predicted WER < 5 %
        if stoi > STOI_ACCEPTABLE:
            return "acceptable"  # dependent on DeepFilterNet
        return "critical"        # procurement risk

    # ------------------------------------------------------------------
    # Channel count

    def compute_channel_count(self) -> ChannelCountSummary:
        mx412 = [m for m in self.microphones if isinstance(m, MX412Gooseneck)]
        mxa310 = [m for m in self.microphones if isinstance(m, MXA310Tabletop)]
        mxa920 = [m for m in self.microphones if isinstance(m, MXA920CeilingArray)]

        g_ch = len(mx412)
        t_ch = len(mxa310) * 4
        c_ch = sum(m.channel_count for m in mxa920)
        total = g_ch + t_ch + c_ch

        parts: list[str] = []
        if mxa310:
            parts.append(f"{len(mxa310)}×MXA310 ({t_ch}ch)")
        if mxa920:
            parts.append(f"{len(mxa920)}×MXA920 ({c_ch}ch)")
        if mx412:
            parts.append(f"{len(mx412)}×MX412 ({g_ch}ch)")

        return ChannelCountSummary(
            mx412_units=len(mx412),
            mxa310_units=len(mxa310),
            mxa920_units=len(mxa920),
            gooseneck_channels=g_ch,
            tabletop_channels=t_ch,
            ceiling_channels=c_ch,
            total_alsa_streams=total,
            description=" + ".join(parts) + f" = {total} streams",
        )

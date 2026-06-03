"""
Core simulation engine.

Each microphone device gets its own PyRoomAcoustics ShoeBox instance because
PRA's simulate() call aggregates all capsules into a single array — there is no
additive mic array API.  One room per device is the correct pattern.

Per-source signals are extracted by convolving each source's clean signal with
its individual RIR (room.rir[mic_idx][src_idx]).  These are stored in
SimulationResult and used by MetricsCalculator for proper zone SINR computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve

from config.defaults import SAMPLE_RATE
from models.microphone import MicrophoneProfile, MXA310Tabletop
from models.participant import Participant
from models.room import RoomGeometry
from simulation.noise import TableNoiseInjector


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Output of SimulationEngine.run_single() for one iteration."""
    participant_positions: list[np.ndarray]
    # mic_id -> list of per-channel combined signals (all sources summed)
    mic_signals: dict[str, list[np.ndarray]]
    # mic_id -> per-source signals at channel 0 (directivity-weighted).
    # per_source_signals[mic_id][src_idx] is used for zone SINR calculation.
    per_source_signals: dict[str, list[np.ndarray]]
    room_geometry: RoomGeometry
    rt60: float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SimulationEngine:
    """
    Orchestrates PRA simulation for a complete wargaming room layout.

    Participants and microphones are configured at construction time; run_single()
    can accept overridden participant positions for Monte Carlo iteration.

    clean_signal: reference speech used as the PRA source signal.  If not
      provided, a synthetic band-limited speech-shaped signal is generated.
    """

    def __init__(
        self,
        room: RoomGeometry,
        participants: list[Participant],
        microphones: list[MicrophoneProfile],
        noise_injector: TableNoiseInjector | None = None,
        fs: int = SAMPLE_RATE,
        clean_signal: np.ndarray | None = None,
    ) -> None:
        self.room = room
        self.participants = participants
        self.microphones = microphones
        self.noise_injector = noise_injector or TableNoiseInjector(0.0, fs=fs)
        self.fs = fs
        self._clean_signal = (
            clean_signal if clean_signal is not None else _synthesise_speech(fs)
        )

    # Small inset from each wall — PRA rejects sources exactly on the boundary.
    _WALL_MARGIN: float = 0.05

    def run_single(
        self,
        participant_positions: list[np.ndarray] | None = None,
    ) -> SimulationResult:
        """
        Run one simulation pass.
        participant_positions overrides stored positions (used by MonteCarloRunner).
        """
        raw = participant_positions or [p.position_vector() for p in self.participants]
        positions = [self._clamp_to_room(p) for p in raw]
        rt60 = self.room.sabine_rt60()

        mic_signals: dict[str, list[np.ndarray]] = {}
        per_source_signals: dict[str, list[np.ndarray]] = {}

        for mic in self.microphones:
            combined, per_src = self._simulate_mic(mic, positions)
            mic_signals[mic.mic_id] = combined
            per_source_signals[mic.mic_id] = per_src

        return SimulationResult(
            participant_positions=positions,
            mic_signals=mic_signals,
            per_source_signals=per_source_signals,
            room_geometry=self.room,
            rt60=rt60,
        )

    def _clamp_to_room(self, pos: np.ndarray) -> np.ndarray:
        """Clamp a position to lie strictly inside the room, with a small wall margin."""
        m = self._WALL_MARGIN
        clamped = pos.copy()
        clamped[0] = float(np.clip(pos[0], m, self.room.length - m))
        clamped[1] = float(np.clip(pos[1], m, self.room.width - m))
        clamped[2] = float(np.clip(pos[2], m, self.room.height - m))
        return clamped

    # ------------------------------------------------------------------

    def _simulate_mic(
        self,
        mic: MicrophoneProfile,
        participant_positions: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Run PRA for one mic device.

        Returns:
            combined: one directivity-weighted signal per channel (all sources summed).
            per_src:  one directivity-weighted signal per source (channel 0 only),
                      for SINR calculation.
        """
        pra_room = self.room.to_pra_shoebox(self.fs)

        for pos in participant_positions:
            pra_room.add_source(pos.tolist(), signal=self._clean_signal)

        capsule_positions = mic.channel_positions()
        R = np.column_stack(capsule_positions)   # shape (3, n_channels)
        pra_room.add_microphone(R)
        pra_room.simulate()

        n_clean = len(self._clean_signal)
        n_channels = mic.channel_count
        n_sources = len(participant_positions)

        # ---- combined per-channel signals ----
        combined: list[np.ndarray] = []
        for ch_idx in range(n_channels):
            raw = pra_room.mic_array.signals[ch_idx, :n_clean]
            # Apply directivity for the nearest (dominant) source
            if participant_positions:
                g = mic.gain(participant_positions[0], channel=ch_idx)
                raw = raw * g
            # Inject table noise into MXA310 channels
            if isinstance(mic, MXA310Tabletop) and mic.noise_coefficient > 0.0:
                injector = TableNoiseInjector(
                    noise_coefficient=mic.noise_coefficient,
                    fs=self.fs,
                )
                raw = injector.inject(raw)
            combined.append(raw)

        # ---- per-source signals at channel 0 (for SINR) ----
        # Convolve each source's clean signal with its individual RIR at capsule 0.
        per_src: list[np.ndarray] = []
        for src_idx in range(n_sources):
            rir = pra_room.rir[0][src_idx]   # channel 0, source src_idx
            convolved = fftconvolve(self._clean_signal, rir)[:n_clean]
            g = mic.gain(participant_positions[src_idx], channel=0)
            per_src.append(convolved * g)

        return combined, per_src


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthesise_speech(fs: int, duration_s: float = 1.0) -> np.ndarray:
    """
    Band-limited speech-shaped noise when no real WAV is provided.
    Shaped to 100–8000 Hz via FFT masking.  Seeded for reproducibility.
    """
    n = int(fs * duration_s)
    rng = np.random.default_rng(0)
    white = rng.normal(0.0, 1.0, n)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spectrum = np.fft.rfft(white)
    mask = (freqs >= 100.0) & (freqs <= 8000.0)
    spectrum[~mask] = 0.0
    signal = np.fft.irfft(spectrum, n=n)
    peak = np.max(np.abs(signal))
    if peak > 1e-12:
        signal /= peak
    return signal

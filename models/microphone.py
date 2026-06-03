"""
Microphone profiles for the three Shure hardware candidates.

Directivity gain is computed post-PRA-simulation as a multiplicative weight on
the convolved signals.  PRA models omnidirectional capsules; the pattern shapes
defined here are applied analytically to the resulting signals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from config.defaults import (
    MX412_EFFECTIVE_RADIUS,
    MXA310_CAPSULE_RADIUS,
    MXA310_CHANNELS,
    MXA310_EFFECTIVE_RADIUS,
    MXA310_HEIGHT,
    MXA920_BEAM_HALF_ANGLE_DEG,
    MXA920_EFFECTIVE_RADIUS,
    MXA920_HEIGHT,
)
from simulation.directivity import (
    angle_between,
    cardioid_gain,
    conical_beam_gain,
)


class MicrophoneProfile(ABC):
    """Abstract base class shared by all microphone types."""

    mic_id: str
    position: np.ndarray
    channel_count: int
    effective_radius: float
    mic_type: str

    @abstractmethod
    def gain(self, source_pos: np.ndarray, channel: int = 0) -> float:
        """Linear gain [0, 1] for a source at source_pos on the given channel."""
        ...

    @abstractmethod
    def channel_positions(self) -> list[np.ndarray]:
        """3-D position of each capsule, in channel order.  len == channel_count."""
        ...

    @property
    def is_ceiling_mounted(self) -> bool:
        return float(self.position[2]) >= 2.5


class MX412Gooseneck(MicrophoneProfile):
    """
    Shure MX412 desktop gooseneck microphone.

    Modelled as a single-channel tight cardioid: E(θ) = 0.5 + 0.5·cos(θ).
    Placed in close proximity to a fixed seat, aimed downward toward the table.
    Effective capture radius ~0.6 m.
    """

    mic_type = "MX412"
    channel_count = 1

    def __init__(
        self,
        mic_id: str,
        position: np.ndarray,
        aim_vector: np.ndarray | None = None,
    ) -> None:
        self.mic_id = mic_id
        self.position = np.asarray(position, dtype=float)
        self.effective_radius = MX412_EFFECTIVE_RADIUS
        raw_aim = (
            np.asarray(aim_vector, dtype=float)
            if aim_vector is not None
            else np.array([0.0, 0.0, -1.0])   # pointing slightly downward by default
        )
        self.aim_vector = raw_aim / np.linalg.norm(raw_aim)

    def gain(self, source_pos: np.ndarray, channel: int = 0) -> float:
        direction = np.asarray(source_pos, dtype=float) - self.position
        theta = angle_between(direction, self.aim_vector)
        return cardioid_gain(theta)

    def channel_positions(self) -> list[np.ndarray]:
        return [self.position.copy()]


class MXA310Tabletop(MicrophoneProfile):
    """
    Shure MXA310 tabletop boundary array.

    4 independent cardioid channels at 90° azimuth increments (0°, 90°, 180°, 270°).
    Sits at table height z = 0.75 m (enforced).

    noise_coefficient: Mechanical Interference Coefficient [0, 1].
      0.0 = disabled; 1.0 = maximum table noise injection.
      Used by TableNoiseInjector in the engine.
    """

    mic_type = "MXA310"
    channel_count = MXA310_CHANNELS
    _AZIMUTHS_DEG: ClassVar[list[float]] = [0.0, 90.0, 180.0, 270.0]

    def __init__(
        self,
        mic_id: str,
        position: np.ndarray,
        noise_coefficient: float = 0.0,
    ) -> None:
        pos = np.asarray(position, dtype=float).copy()
        pos[2] = MXA310_HEIGHT
        self.mic_id = mic_id
        self.position = pos
        self.effective_radius = MXA310_EFFECTIVE_RADIUS
        if not (0.0 <= noise_coefficient <= 1.0):
            raise ValueError(f"noise_coefficient must be in [0, 1], got {noise_coefficient}")
        self.noise_coefficient = noise_coefficient

    def gain(self, source_pos: np.ndarray, channel: int = 0) -> float:
        """
        Cardioid evaluated in the horizontal plane.
        Vertical component is ignored — the toroidal pattern is omnidirectional
        within ±45° of elevation.
        """
        azimuth_rad = np.deg2rad(self._AZIMUTHS_DEG[channel])
        aim = np.array([np.cos(azimuth_rad), np.sin(azimuth_rad), 0.0])
        direction = np.asarray(source_pos, dtype=float) - self.position
        direction[2] = 0.0   # collapse to horizontal plane
        theta = angle_between(direction, aim)
        return cardioid_gain(theta)

    def channel_positions(self) -> list[np.ndarray]:
        positions = []
        for deg in self._AZIMUTHS_DEG:
            rad = np.deg2rad(deg)
            offset = np.array([
                MXA310_CAPSULE_RADIUS * np.cos(rad),
                MXA310_CAPSULE_RADIUS * np.sin(rad),
                0.0,
            ])
            positions.append(self.position + offset)
        return positions


class MXA920CeilingArray(MicrophoneProfile):
    """
    Shure MXA920 (or MXA910) ceiling array.

    Overhead at z = 3.0 m (enforced).  Each channel is an independently steerable
    conical beam (7.5° half-angle) aimed at a configured desk cluster centroid.

    beam_targets: one 3-D floor-level coordinate per channel.
      len(beam_targets) determines channel_count.
    """

    mic_type = "MXA920"

    def __init__(
        self,
        mic_id: str,
        position: np.ndarray,
        beam_targets: list[np.ndarray],
        beam_half_angle_deg: float = MXA920_BEAM_HALF_ANGLE_DEG,
    ) -> None:
        if not beam_targets:
            raise ValueError("beam_targets must have at least one entry")
        pos = np.asarray(position, dtype=float).copy()
        pos[2] = MXA920_HEIGHT
        self.mic_id = mic_id
        self.position = pos
        self.beam_targets = [np.asarray(t, dtype=float) for t in beam_targets]
        self.channel_count = len(beam_targets)
        self.effective_radius = MXA920_EFFECTIVE_RADIUS
        self._beam_half_angle_rad = float(np.deg2rad(beam_half_angle_deg))

    def gain(self, source_pos: np.ndarray, channel: int = 0) -> float:
        target = self.beam_targets[channel]
        aim = target - self.position
        direction = np.asarray(source_pos, dtype=float) - self.position
        theta = angle_between(aim, direction)
        return conical_beam_gain(theta, self._beam_half_angle_rad)

    def channel_positions(self) -> list[np.ndarray]:
        """
        Physical capsule ring at ceiling height.
        Beam steering is handled by gain(); capsule positions only affect PRA's ISM.
        """
        array_radius = 0.10   # metres — physical array footprint
        n = self.channel_count
        positions = []
        for i in range(n):
            angle = 2.0 * np.pi * i / n
            offset = np.array([
                array_radius * np.cos(angle),
                array_radius * np.sin(angle),
                0.0,
            ])
            positions.append(self.position + offset)
        return positions

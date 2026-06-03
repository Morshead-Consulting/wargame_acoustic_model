from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from config.defaults import Z_SEATED, Z_STANDING


@dataclass
class Participant:
    """
    A single wargame player modelled as a point sound source.

    z defaults to Z_SEATED (1.2 m) for seated players.  Pass posture="standing"
    and the default z is automatically promoted to Z_STANDING (1.6 m).
    zone is used by MetricsCalculator to group participants for SINR calculation.
    """

    participant_id: str
    x: float
    y: float
    z: float = Z_SEATED
    zone: str = "default"
    posture: Literal["seated", "standing"] = "seated"

    def __post_init__(self) -> None:
        if self.posture == "standing" and self.z == Z_SEATED:
            self.z = Z_STANDING

    def position_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def perturbed(self, rng: np.random.Generator, sigma: float) -> "Participant":
        """
        Return a copy displaced by N(0, σ²) in x and y.
        z is not perturbed — vertical movement is not modelled.
        """
        dx, dy = rng.normal(0.0, sigma, 2)
        return Participant(
            participant_id=self.participant_id,
            x=self.x + dx,
            y=self.y + dy,
            z=self.z,
            zone=self.zone,
            posture=self.posture,
        )

    def distance_to(self, point: np.ndarray) -> float:
        return float(np.linalg.norm(self.position_vector() - np.asarray(point)))

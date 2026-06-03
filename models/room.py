from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from config.defaults import (
    CEILING_PENALTY_REF_HEIGHT,
    CEILING_PENALTY_SCALE,
    DEFAULT_ABSORPTION,
    MAX_ORDER,
    SAMPLE_RATE,
)

if TYPE_CHECKING:
    import pyroomacoustics as pra


@dataclass
class RoomGeometry:
    """
    Defines the physical space and its acoustic properties.

    absorption: energy absorption coefficient α ∈ (0, 1].
      0.15 → echoey ring-fenced room; 0.45 → heavily treated space.
    rt60_formula: "sabine" (default) or "eyring" (more accurate for α > 0.3).
    """

    length: float
    width: float
    height: float
    absorption: float = DEFAULT_ABSORPTION
    name: str = "wargame_room"
    rt60_formula: Literal["sabine", "eyring"] = "sabine"

    def __post_init__(self) -> None:
        for attr, val in (("length", self.length), ("width", self.width), ("height", self.height)):
            if val <= 0:
                raise ValueError(f"{attr} must be positive, got {val}")
        if not (0 < self.absorption <= 1.0):
            raise ValueError(f"absorption must be in (0, 1], got {self.absorption}")

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    @property
    def surface_area(self) -> float:
        L, W, H = self.length, self.width, self.height
        return 2.0 * (L * W + L * H + W * H)

    def sabine_rt60(self) -> float:
        """
        Calculate reverberation time.
        Sabine:  RT60 = 0.161 · V / (α · S)
        Eyring:  RT60 = -0.161 · V / (S · ln(1 - α))
        """
        if self.rt60_formula == "eyring":
            return -0.161 * self.volume / (self.surface_area * math.log(1.0 - self.absorption))
        return 0.161 * self.volume / (self.absorption * self.surface_area)

    def drr_penalty(self, mic_height: float) -> float:
        """
        STOI penalty applied to ceiling-mounted arrays.
        Models the degraded direct-to-reverberant ratio at greater heights.
        Returns 0 for mics at or below CEILING_PENALTY_REF_HEIGHT.
        """
        excess = max(0.0, mic_height - CEILING_PENALTY_REF_HEIGHT)
        return CEILING_PENALTY_SCALE * excess

    def to_pra_shoebox(self, fs: int = SAMPLE_RATE) -> "pra.ShoeBox":
        """
        Single factory for PyRoomAcoustics rooms.
        The only place in the codebase that imports or instantiates PRA.
        """
        import pyroomacoustics as pra  # deferred so non-PRA tests import cleanly

        materials = pra.Material(energy_absorption=self.absorption)
        return pra.ShoeBox(
            [self.length, self.width, self.height],
            fs=fs,
            max_order=MAX_ORDER,
            materials=materials,
            air_absorption=True,
            ray_tracing=False,
        )

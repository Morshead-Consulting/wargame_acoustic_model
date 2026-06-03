"""
Pure directivity and distance-attenuation functions.
All functions are stateless — no side-effects, no class state.
Imported by models.microphone to keep physics separate from data modelling.
"""

from __future__ import annotations

import numpy as np


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Angle in radians between two 3-D vectors.
    Returns 0 when either vector is effectively zero-length (avoids NaN).
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def cardioid_gain(theta_rad: float) -> float:
    """
    Standard cardioid: E(θ) = 0.5 + 0.5·cos(θ), clipped to [0, 1].
    Maximum gain at θ=0°, zero at θ=180°.
    """
    return float(np.clip(0.5 + 0.5 * np.cos(theta_rad), 0.0, 1.0))


def supercardioid_gain(theta_rad: float) -> float:
    """
    Supercardioid (narrower null at ≈126°): 0.37 + 0.63·cos(θ), clipped to [0, 1].
    Available for MX412 variant modelling.
    """
    return float(np.clip(0.37 + 0.63 * np.cos(theta_rad), 0.0, 1.0))


def conical_beam_gain(theta_rad: float, half_angle_rad: float) -> float:
    """
    Inside the beam cone: full gain (1.0).
    Outside: cos(θ)^6 roll-off — approximates ~40 dB/octave rejection.
    """
    if theta_rad <= half_angle_rad:
        return 1.0
    return float(max(0.0, np.cos(theta_rad) ** 6))


def inverse_square_attenuation(distance_m: float, ref_distance_m: float = 1.0) -> float:
    """
    Pressure ratio from inverse-square law.
    Clamped at 0.01 m to prevent divide-by-zero at the source position.
    """
    d = max(distance_m, 0.01)
    return float((ref_distance_m / d) ** 2)

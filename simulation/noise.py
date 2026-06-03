"""
Table noise injection for MXA310 boundary-layer channels.

Models pen tapping, map shuffling, and laptop vibrations as Poisson-distributed
low-frequency impulse bursts.  The coefficient is set per MXA310 instance and
injected by SimulationEngine after PRA convolution.
"""

from __future__ import annotations

import numpy as np

from config.defaults import (
    SAMPLE_RATE,
    TABLE_NOISE_AMPLITUDE,
    TABLE_NOISE_IMPULSE_RATE_HZ,
    TABLE_NOISE_LOW_FREQ_HZ,
)


class TableNoiseInjector:
    """
    Adds simulated mechanical interference to a signal array.

    noise_coefficient = 0.0  → inject() is a pure no-op (zero overhead).
    noise_coefficient = 1.0  → maximum injection at TABLE_NOISE_AMPLITUDE level.
    """

    def __init__(
        self,
        noise_coefficient: float = 0.0,
        fs: int = SAMPLE_RATE,
        low_freq_hz: float = TABLE_NOISE_LOW_FREQ_HZ,
        impulse_rate_hz: float = TABLE_NOISE_IMPULSE_RATE_HZ,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not (0.0 <= noise_coefficient <= 1.0):
            raise ValueError(f"noise_coefficient must be in [0, 1], got {noise_coefficient}")
        self.noise_coefficient = noise_coefficient
        self.fs = fs
        self.low_freq_hz = low_freq_hz
        self.impulse_rate_hz = impulse_rate_hz
        self._rng = rng if rng is not None else np.random.default_rng()

    def generate(self, n_samples: int) -> np.ndarray:
        """
        Build a noise signal of length n_samples.
        Returns a zero array when noise_coefficient == 0 (fast path).
        """
        if self.noise_coefficient == 0.0 or n_samples == 0:
            return np.zeros(n_samples, dtype=float)

        signal = np.zeros(n_samples, dtype=float)
        duration_s = n_samples / self.fs
        n_impulses = int(self._rng.poisson(self.impulse_rate_hz * duration_s))

        if n_impulses == 0:
            return signal

        impulse_samples = self._rng.integers(0, n_samples, size=n_impulses)
        t = np.arange(n_samples) / self.fs
        omega = 2.0 * np.pi * self.low_freq_hz
        envelope_width = 0.02   # 20 ms Gaussian decay (pen-tap / map-shuffle signature)

        for idx in impulse_samples:
            t0 = idx / self.fs
            envelope = np.exp(-((t - t0) ** 2) / (2.0 * envelope_width ** 2))
            signal += envelope * np.sin(omega * (t - t0))

        peak = np.max(np.abs(signal))
        if peak > 1e-12:
            signal *= (self.noise_coefficient * TABLE_NOISE_AMPLITUDE) / peak

        return signal

    def inject(self, signal: np.ndarray) -> np.ndarray:
        """Return signal + noise.  No copy made when noise_coefficient == 0."""
        if self.noise_coefficient == 0.0:
            return signal
        return signal + self.generate(len(signal))

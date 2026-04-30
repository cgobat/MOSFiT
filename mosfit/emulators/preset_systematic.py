"""Preset systematic uncertainty (mag) vs rest-frame time for emulators.

This is a *fixed* floor in magnitude space (not a fitted parameter), intended
to inflate errors where emulators are less trustworthy (e.g. near time edges).
Any SED module may emit ``emulator_preset_systematic_mag``; :class:`Diagonal`
adds ``sigma**2`` to the diagonal only for magnitude-like observations.
"""
from __future__ import annotations

import numpy as np


def preset_systematic_mag(
    t_rest_days,
    t_emulator_min: float,
    t_emulator_max: float,
    sys_min_mag: float = 0.2,
    sys_max_mag: float = 0.7,
    peak_t_rest_day: float = 20.0,
) -> np.ndarray:
    """
    Smooth U-shaped systematic error (mag) across an emulator time domain.

    ``sys_max_mag`` at ``t_emulator_min`` and ``t_emulator_max``;
    ``sys_min_mag`` at ``peak_t_rest_day`` (clipped to the emulator window);
    cosine-smooth between.
    """
    t = np.asarray(t_rest_days, dtype=np.float64)
    flat = t.ndim == 0
    if flat:
        t = t.reshape(1)

    lo = float(min(sys_min_mag, sys_max_mag))
    hi = float(max(sys_min_mag, sys_max_mag))
    t0 = float(np.clip(peak_t_rest_day, t_emulator_min, t_emulator_max))
    eps = 1e-12

    out = np.empty_like(t, dtype=np.float64)
    left = t <= t0
    if np.any(left):
        den = max(t0 - t_emulator_min, eps)
        x = np.clip((t[left] - t_emulator_min) / den, 0.0, 1.0)
        out[left] = hi - (hi - lo) * 0.5 * (1.0 - np.cos(np.pi * x))
    if np.any(~left):
        den = max(t_emulator_max - t0, eps)
        x = np.clip((t[~left] - t0) / den, 0.0, 1.0)
        out[~left] = lo + (hi - lo) * 0.5 * (1.0 - np.cos(np.pi * x))

    if flat:
        return out.reshape(())
    return out

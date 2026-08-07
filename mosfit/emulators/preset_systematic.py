"""Preset systematic uncertainty (mag) vs rest-frame time for emulators.

Combines an exponential fit to emulator performance on Sedona models with an
RMS floor from Sedona–CMFGen light-curve comparisons, in quadrature.

Any SED module may emit ``emulator_preset_systematic_mag``; :class:`Diagonal`
adds ``sigma**2`` to the diagonal only for magnitude-like observations.
"""
from __future__ import annotations

import numpy as np


def rms_sedona_uncertainty(t):
    """RMS uncertainty between Sedona and CMFGen SESN light curves (mag)."""
    gauss1 = 0.047736 * np.exp(-(t - 4.33)**2 / (2 * 0.51**2))
    gauss2 = -0.152869 * np.exp(-(t - 19.09)**2 / (2 * 3.69**2))
    gauss3 = 0.174996 * np.exp(-(t - 69.75)**2 / (2 * 8.05**2))
    return 0.357635 + gauss1 + gauss2 + gauss3


def preset_systematic_mag(
    t_rest_days_eval,
    t_emulator_min: float = None,
    t_emulator_max: float = None,
    sys_min_mag: float = None,
    sys_max_mag: float = None,
    peak_t_rest_day: float = None,
) -> np.ndarray:
    """
    Systematic uncertainty (mag): emulator performance and Sedona–CMFGen RMS
    combined in quadrature.

    The optional ``t_emulator_min`` / ``sys_*`` arguments are accepted for
    compatibility with :class:`~mosfit.modules.seds.sesn_sedona.SESNSedona` but are
    not used (uncertainty is data-driven, not the old U-shaped preset).
    """
    t = np.asarray(t_rest_days_eval, dtype=np.float64)
    flat = t.ndim == 0
    if flat:
        t = t.reshape(1)

    emu = 0.124837 + 1.703078 * np.exp(-0.278435 * t)
    out = np.sqrt(emu**2 + rms_sedona_uncertainty(t)**2)

    if flat:
        return out.reshape(())
    return out

"""Definitions for the `Viscous` class."""
import numpy as np

from mosfit.modules.transforms.transform import Transform

CLASS_NAME = 'Viscous'


class Viscous(Transform):
    """Viscous delay transform."""

    N_INT_TIMES = 1000
    MIN_LOG_SPACING = -3

    def _quadrature_nodes(self, tvisc, t_end):
        """Abscissae on [0, 1]; cached while ``tvisc`` and span are unchanged."""
        key = (float(tvisc), float(t_end))
        if getattr(self, '_xm_key', None) == key:
            return self._xm
        num = int(self.N_INT_TIMES / 2.0)
        lsp = np.logspace(
            np.log10(tvisc / t_end) + self.MIN_LOG_SPACING, 0, num)
        self._xm = np.unique(np.concatenate((lsp, 1 - lsp)))
        self._xm_key = key
        return self._xm

    def process(self, **kwargs):
        """Process module."""
        Transform.process(self, **kwargs)

        tvisc = kwargs['Tviscous']

        new_lums = np.zeros_like(self._times_to_process)
        if len(self._dense_times_since_exp) < 2:
            return {self.dense_key('luminosities'): new_lums}
        dense_t = np.asarray(self._dense_times_since_exp, dtype=float)
        dense_l = np.asarray(self._dense_luminosities, dtype=float)
        min_te = min(self._dense_times_since_exp)
        tb = max(0.0, min_te)
        t_end = float(dense_t[-1])

        uniq_times = np.unique(self._times_to_process[
            (self._times_to_process >= tb) & (
                self._times_to_process <= t_end)])
        lu = len(uniq_times)

        xm = self._quadrature_nodes(tvisc, t_end)

        int_times = np.clip(tb + (uniq_times.reshape(lu, 1) - tb) * xm, tb,
                            t_end)

        int_tes = int_times[:, -1]
        int_lums = np.interp(int_times, dense_t, dense_l)

        int_args = int_lums * np.exp(
            (int_times - int_tes.reshape(lu, 1)) / tvisc)
        int_args[np.isnan(int_args)] = 0.0

        uniq_lums = np.trapezoid(int_args, int_times) / tvisc
        new_lums = uniq_lums[np.searchsorted(uniq_times,
                                             self._times_to_process)]

        return {self.dense_key('luminosities'): new_lums}

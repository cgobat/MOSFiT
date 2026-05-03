"""Definitions for the `Likelihood` class."""
from math import isnan

import numpy as np

from mosfit.constants import LIKELIHOOD_FLOOR
from mosfit.modules.module import Module


# Important: Only define one ``Module`` class per file.


def _sesn_valid_mask_on_residual_stream(kwargs, residual_n):
    """``valid_mask`` is dense like ``observed``; diagonal uses ``mask[observed]``."""
    vm = kwargs.get('valid_mask', None)
    obs = kwargs.get('observed', None)
    if vm is None or obs is None:
        return None
    vm = np.asarray(vm, dtype=bool)
    obs = np.asarray(obs, dtype=bool)
    if vm.shape != obs.shape:
        return None
    vp = vm[obs].ravel()
    if vp.size != int(residual_n):
        return None
    return vp


class Likelihood(Module):
    """Calculate the maximum likelihood score for a model."""

    MIN_COV_TERM = 1.0e-30

    def __init__(self, **kwargs):
        """Initialize `Likelihood` module."""
        super(Likelihood, self).__init__(**kwargs)
        self._cuda_reported = False
        self._use_cpu = None

        if not self._model._fitter._cuda:
            self._use_cpu = True

    def process(self, **kwargs):
        """Calculate the likelihood, returning ln(likelihood)."""
        ret = {'value': LIKELIHOOD_FLOOR}

        self._fractions = kwargs.get('fractions', [])
        if not self._fractions:
            return ret

        model_full = np.asarray(kwargs['model_observations'])
        self._score_modifier = kwargs.get(self.key('score_modifier'), 0.0)

        dense_ul = np.array(kwargs.get('upperlimits', []), dtype=bool)
        dense_observed = kwargs.get('observed', None)
        if dense_observed is not None:
            dense_observed = np.asarray(dense_observed, dtype=bool)
        dense_vm = kwargs.get('valid_mask', None)
        if dense_vm is not None:
            dense_vm = np.asarray(dense_vm, dtype=bool)
            if dense_vm.shape != model_full.shape:
                dense_vm = None

        value = ret['value']

        # Basic sanity on fractions
        if min(self._fractions) < 0.0 or max(self._fractions) > 1.0:
            return ret

        if dense_observed is None or dense_observed.shape != model_full.shape:
            use_dense = np.ones_like(model_full, dtype=bool)
        elif dense_vm is None:
            use_dense = dense_observed
        else:
            use_dense = dense_observed & dense_vm
            if not np.any(use_dense):
                return ret

        for ii in np.flatnonzero(use_dense):
            md = float(model_full[ii])
            ul = dense_ul[ii] if ii < dense_ul.size else False
            if (not ul) and (isnan(md) or not np.isfinite(md)):
                return ret

        if dense_observed is not None and dense_observed.shape == model_full.shape:
            self._upper_limits = dense_ul[dense_observed]
        else:
            self._upper_limits = dense_ul

        self._model_observations = (
            model_full[dense_observed]
            if dense_observed is not None
            and dense_observed.shape == model_full.shape
            else model_full)

        # Covariance diagonal and residuals (1D arrays aligned with data)
        diag = kwargs.get('kdiagonal', None)
        residuals = kwargs.get('kresiduals', None)

        if diag is None or residuals is None:
            return ret

        diag = np.asarray(diag)
        residuals = np.asarray(residuals)

        valid_mask_residual = _sesn_valid_mask_on_residual_stream(
            kwargs, residuals.size)

        if not np.any(diag.shape) or not np.any(residuals.shape):
            return ret

        # Evaluate only the diagonal Gaussian likelihood.
        if 'obandvs' not in kwargs:
            return ret
        self._o_band_vs = np.asarray(kwargs['obandvs'])

        vp = valid_mask_residual
        if vp is not None:
            if not np.any(vp):
                return ret
            n_vp = int(vp.shape[0])
            if not (diag.shape[0] == n_vp and residuals.shape[0] == n_vp
                    and self._o_band_vs.shape[0] == n_vp):
                return ret
            self._o_band_vs = self._o_band_vs[vp]
            diag = diag[vp]
            residuals = residuals[vp]
            if self._upper_limits.shape[0] == n_vp:
                self._upper_limits = self._upper_limits[vp]
        elif dense_vm is not None:
            if dense_vm.shape == self._o_band_vs.shape:
                self._o_band_vs = self._o_band_vs[dense_vm]
            if dense_vm.shape == diag.shape:
                diag = diag[dense_vm]
                residuals = residuals[dense_vm]
                if dense_ul.shape[0] == dense_vm.shape[0]:
                    self._upper_limits = dense_ul[dense_vm]

        var = self._o_band_vs ** 2 + diag
        var = np.maximum(var, Likelihood.MIN_COV_TERM)
        value = -0.5 * np.sum(
            residuals ** 2 / var + np.log(2.0 * np.pi * var)
        )

        ret['kdiagonal'] = diag
        ret['kresiduals'] = residuals

        score = self._score_modifier + value
        if isnan(score) or not np.isfinite(score):
            return ret
        ret['value'] = max(LIKELIHOOD_FLOOR, score)
        return ret

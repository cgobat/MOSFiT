"""Definitions for the `Likelihood` class."""
from math import isnan

import numpy as np
from scipy import linalg as spla

from mosfit.constants import LIKELIHOOD_FLOOR
from mosfit.modules.module import Module


# Important: Only define one ``Module`` class per file.


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

        # Model predictions and upper-limit flags
        self._model_observations = np.asarray(kwargs['model_observations'])
        self._score_modifier = kwargs.get(self.key('score_modifier'), 0.0)
        self._upper_limits = np.asarray(kwargs.get('upperlimits', []), dtype=bool)

        # NEW: per-point validity mask from upstream (Sampson→Photometry)
        valid_mask = kwargs.get('valid_mask')


        if valid_mask is not None:
            #print("DEBUG Likelihood: N_used =", np.count_nonzero(valid_mask))
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if valid_mask.shape != self._model_observations.shape:
                # shape mismatch: ignore mask
                valid_mask = None
            # Strict: if no valid points, give a very bad finite score.
            elif not np.any(valid_mask):
                return ret

        # Basic sanity on fractions
        if min(self._fractions) < 0.0 or max(self._fractions) > 1.0:
            return ret

        # Covariance diagonal and residuals (1D arrays aligned with data)
        diag = kwargs.get('kdiagonal', None)
        residuals = kwargs.get('kresiduals', None)

        if diag is None or residuals is None:
            return ret

        diag = np.asarray(diag)
        residuals = np.asarray(residuals)

        # Apply mask to diag and residuals as well
        if valid_mask is not None:
            # Apply mask to model_obs and upper_limits
            self._model_observations = self._model_observations[valid_mask]
            diag = diag[valid_mask]
            residuals = residuals[valid_mask]
            if self._upper_limits.size:
                self._upper_limits = self._upper_limits[valid_mask]

        if self._upper_limits.size:
            finite_mask = self._upper_limits | np.isfinite(self._model_observations)
        else:
            finite_mask = np.isfinite(self._model_observations)
        if not np.all(finite_mask):
            return ret

        value = ret["value"]

        # Full covariance matrix case
        if kwargs.get('kmat') is not None:
            kmat = np.asarray(kwargs['kmat'])

            # Apply mask to the covariance matrix: select rows/cols
            if valid_mask is not None:
                kmat = kmat[np.ix_(valid_mask, valid_mask)]

            # Add observed errors to diagonal
            kmat[np.diag_indices_from(kmat)] += diag

            if self._use_cpu is not True and self._model._fitter._cuda:
                try:
                    import pycuda.gpuarray as gpuarray
                    import skcuda.linalg as skla
                except ImportError:
                    self._use_cpu = True
                    if not self._cuda_reported:
                        self._printer.message(
                            'cuda_not_enabled',
                            master_only=True,
                            warning=True
                        )
                else:
                    self._use_cpu = False
                    if not self._cuda_reported:
                        self._printer.message('cuda_enabled', master_only=True)
                        self._cuda_reported = True

                    kmat_gpu = gpuarray.to_gpu(kmat)
                    skla.cholesky(kmat_gpu, lib='cusolver')
                    value = -np.log(skla.det(kmat_gpu, lib='cusolver'))
                    res_gpu = gpuarray.to_gpu(residuals.reshape(
                        len(residuals), 1))
                    cho_mat_gpu = res_gpu.copy()
                    skla.cho_solve(kmat_gpu, cho_mat_gpu, lib='cusolver')
                    value -= 0.5 * (
                        skla.mdot(skla.transpose(res_gpu),
                                  cho_mat_gpu).get())[0][0]

            if self._use_cpu:
                try:
                    chol_kmat = spla.cholesky(kmat, lower=False, check_finite=False)
                    value = -np.sum(np.log(np.diag(chol_kmat)))
                    solved = spla.cho_solve(
                        (chol_kmat, False), residuals, check_finite=False
                    )
                    value -= 0.5 * np.dot(residuals, solved)
                except spla.LinAlgError:
                    return ret

            ret['kdiagonal'] = diag
            ret['kresiduals'] = residuals

        elif 'kfmat' in kwargs:
            raise RuntimeError('Should not have kfmat in likelihood!')

        else:
            # Shortcut when matrix is diagonal.
            self._o_band_vs = np.asarray(kwargs['obandvs'])

            # Apply mask to variances as well
            if valid_mask is not None:
                self._o_band_vs = self._o_band_vs[valid_mask]

            # diag and residuals already masked above
            variance_terms = self._o_band_vs**2 + diag
            value = -0.5 * np.sum(
                residuals**2 / variance_terms + np.log(variance_terms)
            )

        score = self._score_modifier + value
        if isnan(score) or not np.isfinite(score):
            return ret
        ret['value'] = max(LIKELIHOOD_FLOOR, score)
        return ret

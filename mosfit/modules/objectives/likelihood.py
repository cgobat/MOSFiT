"""Definitions for the `Likelihood` class."""
from math import isnan

import numpy as np
import scipy

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

        # Full covariance matrix case (slice kmat and matching 1D arrays together)
        if kwargs.get('kmat', None) is not None:
            kmat = np.asarray(kwargs['kmat'])

            mask_k = None
            if (valid_mask_residual is not None and
                    kmat.shape[0] == valid_mask_residual.shape[0]):
                mask_k = valid_mask_residual
            elif (dense_vm is not None and
                  kmat.shape[0] == dense_vm.shape[0]):
                mask_k = dense_vm

            dk = diag
            rk = residuals
            if mask_k is not None:
                if not np.any(mask_k):
                    return ret
                kmat = kmat[np.ix_(mask_k, mask_k)]
                dk = dk[mask_k]
                rk = rk[mask_k]

            # Add observed errors to diagonal
            kmat[np.diag_indices_from(kmat)] += dk

            condn = np.linalg.cond(kmat)
            if condn > 1.0e10:
                return ret

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
                        self._printer.message(
                            'cuda_enabled',
                            master_only=True
                        )
                        self._cuda_reported = True

                    kmat_gpu = gpuarray.to_gpu(kmat)
                    skla.cholesky(kmat_gpu, lib='cusolver')
                    value = -np.log(skla.det(kmat_gpu, lib='cusolver'))
                    res_gpu = gpuarray.to_gpu(rk.reshape(
                        len(rk), 1))
                    cho_mat_gpu = res_gpu.copy()
                    skla.cho_solve(kmat_gpu, cho_mat_gpu, lib='cusolver')
                    value -= (0.5 * (
                        skla.mdot(skla.transpose(res_gpu),
                                  cho_mat_gpu)).get())[0][0]

            if self._use_cpu:
                try:
                    import scipy
                    chol_kmat = scipy.linalg.cholesky(
                        kmat,
                        check_finite=False
                    )

                    value = -np.linalg.slogdet(chol_kmat)[-1]
                    value -= 0.5 * (
                        np.matmul(
                            rk.T,
                            scipy.linalg.cho_solve(
                                (chol_kmat, False),
                                rk,
                                check_finite=False
                            )
                        )
                    )
                except Exception:
                    try:
                        import scipy
                        value = -0.5 * (
                            np.matmul(
                                np.matmul(
                                    rk.T, scipy.linalg.inv(kmat)
                                ),
                                rk
                            ) + np.log(scipy.linalg.det(kmat))
                        )
                    except scipy.linalg.LinAlgError:
                        return ret

            ret['kdiagonal'] = dk
            ret['kresiduals'] = rk

        elif 'kfmat' in kwargs:
            raise RuntimeError('Should not have kfmat in likelihood!')

        else:
            # Shortcut when matrix is diagonal.
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

            # Full diagonal Gaussian: -½ Σ ( r²/σ² + log(2π σ²) ), σ² = kernel + obs.
            var = self._o_band_vs ** 2 + diag
            var = np.maximum(var, Likelihood.MIN_COV_TERM)
            value = -0.5 * np.sum(
                residuals ** 2 / var + np.log(2.0 * np.pi * var)
            )

        score = self._score_modifier + value
        if isnan(score) or not np.isfinite(score):
            return ret
        ret['value'] = max(LIKELIHOOD_FLOOR, score)
        return ret
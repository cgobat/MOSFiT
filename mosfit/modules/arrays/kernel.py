"""Definitions for the `Kernel` class."""
from collections import OrderedDict

import numpy as np
from six import string_types

from mosfit.constants import ANG_CGS, C_CGS
from mosfit.modules.arrays.array import Array

# Important: Only define one ``Module`` class per file.


class Kernel(Array):
    """Calculate the kernel for use in computing the likelihood score."""

    MIN_COV_TERM = 1.0e-30

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Kernel, self).__init__(**kwargs)
        self._times = np.array([])

    def process(self, **kwargs):
        """Process module."""
        self.preprocess(**kwargs)

        ret = OrderedDict()

        # Get band variances
        self._variance = kwargs.get(self.key('variance'), 0.0)

        # Get array of real observations.
        self._observations = np.array([
            ct if (t == 'countrate' or t == 'magcount') else y if
            (t == 'magnitude') else fd if t == 'fluxdensity' else None
            for y, ct, fd, t in zip(self._mags, self._cts, self._fds,
                                    self._o_otypes)
        ])

        # Get array of model observations.
        self._model_observations = kwargs.get('model_observations', [])

        # Handle band-specific variances if that option is enabled.
        self._band_v_vars = OrderedDict()
        for key in kwargs:
            if key.startswith('variance-band-'):
                self._band_v_vars[key.split('-')[-1]] = kwargs[key]

        if self._variance_bands:
            self._o_variance_bands = [
                self._variance_bands[i] for i in self._all_band_indices
            ]

            self._band_vs = np.array([
                self._band_v_vars.get(i, self._variance) if isinstance(
                    i, string_types) else
                (i[0] * self._band_v_vars.get(i[1][0], self._variance) +
                 (1.0 - i[0]) * self._band_v_vars.get(i[1][0], self._variance))
                for i in self._o_variance_bands
            ])
        else:
            self._band_vs = np.full(
                len(self._all_band_indices), self._variance)

        # Compute relative errors for count-based observations.
        self._band_vs[self._count_inds] = (
            10.0**(self._band_vs[self._count_inds] / 2.5) -
            1.0) * self._model_observations[self._count_inds]

        self._o_band_vs = self._band_vs[self._observed]

        ret['abandvs'] = self._band_vs
        ret['obandvs'] = self._o_band_vs

        return ret

    def receive_requests(self, **requests):
        """Receive requests from other ``Module`` objects."""
        self._average_wavelengths = requests.get('average_wavelengths', [])
        self._variance_bands = requests.get('variance_bands', [])

    def preprocess(self, **kwargs):
        """Construct kernel distance arrays."""
        new_times = np.array(kwargs.get('all_times', []), dtype=float)
        if np.array_equiv(new_times, self._times) and self._preprocessed:
            return
        self._times = new_times
        self._all_band_indices = kwargs.get('all_band_indices', [])
        self._are_bands = np.array(self._all_band_indices) >= 0
        self._freqs = kwargs.get('all_frequencies', [])
        self._mags = np.array(kwargs.get('magnitudes', []))
        self._fds = np.array(kwargs.get('fluxdensities', []))
        self._cts = np.array(kwargs.get('countrates', []))
        self._u_freqs = kwargs.get('all_u_frequencies', [])
        self._waves = np.array([
            self._average_wavelengths[bi]
            if bi >= 0 else C_CGS / self._freqs[i] / ANG_CGS
            for i, bi in enumerate(self._all_band_indices)
        ])
        self._observed = np.array(kwargs.get('observed', []), dtype=bool)
        self._observation_types = kwargs.get('observation_types')
        self._n_obs = len(self._observed)
        self._count_inds = self._observation_types != 'magnitude'

        self._o_times = self._times[self._observed]
        self._o_waves = self._waves[self._observed]
        self._o_otypes = self._observation_types[self._observed]

        self._preprocessed = True

"""Numeric checks for vectorized SED/photometry/diagonal (not a pytest suite)."""
from __future__ import print_function

import os
import sys

import numpy as np

# Keep this import-only check independent of the rest of the file.
os.environ.setdefault('MOSFIT_PHOTOMETRY_DEBUG', '')


def test_import_no_torch():
    import sys as _sys
    before = {k for k in _sys.modules if k == 'torch' or k.startswith('torch.')}
    import mosfit  # noqa: F401
    after = {k for k in _sys.modules if k == 'torch' or k.startswith('torch.')}
    leaked = after - before
    assert not leaked, leaked
    print('import mosfit does not import torch')


def test_blackbody_matches_serial():
    from mosfit.modules.seds.blackbody import Blackbody
    from mosfit.constants import BOL_BAND_INDEX

    class DummyPool(object):
        size = 0
        comm = None

        def is_master(self):
            return True

    class DummyPrinter(object):
        def message(self, *a, **k):
            return ''

        def text(self, *a, **k):
            return ''

        def prt(self, *a, **k):
            return ''

    class DummyModel(object):
        def pool(self):
            return DummyPool()

        def printer(self):
            return DummyPrinter()

    bb = Blackbody(name='blackbody', model=DummyModel())
    rng = np.random.RandomState(0)
    n_wav = 17
    sample = np.array([
        np.linspace(3000.0, 8000.0, n_wav),
        np.linspace(4000.0, 9000.0, n_wav),
    ])
    bb._sample_wavelengths = sample
    n = 40
    bi = np.array([0, 1, 0, 1, BOL_BAND_INDEX] * 8, dtype=int)
    lum = rng.uniform(1e40, 1e43, size=n)
    lum[3] = 0.0
    radius = rng.uniform(1e14, 1e15, size=n)
    temp = rng.uniform(5000.0, 20000.0, size=n)
    kwargs = {
        'luminosities': lum,
        'all_bands': ['g'] * n,
        'all_band_indices': bi,
        'all_frequencies': np.zeros(n),
        'radiusphot': radius,
        'temperaturephot': temp,
        'rest_times': np.linspace(0, 10, n),
        'redshift': 0.17,
    }
    out = bb.process(**kwargs)
    seds = out[bb.key('seds')]

    xc = bb.X_CONST
    fc = bb.FLUX_CONST
    zp1 = 1.17
    Azp1 = __import__('astropy').units.Angstrom.cgs.scale / zp1
    ref = []
    for li in range(n):
        bii = bi[li]
        if bii == BOL_BAND_INDEX:
            ref.append(np.zeros(1))
            continue
        if lum[li] == 0.0:
            ref.append(np.zeros(n_wav))
            continue
        rest_wavs = sample[bii] * Azp1
        rp = radius[li]
        tp = temp[li]
        row = fc * rp ** 2 / rest_wavs ** 5 / np.expm1(xc / rest_wavs / tp)
        row[np.isnan(row)] = 0.0
        ref.append(row)
    for li in range(n):
        np.testing.assert_allclose(np.asarray(seds[li]), ref[li], rtol=1e-10,
                                   atol=1e-20)
    print('Blackbody unique-band Planck matches serial formula')


def test_photometry_trapz():
    from mosfit.modules.observables.photometry import Photometry
    from mosfit.constants import MAG_FAC

    class DummyPool(object):
        size = 0
        comm = None

        def is_master(self):
            return True

    class DummyPrinter(object):
        def message(self, *a, **k):
            return ''

        def text(self, *a, **k):
            return ''

        def prt(self, *a, **k):
            return ''

    class DummyModel(object):
        def pool(self):
            return DummyPool()

        def printer(self):
            return DummyPrinter()

    phot = Photometry.__new__(Photometry)
    phot._model = DummyModel()
    phot._pool = DummyPool()
    phot._printer = DummyPrinter()
    phot._preprocessed = True
    phot._name = 'photometry'
    phot._replacements = {}
    n_wav = 17
    wavs = np.linspace(4000.0, 8000.0, n_wav)
    trans = np.exp(-((wavs - 6000.0) / 800.0) ** 2)
    phot._band_wavelengths = [wavs]
    phot._transmissions = [trans]
    phot._band_areas = [[]]
    phot._band_offsets = np.array([0.0])
    phot._filter_integrals = np.array([
        Photometry.FLUX_STD * np.trapezoid(trans / wavs ** 2, wavs)])
    n = 12
    seds = np.stack([np.linspace(1e38, 2e38, n_wav) * (i + 1) for i in range(n)])
    kwargs = {
        'luminosities': np.ones(n),
        'all_band_indices': np.zeros(n, dtype=int),
        'observation_types': np.array(['magnitude'] * n),
        'observed': np.ones(n, dtype=bool),
        'lumdist': 100.0,
        'all_frequencies': np.zeros(n),
        'redshift': 0.1,
        'sample_wavelengths': [wavs],
        'seds': seds,
    }
    out = phot.process(**kwargs)
    zp1 = 1.1
    trans_i = np.interp(wavs, wavs, trans)
    yvals = trans_i * seds / zp1
    eff = np.trapezoid(yvals, wavs, axis=-1) / phot._filter_integrals[0]
    from mosfit.constants import FOUR_PI, MPC_CGS
    ldist = np.log10(FOUR_PI * (100.0 * MPC_CGS) ** 2)
    mags = -0.0 - MAG_FAC * (np.log10(eff) - ldist)
    np.testing.assert_allclose(out['model_observations'], mags, rtol=1e-10)
    print('Photometry band-grouped trapz matches serial formula')


def test_diagonal_residuals():
    from mosfit.modules.arrays.diagonal import Diagonal

    class DummyPool(object):
        size = 0

        def is_master(self):
            return True

    class DummyPrinter(object):
        def message(self, *a, **k):
            return ''

        def prt(self, *a, **k):
            return ''

    class DummyModel(object):
        def pool(self):
            return DummyPool()

        def printer(self):
            return DummyPrinter()

    d = Diagonal(name='diagonal', model=DummyModel())
    n = 8
    model = np.array([18.0, 19.5, 17.0, 20.0, 18.2, 16.0, 21.0, 18.5])
    mags = np.array([18.1, 19.0, 17.2, 21.0, 18.0, 16.5, 22.0, 18.4])
    ul = np.array([False, True, False, True, False, False, True, False])
    d._preprocessed = True
    d._observed = np.ones(n, dtype=bool)
    d._o_types = np.array(['magnitude'] * n)
    d._mags_f = mags.astype(float)
    d._lums_f = np.full(n, np.nan)
    d._cts_f = np.full(n, np.nan)
    d._fds_f = np.full(n, np.nan)
    d._upper_limits = ul
    d._e_l_mags_f = np.full(n, 0.1)
    d._e_u_mags_f = np.full(n, 0.2)
    d._e_l_lums_f = np.full(n, 0.1)
    d._e_u_lums_f = np.full(n, 0.2)
    d._e_l_cts_f = np.full(n, 0.1)
    d._e_u_cts_f = np.full(n, 0.2)
    d._e_l_fds_f = np.full(n, 0.1)
    d._e_u_fds_f = np.full(n, 0.2)
    out = d.process(model_observations=model)
    from math import isnan
    ref = []
    for x, y, u in zip(model, mags, ul):
        if (not u and y is not None) or (not isnan(x) and y is not None and x < y):
            ref.append(abs(x - y))
        else:
            ref.append(0.0)
    np.testing.assert_allclose(out['kresiduals'], np.array(ref), rtol=0, atol=0)
    print('Diagonal residuals match serial definition')


def test_mm83():
    from mosfit.modules.seds.losextinction import LOSExtinction

    class DummyPool(object):
        size = 0

        def is_master(self):
            return True

    class DummyPrinter(object):
        def message(self, *a, **k):
            return ''

        def prt(self, *a, **k):
            return ''

    class DummyModel(object):
        def pool(self):
            return DummyPool()

        def printer(self):
            return DummyPrinter()

    los = LOSExtinction(name='ext', model=DummyModel())
    waves = np.array([2.0, 50.0, 200.0, 800.0, 2000.0])
    nh = 1.0e21
    vec = los.mm83(nh, waves)
    y = np.array([los.H_C_CGS / (x * los.ANG_CGS * los.KEV_CGS) for x in waves])
    i = np.array([np.searchsorted(los._mm83[:, 0], x) - 1 for x in y])
    al = [1.0e-24 * (los._mm83[x, 1] + los._mm83[x, 2] * y[j] +
                     los._mm83[x, 3] * y[j] ** 2) / y[j] ** 3
          for j, x in enumerate(i)]
    al = [al[j] if x < los._min_xray
          else los._almin * (los._min_xray / x) ** 3
          for j, x in enumerate(y)]
    al = [al[j] if x > los._max_xray
          else los._almax * (los._max_xray / x) ** 3
          for j, x in enumerate(y)]
    ref = nh * np.array(al)
    np.testing.assert_allclose(vec, ref, rtol=1e-12)
    print('mm83 vectorized matches list-comprehension formula')


if __name__ == '__main__':
    test_import_no_torch()
    test_blackbody_matches_serial()
    test_photometry_trapz()
    test_diagonal_residuals()
    test_mm83()
    print('all numeric tests passed')
    sys.exit(0)

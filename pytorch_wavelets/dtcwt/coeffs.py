from __future__ import absolute_import

import io
from importlib import resources
from numpy import load
try:
    import pywt
    _HAVE_PYWT = True
except ImportError:
    _HAVE_PYWT = False

COEFF_CACHE = {}


def _load_from_file(basename, varnames):

    try:
        mat = COEFF_CACHE[basename]
    except KeyError:
        try:
            raw = resources.read_binary('pytorch_wavelets.dtcwt.data', basename + '.npz')
            mat = dict(load(io.BytesIO(raw)))
        except Exception:
            from pkg_resources import resource_stream
            with resource_stream('pytorch_wavelets.dtcwt.data', basename + '.npz') as f:
                mat = dict(load(f))
        COEFF_CACHE[basename] = mat

    try:
        return tuple(mat[k] for k in varnames)
    except KeyError:
        raise ValueError(
            'Wavelet does not define ({0}) coefficients'.format(
                ', '.join(varnames)))


def biort(name):


    return level1(name, compact=True)


def level1(name, compact=False):


    if compact:
        if name == 'near_sym_b_bp':
            return _load_from_file(name, ('h0o', 'g0o', 'h1o', 'g1o', 'h2o', 'g2o'))
        else:
            return _load_from_file(name, ('h0o', 'g0o', 'h1o', 'g1o'))
    else:
        return _load_from_file(name, ('h0a', 'h0b', 'g0a', 'g0b', 'h1a', 'h1b',
                                      'g1a', 'g1b'))


def qshift(name):


    if name == 'qshift_b_bp':
        return _load_from_file(name, ('h0a', 'h0b', 'g0a', 'g0b', 'h1a', 'h1b',
                                      'g1a', 'g1b', 'h2a', 'h2b', 'g2a','g2b'))
    else:
        return _load_from_file(name, ('h0a', 'h0b', 'g0a', 'g0b', 'h1a', 'h1b',
                                      'g1a', 'g1b'))


def pywt_coeffs(name):

    if not _HAVE_PYWT:
        raise ImportError("Could not find PyWavelets module")
    return pywt.Wavelet(name)

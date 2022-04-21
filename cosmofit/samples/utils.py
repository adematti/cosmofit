"""Utilities for confidence level conversions."""

import math

import numpy as np
from scipy import stats

from cosmofit import utils


def nsigmas_to_quantiles_1d(nsigmas):
    r"""
    Turn number of Gaussian sigmas ``nsigmas`` into quantiles,
    e.g. :math:`\simeq 0.68` for :math:`1 \sigma`.
    """
    return stats.norm.cdf(nsigmas, loc=0, scale=1) - stats.norm.cdf(-nsigmas, loc=0, scale=1)


def nsigmas_to_quantiles_1d_sym(nsigmas):
    r"""
    Turn number of Gaussian sigmas ``nsigmas`` into lower and upper quantiles,
    e.g. :math:`\simeq 0.16, 0.84` for :math:`1 \sigma`.
    """
    total = nsigmas_to_quantiles_1d(nsigmas)
    out = (1. - total) / 2.
    return out, 1. - out


def nsigmas_to_deltachi2(nsigmas, ddof=1):
    r"""Turn number of Gaussian sigmas ``nsigmas`` into :math:`\chi^{2}` levels at ``ddof`` degrees of freedom."""
    quantile = nsigmas_to_quantiles_1d(nsigmas)
    return stats.chi2.ppf(quantile, ddof)  # inverse of cdf


def metrics_to_latex(name):
    """Turn metrics ``name`` to latex string."""
    toret = utils.txt_to_latex(name)
    for full, symbol in [('loglkl', 'L'), ('logposterior', '\\mathcal{L}'), ('logprior', 'p')]:
        toret = toret.replace(full, symbol)
    return toret


def interval(samples, weights, nsigmas=1., bins=100, method='gaussian_kde', bw_method='scott'):
    """
    Return n-sigmas confidence interval(s).

    Parameters
    ----------
    columns : list, ParameterCollection, default=None
        Parameters to compute confidence interval for.

    nsigmas : int
        Return interval for this number of sigmas.

    bins : int, default=100
        Number of bins i.e. mesh nodes.
        See :meth:`Mesh.from_samples`.

    method : string
        Method to interpolate (weighted) samples on mesh.
        See :meth:`Mesh.from_samples`.

    bw_method : string, default='scott'
        If ``method`` is ``'gaussian_kde'``, method to determine KDE bandwidth, see :class:`scipy.stats.gaussian_kde`.

    Returns
    -------
    interval : tuple
    """
    idx = np.argsort(samples)
    x = samples[idx]
    weights = weights[idx]
    nquantile = nsigmas_to_quantiles_1d(nsigmas)
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    cdfpq = cdf + nquantile
    ixmaxup = np.searchsorted(cdf, cdfpq, side='left')
    mask = ixmaxup < len(x)
    indices = np.array([np.flatnonzero(mask), ixmaxup[mask]])
    xmin, xmax = x[indices]
    argmin = np.argmin(xmax - xmin)
    return (xmin[argmin], xmax[argmin])


def std_notation(value, sigfigs, extra=None):
    """
    Standard notation (US version).
    Return a string corresponding to value with the number of significant digits ``sigfigs``.

    >>> std_notation(5, 2)
    '5.0'
    >>> std_notation(5.36, 2)
    '5.4'
    >>> std_notation(5360, 2)
    '5400'
    >>> std_notation(0.05363, 3)
    '0.0536'

    Created by William Rusnack:
      github.com/BebeSparkelSparkel
      linkedin.com/in/williamrusnack/
      williamrusnack@gmail.com
    """
    sig_digits, power, is_neg = _number_profile(value, sigfigs)
    if is_neg and all(d == '0' for d in sig_digits): is_neg = False

    return ('-' if is_neg else '') + _place_dot(sig_digits, power)


def sci_notation(value, sigfigs, filler='e'):
    """
    Scientific notation.

    Return a string corresponding to value with the number of significant digits ``sigfigs``,
    with 10s exponent filler ``filler`` placed between the decimal value and 10s exponent.

    >>> sci_notation(123, 1, 'e')
    '1e2'
    >>> sci_notation(123, 3, 'e')
    '1.23e2'
    >>> sci_notation(0.126, 2, 'e')
    '1.3e-1'

    Created by William Rusnack
      github.com/BebeSparkelSparkel
      linkedin.com/in/williamrusnack/
      williamrusnack@gmail.com
    """
    sig_digits, power, is_neg = _number_profile(value, sigfigs)
    if is_neg and all(d == '0' for d in sig_digits): is_neg = False

    dot_power = min(-(sigfigs - 1), 0)
    ten_power = power + sigfigs - 1
    return ('-' if is_neg else '') + _place_dot(sig_digits, dot_power) + filler + str(ten_power)


def _place_dot(digits, power):
    """
    Place dot in the correct spot, given by integer ``power`` (starting from the right of ``digits``)
    in the string ``digits``.
    If the dot is outside the range of the digits zeros will be added.

    >>> _place_dot('123', 2)
    '12300'
    >>> _place_dot('123', -2)
    '1.23'
    >>> _place_dot('123', 3)
    '0.123'
    >>> _place_dot('123', 5)
    '0.00123'

    Created by William Rusnack
      github.com/BebeSparkelSparkel
      linkedin.com/in/williamrusnack/
      williamrusnack@gmail.com
    """
    if power > 0: out = digits + '0' * power

    elif power < 0:
        power = abs(power)
        sigfigs = len(digits)

        if power < sigfigs:
            out = digits[:-power] + '.' + digits[-power:]

        else:
            out = '0.' + '0' * (power - sigfigs) + digits

    else:
        out = digits + ('.' if digits[-1] == '0' else '')

    return out


def _number_profile(value, sigfigs):
    """
    Return elements to turn number into string representation.

    Created by William Rusnack
      github.com/BebeSparkelSparkel
      linkedin.com/in/williamrusnack/
      williamrusnack@gmail.com

    Parameters
    ----------
    value : float
        Number.

    sigfigs : int
        Number of significant digits.

    Returns
    -------
    sig_digits : string
        Significant digits.

    power : int
        10s exponent to get the dot to the proper location in the significant digits

    is_neg : bool
        ``True`` if value is < 0 else ``False``
    """
    if value == 0:
        sig_digits = '0' * sigfigs
        power = -(1 - sigfigs)
        is_neg = False

    else:
        if value < 0:
            value = abs(value)
            is_neg = True
        else:
            is_neg = False

        power = -1 * math.floor(math.log10(value)) + sigfigs - 1
        sig_digits = str(int(round(abs(value) * 10.0**power)))

    return sig_digits, int(-power), is_neg


def round_measurement(x, u=0.1, v=None, sigfigs=2, notation='auto'):
    """
    Return string representation of input central value ``x`` with uncertainties ``u`` and ``v``.

    Parameters
    ----------
    x : float
        Central value.

    u : float, default=0.1
        Upper uncertainty on ``x`` (positive).

    v : float, default=None
        Lower uncertainty on ``v`` (negative).
        If ``None``, only returns string representation for ``x`` and ``u``.

    sigfigs : int, default=2
        Number of digits to keep for the uncertainties (hence fixing number of digits for ``x``).

    Returns
    -------
    xr : string
        String representation for central value ``x``.

    ur : string
        String representation for upper uncertainty ``u``.

    vr : string
        If ``v`` is not ``None``, string representation for lower uncertainty ``v``.
    """
    x, u = float(x), float(u)
    return_v = True
    if v is None:
        return_v = False
        v = -abs(u)
    else:
        v = float(v)
    logx = 0
    if x != 0.: logx = math.floor(math.log10(abs(x)))
    if u == 0.: logu = logx
    else: logu = math.floor(math.log10(abs(u)))
    if v == 0.: logv = logx
    else: logv = math.floor(math.log10(abs(v)))
    if x == 0.: logx = max(logu, logv)

    def round_notation(val, sigfigs, notation='auto', center=False):
        if notation == 'auto':
            # if 1e-3 < abs(val) < 1e3 or center and (1e-3 - abs(u) < abs(x) < 1e3 + abs(v)):
            if (1e-3 - abs(u) < abs(x) < 1e3 + abs(v)):
                notation = 'std'
            else:
                notation = 'sci'
        notation_dict = {'std': std_notation, 'sci': sci_notation}
        if notation in notation_dict:
            return notation_dict[notation](val, sigfigs=sigfigs)
        return notation(val, sigfigs=sigfigs)

    if logv > logu:
        xr = round_notation(x, sigfigs=logx - logu + sigfigs, notation=notation, center=True)
        ur = round_notation(u, sigfigs=sigfigs, notation=notation)
        vr = round_notation(v, sigfigs=logv - logu + sigfigs, notation=notation)
    else:
        xr = round_notation(x, sigfigs=logx - logv + sigfigs, notation=notation, center=True)
        ur = round_notation(u, sigfigs=logu - logv + sigfigs, notation=notation)
        vr = round_notation(v, sigfigs=sigfigs, notation=notation)

    if return_v: return xr, ur, vr
    return xr, ur

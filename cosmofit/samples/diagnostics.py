import logging

import numpy as np
from scipy import stats

from cosmofit import utils


logger = logging.getLogger('Diagnostics')
log_info = logger.info
log_warning = logger.warning


def gelman_rubin(chains, params=None, statistic='mean', method='eigen', return_matrices=False, check=True):
    """
    Estimate Gelman-Rubin statistics, which compares covariance of chain means to (mean of) intra-chain covariances.

    Parameters
    ----------
    chains : list
        List of :class:`Chain` instances.

    columns : list, ParameterCollection
        Parameters to compute Gelman-Rubin statistics for.
        Defaults to all parameters.

    statistic : string, callable, default='mean'
        If 'mean', compares covariance of chain means to (mean of) intra-chain covariances.
        Else, must be a callable taking :class:`Chain` instance and parameter list as input
        and returning array of values (one for each parameter).

    method : string, default='eigen'
        If 'eigen', return eigenvalues of covariance ratios, else diagonal.

    return_matrices : bool, default=True
        If ``True``, also return pair of covariance matrices.

    check : bool, default=True
        Whether to check for reliable inverse of intra-chain covariances.

    Reference
    ---------
    http://www.stat.columbia.edu/~gelman/research/published/brooksgelman2.pdf
    """
    if params is None: params = chains[0].params(varied=True)
    isscalar = not utils.is_sequence(params)

    if not utils.is_sequence(chains):
        raise ValueError('Provide a list of at least 2 chains to estimate Gelman-Rubin')
    nchains = len(chains)
    if nchains < 2:
        raise ValueError('{:d} chains provided; one needs at least 2 to estimate Gelman-Rubin'.format(nchains))

    if statistic == 'mean':

        def statistic(chain, params):
            return chain.mean(params)

    means = np.asarray([statistic(chain, params) for chain in chains])
    covs = np.asarray([chain.cov(params) for chain in chains])
    nsteps = np.asarray([chain.weight.sum() for chain in chains])
    # W = "within"
    Wn1 = np.average(covs, weights=nsteps, axis=0)
    Wn = np.average(((nsteps - 1.) / nsteps)[:, None, None] * covs, weights=nsteps, axis=0)
    # B = "between"
    # We don't weight with the number of samples in the chains here:
    # shorter chains will likely be outliers, and we want to notice them
    B = np.cov(means.T, ddof=1)
    V = Wn + (nchains + 1.) / nchains * B
    if method == 'eigen':
        # divide by stddev for numerical stability
        stddev = np.sqrt(np.diag(V).real)
        V = V / stddev[:, None] / stddev[None, :]
        invWn1 = utils.inv(Wn1 / stddev[:, None] / stddev[None, :], check=check)
        toret = np.linalg.eigvalsh(invWn1.dot(V))
    else:
        toret = np.diag(V) / np.diag(Wn1)
    if isscalar:
        toret = toret[0]
    if return_matrices:
        return toret, (V, Wn1)
    return toret


def autocorrelation(chains, params=None):
    """
    Return weighted autocorrelation.
    Adapted from https://github.com/dfm/emcee/blob/main/src/emcee/autocorr.py

    Parameters
    ----------
    params : string, Parameter
        Parameters to compute autocorrelation for.
        Defaults to all parameters.

    Returns
    -------
    autocorr : array
    """
    if not utils.is_sequence(chains):
        chains = [chains]

    if params is None: params = chains[0].params(varied=True)
    if utils.is_sequence(params):
        return np.array([autocorrelation(chains[0], param) for param in params])

    toret = 0
    for chain in chains:
        value = chain[params]
        weight = chain.weight
        x = (value - np.average(value, weights=weight)) * weight
        toret += _autocorrelation_1d(x)
    return toret / len(chains)


def integrated_autocorrelation_time(chains, params=None, min_corr=None, c=5, reliable=50, check=False):
    """
    Return integrated autocorrelation time.
    Adapted from https://github.com/dfm/emcee/blob/main/src/emcee/autocorr.py

    Parameters
    ----------
    chains : list
        List of :class:`Chain` instances.

    columns : list, ParameterCollection
        Parameters to compute integrated autocorrelation time for.

    min_corr : float, default=None
        Integrate starting from this lower autocorrelation threshold.
        If ``None``, use ``c``.

    c : float, int
        Step size for the window search.

    reliable : float, int, default=50
        Minimum ratio between the chain length and estimated autocorrelation time
        for it to be considered reliable.

    check : bool, default=False
        Whether to check for reliable estimate of autocorrelation time (based on ``reliable``).

    Returns
    -------
    iat : scalar, array
    """
    if not utils.is_sequence(chains):
        chains = [chains]

    if params is None: params = chains[0].params(varied=True)

    if utils.is_sequence(params):
        return np.array([integrated_autocorrelation_time(chains, param, min_corr=min_corr, c=c, reliable=reliable, check=check) for param in params])

    # Automated windowing procedure following Sokal (1989)
    def auto_window(taus, c):
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    size = chains[0].size
    for chain in chains:
        if chain.size != size:
            raise ValueError('Input chains must have same length')
    corr = autocorrelation(chains, params)
    toret = None
    if min_corr is not None:
        ix = np.argmin(corr > min_corr * corr[0])
        toret = 1 + 2 * np.sum(corr[1:ix])
    elif c is not None:
        taus = 2 * np.cumsum(corr) - 1  # 1 + 2 sum_{i=1}^{N} f_{i}
        window = auto_window(taus, c)
        toret = taus[window]
    else:
        raise ValueError('A criterion must be provided to stop integration of correlation time')
    if check and reliable * toret > size:
        msg = 'The chain is shorter than {:d} times the integrated autocorrelation time for {}. Use this estimate with caution and run a longer chain!\n'.format(reliable, params)
        msg += 'N/{:d} = {:.0f};\ntau: {}'.format(reliable, size / reliable, toret)
        log_warning(msg)
    return toret


def _autocorrelation_1d(x):
    """
    Estimate the normalized autocorrelation function.
    Taken from https://github.com/dfm/emcee/blob/main/src/emcee/autocorr.py

    Parameters
    ----------
    x : array
        1D time series.

    Returns
    -------
    acf : array
        The autocorrelation function of the time series.
    """
    from numpy import fft
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError('invalid dimensions for 1D autocorrelation function')

    n = 2**(2 * len(x) - 1).bit_length()

    # Compute the FFT and then (from that) the auto-correlation function
    f = fft.fft(x, n=n)
    acf = fft.ifft(f * np.conjugate(f))[:len(x)].real

    acf /= acf[0]
    return acf


def geweke(chains, params=None, first=0.25, last=0.75):

    if not utils.is_sequence(chains):
        chains = [chains]

    if params is None: params = chains[0].params(varied=True)
    if utils.is_sequence(params):
        return np.array([geweke(chains[0], parameter, first=first, last=last) for parameter in params])

    toret = []
    for chain in chains:
        value, aweight, fweight = chain[params].ravel(), chain.aweight.ravel(), chain.fweight.ravel()
        ifirst, ilast = int(first * value.size + 0.5), int(last * value.size + 0.5)
        value_first, value_last = value[:ifirst], value[ilast:]
        aweight_first, aweight_last = aweight[:ifirst], aweight[ilast:]
        fweight_first, fweight_last = fweight[:ifirst], fweight[ilast:]
        diff = abs(np.average(value_first, weights=aweight_first * fweight_first) - np.average(value_last, weights=aweight_last * fweight_last))
        diff /= (np.cov(value_first, aweights=aweight_first, fweights=fweight_first) + np.cov(value_last, aweights=aweight_last, fweights=fweight_last))**0.5
        toret.append(diff)

    return stats.normaltest(toret)

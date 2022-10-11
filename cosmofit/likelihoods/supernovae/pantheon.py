import numpy as np
import scipy as sp

from cosmofit import plotting
from .base import SNLikelihood


class PantheonSNLikelihood(SNLikelihood):

    """Pantheon type Ia supernova sample."""

    def __init__(self, *args, **kwargs):
        super(PantheonSNLikelihood, self).__init__(*args, **kwargs)
        # Add statistical error
        self.covariance += np.diag(self.light_curve_params['dmb']**2)
        self.std = np.diag(self.covariance)**0.5
        self.covariance = sp.linalg.cholesky(self.covariance, lower=True, overwrite_a=True)

    def run(self, Mb=0):
        z = self.light_curve_params['zcmb']
        self.flatmodel = 5 * np.log10(self.cosmo.luminosity_distance(z)) + 25
        self.flatdata = self.light_curve_params['mb'] - Mb - 5 * np.log10((1 + self.light_curve_params['zhel']) / (1 + z))
        diff = self.flatdata - self.flatmodel
        # Solve the triangular system, also time expensive (0.02 seconds)
        diff = sp.linalg.solve_triangular(self.covariance, diff, lower=True, check_finite=False)
        # Finally, compute the chi2 as the sum of the squared residuals
        self.loglikelihood = -0.5 * np.sum(diff**2)

    def plot(self, fn, kw_save=None):
        from matplotlib import pyplot as plt
        fig, lax = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'height_ratios': (3, 1)}, figsize=(6, 6), squeeze=True)
        fig.subplots_adjust(hspace=0)
        alpha = 0.3
        argsort = np.argsort(self.light_curve_params['zcmb'])
        zdata = self.light_curve_params['zcmb'][argsort]
        flatdata, flatmodel, std = self.flatdata[argsort], self.flatmodel[argsort], self.std[argsort]
        lax[0].plot(zdata, flatdata, marker='o', markeredgewidth=0., linestyle='none', alpha=alpha, color='b')
        lax[0].plot(zdata, flatmodel, linestyle='-', marker=None, color='k')
        lax[0].set_xscale('log')
        lax[1].errorbar(zdata, flatdata - flatmodel, yerr=std, linestyle='none', marker='o', alpha=alpha, color='b')
        lax[0].set_ylabel(r'distance modulus [$\mathrm{mag}$]')
        lax[1].set_ylabel(r'Hubble res. [$\mathrm{mag}$]')
        lax[1].set_xlabel('$z$')
        if fn is not None:
            plotting.savefig(fn, fig=fig, **(kw_save or {}))
        return lax

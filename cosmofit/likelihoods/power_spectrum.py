import glob

import numpy as np

from .base import BaseGaussianLikelihood
from cosmofit import plotting, utils


class PowerSpectrumMultipolesLikelihood(BaseGaussianLikelihood):

    def __init__(self, covariance=None, data=None, klim=None, kstep=None, krebin=None, zeff=None, fiducial=None, wmatrix=None):

        def load_data(fn):
            from pypower import PowerSpectrumStatistics
            return PowerSpectrumStatistics.load(fn)

        def lim_data(power, klim=klim, kstep=kstep, krebin=krebin):
            if krebin is None:
                krebin = 1
                if kstep is not None:
                    krebin = int(np.rint(kstep / np.diff(power.kedges).mean()))
            power = power[:(power.shape[0] // krebin) * krebin:krebin]
            data = power.get_power(complex=False)
            nells = len(power.ells)
            assert len(data) == nells
            if klim is None:
                return [np.array(power.k)] * nells, tuple(power.ells), list(data)
            if utils.is_sequence(klim):
                if not utils.is_sequence(klim[0]):
                    klim = [klim] * nells
                if len(klim) > nells:
                    raise ValueError('{:d} limits provided but only {:d} poles computed'.format(len(klim), nells))
                klim = {ell: klim[ill] for ill, ell in enumerate(power.ells)}
            if isinstance(klim, dict):
                list_k, list_data, ells = [], [], []
                for ell, lim in klim.items():
                    mask = (power.k >= lim[0]) & (power.k < lim[1])
                    list_k.append(power.k[mask])
                    list_data.append(data[power.ells.index(ell)][mask])
                    ells.append(ell)
                return list_k, tuple(ells), list_data
            raise ValueError('Unknown klim format; provide e.g. {0: (0.01, 0.2), 2: (0.01, 0.15)}')

        self.k, poles, nobs = None, None, None

        if data is not None:
            if isinstance(data, str):
                data = load_data(data)
            self.k, self.ells, poles = lim_data(data)

        if isinstance(covariance, str):
            covariance = [covariance]

        if utils.is_sequence(covariance) and isinstance(covariance[0], str):
            if self.mpicomm.rank == 0:
                list_data = []
                for fn in covariance:
                    for fn in sorted(glob.glob(fn)):
                        k, ells, data = lim_data(load_data(fn))
                        if self.k is None:
                            self.k, self.ells = k, ells
                        if not all(np.allclose(kk, skk) for kk, skk in zip(self.k, k)):
                            raise ValueError('{} does not have expected k-binning (based on previous data)'.format(fn))
                        if ells != self.ells:
                            raise ValueError('{} does not have expected poles (based on previous data)'.format(fn))
                        list_data.append(np.ravel(data))
                nobs = len(list_data)
                covariance = np.cov(list_data, rowvar=False, ddof=1)
            covariance = self.mpicomm.bcast(covariance if self.mpicomm.rank == 0 else None, root=0)

        super(PowerSpectrumMultipolesLikelihood, self).__init__(covariance=covariance, data=np.concatenate(poles, axis=0) if poles is not None else None, nobs=nobs)
        self.requires['theory'] = ('WindowedPowerSpectrumMultipoles', {'k': self.k, 'ellsout': self.ells, 'zeff': zeff, 'fiducial': fiducial, 'wmatrix': wmatrix})

    def plot(self, fn=None, kw_save=None):
        from matplotlib import pyplot as plt
        height_ratios = [max(len(self.ells), 3)] + [1] * len(self.ells)
        figsize = (6, 1.5 * sum(height_ratios))
        fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
        fig.subplots_adjust(hspace=0)
        data, model, std = self.data, self.model, self.std
        for ill, ell in enumerate(self.ells):
            lax[0].errorbar(self.k[ill], self.k[ill] * data[ill], yerr=self.k[ill] * std[ill], color='C{:d}'.format(ill), label=r'$\ell = {:d}$'.format(ell))
        for ill, ell in enumerate(self.ells):
            lax[0].plot(self.k[ill], self.k[ill] * model[ill], color='C{:d}'.format(ill))
        for ill, ell in enumerate(self.ells):
            lax[ill + 1].plot(self.k[ill], (data[ill] - model[ill]) / std[ill], color='C{:d}'.format(ill))
            lax[ill + 1].set_ylim(-4, 4)
            for offset in [-2., 2.]: lax[ill + 1].axhline(offset, color='k', linestyle='--')
            lax[ill + 1].set_ylabel(r'$\Delta P_{{{0:d}}} / \sigma_{{ P_{{{0:d}}} }}$'.format(ell))
        for ax in lax: ax.grid(True)
        lax[0].legend()
        lax[0].set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        lax[-1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        if fn is not None:
            plotting.savefig(fn, fig=fig, **(kw_save or {}))
        return lax

    def unpack(self, array):
        toret = []
        nout = 0
        for k in self.k:
            sl = slice(nout, nout + len(k))
            toret.append(array[sl])
            nout = sl.stop
        return toret

    @property
    def flatmodel(self):
        return self.theory.flatpower

    @property
    def model(self):
        return self.theory.power

    @property
    def data(self):
        return self.unpack(self.flatdata)

    @property
    def std(self):
        return self.unpack(np.diag(self.covariance) ** 0.5)

    def __getstate__(self):
        state = super(PowerSpectrumMultipolesLikelihood, self).__getstate__()
        for name in ['k', 'ells']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

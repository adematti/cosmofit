import glob

import numpy as np

from .base import BaseGaussianLikelihood
from cosmofit import plotting, utils


class CorrelationFunctionMultipolesLikelihood(BaseGaussianLikelihood):

    def __init__(self, covariance=None, data=None, slim=None, sstep=None, srebin=None, zeff=None, fiducial=None):

        def load_data(fn):
            from pycorr import TwoPointCorrelationFunction
            return TwoPointCorrelationFunction.load(fn)

        def lim_data(corr, slim=slim, sstep=sstep, srebin=srebin):
            if srebin is None:
                srebin = 1
                if sstep is not None:
                    srebin = int(np.rint(sstep / np.diff(corr.edges[0]).mean()))
            corr = corr[:(corr.shape[0] // srebin) * srebin:srebin]
            if slim is None:
                slim = {ell: [0, np.inf] for ell in (0, 2, 4)}
            if utils.is_sequence(slim):
                if not utils.is_sequence(slim[0]):
                    slim = [slim] * nells
                slim = {ell: slim[ill] for ill, ell in enumerate(corr.ells)}
            elif not isinstance(slim, dict):
                raise ValueError('Unknown slim format; provide e.g. {0: (0.01, 0.2), 2: (0.01, 0.15)}')
            ells = tuple(slim.keys())
            s, data = corr(ells=ells, return_sep=True, return_std=False)
            list_s, list_data = [], []
            for ell, lim in slim.items():
                mask = (s >= lim[0]) & (s < lim[1])
                list_s.append(s[mask])
                list_data.append(data[ells.index(ell)][mask])
            return list_s, ells, list_data

        self.s, self.ells, flatdata, nobs = None, None, None, None
        data_is_mean = data == 'mean'
        if isinstance(covariance, str):
            covariance = [covariance]
        has_mocks = utils.is_sequence(covariance) and isinstance(covariance[0], str)

        if data_is_mean:
            if not has_mocks:
                raise ValueError('data is mean of mocks, but no mocks provided')
        elif data is not None:
            if self.mpicomm.rank == 0:
                if isinstance(data, str):
                    data = load_data(data)
                self.s, self.ells, flatdata = lim_data(data)
                flatdata = np.ravel(flatdata)

        if has_mocks:
            if self.mpicomm.rank == 0:
                list_mock = []
                for mocks in covariance:
                    if isinstance(mocks, str):
                        mocks = [load_data(mock) for mock in glob.glob(mocks)]
                    else:
                        mocks = [mocks]
                    for mock in mocks:
                        mock_s, mock_ells, mock = lim_data(mock)
                        if self.s is None:
                            self.s, self.ells = mock_s, mock_ells
                        if not all(np.allclose(ss, ms) for ss, ms in zip(self.s, mock_s)):
                            raise ValueError('{} does not have expected s-binning (based on previous data)'.format(fn))
                        if mock_ells != self.ells:
                            raise ValueError('{} does not have expected poles (based on previous data)'.format(fn))
                        list_mock.append(np.ravel(mock))
                nobs = len(list_mock)
                if data_is_mean:
                    flatdata = np.mean(list_mock, axis=0)
                covariance = np.cov(list_mock, rowvar=False, ddof=1)
            covariance = self.mpicomm.bcast(covariance if self.mpicomm.rank == 0 else None, root=0)

        self.s, self.ells, flatdata, nobs = self.mpicomm.bcast((self.s, self.ells, flatdata, nobs) if self.mpicomm.rank == 0 else None, root=0)

        super(CorrelationFunctionMultipolesLikelihood, self).__init__(covariance=covariance, data=flatdata, nobs=nobs)
        self.requires['theory'] = ('cosmofit.theories.base.WindowedCorrelationFunctionMultipoles',
                                   {'s': self.s, 'ells': self.ells, 'theory': {'init': {'zeff': zeff, 'fiducial': fiducial}}})

    def plot(self, fn=None, kw_save=None):
        from matplotlib import pyplot as plt
        height_ratios = [max(len(self.ells), 3)] + [1] * len(self.ells)
        figsize = (6, 1.5 * sum(height_ratios))
        fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
        fig.subplots_adjust(hspace=0)
        data, model, std = self.data, self.model, self.std
        for ill, ell in enumerate(self.ells):
            lax[0].errorbar(self.s[ill], self.s[ill]**2 * data[ill], yerr=self.s[ill]**2 * std[ill], color='C{:d}'.format(ill), linestyle='none', marker='o', label=r'$\ell = {:d}$'.format(ell))
            lax[0].plot(self.s[ill], self.s[ill]**2 * model[ill], color='C{:d}'.format(ill))
        for ill, ell in enumerate(self.ells):
            lax[ill + 1].plot(self.s[ill], (data[ill] - model[ill]) / std[ill], color='C{:d}'.format(ill))
            lax[ill + 1].set_ylim(-4, 4)
            for offset in [-2., 2.]: lax[ill + 1].axhline(offset, color='k', linestyle='--')
            lax[ill + 1].set_ylabel(r'$\Delta \xi_{{{0:d}}} / \sigma_{{ \xi_{{{0:d}}} }}$'.format(ell))
        for ax in lax: ax.grid(True)
        lax[0].legend()
        lax[0].set_ylabel(r'$s^{2} \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        lax[-1].set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
        if fn is not None:
            plotting.savefig(fn, fig=fig, **(kw_save or {}))
        return lax

    def unpack(self, array):
        toret = []
        nout = 0
        for s in self.s:
            sl = slice(nout, nout + len(s))
            toret.append(array[sl])
            nout = sl.stop
        return toret

    def flatmodel(self):
        return self.theory.flatcorr

    @property
    def model(self):
        return self.theory.corr

    @property
    def data(self):
        return self.unpack(self.flatdata)

    @property
    def std(self):
        return self.unpack(np.diag(self.covariance) ** 0.5)

    def __getstate__(self):
        state = super(CorrelationFunctionMultipolesLikelihood, self).__getstate__()
        for name in ['s', 'ells']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

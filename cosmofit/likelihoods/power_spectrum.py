import glob

import numpy as np

from .base import BaseGaussianLikelihood
from cosmofit import utils


class PowerSpectrumMultipolesLikelihood(BaseGaussianLikelihood):

    def __init__(self, covariance=None, data=None, klim=None, zeff=None, fiducial=None, wmatrix=None):

        def load_data(fn):
            from pypower import PowerSpectrumStatistics
            return PowerSpectrumStatistics.load(fn)

        def lim_data(power, klim=klim):
            data = power.get_power(complex=False)
            nells = len(power.ells)
            assert len(data) == nells
            if klim is None:
                return np.array(power.k), np.ones_like(power.k, dtype='?'), tuple(power.ells), np.ravel(data)
            if utils.is_sequence(klim):
                if not utils.is_sequence(klim[0]):
                    klim = [klim] * nells
                if len(klim) > nells:
                    raise ValueError('{:d} limits provided but only {:d} poles computed'.format(len(klim), nells))
                klim = {ell: klim[ill] for ill, ell in enumerate(power.ells)}
            if isinstance(klim, dict):
                list_mask, list_data, ells = [], [], []
                for ell, lim in klim.items():
                    ill = power.ells.index(ell)
                    mask = (power.k >= lim[0]) & (power.k < lim[1])
                    if np.any(mask):
                        list_mask.append(mask)
                        list_data.append(data[ill][mask])
                        ells.append(ell)
                common_mask = np.any(list_mask, axis=0)
                mask = np.concatenate([mask[common_mask] for mask in list_mask], axis=0)
                k = power.k[common_mask]
                return k, mask, tuple(ells), np.concatenate(list_data, axis=0)
            raise ValueError('Unknown klim format; provide e.g. {0: (0.01, 0.2), 2: (0.01, 0.15)}')

        self.k, self.kmask, nobs = None, None, None

        if data is not None:
            if isinstance(data, str):
                data = load_data(data)
            self.k, self.kmask, self.ells, data = lim_data(data)

        if isinstance(covariance, str):
            covariance = [covariance]

        if utils.is_sequence(covariance) and isinstance(covariance[0], str):
            if self.mpicomm.rank == 0:
                list_data = []
                for fn in covariance:
                    for fn in sorted(glob.glob(fn)):
                        k, kmask, ells, data = lim_data(load_data(fn))
                        if self.k is None:
                            self.k, self.kmask, self.ells = k, kmask, ells
                        if not np.allclose(np.repeat(k, len(ells))[kmask], np.repeat(self.k, len(ells))[self.kmask]):
                            raise ValueError('{} does not have expected k-binning (based on previous data)'.format(fn))
                        if ells != self.ells:
                            raise ValueError('{} does not have expected poles (based on previous data)'.format(fn))
                        list_data.append(data)
                nobs = len(list_data)
                covariance = np.cov(list_data, rowvar=False, ddof=1)
            covariance = self.mpicomm.bcast(covariance if self.mpicomm.rank == 0 else None, root=0)

        super(PowerSpectrumMultipolesLikelihood, self).__init__(covariance=covariance, data=data, nobs=nobs)
        self.requires['power'] = ('WindowedPowerSpectrumMultipoles', {'kout': self.k, 'ellsout': self.ells, 'zeff': zeff, 'fiducial': fiducial, 'wmatrix': wmatrix})

    @property
    def model(self):
        return self.power.power

    def __getstate__(self):
        state = {}
        for name in ['k', 'kmask', 'data', 'covariance', 'precision', 'loglikelihood']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

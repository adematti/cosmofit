import numpy as np

from cosmofit.base import BaseCalculator
from cosmofit import utils


class GaussianSyntheticDataGenerator(BaseCalculator):

    def __init__(self, covariance, seed=None):
        self.covariance = np.atleast_2d(covariance)
        if self.covariance.shape != (self.covariance.shape[0],) * 2:
            raise ValueError('Covariance must be a square matrix')
        self.seed = seed
        if self.seed is not None:
            self.rng = np.random.RandomState(seed=self.seed)
        self.zeros = np.zeros(self.covariance.shape[0], dtype='f8')

    def run(self):
        if self.seed is not None:
            self.flatdata = self.mpicomm.bcast(self.rng.multivariate_normal(self.zeros, self.covariance), root=0)
        self.flatdata = self.zeros.copy()


class BaseGaussianLikelihood(BaseCalculator):

    def __init__(self, covariance, data=None, nobs=None):
        self.covariance = np.atleast_2d(covariance)
        if self.covariance.shape != (self.covariance.shape[0],) * 2:
            raise ValueError('Covariance must be a square matrix')
        self.flatdata = data
        if data is not None:
            self.flatdata = np.ravel(data)
            if self.covariance.shape != (self.flatdata.size,) * 2:
                raise ValueError('Based on provided data, covariance expected to be a matrix of shape ({0:d}, {0:d})'.format(self.flatdata.size))
        self.precision = utils.inv(self.covariance)
        self.nobs = nobs
        if nobs is not None:
            self.nobs = int(nobs)
            size = self.precision.shape[0]
            self.hartlap = (self.nobs - size - 2.) / (self.nobs - 1.)
            if self.mpicomm.rank == 0:
                self.log_info('Covariance matrix with {:d} points built from {:d} observations.'.format(size, self.nobs))
                self.log_info('...resulting in Hartlap factor of {:.4f}.'.format(self.hartlap))
            self.precision *= self.hartlap

        self.requires = {}
        if self.flatdata is None:
            self.requires = {'synthetic': ('GaussianSyntheticDataGenerator', {'covariance': self.covariance})}

    def run(self):
        flatmodel = self.flatmodel
        if self.flatdata is None:
            if self.mpicomm.rank == 0:
                self.log_info('Using synthetic data.')
            self.flatdata = self.synthetic.flatdata + flatmodel
        diff = self.flatdata - flatmodel
        self.loglikelihood = -0.5 * diff.dot(self.precision).T.dot(diff)

    def __getstate__(self):
        state = {}
        for name in ['flatdata', 'covariance', 'precision', 'loglikelihood']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

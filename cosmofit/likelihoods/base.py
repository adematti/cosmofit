import numpy as np

from cosmofit.base import BaseCalculator
from cosmofit.samples import load_samples
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

    def __init__(self, covariance, data=None, nobs=None, project=None):
        self.covariance = np.atleast_2d(covariance)
        if self.covariance.shape != (self.covariance.shape[0],) * 2:
            raise ValueError('Covariance must be a square matrix')
        self.flatdata = data
        if data is not None:
            self.flatdata = np.ravel(data)
            if self.covariance.shape != (self.flatdata.size,) * 2:
                raise ValueError('Based on provided data, covariance expected to be a matrix of shape ({0:d}, {0:d})'.format(self.flatdata.size))
        if project is not None:
            if not isinstance(project, dict):
                self.eigenvectors = np.array(project)
            else:
                samples = load_samples(source='samples', fn=project['samples'], burnin=project.get('burnin', None))
                samples = samples[0].concatenate(samples)
                method = project.get('fiducial', 'diag')
                if method == 'diag':
                    fiducial = np.diag(np.diag(self.covariance))
                elif method == 'full':
                    fiducial = self.covariance
                else:
                    raise ValueError('fiducial must be one of ["diag", "full"]')
                precision = utils.inv(fiducial)
                self.eigenvectors = utils.subspace(samples[project.get('name', 'flatmodel')], precision=precision, chi2min=project.get('chi2min', 0.1))
            self.precision = utils.inv(self.eigenvectors.T.dot(self.covariance).dot(self.eigenvectors))
        else:
            self.eigenvectors = 1.
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
        if self.flatdata is None:
            if self.mpicomm.rank == 0:
                self.log_info('Using synthetic data.')
            self.flatdata = self.synthetic.flatdata + self.flatmodel
        flatdiff = self.flatdiff
        self.loglikelihood = -0.5 * flatdiff.dot(self.precision).dot(flatdiff)

    @property
    def flatdiff(self):
        return (self.flatmodel - self.flatdata).dot(self.eigenvectors)

    def __getstate__(self):
        state = {}
        for name in ['flatdata', 'covariance', 'eigenvectors', 'precision', 'loglikelihood']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

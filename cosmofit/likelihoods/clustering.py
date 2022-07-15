import glob
import numpy as np

from .base import BaseGaussianLikelihood


class FullParameterizationLikelihood(BaseGaussianLikelihood):

    def __init__(self, chains, select=None, burnin=None):
        if isinstance(chains, str):
            chains = [chains]
        if utils.is_sequence(chains) and isinstance(chains[0], str)
            chains = [Chain.load(fn) for fn in glob.glob(ff) for ff in chains]
        if burnin is not None:
            chains = [chain.remove_burnin(burnin) for chain in chains]
        chain = Chain.concatenate(chains)
        params = chain.params()
        if select is not None:
            params = params.select(**select)
        self.params = params
        mean = np.array([chain.mean(param) for param in self.params])
        covariance = chain.cov(self.params)
        super(FullParameterizationLikelihood, self).__init__(covariance=covariance, data=mean)
        self.requires = {'cosmo': ('BasePrimordialCosmology', {})}

    def _update_params(self, params):
        indices = [self.params.index(param) for param in params]
        super(FullParameterizationLikelihood, self).__init__(covariance=self.covariance[np.ix_(indices, indices)], data=self.flatdata[indices])

    @property
    def flatmodel(self):
        params, values = [], []
        for param in self.params:
            value = getattr(self.cosmo, param.basename)
            if value is not None:
                params.append(param)
                values.append(value)
        if len(params) != len(self.params):
            self._update_params(params)
        return np.array(values, dtype='f8')

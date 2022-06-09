import numpy as np

from .base import BaseEmulatorEngine
from .qmc_engine import get_qmc_engine


class MLPEmulatorEngine(BaseEmulatorEngine):

    def __init__(self, pipeline, qmc_engine, nsamples=100, **kwargs):
        self.qmc_engine = qmc_engine
        super(MLPEmulatorEngine, self).__init__(pipeline=pipeline, **kwargs)

    def sample(self):
        samples = self.qmc_engine.random(n=self.nsamples)
        self.varied_X = self.qmc_engine.scale(samples, [self.limits[param][0] for param in self.centers], [self.limits[param][1] for param in self.centers])
        self.varied_Y, self.fixed_Y = self.mpirun_pipeline(**{str(param): value for param, values in zip(self.varied_params, self.varied_X)})
    
    def fit(self):
        pass

    def predict(self, **params):
        diff = [params[param] - self.centers[param] for param in self.varied_names]
        ndim = len(diff)
        toret = self.fixed_Y.copy()
        for name in self.derivatives:
            toret[name] = self.derivatives[name][0].copy()
            prefactor = 1
            for order in range(1, self.order + 1):
                prefactor *= order
                for ideriv, indices in enumerate(itertools.product(np.arange(ndim), repeat=order)):
                    toret[name] += 1. / prefactor * self.derivatives[name][order][ideriv] * np.prod([diff[ii] for ii in indices])
        return toret

    def __getstate__(self):
        state = super(MLPEmulatorEngine, self).__getstate__()
        for name in ['derivatives', 'order']:
            state[name] = getattr(self, name)
        return state
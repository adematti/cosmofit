import numpy as np

from .base import BaseEmulatorEngine
from .qmc_engine import get_qmc_engine


class MLPEmulatorEngine(BaseEmulatorEngine):

    def __init__(self, pipeline, qmc_engine, **kwargs):
        self.qmc_engine = qmc_engine
        super(MLPEmulatorEngine, self).__init__(pipeline=pipeline, **kwargs)

    def sample(self, nsamples=100):
        self.samples = self.qmc_engine.random(n=nsamples)
        self.samples = self.mpirun_pipeline(self.qmc_engine.scale(self.samples, [self.limits[param][0] for param in self.centers], [self.limits[param][1] for param in self.centers]))

    def fit(self):
        pass

    def predict(self, **params):
        pass

    def __getstate__(self):
        state = super(MLPEmulatorEngine, self).__getstate__()
        for name in []:
            state[name] = getattr(self, name)
        return state

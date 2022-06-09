import itertools

import numpy as np
from findiff import FinDiff

from .base import BaseEmulatorEngine


class TaylorEmulatorEngine(BaseEmulatorEngine):

    def __init__(self, pipeline, order=4, step_frac=1e-2, **kwargs):
        self.order = int(order)
        self.step_frac = float(step_frac)
        super(TaylorEmulatorEngine, self).__init__(pipeline=pipeline, **kwargs)
    
    def sample(self):
        delta = [(self.limits[param][1] - self.limits[param][0]) / 2. * self.step_frac for param in self.centers]
        grid = [self.centers[param] + delta[iparam] * np.arange(-self.order, self.order + 1) for iparam, param in enumerate(self.centers)]
        try:
            grid = [value.ravel() for value in np.meshgrid(*grid, indexing='ij')]
        except np.core._exceptions._ArrayMemoryError as exc:
            raise ValueError('Memory error: try decreasing the number of varied parameters or the order of the Taylor expansion') from exc
        self.varied_X = grid
        self.varied_Y, self.fixed_Y = self.mpirun_pipeline(**{str(param): values for param, values in zip(self.varied, self.varied_X)})

    def fit(self):
        shape = (2 * self.order + 1,) * len(self.varied_X) 
        ndim = len(shape)
        self.derivatives = {}
        for name, values in self.varied_Y.items():
            values = values.reshape(shape + values.shape[1:])
            center_index = (self.order,) * ndim
            self.derivatives[name] = [values[center_index]]  # F(x=center)
            for order in range(1, self.order + 1):
                deriv = []
                for indices in itertools.product(np.arange(ndim), repeat=order):
                    dx = FinDiff(*[(ii, delta[ii], 1) for ii in indices])
                    deriv.append(dx(values)[center_index])
                self.derivatives[name].append(deriv)

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
        state = super(TaylorEmulatorEngine, self).__getstate__()
        for name in ['derivatives', 'order']:
            state[name] = getattr(self, name)
        return state
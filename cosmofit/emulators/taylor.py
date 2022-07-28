import itertools

import numpy as np

from .base import BaseEmulatorEngine


class TaylorEmulatorEngine(BaseEmulatorEngine):

    def __init__(self, pipeline, order=4, **kwargs):
        self.order = int(order)
        super(TaylorEmulatorEngine, self).__init__(pipeline=pipeline, **kwargs)

    def get_default_samples(self, scale=1e-2):
        from cosmofit.samplers import GridSampler
        sampler = GridSampler(self.pipeline, ngrid=2 * self.order + 1, scale=scale)
        sampler.run()
        return sampler.samples

    def fit(self):
        from findiff import FinDiff
        self.centers, self.derivatives = {}, {}
        if self.mpicomm.rank == 0:
            ndim = len(self.varied_params)
            shape = tuple(self.samples.attrs.get('ngrid', self.samples.shape))
            assert len(shape) == ndim
            center_index = tuple([s // 2 for s in shape])
            axes, delta = [], []
            for name in self.varied_params:
                values = self.samples[name]
                values = values.reshape(shape)
                for axis in range(len(shape)):
                    if shape[axis] > 1:
                        dd = (np.take(values, 1, axis=axis) - np.take(values, 0, axis=axis)).flat[0]
                        found = dd > 0
                        if found:
                            delta.append(dd)
                            axes.append(axis)
                            break
                if found and shape[axis] < 2 * self.order + 1:
                    raise ValueError('Grid is not large enough ({:d}) for parameter {} (axis {:d}) to estimate {:d}-nth order derivative'.format(shape[axis], name, axis, self.order))
                if not found and self.order > 0:
                    raise ValueError('Parameter {} has not been sampled, hence impossible to estimate {:d}-nth order derivative'.format(name, self.order))
                self.centers[name] = values[center_index]

            for name in self.varied:
                values = self.samples[name]
                values = values.reshape(shape + values.shape[len(self.samples.shape):])
                self.derivatives[name] = [values[center_index]]  # F(x=center)
                for order in range(1, self.order + 1):
                    deriv = []
                    for indices in itertools.product(np.arange(ndim), repeat=order):
                        dx = FinDiff(*[(axes[ii], delta[ii], 1) for ii in indices])
                        deriv.append(dx(values)[center_index])
                    self.derivatives[name].append(deriv)
                    #if order == 1: print(name, self.varied_params, deriv)

        self.derivatives = self.mpicomm.bcast(self.derivatives, root=0)
        self.centers = self.mpicomm.bcast(self.centers, root=0)

    def predict(self, **params):
        diff = [params[param] - self.centers[param] for param in self.varied_params]
        ndim = len(diff)
        toret = {}
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
        for name in ['centers', 'derivatives', 'order']:
            state[name] = getattr(self, name)
        return state

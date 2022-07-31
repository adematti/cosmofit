import itertools

import numpy as np
import mpytools as mpy

from .base import BaseEmulatorEngine


class TaylorEmulatorEngine(BaseEmulatorEngine):

    def __init__(self, pipeline, order=4, **kwargs):
        super(TaylorEmulatorEngine, self).__init__(pipeline=pipeline, **kwargs)
        if not isinstance(order, dict):
            order = {'*': order}
        self.order = {str(param): None for param in self.varied_params}
        for name, value in order.items():
            for tmpname in self.params.names(name=name, varied=True, derived=False):
                self.order[tmpname] = int(value)
        for name, value in self.order.items():
            if value is None:
                raise ValueError('order not specified for parameter {}'.format(name))
            elif value < 0:
                raise ValueError('order is {:d} < 0 for parameter {}'.format(value, name))

    def get_default_samples(self, scale=1e-2):
        from cosmofit.samplers import GridSampler
        # We are sampling a hypercube though only require a hypersphere
        sampler = GridSampler(self.pipeline, ngrid={name: 2 * self.order[name] + 1 for name in self.order}, scale=scale)
        sampler.run()
        return sampler.samples

    def fit(self):
        from findiff import FinDiff
        self.centers, self.derivatives, self.powers = {}, {}, {}
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
                if found and shape[axis] < 2 * self.order[name] + 1:
                    raise ValueError('Grid is not large enough ({:d}) for parameter {} (axis {:d}) to estimate {:d}-nth order derivative'.format(shape[axis], name, axis, self.order[name]))
                if not found and self.order[name] > 0:
                    raise ValueError('Parameter {} has not been sampled, hence impossible to estimate {:d}-nth order derivative'.format(name, self.order[name]))
                self.centers[name] = values[center_index]

            for name in self.varied:
                values = self.samples[name]
                values = values.reshape(shape + values.shape[len(self.samples.shape):])
                self.powers[name] = [[0,] * ndim]
                self.derivatives[name] = [values[center_index]]  # F(x=center)
                prefactor = 1.
                for order in range(1, max(self.order.values()) + 1):
                    prefactor /= order
                    for indices in itertools.product(range(ndim), repeat=order):
                        powers = list(np.bincount(indices, minlength=ndim))
                        if any(powers[ii] > self.order[name] for ii, name in enumerate(self.varied_params)):
                            continue # The power-th derivative is zero
                        dx = FinDiff(*[(axes[ii], delta[ii], power) for ii, power in enumerate(powers) if power > 0])
                        self.powers[name].append(powers)
                        self.derivatives[name].append(prefactor * dx(values)[center_index])

        self.derivatives = {name: mpy.bcast(self.derivatives[name] if self.mpicomm.rank == 0 else None, mpicomm=self.mpicomm, mpiroot=0) for name in self.varied}
        self.powers = self.mpicomm.bcast(self.powers, root=0)
        self.centers = self.mpicomm.bcast(self.centers, root=0)

    def predict(self, **params):
        diffs = [params[param] - self.centers[param] for param in self.varied_params]
        toret = {}
        for name in self.derivatives:
            toret[name] = sum(self.derivatives[name][ii] * np.prod([d**p if p != 0 else 1. for d, p in zip(diffs, powers)]) for ii, powers in enumerate(self.powers[name]))
        return toret

    def __getstate__(self):
        state = super(TaylorEmulatorEngine, self).__getstate__()
        for name in ['centers', 'derivatives', 'powers', 'order']:
            state[name] = getattr(self, name)
        return state

import itertools

import numpy as np
import mpytools as mpy

from cosmofit.utils import jnp
from .base import BaseEmulator


def deriv_ncoeffs(order, acc=2):
    return 2 * ((order + 1) // 2) - 1 + acc


def coefficients(order, acc, coords, idx):
    """
    Calculates the finite difference coefficients for given derivative order and accuracy order.
    Assumes that the underlying grid is non-uniform.

    Taken from https://github.com/maroba/findiff/blob/master/findiff/coefs.py

    Parameters
    ----------

    order : int
        The derivative order (positive integer).

    acc : int
        The accuracy order (even positive integer).

    coords : np.ndarray
        The coordinates of the axis for the partial derivative.

    idx : int
        Index of the grid position where to calculate the coefficients.

    Returns
    -------
    coeffs, offsets
    """
    import math

    if acc % 2 or acc <= 0:
        raise ValueError('Accuracy order acc must be positive EVEN integer')

    if order < 0:
        raise ValueError('Derive degree must be positive integer')

    order, acc = int(order), int(acc)

    ncoeffs = deriv_ncoeffs(order, acc=acc)
    nside = ncoeffs // 2
    ncoeffs += (order % 2 == 0)

    def _build_rhs(offsets, order):
        """The right hand side of the equation system matrix"""
        b = [0 for _ in offsets]
        b[order] = math.factorial(order)
        return np.array(b, dtype='float')

    def _build_matrix_non_uniform(p, q, coords, k):
        """Constructs the equation matrix for the finite difference coefficients of non-uniform grids at location k"""
        A = [[1] * (p + q + 1)]
        for i in range(1, p + q + 1):
            line = [(coords[k + j] - coords[k])**i for j in range(-p, q + 1)]
            A.append(line)
        return np.array(A, dtype='float')

    if idx < nside:
        matrix = _build_matrix_non_uniform(0, ncoeffs - 1, coords, idx)

        offsets = list(range(ncoeffs))
        rhs = _build_rhs(offsets, order)

        return np.linalg.solve(matrix, rhs), np.array(offsets)

    if idx >= len(coords) - nside:
        matrix = _build_matrix_non_uniform(ncoeffs - 1, 0, coords, idx)

        offsets = list(range(-ncoeffs + 1, 1))
        rhs = _build_rhs(offsets, order)

        return np.linalg.solve(matrix, rhs), np.array(offsets)

    matrix = _build_matrix_non_uniform(nside, nside, coords, idx)

    offsets = list(range(-nside, nside + 1))
    rhs = _build_rhs(offsets, order)

    return np.linalg.solve(matrix, rhs), np.array([p for p in range(-nside, nside + 1)])


def deriv_nd(X, Y, orders, center=None):
    uorders = []
    for axis, order, acc in orders:
        if not order: continue
        uorders.append((axis, order, acc))
    orders = uorders
    if center is None:
        center = [np.median(np.unique(x)) for x in X]
    if not len(orders):
        toret = Y[np.all([x == c for x, c in zip(X, center)], axis=0)]
        if not toret.size:
            raise ValueError('Global center point not found')
        return toret[0]
    axis, order, acc = orders[-1]
    ncoeffs = deriv_ncoeffs(order, acc=acc)
    coord = np.unique(X[axis])
    if coord.size < ncoeffs:
        raise ValueError('Grid is not large enough ({:d} < {:d}) to estimate {:d}-th order derivative'.format(coord.size, ncoeffs, order))
    cidx = np.flatnonzero(coord == center[axis])
    if not cidx.size:
        raise ValueError('Global center point not found')
    cidx = cidx[0]
    toret = 0.
    for coeff, offset in zip(*coefficients(order, acc, coord, cidx)):
        mask = X[axis] == coord[cidx + offset]
        y = deriv_nd([x[mask] for x in X], Y[mask], orders[:-1])
        toret += y * coeff
    return toret


class TaylorEmulator(BaseEmulator):

    def __init__(self, pipeline, order=4, accuracy=2, **kwargs):
        super(TaylorEmulator, self).__init__(pipeline=pipeline, **kwargs)
        for name, item in zip(['order', 'accuracy'], [order, accuracy]):
            if not isinstance(item, dict):
                item = {'*': item}
            tmp = {str(param): None for param in self.varied_params}
            for template, value in item.items():
                for tmpname in self.params.names(name=template, varied=True, derived=False):
                    tmp[tmpname] = int(value)
            for param, value in tmp.items():
                if value is None:
                    raise ValueError('{} not specified for parameter {}'.format(name, param))
                elif value < 1:
                    raise ValueError('{} is {:d} < 1 for parameter {}'.format(name, value, param))
            setattr(self, name, tmp)
        for name, acc in self.accuracy.items():
            if acc % 2 or acc <= 0:
                raise ValueError('Accuracy is {:d} for parameter {}, but it must be positive EVEN integer'.format(acc, name))

    def get_default_samples(self, scale=1e-1):
        from cosmofit.samplers import GridSampler
        sampler = GridSampler(self.pipeline, ngrid={name: deriv_ncoeffs(order, acc=self.accuracy[name]) for name, order in self.order.items()},
                              scale=scale, sphere=self.order)
        sampler.run()
        return sampler.samples

    def fit(self):
        self.center, self.derivatives, self.powers = {}, {}, {}
        if self.mpicomm.rank == 0:
            ndim = len(self.varied_params)
            X, center, self.center = [], [], {}
            for name in self.varied_params:
                X.append(self.samples[name])
                center.append(np.median(np.unique(X[-1])))
                self.center[name] = center[-1]
            cidx = np.flatnonzero(np.all([x == c for x, c in zip(X, center)], axis=0))
            if not cidx.size:
                raise ValueError('Global center point not found')
            cidx = cidx[0]
            for name in self.varied:
                Y = self.samples[name]
                self.powers = [np.zeros(ndim, dtype='i4')]
                self.derivatives[name] = [Y[cidx]]  # F(x=center)
                prefactor = 1.
                for order in range(1, max(self.order.values()) + 1):
                    prefactor /= order
                    for indices in itertools.product(range(ndim), repeat=order):
                        degrees = np.bincount(indices, minlength=ndim).astype('i4')
                        #if any(degrees[ii] > self.order[name] for ii, name in enumerate(self.varied_params)):
                        if sum(degrees) > min(self.order[name] for ii, name in enumerate(self.varied_params) if degrees[ii]):
                            continue
                        orders = [(ii, degree, self.accuracy[self.varied_params[ii]]) for ii, degree in enumerate(degrees) if degree > 0]
                        dx = prefactor * deriv_nd(X, Y, orders, center=center)
                        if np.isnan(dx).any():
                            raise ValueError('Some derivatives are NaN: NaN in {}'.format(name))
                        self.powers.append(degrees)
                        self.derivatives[name].append(dx)
                self.powers = np.array(self.powers)
                self.derivatives[name] = np.array(self.derivatives[name])

        self.derivatives = {name: mpy.bcast(self.derivatives[name] if self.mpicomm.rank == 0 else None, mpicomm=self.mpicomm, mpiroot=0) for name in self.varied}
        self.powers = self.mpicomm.bcast(self.powers, root=0)
        self.center = self.mpicomm.bcast(self.center, root=0)

    def predict(self, **params):
        diffs = jnp.array([params[param] - self.center[param] for param in self.varied_params])
        powers = jnp.prod(jnp.where(self.powers > 0, diffs ** self.powers, 1.), axis=-1)
        toret = {}
        for name in self.derivatives:
            toret[name] = jnp.tensordot(self.derivatives[name], powers, axes=(0, 0))
        return toret

    def __getstate__(self):
        state = super(TaylorEmulator, self).__getstate__()
        for name in ['center', 'derivatives', 'powers', 'order']:
            state[name] = getattr(self, name)
        return state

"""Definition of :class:`Profiles`, to hold products of likelihood profiling."""

import os

import numpy as np
from mpi4py import MPI

from cosmofit.parameter import Parameter, ParameterArray, ParameterCollection, BaseParameterCollection
from . import utils


def _reshape(array, shape, previous=None):
    if np.ndim(shape) == 0:
        shape = (shape, )
    shape = tuple(shape)
    ashape = array.shape
    if previous is not None:
        return array.reshape(shape + ashape[len(previous):])
    for i in range(1, array.ndim + 1):
        try:
            return array.reshape(shape + ashape[i:])
        except ValueError:
            continue
    raise ValueError('Cannot match array of shape {} into shape {}'.format(ashape, shape))


class ParameterValues(BaseParameterCollection):

    """Class that holds samples drawn from likelihood."""

    _type = ParameterArray
    _attrs = BaseParameterCollection._attrs + ['_enforce', 'outputs']

    def __init__(self, data=None, params=None, enforce=None, outputs=None, attrs=None):
        self.attrs = dict(attrs or {})
        self.data = []
        outputs = list(outputs or [])
        self.outputs = set([str(name) for name in outputs])
        self._enforce = enforce or {'ndmin': 1}
        if params is not None:
            if len(params) != len(data):
                raise ValueError('Provide as many parameters as arrays')
            for param, value in zip(params, data):
                self[param] = value
        else:
            super(ParameterValues, self).__init__(data=data, attrs=attrs)

    @property
    def inputs(self):
        return [name for name in self.names() if name not in self.outputs]

    @staticmethod
    def _get_param(item):
        return item.param

    @property
    def shape(self):
        if len(self.inputs):
            return self[self.inputs[0]].shape
        return ()

    @shape.setter
    def shape(self, shape):
        previous = self.shape or None
        for array in self:
            self.set(_reshape(array, shape, previous=previous))

    def reshape(self, *args):
        new = self.copy()
        if len(args) == 1:
            shape = args[0]
        else:
            shape = args
        new.shape = shape
        return new

    def ravel(self):
        # Flatten along iteration axis
        return self.reshape(self.size)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.prod(self.shape, dtype='intp')

    def __len__(self):
        if self.shape:
            return self.shape[0]
        return 0

    def select(self, output=None, **kwargs):
        toret = super(ParameterValues, self).select(**kwargs)
        if output is not None:
            for name in toret.names():
                if (output and name not in self.outputs) or (not output and name in self.outputs):
                    del toret[name]
        return toret

    @classmethod
    def concatenate(cls, *others):
        """
        Concatenate input collections.
        Unique items only are kept.
        """
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        if not others: return cls()
        new = cls()
        new_params = others[0].params()
        new_names = new_params.names()
        for other in others:
            other_names = other.names()
            if new_names and other_names and set(other_names) != set(new_names):
                raise ValueError('Cannot concatenate values as parameters do not match: {} != {}.'.format(new_names, other_names))
        for param in new_params:
            #print(param, [np.atleast_1d(other[param]) for other in others], len(others))
            new[param] = np.concatenate([np.atleast_1d(other[param]) for other in others], axis=0)
            #print('2', param, new[param], [np.atleast_1d(other[param]) for other in others], len(others))
        return new

    def set(self, item, output=None):
        super(ParameterValues, self).set(item)
        if output is None:
            param = self._get_param(item)
            output = param.derived and param.fixed
        if output:
            self.outputs.add(self._get_name(item))

    def update(self, *args, **kwargs):
        """Update collection with new one."""
        if len(args) == 1 and isinstance(args[0], self.__class__):
            other = args[0]
        else:
            other = self.__class__(*args, **kwargs)
        self.outputs |= other.outputs
        for item in other:
            self.set(item, output=False)

    def __setitem__(self, name, item):
        """
        Update parameter in collection (a parameter with same name must already exist).
        See :meth:`set` to set a new parameter.

        Parameters
        ----------
        name : Parameter, string, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
            If integer, index in collection.

        item : Parameter
            Parameter.
        """
        if not isinstance(item, self._type):
            try:
                name = self.data[name].param  # list index
            except TypeError:
                pass
            is_output = str(name) in self.outputs
            param = Parameter(name, latex=utils.outputs_to_latex(str(name)) if is_output else None, derived=is_output)
            if param in self:
                param = self[param].param.clone(param)
            item = ParameterArray(item, param, **self._enforce)
        try:
            self.data[name] = item  # list index
        except TypeError:
            item_name = str(self._get_name(item))
            if str(name) != item_name:
                raise KeyError('Parameter {} must be indexed by name (incorrect {})'.format(item_name, name))
            self.set(item)

    def __getitem__(self, name):
        """
        Get samples parameter ``name`` if :class:`Parameter` or string,
        else return copy with local slice of samples.
        """
        if isinstance(name, (Parameter, str)):
            return self.get(name)
        new = self.copy()
        try:
            new.data = [column[name] for column in self.data]
        except IndexError as exc:
            raise IndexError('Unrecognized indices {}'.format(name)) from exc
        return new

    def __repr__(self):
        """Return string representation, including shape and columns."""
        return '{}(shape={}, params={})'.format(self.__class__.__name__, self.shape, self.params())

    def to_array(self, params=None, struct=True):
        """
        Return samples as numpy array.

        Parameters
        ----------
        columns : list, default=None
            Columns to use. Defaults to all columns.

        struct : bool, default=True
            Whether to return structured array, with columns accessible through e.g. ``array['Position']``.
            If ``False``, numpy will attempt to cast types of different columns.

        Returns
        -------
        array : array
        """
        if params is None: params = self.params()
        names = [str(param) for param in params]
        if struct:
            toret = np.empty(len(self), dtype=[(name, self[name].dtype, self.shape[1:]) for name in names])
            for name in names: toret[name] = self[name]
            return toret
        return np.array([self[name] for name in names])

    def to_dict(self, params=None):
        if params is None: params = self.params()
        return {str(param): self[param] for param in params}

    @classmethod
    def bcast(cls, value, mpicomm=None, mpiroot=0):
        import mpytools as mpy
        if mpicomm is None:
            mpicomm = mpy.CurrentMPIComm.get()
        state = None
        if mpicomm.rank == mpiroot:
            state = value.__getstate__()
            state['data'] = [array['param'] for array in state['data']]
        state = mpicomm.bcast(state, root=mpiroot)
        for ivalue, param in enumerate(state['data']):
            state['data'][ivalue] = {'value': mpy.bcast(value.data[ivalue] if mpicomm.rank == mpiroot else None, mpicomm=mpicomm, mpiroot=mpiroot), 'param': param}
        return cls.from_state(state)

    def send(self, dest, tag=0, mpicomm=None):
        import mpytools as mpy
        if mpicomm is None:
            mpicomm = mpy.CurrentMPIComm.get()
        state = self.__getstate__()
        state['data'] = [array['param'] for array in state['data']]
        mpicomm.send(state, dest=dest, tag=tag)
        for array in self:
            mpy.send(array, dest=dest, tag=tag, mpicomm=mpicomm)

    @classmethod
    def recv(cls, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, mpicomm=None):
        import mpytools as mpy
        if mpicomm is None:
            mpicomm = mpy.CurrentMPIComm.get()
        state = mpicomm.recv(source=source, tag=tag)
        for ivalue, param in enumerate(state['data']):
            state['data'][ivalue] = {'value': mpy.recv(source, tag=tag, mpicomm=mpicomm), 'param': param}
        return cls.from_state(state)

    @classmethod
    def sendrecv(cls, value, source=0, dest=0, tag=0, mpicomm=None):
        import mpytools as mpy
        if mpicomm is None:
            mpicomm = mpy.CurrentMPIComm.get()
        if dest == source:
            return value.copy()
        if mpicomm.rank == source:
            value.send(dest=dest, tag=tag, mpicomm=mpicomm)
        toret = None
        if mpicomm.rank == dest:
            toret = cls.recv(source=source, tag=tag, mpicomm=mpicomm)
        return toret

    def match(self, other, eps=1e-8, params=None):
        if params is None:
            params = set(self.names(output=False)) & set(other.names(output=False))
        from scipy import spatial
        kdtree = spatial.cKDTree(np.column_stack([self[name].ravel() for name in params]), leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None)
        array = np.column_stack([other[name].ravel() for name in params])
        dist, indices = kdtree.query(array, k=1, eps=0, p=2, distance_upper_bound=eps)
        mask = indices < self.size
        return np.unravel_index(np.flatnonzero(mask), shape=other.shape), np.unravel_index(indices[mask], shape=self.shape)


class Samples(ParameterValues):

    pass

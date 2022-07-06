"""Definition of :class:`Profiles`, to hold products of likelihood profiling."""

import os

import numpy as np
from mpi4py import MPI

from cosmofit.utils import BaseClass, is_sequence
from cosmofit.parameter import Parameter, ParameterArray, ParameterCollection, BaseParameterCollection
from .utils import outputs_to_latex
from . import utils


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
        if len(self.data):
            return self[self.inputs[0]].shape
        return ()

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.prod(self.shape, dtype='intp')

    def __len__(self):
        return self.shape[0]

    def params(self, output=None, **kwargs):
        params = super(ParameterValues, self).params(**kwargs)
        if output is not None:
            if output:
                params = [param for param in params if str(param) in self.outputs]
            else:
                params = [param for param in params if str(param) not in self.outputs]
        return params

    def ravel(self):
        new = self.copy()
        for name in self.names():  # flatten along iteration axis
            array = self[name]
            new[name] = array.reshape((self.size,) + array.shape[self.ndim:])
        return new

    @classmethod
    def concatenate(cls, *others):
        """
        Concatenate input collections.
        Unique items only are kept.
        """
        if not others: return cls()
        if len(others) == 1 and is_sequence(others[0]):
            others = others[0]
        new = cls(others[0])
        new_names = new.names()
        for other in others:
            other_names = other.names()
            if new_names and other_names and set(other_names) != set(new_names):
                raise ValueError('Cannot concatenate values as parameters do not match: {} != {}.'.format(other_names, new_names))
        for param in new.params():
            new[param] = np.concatenate([np.atleast_1d(other[param]) for other in others], axis=0)
        return new

    def set(self, item, output=False):
        super(ParameterValues, self).set(item)
        if output:
            self.outputs.add(self._get_name(item))

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
            is_output = str(name) in self.outputs
            param = Parameter(name, latex=outputs_to_latex(str(name)) if is_output else None, derived=is_output)
            if param in self:
                param = self[param].param.clone(param)
            item = ParameterArray(item, param, **self._enforce)
        try:
            self.data[name] = item
        except TypeError:
            item_name = str(self._get_name(item))
            if str(name) != item_name:
                raise KeyError('Parameter {} must be indexed by name (incorrect {})'.format(item_name, name))
            try:
                self.data[self._index_name(name)] = item
            except IndexError:
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
        return 'ParameterValues(shape={}, params={})'.format(self.shape, self.params())

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
        if params is None:
            params = self.params()
        names = [self._get_name(name) for name in params]
        if struct:
            toret = np.empty(len(self), dtype=[(name, self[name].dtype, self.shape[1:]) for name in names])
            for name in names: toret[name] = self[name]
            return toret
        return np.array([self[name] for name in names])

    def to_dict(self, params=None):
        if params is None:
            params = self.params()
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
            mpy.send(array, dest=dest, tag=tag)

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

    def match(self, other, eps=1e-8, **kwargs):
        names = self.names(**kwargs)
        from scipy import spatial
        kdtree = spatial.cKDTree(np.array([self[name].ravel() for name in names]).T, leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None)
        array = np.array([other[name].ravel() for name in names]).T
        dist, indices = kdtree.query(array, k=1, eps=0, p=2, distance_upper_bound=eps)
        mask = indices < self.size
        return np.unravel_index(np.flatnonzero(mask), shape=other.shape), np.unravel_index(indices[mask], shape=self.shape)


class ParameterBestFit(ParameterValues):

    _attrs = ParameterValues._attrs + ['_logposterior']

    def __init__(self, *args, logposterior='logposterior', **kwargs):
        self._logposterior = logposterior
        super(ParameterBestFit, self).__init__(*args, **kwargs)
        self.outputs.add(self._logposterior)

    @property
    def logposterior(self):
        if self._logposterior not in self:
            self[self._logposterior] = np.zeros(self.shape, dtype='f8')
        return self[self._logposterior]


class ParameterCovariance(BaseClass):

    """Class that represents a parameter covariance."""

    def __init__(self, covariance, params):
        """
        Initialize :class:`ParameterCovariance`.

        Parameters
        ----------
        covariance : array
            2D array representing covariance.

        params : list, ParameterCollection
            Parameters corresponding to input ``covariance``.
        """
        self._value = np.atleast_2d(covariance)
        self._params = ParameterCollection(params)

    def params(self, *args, **kwargs):
        return self._params.params(*args, **kwargs)

    def cov(self, params=None):
        """Return covariance matrix for input parameters ``params``."""
        if params is None:
            params = self.params
        idx = np.array([self.params.index(param) for param in params])
        toret = self._value[np.ix_(idx, idx)]
        return toret

    def invcov(self, params=None):
        """Return inverse covariance matrix for input parameters ``params``."""
        return utils.inv(self.cov(params))

    def corrcoef(self, params=None):
        """Return correlation matrix for input parameters ``params``."""
        return utils.cov_to_corrcoef(self.cov(parmeters=params))

    def __getstate__(self):
        """Return this class state dictionary."""
        return {'value': self._value, 'params': self._params.__getstate__()}

    def __setstate__(self, state):
        """Set this class state dictionary."""
        self._value = state['value']
        self._params = ParameterCollection.from_state(state['params'])

    def __repr__(self):
        """Return string representation of parameter covariance, including parameters."""
        return '{}({})'.format(self.__class__.__name__, self.params)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == type(self) and other.params() == self.params() and np.all(other._value == self._value)

    @classmethod
    def bcast(cls, value, mpicomm=None, mpiroot=0):
        import mpytools as mpy
        if mpicomm is None:
            mpicomm = mpy.CurrentMPIComm.get()
        state = None
        if mpicomm.rank == mpiroot:
            state = value.__getstate__()
            state['value'] = None
        state = mpicomm.bcast(state, root=mpiroot)
        state['value'] = mpy.bcast(value._value if mpicomm.rank == mpiroot else None, mpicomm=mpicomm, mpiroot=mpiroot)
        return cls.from_state(state)


class Profiles(BaseClass):
    r"""
    Class holding results of likelihood profiling.

    Attributes
    ----------
    start : ParamDict
        Initial parameter values.

    bestfit : ParamDict
        Best fit parameters.

    parabolic_errors : ParamDict
        Parameter parabolic errors.

    deltachi2_errors : ParamDict
        Lower and upper errors corresponding to :math:`\Delta \chi^{2} = 1`.

    covariance : ParameterCovariance
        Parameter covariance at best fit.
    """
    _attrs = {'start': ParameterValues, 'bestfit': ParameterBestFit, 'parabolic_errors': ParameterBestFit, 'deltachi2_errors': ParameterBestFit, 'covariance': ParameterCovariance}

    def __init__(self, attrs=None, **kwargs):
        """
        Initialize :class:`Profiles`.

        Parameters
        ----------
        attrs : dict, default=None
            Other attributes.
        """
        self.attrs = attrs or {}
        self.set(**kwargs)

    def params(self, *args, **kwargs):
        return self.start.params(*args, **kwargs)

    def set(self, params=None, **kwargs):
        for name, cls in self._attrs.items():
            if name in kwargs:
                item = kwargs[name]
                if name == 'covariance':
                    if not isinstance(item, cls):
                        item = cls(item, params=self.params() if params is None else params)
                else:
                    item = cls(item)
                setattr(self, name, item)

    def get(self, name):
        """Access attribute by name."""
        return getattr(self, name)

    def has(self, name):
        """Has this attribute?"""
        return hasattr(self, name)

    @classmethod
    def concatenate(cls, *others):
        """
        Concatenate profiles together.

        Parameters
        ----------
        others : list
            List of :class:`Profiles` instances.

        Returns
        -------
        new : Profiles

        Warning
        -------
        :attr:`attrs` of returned profiles contains, for each key, the last value found in ``others`` :attr:`attrs` dictionaries.
        """
        if not others: return cls()
        if len(others) == 1 and is_sequence(others[0]):
            others = others[0]
        new = others[0].copy()
        attrs = [name for name in new._attrs if new.has(name) and name != 'covariance']
        for other in others:
            if [name for name in other._attrs if other.has(name) and name != 'covariance'] != attrs:
                raise ValueError('Cannot concatenate two profiles if both do not have same attributes.')
        for name in attrs:
            setattr(new, name, new._attrs[name].concatenate(*[other.get(name) for other in others]))
        return new

    def extend(self, other):
        """Extend profiles with ``other``."""
        new = self.concatenate(self, other)
        self.__dict__.update(new.__dict__)

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        state['attrs'] = self.attrs
        for name in self._attrs:
            if hasattr(self, name):
                state[name] = getattr(self, name).__getstate__()
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        self.attrs = state.get('attrs', {})
        for name, cls in self._attrs.items():
            if name in state:
                setattr(self, name, cls.from_state(state[name]))

    def to_stats(self, params=None, quantities=None, sigfigs=2, tablefmt='latex_raw', filename=None):
        """
        Export profiling quantities.

        Parameters
        ----------
        params : list, ParameterCollection
            Parameters to export quantities for.
            Defaults to all parameters.

        quantities : list, default=None
            Quantities to export. Defaults to ``['bestfit','parabolic_errors','deltachi2_errors']``.

        sigfigs : int, default=2
            Number of significant digits.
            See :func:`utils.round_measurement`.

        tablefmt : string, default='latex_raw'
            Format for summary table.
            See :func:`tabulate.tabulate`.

        filename : string default=None
            If not ``None``, file name where to save summary table.

        Returns
        -------
        tab : string
            Summary table.
        """
        import tabulate
        if params is None: params = self.params()
        data = []
        if quantities is None: quantities = [quantity for quantity in ['bestfit', 'parabolic_errors', 'deltachi2_errors'] if self.has(quantity)]
        is_latex = 'latex_raw' in tablefmt
        argmax = self.bestfit.logposterior.argmax()

        def round_errors(low, up):
            low, up = utils.round_measurement(0.0, low, up, sigfigs=sigfigs)[1:]
            if is_latex: return '${{}}_{{{}}}^{{+{}}}$'.format(low, up)
            return '{}/+{}'.format(low, up)

        for iparam, param in enumerate(params):
            row = []
            if is_latex: row.append(param.latex(inline=True))
            else: row.append(str(param.name))
            row.append(str(param.varied))
            ref_error = self.parabolic_errors[param][argmax]
            for quantity in quantities:
                if quantity in ['bestfit', 'parabolic_errors']:
                    value = self.get(quantity)[param][argmax]
                    value = utils.round_measurement(value, ref_error, sigfigs=sigfigs)[0]
                    if is_latex: value = '${}$'.format(value)
                    row.append(value)
                elif quantity == 'deltachi2_errors':
                    low, up = self.get(quantity)[param][argmax]
                    row.append(round_errors(low, up))
                else:
                    raise ValueError('Unknown quantity {}.'.format(quantity))
            data.append(row)
        headers = []
        chi2min = '{:.2f}'.format(-2. * self.bestfit.logposterior[argmax])
        headers.append((r'$\chi^{{2}} = {}$' if is_latex else 'chi2 = {}').format(chi2min))
        headers.append('varied')
        headers += [quantity.replace('_', ' ') for quantity in quantities]
        tab = tabulate.tabulate(data, headers=headers, tablefmt=tablefmt)
        if filename is not None:
            utils.mkdir(os.path.dirname(filename))
            self.log_info('Saving to {}.'.format(filename))
            with open(filename, 'w') as file:
                file.write(tab)
        return tab

    @classmethod
    def bcast(cls, value, mpicomm=None, mpiroot=0):
        import mpytools as mpy
        if mpicomm is None:
            mpicomm = mpy.CurrentMPIComm.get()
        state = None
        if mpicomm.rank == mpiroot:
            state = value.__getstate__()
            for name in cls._attrs: state[name] = None
        state = mpicomm.bcast(state, root=mpiroot)
        for name, acls in cls._attrs.items():
            if mpicomm.bcast(value.has(name) if mpicomm.rank == mpiroot else None, root=mpiroot):
                state[name] = acls.bcast(value.get(name) if mpicomm.rank == mpiroot else None, mpiroot=mpiroot).__getstate__()
            else:
                del state[name]
        return cls.from_state(state)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return all(getattr(other, name) == getattr(self, name) for name in self._attrs)

"""Classes to handle parameters."""

import sys
import re
import fnmatch

import numpy as np
from scipy import stats

from . import base
from .base import Decoder
from .utils import BaseClass, is_sequence


def decode_name(name, default_start=0, default_stop=None, default_step=1):
    """
    Split ``name`` into strings and allowed index ranges.

    >>> decode_name('a_[-4:5:2]_b_[0:2]')
    ['a_', '_b_'], [range(-4, 5, 2), range(0, 2, 1)]

    Parameters
    ----------
    name : string
        Parameter name, e.g. ``a_[-4:5:2]``.

    default_start : int, default=0
        Range start to use as a default.

    default_stop : int, default=None
        Range stop to use as a default.

    default_step : int, default=1
        Range step to use as a default.

    Returns
    -------
    strings : list
        List of strings.

    ranges : list
        List of ranges.
    """
    name = str(name)
    replaces = re.finditer(r'\[(-?\d*):(\d*):*(-?\d*)\]', name)
    strings, ranges = [], []
    string_start = 0
    for ireplace, replace in enumerate(replaces):
        start, stop, step = replace.groups()
        if not start:
            start = default_start
            if start is None:
                raise ValueError('You must provide a lower limit to parameter index')
        else: start = int(start)
        if not stop:
            stop = default_stop
            if stop is None:
                raise ValueError('You must provide an upper limit to parameter index')
        else: stop = int(stop)
        if not step:
            step = default_step
            if step is None:
                raise ValueError('You must provide a step for parameter index')
        else: step = int(step)
        strings.append(name[string_start:replace.start()])
        string_start = replace.end()
        ranges.append(range(start, stop, step))

    strings += [name[string_start:]]

    return strings, ranges


def yield_names_latex(name, latex=None, **kwargs):
    r"""
    Yield parameter name and latex strings with template forms ``[::]`` replaced.

    >>> yield_names_latex('a_[-4:3:2]', latex='\alpha_[-4:5:2]')
    a_-4, \alpha_{-4}
    a_-2, \alpha_{-2}
    a_-0, \alpha_{-0}
    a_2, \alpha_{-2}

    Parameters
    ----------
    name : string
        Parameter name.

    latex : string, default=None
        Latex for parameter.

    kwargs : dict
        Arguments for :func:`decode_name`

    Returns
    -------
    name : string
        Parameter name with template forms ``[::]`` replaced.

    latex : string, None
        If input ``latex`` is ``None``, ``None``.
        Else latex string with template forms ``[::]`` replaced.
    """
    strings, ranges = decode_name(name, **kwargs)

    if not ranges:
        yield strings[0], latex

    else:
        import itertools

        template = '{:d}'.join(strings)
        if latex is not None:
            latex = latex.replace('[]', '{{{:d}}}')

        for nums in itertools.product(*ranges):
            yield template.format(*nums), latex.format(*nums) if latex is not None else latex


def find_names_latex(allnames, name, latex=None, quiet=True):
    r"""
    Search parameter name ``name`` in list of names ``allnames``,
    matching template forms ``[::]``;
    return corresponding parameter names and latex.

    >>> find_names_latex(['a_1', 'a_2', 'b_1'], 'a_[:]', latex='\alpha_[:]')
    [('a_1', '\alpha_{1}'), ('a_2', '\alpha_{2}')]

    Parameters
    ----------
    allnames : list
        List of parameter names (strings).

    name : string
        Parameter name to match in ``allnames``.

    latex : string, default=None
        Latex for parameter.

    quiet : bool, default=True
        If ``False`` and no match for ``name`` was found is ``allnames``, raise :class:`ParameterError`.

    Returns
    -------
    toret : list
        List of string tuples ``(name, latex)``.
        ``latex`` is ``None`` if input ``latex`` is ``None``.
    """
    name = str(name)
    error = ParameterError('No match found for {}'.format(name))
    strings, ranges = decode_name(name, default_start=-sys.maxsize, default_stop=sys.maxsize)
    if not ranges:
        if strings[0] in allnames:
            return [(strings[0], latex)]
        if not quiet:
            raise error
        return []
    pattern = re.compile(r'(-?\d*)'.join(strings))
    toret = []
    if latex is not None:
        latex = latex.replace('[]', '{{{:d}}}')
    for paramname in allnames:
        match = re.match(pattern, paramname)
        if match:
            add = True
            nums = []
            for s, ra in zip(match.groups(), ranges):
                idx = int(s)
                nums.append(idx)
                add = idx in ra  # ra not in memory
                if not add: break
            if add:
                toret.append((paramname, latex.format(*nums) if latex is not None else latex))
    if not toret and not quiet:
        raise error
    return toret


def find_names(allnames, name, quiet=True):
    """
    Search parameter name ``name`` in list of names ``allnames``,
    matching template forms ``[::]``;
    return corresponding parameter names.
    Contrary to :func:`find_names_latex`, it does not handle latex strings,
    but can take a list of parameter names as ``name``
    (thus returning the concatenated list of matching names in ``allnames``).

    >>> find_names(['a_1', 'a_2', 'b_1', 'c_2'], ['a_[:]', 'b_[:]'])
    ['a_1', 'a_2', 'b_1']

    Parameters
    ----------
    allnames : list
        List of parameter names (strings).

    name : list, string
        List of parameter name(s) to match in ``allnames``.

    quiet : bool, default=True
        If ``False`` and no match for parameter name was found is ``allnames``, raise :class:`ParameterError`.

    Returns
    -------
    toret : list
        List of parameter names (strings).
    """
    if isinstance(name, list):
        toret = []
        for name_ in name: toret += find_names(allnames, name_, quiet=quiet)
        return toret

    name = str(name)
    error = ParameterError('No match found for {}'.format(name))

    name = fnmatch.translate(name)
    strings, ranges = decode_name(name, default_start=-sys.maxsize, default_stop=sys.maxsize)
    pattern = re.compile(r'(-?\d*)'.join(strings))
    toret = []
    for paramname in allnames:
        match = re.match(pattern, paramname)
        if match:
            add = True
            nums = []
            for s, ra in zip(match.groups(), ranges):
                idx = int(s)
                nums.append(idx)
                add = idx in ra  # ra not in memory
                if not add: break
            if add:
                toret.append(paramname)
    if not toret and not quiet:
        raise error
    return toret


class ParameterError(Exception):

    """Exception raised when issue with :class:`ParameterError`."""


class ParameterArray(np.ndarray):

    def __new__(cls, value, parameter, copy=False, dtype=None):
        """
        Initalize :class:`array`.

        Parameters
        ----------
        value : array
            Local array value.

        copy : bool, default=False
            Whether to copy input array.

        dtype : dtype, default=None
            If provided, enforce this dtype.
        """
        obj = value.view(cls)
        obj.parameter = parameter
        return obj

    def __array_finalize__(self, obj):
        self.parameter = getattr(obj, 'parameter', None)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        args = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, ParameterArray):
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, ParameterArray):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            if isinstance(inputs[0], ParameterArray):
                inputs[0].parameter = self.parameter
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(ParameterArray)
                         if output is None else output)
                        for result, output in zip(results, outputs))

        for result in results:
            if isinstance(result, ParameterArray):
                result.parameter = self.parameter

        return results[0] if len(results) == 1 else results

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self, self.parameter)


class Parameter(BaseClass):
    """
    Class that represents a parameter.

    Attributes
    ----------
    name : string
        Parameter name.

    value : float
        Default value for parameter.

    fixed : bool
        Whether parameter is fixed.

    prior : ParameterPrior
        Prior distribution.

    ref : ParameterPrior
        Reference distribution.
        This is supposed to represent the expected posterior for this parameter.

    proposal : float
        Proposal uncertainty.

    latex : string, default=None
        Latex for parameter.
    """
    _attrs = ['basename', 'namespace', 'value', 'fixed', 'prior', 'ref', 'proposal', 'latex']

    def __init__(self, basename, namespace=None, value=None, fixed=None, prior=None, ref=None, proposal=None, latex=None):
        """
        Initialize :class:`Parameter`.

        Parameters
        ----------
        name : string, Parameter
            If :class:`Parameter`, update ``self`` attributes.

        value : float, default=False
            Default value for parameter.

        fixed : bool, default=None
            Whether parameter is fixed.
            If ``None``, defaults to ``True`` if ``prior`` or ``ref`` is not ``None``, else ``False``.

        prior : ParameterPrior, dict, default=None
            Prior distribution for parameter, arguments for :class:`ParameterPrior`.

        ref : Prior, dict, default=None
            Reference distribution for parameter, arguments for :class:`ParameterPrior`.
            This is supposed to represent the expected posterior for this parameter.
            If ``None``, defaults to ``prior``.

        proposal : float, default=None
            Proposal uncertainty for parameter.
            If ``None``, defaults to scale (or half of limiting range) of ``ref``.

        latex : string, default=None
            Latex for parameter.
        """
        if isinstance(basename, Parameter):
            self.__dict__.update(basename.__dict__)
            return
        basename = str(basename)
        names = basename.split(base.namespace_delimiter)
        if namespace is not None: names.append(namespace)
        if len(names) > 2:
            raise ParameterError('Single namespace accepted')
        if len(names) == 2:
            self.basename, self.namespace = names
        else:
            self.basename, self.namespace = names[0], None
        self.value = value
        self.prior = prior if isinstance(prior, ParameterPrior) else ParameterPrior(**(prior or {}))
        if value is None:
            if self.prior.is_proper():
                self.value = np.mean(self.prior.limits)
        if ref is not None:
            self.ref = ref if isinstance(ref, ParameterPrior) else ParameterPrior(**(ref or {}))
        else:
            self.ref = self.prior.copy()
        if value is None:
            if (ref is not None or prior is not None):
                if hasattr(self.ref, 'loc'):
                    self.value = self.ref.loc
                elif self.ref.is_proper():
                    self.value = (self.ref.limits[1] - self.ref.limits[0]) / 2.
        self.latex = latex
        if fixed is None:
            fixed = prior is None and ref is None
        self.fixed = bool(fixed)
        self.proposal = proposal
        if proposal is None:
            if (ref is not None or prior is not None):
                if hasattr(self.ref, 'scale'):
                    self.proposal = self.ref.scale
                elif self.ref.is_proper():
                    self.proposal = (self.ref.limits[1] - self.ref.limits[0]) / 2.

    @property
    def name(self):
        if self.namespace:
            return base.namespace_delimiter.join(self.namespace, self.basename)
        return self.basename

    def clone(self, *args, **kwargs):
        state = {}
        if len(args) == 1 and isinstance(args[0], self.__class__):
            state.update({key: getattr(args[0], key) for key in args[0]._attrs})
        elif len(args):
            raise ValueError('Unrecognized arguments {}'.format(args))
        state.update({key: getattr(self, key) for key in self._attrs})
        state.update(kwargs)
        return self.__class__(**state)

    def update(self, *args, **kwargs):
        """Update parameter attributes with new arguments ``kwargs``."""
        self.__dict__.update(self.clone(*args, **kwargs).__dict__)

    @property
    def varied(self):
        """Whether parameter is varied (i.e. not fixed)."""
        return (not self.fixed)

    def latex(self, namespace=False, inline=False):
        """If :attr:`latex` is specified (i.e. not ``None``), return :attr:`latex` surrounded by '$' signs, else :attr:`name`."""
        if namespace:
            namespace = self.namespace
        if self.latex is not None:
            if namespace:
                match1 = re.match('(.*)_(.)$', self.latex)
                match2 = re.match('(.*)_{(.*)}$', self.latex)
                if match1 is not None:
                    latex = r'%s_{%s,\mathrm{%s}}' % (match1.group(1), match1.group(2), namespace)
                elif match2 is not None:
                    latex = r'%s_{%s,\mathrm{%s}}' % (match2.group(1), match2.group(2), namespace)
                else:
                    latex = r'%s_{\mathrm{%s}}' % (self.latex, namespace)
            else:
                latex = self.latex
            if inline:
                latex = '${}$'.format(latex)
            return latex
        return str(self)

    @property
    def limits(self):
        """Parameter limits."""
        return self.prior.limits

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for key in self._attrs:
            state[key] = getattr(self, key)
            if hasattr(state[key], '__getstate__'):
                state[key] = state[key].__getstate__()
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        super(Parameter, self).__setstate__(state)
        for key in ['prior', 'ref']:
            setattr(self, key, ParameterPrior.from_state(state[key]))

    def __repr__(self):
        """Represent parameter as string (name and fixed or varied)."""
        return '{}({}, {})'.format(self.__class__.__name__, self.name, 'fixed' if self.fixed else 'varied')

    def __str__(self):
        """Return parameter as string (name)."""
        return str(self.name)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == type(self) and all(getattr(other, key) == getattr(self, key) for key in self._attrs)


class BaseParameterCollection(BaseClass):

    """Class holding a collection of parameters."""

    _type = Parameter
    _attrs = []

    @staticmethod
    def _get_name(param):
        return getattr(param, 'name', str(param))

    @staticmethod
    def _get_param(item):
        return item

    def __init__(self, data=None):
        """
        Initialize :class:`ParameterCollection`.

        Parameters
        ----------
        data : list, tuple, string, dict, ParameterCollection
            Can be:

            - list (or tuple) of parameters (:class:`Parameter` or dictionary to initialize :class:`Parameter`).
            - dictionary of name: parameter
            - :class:`ParameterCollection` instance

        string : string
            If not ``None``, *yaml* format string to decode.
            Added on top of ``data``.
        """
        if isinstance(data, self.__class__):
            self.__dict__.update(data.copy().__dict__)
            return

        self.data = []

        if is_sequence(data):
            data_ = data
            data = {}
            for item in data_:
                data[self._get_name(item)] = item  # only name is provided

        for name, item in data.items():
            self[name] = item

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
            raise TypeError('{} is not a {} instance.'.format(item, self._type))
        try:
            self.data[name] = item
        except TypeError:
            name = self._get_name(item)
            if self._get_name(item) != name:
                raise KeyError('Parameter {} must be indexed by name (incorrect {})'.format(self._get_name(item), name))
            self.data[self._index_name(name)] = item

    def __getitem__(self, name):
        """
        Return parameter ``name``.

        Parameters
        ----------
        name : Parameter, string, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
            If integer, index in collection.

        Returns
        -------
        param : Parameter
        """
        try:
            return self.data[name]
        except TypeError:
            return self.data[self.index(name)]

    def __delitem__(self, name):
        """
        Delete parameter ``name``.

        Parameters
        ----------
        name : Parameter, string, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
            If integer, index in collection.
        """
        try:
            del self.data[name]
        except TypeError:
            del self.data[self.index(name)]

    def get(self, name, *args, **kwargs):
        """
        Return parameter of name ``name`` in collection.

        Parameters
        ----------
        name : Parameter, string
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.

        Returns
        -------
        param : Parameter
        """
        has_default = False
        if args:
            if len(args) > 1:
                raise SyntaxError('Too many arguments!')
            has_default = True
            default = args[0]
        if kwargs:
            if len(kwargs) > 1:
                raise SyntaxError('Too many arguments!')
            has_default = True
            default = kwargs['default']
        try:
            return self.data[self.index(name)]
        except IndexError:
            if has_default:
                return default
                raise KeyError('Column {} does not exist'.format(name))

    def set(self, item):
        """
        Set parameter ``param`` in collection.
        If there is already a parameter with same name in collection, replace this stored parameter by the input one.
        Else, append parameter to collection.
        """
        if not isinstance(item, self._type):
            raise TypeError('{} is not a {} instance.'.format(item, self._type))
        if item in self:
            self[self._get_name(item)] = item
        else:
            self.data.append(item)

    def index(self, name):
        """
        Return index of parameter ``name``.

        Parameters
        ----------
        name : Parameter, string, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
            If integer, index in collection.

        Returns
        -------
        index : int
        """
        return self._index_name(self._get_name(name))

    def _index_name(self, name):
        # get index of parameter name ``name``
        for ii, item in enumerate(self.data):
            if self._get_name(item) == name:
                return ii
        raise IndexError('Parameter {} not found'.format(name))

    def __contains__(self, name):
        """Whether collection contains parameter ``name``."""
        return self._get_name(name) in self.names()

    def setdefault(self, item):
        """Set parameter ``param`` in collection if not already in it."""
        if not isinstance(item, self._type):
            raise TypeError('{} is not a {} instance.'.format(item, self._type))
        if item in self:
            self.set(item)

    def params(self, **kwargs):
        names = [self._get_name(item) for item in self.data]
        name = kwargs.pop('name', None)
        if name is not None:
            names = find_names(names, name)
            if not names: return []  # no match
        toret = []
        for name in names:
            param = self._get_name(self[name])
            if all(getattr(param, key) == value for key, value in kwargs.items()):
                toret.append(param)
        return toret

    def names(self, **kwargs):
        """Return parameter names in collection."""
        params = self.params(**kwargs)
        return [str(param) for param in params]

    def select(self, **kwargs):
        """
        Return new collection, after selection of parameters whose attribute match input values::

            collection.select(fixed=True)

        returns collection of fixed parameters.
        If 'name' is provided, consider all matching parameters, e.g.::

            collection.select(varied=True, name='a_[0:2]')

        returns a collection of varied parameters, with name in ``['a_0', 'a_1']``.
        """
        toret = self.__class__()
        names = self.names(**kwargs)
        for name in names:
            if name in self:
                toret.set(self[name])
        return toret

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
        for other in others[1:]:
            other = cls(other)
            for item in other.data:
                new.set(item)
        return new

    def extend(self, other):
        """
        Extend collection with ``other``.
        Unique items only are kept.
        """
        new = self.concatenate(self, other)
        self.__dict__.update(new.__dict__)

    def __len__(self):
        """Collection length, i.e. number of items."""
        return len(self.data)

    def __iter__(self):
        """Iterator on collection."""
        return iter(self.data)

    def __getstate__(self):
        """Return this class state dictionary."""
        toret = {'data': [item.__getstate__() for item in self]}
        for name in self._attrs:
            if hasattr(self, name):
                toret[name] = getattr(self, name)

    def __setstate__(self, state):
        """Set this class state dictionary."""
        self.data = [self._type.from_state(item) for item in state['data']]

    def clear(self):
        """Empty collection."""
        self.data.clear()

    def update(self, *args, **kwargs):
        """Update collection with new one."""
        if len(args) == 1 and isinstance(args[0], self.__class__):
            other = args[0]
        else:
            other = self.__class__(*args, **kwargs)
        for item in other:
            self.set(item)


class ParameterCollection(BaseParameterCollection):

    """Class holding a collection of parameters."""

    def __init__(self, data=None, namespace=None):
        """
        Initialize :class:`ParameterCollection`.

        Parameters
        ----------
        data : list, tuple, string, dict, ParameterCollection
            Can be:

            - list (or tuple) of parameters (:class:`Parameter` or dictionary to initialize :class:`Parameter`).
            - dictionary of name: parameter
            - :class:`ParameterCollection` instance

        string : string
            If not ``None``, *yaml* format string to decode.
            Added on top of ``data``.
        """
        if isinstance(data, self.__class__):
            self.__dict__.update(data.copy().__dict__)
            return

        self.data = []

        if isinstance(data, str):
            data = Decoder(data)

        if is_sequence(data):
            data_ = data
            data = {}
            for name in data_:
                if isinstance(name, Parameter):
                    data[name.name] = name
                elif isinstance(name, dict):
                    data[name['name']] = name
                else:
                    data[name] = {}  # only name is provided
        data = {name: conf for name, conf in data.items()}
        list_varied = data.pop('varied', None)
        list_fixed = data.pop('fixed', None)
        list_namespace = data.pop('namespace', None)
        for name, conf in data.items():
            if isinstance(conf, Parameter):
                self.set(conf)
            else:
                latex = conf.pop('latex', None)
                paramnamespace = conf.get('namespace', None)
                if paramnamespace is True:
                    conf['namespace'] = namespace
                elif paramnamespace is None:
                    if namespace is not None or name in list_namespace:
                        conf['namespace'] = namespace
                if conf.get('fixed', None) is None:
                    if name in list_varied:
                        conf['fixed'] = False
                    elif name in list_fixed:
                        conf['fixed'] = True
                for name, latex in yield_names_latex(name, latex=latex):
                    param = Parameter(name=name, latex=latex, **conf)
                    self.set(param)

    def update(self, *args, **kwargs):
        """Update collection with new one."""
        if len(args) == 1 and isinstance(args[0], self.__class__):
            other = args[0]
        else:
            other = self.__class__(*args, **kwargs)
        for item in other:
            self[item].update(item)


class ParameterPriorError(Exception):

    """Exception raised when issue with prior."""


class ParameterPrior(BaseClass):
    """
    Class that describes a 1D prior distribution.

    Parameters
    ----------
    dist : string
        Distribution name.

    rv : scipy.stats.rv_continuous
        Random variate.

    attrs : dict
        Arguments used to initialize :attr:`rv`.
    """

    def __init__(self, dist='uniform', limits=None, **kwargs):
        """
        Initialize :class:`ParameterPrior`.

        Parameters
        ----------
        dist : string
            Distribution name in :mod:`scipy.stats`

        limits : tuple, default=None
            Limits. See :meth:`set_limits`.

        kwargs : dict
            Arguments for :func:`scipy.stats.dist`, typically ``loc``, ``scale``
            (mean and standard deviation in case of a normal distribution ``'dist' == 'norm'``)
        """
        if isinstance(dist, ParameterPrior):
            self.__dict__.update(dist.__dict__)
            return

        self.set_limits(limits)
        self.dist = dist
        self.attrs = kwargs

        # improper prior
        if not self.is_proper():
            return

        if self.is_limited():
            dist = getattr(stats, self.dist if self.dist.startswith('trunc') or self.dist == 'uniform' else 'trunc{}'.format(self.dist))
            if self.dist == 'uniform':
                self.rv = dist(self.limits[0], self.limits[1] - self.limits[0])
            else:
                self.rv = dist(*self.limits, **kwargs)
        else:
            self.rv = getattr(stats, self.dist)(**kwargs)

    def set_limits(self, limits=None):
        r"""
        Set limits.

        Parameters
        ----------
        limits : tuple, default=None
            Tuple corresponding to lower, upper limits.
            ``None`` means :math:`-\infty` for lower bound and :math:`\infty` for upper bound.
            Defaults to :math:`-\infty, \infty`.
        """
        if not limits:
            limits = (-np.inf, np.inf)
        self.limits = list(limits)
        if self.limits[0] is None: self.limits[0] = -np.inf
        if self.limits[1] is None: self.limits[1] = np.inf
        self.limits = tuple(self.limits)
        if self.limits[1] <= self.limits[0]:
            raise ParameterPriorError('ParameterPrior range {} has min greater than max'.format(self.limits))
        if np.isinf(self.limits).any():
            return 1
        return 0

    def isin(self, x):
        """Whether ``x`` is within prior, i.e. within limits - strictly positive probability."""
        return self.limits[0] < x < self.limits[1]

    def __call__(self, x):
        """Return probability density at ``x``."""
        if not self.is_proper():
            return 1. * self.isin(x)
        return self.logpdf(x)

    def sample(self, size=None, random_state=None):
        """
        Draw ``size`` samples from prior. Possible only if prior is proper.

        Parameters
        ---------
        size : int, default=None
            Number of samples to draw.
            If ``None``, return one sample (float).

        random_state : int, numpy.random.Generator, numpy.random.RandomState, default=None
            If integer, a new :class:`numpy.random.RandomState` instance is used, seeded with ``random_state``.
            If ``random_state`` is a :class:`numpy.random.Generator` or :class:`numpy.random.RandomState` instance then that instance is used.
            If ``None``, the :class:`numpy.random.RandomState` singleton is used.

        Returns
        -------
        samples : float, array
            Samples drawn from prior.
        """
        if not self.is_proper():
            raise ParameterPriorError('Cannot sample from improper prior')
        return self.rv.rvs(size=size, random_state=random_state)

    def __str__(self):
        """Return string with distribution name, limits, and attributes (e.g. ``loc`` and ``scale``)."""
        base = self.dist
        if self.is_limited():
            base = '{}[{}, {}]'.format(self.dist, *self.limits)
        return '{}({})'.format(base, self.attrs)

    def __setstate__(self, state):
        """Set this class state dictionary."""
        self.__init__(state['dist'], state['limits'], **state['attrs'])

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for key in ['dist', 'limits', 'attrs']:
            state[key] = getattr(self, key)
        return state

    def is_proper(self):
        """Whether distribution is proper, i.e. has finite integral."""
        return self.dist != 'uniform' or not np.isinf(self.limits).any()

    def is_limited(self):
        """Whether distribution has (at least one) finite limit."""
        return not np.isinf(self.limits).all()

    def __getattribute__(self, name):
        """Make :attr:`rv` attributes directly available in :class:`ParameterPrior`."""
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            attrs = object.__getattribute__(self, 'attrs')
            if name in attrs:
                return attrs[name]
            rv = object.__getattribute__(self, 'rv')
            return getattr(rv, name)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == type(self) and all(getattr(other, key) == getattr(self, key) for key in ['dist', 'limits', 'attrs'])

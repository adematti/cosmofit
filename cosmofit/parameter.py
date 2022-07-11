"""Classes to handle parameters."""

import re
import fnmatch
import functools

import numpy as np
from scipy import stats

from . import base, utils
from .io import BaseConfig
from .utils import BaseClass


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

        template = '%d'.join(strings)
        if latex is not None:
            latex = latex.replace('[]', '%d')

        for nums in itertools.product(*ranges):
            yield template % nums, latex % nums if latex is not None else latex


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
    if not utils.is_sequence(allnames):
        allnames = [allnames]

    if utils.is_sequence(name):
        toret = []
        for nn in name: toret += find_names(allnames, nn, quiet=quiet)
        return toret

    name = str(name)
    error = ParameterError('No match found for {}'.format(name))

    name = fnmatch.translate(name)
    strings, ranges = decode_name(name)
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

    def __new__(cls, value, param, copy=False, dtype=None, **kwargs):
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
        value = np.array(value, copy=copy, dtype=dtype, **kwargs)
        obj = value.view(cls)
        obj.param = param
        return obj

    def __array_finalize__(self, obj):
        self.param = getattr(obj, 'param', None)

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
                inputs[0].param = self.param
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(ParameterArray)
                         if output is None else output)
                        for result, output in zip(results, outputs))

        for result in results:
            if isinstance(result, ParameterArray):
                result.param = self.param

        return results[0] if len(results) == 1 else results

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.param, self)

    def __getstate__(self):
        return {'value': self.view(np.ndarray), 'param': self.param.__getstate__()}

    @classmethod
    def from_state(cls, state):
        return cls(state['value'], Parameter.from_state(state['param']))


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
    _attrs = ['basename', 'namespace', 'value', 'fixed', 'derived', 'prior', 'ref', 'proposal', '_latex']

    def __init__(self, basename, namespace=None, value=None, fixed=None, derived=False, prior=None, ref=None, proposal=None, latex=None):
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
        if namespace: names = [namespace] + names
        if len(names) >= 2:
            self.basename, self.namespace = names[-1], base.namespace_delimiter.join(names[:-1])
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
        self.derived = bool(derived)
        if fixed is None:
            fixed = self.derived or (prior is None and ref is None)
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
            return base.namespace_delimiter.join([self.namespace, self.basename])
        return self.basename

    def update(self, *args, **kwargs):
        """Update parameter attributes with new arguments ``kwargs``."""
        state = {key: getattr(self, key) for key in self._attrs}
        if len(args) == 1 and isinstance(args[0], self.__class__):
            state.update({key: getattr(args[0], key) for key in args[0]._attrs})
        elif len(args):
            raise ValueError('Unrecognized arguments {}'.format(args))
        state['latex'] = state.pop('_latex')
        state.update(kwargs)
        self.__init__(**state)

    def clone(self, *args, **kwargs):
        new = self.copy()
        new.update(*args, **kwargs)
        return new

    @property
    def varied(self):
        """Whether parameter is varied (i.e. not fixed)."""
        return (not self.fixed)

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
        state['latex'] = state.pop('_latex')
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        self.__init__(**state)

    def __repr__(self):
        """Represent parameter as string (name and fixed or varied)."""
        return '{}({}, {})'.format(self.__class__.__name__, self.name, 'fixed' if self.fixed else 'varied')

    def __str__(self):
        """Return parameter as string (name)."""
        return str(self.name)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == type(self) and all(getattr(other, name) == getattr(self, name) for name in self._attrs)


class GetterSetter(object):

    def __init__(self, setter, getter, doc=None):
        self.setter = setter
        self.getter = getter
        self.__doc__ = doc if doc is not None else setter.__doc__

    def __set__(self, obj, value):
        return self.setter(obj, value)

    def __get__(self, obj, cls):
        return functools.partial(self.getter, obj)


def latex_setter(self, latex):
    self._latex = latex


def latex_getter(self, namespace=False, inline=False):
    """If :attr:`latex` is specified (i.e. not ``None``), return :attr:`latex` surrounded by '$' signs, else :attr:`name`."""
    if namespace:
        namespace = self.namespace
    if self._latex is not None:
        if namespace:
            match1 = re.match('(.*)_(.)$', self._latex)
            match2 = re.match('(.*)_{(.*)}$', self._latex)
            if match1 is not None:
                latex = r'%s_{%s,\mathrm{%s}}' % (match1.group(1), match1.group(2), namespace)
            elif match2 is not None:
                latex = r'%s_{%s,\mathrm{%s}}' % (match2.group(1), match2.group(2), namespace)
            else:
                latex = r'%s_{\mathrm{%s}}' % (self._latex, namespace)
        else:
            latex = self._latex
        if inline:
            latex = '${}$'.format(latex)
        return latex
    return str(self)


Parameter.latex = GetterSetter(latex_setter, latex_getter)


class BaseParameterCollection(BaseClass):

    """Class holding a collection of parameters."""

    _type = Parameter
    _attrs = ['attrs']

    @classmethod
    def _get_name(cls, item):
        if isinstance(item, str):
            return item
        if isinstance(item, Parameter):
            param = item
        else:
            param = cls._get_param(item)
        return getattr(param, 'name', str(param))

    @classmethod
    def _get_param(cls, item):
        return item

    def __init__(self, data=None, attrs=None):
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

        self.attrs = dict(attrs or {})
        self.data = []
        if data is None:
            return

        if utils.is_sequence(data):
            dd = datanew.data = self.data.copy()
            data = {}
            for item in dd:
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
            item_name = str(self._get_name(item))
            if str(name) != item_name:
                raise KeyError('Parameter {} must be indexed by name (incorrect {})'.format(item_name, name))
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
        toret = []
        for item in self:
            param = self._get_param(item)
            match = True
            for key, value in kwargs.items():
                param_value = getattr(param, key)
                if key in ['name', 'basename', 'namespace']:
                    key_match = value is None or bool(find_names([param_value], value))
                else:
                    key_match = value == param_value
                match &= key_match
                if not key_match: break
            if match:
                toret.append(param)
        return toret

    def names(self, **kwargs):
        """Return parameter names in collection."""
        params = self.params(**kwargs)
        return [str(param) for param in params]

    def basenames(self, **kwargs):
        """Return base parameter names in collection."""
        params = self.params(**kwargs)
        return [param.basename for param in params]

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
        if len(others) == 1 and utils.is_sequence(others[0]):
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

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.names())

    def __len__(self):
        """Collection length, i.e. number of items."""
        return len(self.data)

    def __iter__(self):
        """Iterator on collection."""
        return iter(self.data)

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {'data': [item.__getstate__() for item in self]}
        for name in self._attrs:
            #if hasattr(self, name):
            state[name] = getattr(self, name)
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        super(BaseParameterCollection, self).__setstate__(state)
        self.data = [self._type.from_state(item) for item in state['data']]

    def __copy__(self):
        new = super(BaseParameterCollection, self).__copy__()
        import copy
        for name in ['data'] + self._attrs:
            # if hasattr(self, name):
            setattr(new, name, copy.copy(getattr(new, name)))
        return new

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

    def clone(self, *args, **kwargs):
        new = self.copy()
        new.update(*args, **kwargs)
        return new

    def items(self):
        return [(self._get_name(item), item) for item in self]

    def deepcopy(self):
        import copy
        return copy.deepcopy(self)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == type(self) and other.params() == self.params() and all(np.all(other_value == self_value) for other_value, self_value in zip(other, self))


class ParameterConfig(BaseConfig):

    _attrs = BaseConfig._attrs + ['fixed', 'derived', 'namespace']
    _keywords = {'fixed': [], 'derived': ['fixed', 'varied'], 'namespace': []}

    def __init__(self, data=None, **kwargs):
        if isinstance(data, self.__class__):
            self.__dict__.update(data.copy().__dict__)
            return
        if isinstance(data, ParameterCollection):
            dd = {}
            for name, param in data.items():
                state = param.__getstate__()
                state.pop('basename')
                state['namespace'] = False
                dd[name] = state
            data = dd
        if utils.is_sequence(data):
            data = {name: {} for name in data}
        super(ParameterConfig, self).__init__(data=data, **kwargs)
        data = self.copy()
        self.fixed, self.derived, self.namespace = {}, {}, {}
        for meta_name in ['fixed', 'varied', 'derived', 'namespace']:
            meta = data.pop(meta_name, [])
            if not utils.is_sequence(meta): meta = [meta]
            if meta_name in ['derived', 'namespace']:
                getattr(self, meta_name).update({name: True for name in meta})
            else:
                self.fixed.update({name: meta_name == 'fixed' for name in meta})
        self.data = {}
        for name, conf in data.items():
            conf = conf.copy()
            latex = conf.pop('latex', None)
            for name, latex in yield_names_latex(name, latex=latex):
                tmp = conf.copy()
                if latex is not None: tmp['latex'] = latex
                tmp['namespace'] = namespace = tmp.get('namespace', None)
                if isinstance(namespace, str):
                    name = base.namespace_delimiter.join([namespace, name])
                    tmp['namespace'] = False
                self[name] = tmp
                for meta_name in ['fixed', 'derived', 'namespace']:
                    meta = getattr(self, meta_name)
                    found = meta_name in tmp
                    param_meta = tmp.pop(meta_name, None)
                    if param_meta is None:
                        for template in meta:
                            if template not in self._keywords[meta_name] and find_names([name], template, quiet=True):
                                param_meta = meta[template]
                                found = True
                    if found:
                        meta[name] = tmp[meta_name] = param_meta

    def update(self, *args, **kwargs):
        other = self.__class__(*args, **kwargs)
        for name in other:
            if name in self:
                self[name].update(other[name])
            else:
                self[name] = other[name]

        def update_order(d1, d2):
            toret = {name: value for name, value in d1.items() if name not in d2}
            for name, value in d2.items():
                toret[name] = value
            return toret

        for meta_name in ['fixed', 'derived', 'namespace']:
            setattr(self, meta_name, update_order(getattr(self, meta_name), getattr(other, meta_name)))
            meta = getattr(self, meta_name)
            for name in self:
                for tmpname in meta:
                    if find_names([name], tmpname, quiet=True):
                        self[name][meta_name] = meta[tmpname]

    def with_namespace(self, namespace=None):
        if namespace is None:
            new = self.deepcopy()
            for name in new:
                new[name]['namespace'] = False
            return new
        new = self.__class__()
        for name, param in self.items():
            newparam = param.copy()
            newparam['namespace'] = False
            if param['namespace'] or param['namespace'] is None:
                name = base.namespace_delimiter.join([namespace, name])
            new[name] = newparam
        for meta_name in ['fixed', 'derived', 'namespace']:
            tmp = {}
            for name, value in getattr(self, meta_name).items():
                key = base.namespace_delimiter.join([namespace, name]) if name not in self._keywords[meta_name] else name
                tmp[key] = value
            setattr(new, meta_name, tmp)
        return new

    def init(self, namespace=None):
        return ParameterCollection(self.with_namespace(namespace=namespace).data)


class ParameterCollection(BaseParameterCollection):

    """Class holding a collection of parameters."""

    def __init__(self, data=None, attrs=None):
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

        self.attrs = dict(attrs or {})
        self.data = []
        if data is None:
            return

        if isinstance(data, str):
            data = ParameterConfig(data)

        if isinstance(data, ParameterConfig):
            self.__dict__.update(data.init())

        if utils.is_sequence(data):
            dd = data
            data = {}
            for name in dd:
                if isinstance(name, Parameter):
                    data[name.name] = name
                elif isinstance(name, dict):
                    data[name['name']] = name
                else:
                    data[name] = {}  # only name is provided
        data = {name: conf for name, conf in data.items()}

        for name, conf in data.items():
            if isinstance(conf, Parameter):
                self.set(conf)
            else:
                if not isinstance(conf, dict):  # parameter value
                    conf = {'value': conf}
                else:
                    conf = conf.copy()
                latex = conf.pop('latex', None)
                for name, latex in yield_names_latex(name, latex=latex):
                    param = Parameter(basename=name, latex=latex, **conf)
                    self.set(param)

    def update(self, *args, name=None, basename=None, **kwargs):
        """Update collection with new one."""
        if len(args) == 1 and isinstance(args[0], self.__class__):
            other = args[0]
            for item in other:
                if item in self:
                    tmp = self[item].clone(item)
                else:
                    tmp = item.copy()
                self.set(tmp)
        elif len(args) <= 1:
            list_update = self.names(name=name, basename=basename)
            for meta_name, fixed in zip(['fixed', 'varied'], [True, False]):
                if meta_name in kwargs:
                    meta = kwargs[meta_name]
                    if isinstance(meta, bool):
                        if meta:
                            meta = list_update
                        else:
                            meta = []
                    for name in meta:
                        for name in self.names(name=name):
                            self[name] = self[name].clone(fixed=fixed)
            if 'namespace' in kwargs:
                namespace = kwargs['namespace']
                indices = [self.index(name) for name in list_update]
                for index in indices:
                    self.data[index] = self.data[index].clone(namespace=namespace)
                names = {}
                for param in self.data: names[param.name] = names.get(param.name, 0) + 1
                duplicates = {name: multiplicity for basename, multiplicity in names.items() if multiplicity > 1}
                if duplicates:
                    raise ValueError('Cannot update namespace, as following duplicates found: {}'.format(duplicates))
        else:
            raise ValueError('Unrecognized arguments {}'.format(args))

    def __add__(self, other):
        return self.concatenate(self, self.__class__(other))

    def __radd__(self, other):
        if other == 0: return self.copy()
        return self.__add__(other)

    def __iadd__(self, other):
        if other == 0: return self.copy()
        return self.__add__(other)


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
        x = np.asarray(x)
        return (self.limits[0] < x) & (x < self.limits[1])

    def __call__(self, x):
        """Return probability density at ``x``."""
        if not self.is_proper():
            toret = np.full_like(x, -np.inf)
            toret[self.isin(x)] = 0.
            return toret
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
        self.__init__(**state)

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for key in ['dist', 'limits']:
            state[key] = getattr(self, key)
        state.update(self.attrs)
        return state

    def is_proper(self):
        """Whether distribution is proper, i.e. has finite integral."""
        return self.dist != 'uniform' or not np.isinf(self.limits).any()

    def is_limited(self):
        """Whether distribution has (at least one) finite limit."""
        return not np.isinf(self.limits).all()

    def __getattr__(self, name):
        """Make :attr:`rv` attributes directly available in :class:`ParameterPrior`."""
        return getattr(object.__getattribute__(self, 'rv'), name)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == type(self) and all(getattr(other, key) == getattr(self, key) for key in ['dist', 'limits', 'attrs'])

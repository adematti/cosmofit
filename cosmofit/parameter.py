"""Classes to handle parameters."""

import re
import fnmatch
import functools
import copy
import numbers

import numpy as np
from scipy import stats

from . import base, utils
from .io import BaseConfig
from .utils import BaseClass, NamespaceDict, deep_eq


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

    error = ParameterError('No match found for {}'.format(name))

    if isinstance(name, re.Pattern):
        pattern = name
        ranges = []
    else:
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

    def __reduce__(self):
        # See https://stackoverflow.com/questions/26598109/preserve-custom-attributes-when-pickling-subclass-of-numpy-array
        # Get the parent's __reduce__ tuple
        pickled_state = super(ParameterArray, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.param.__getstate__(),)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.param = Parameter.from_state(state[-1])  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super(ParameterArray, self).__setstate__(state[:-1])

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
    _attrs = ['basename', 'namespace', 'value', 'fixed', '_derived', 'prior', 'ref', 'proposal', '_latex', 'depends', 'drop', 'saved']
    _allowed_solved = ['.best', '.marg', '.auto']

    def __init__(self, basename, namespace='', value=None, fixed=None, derived=False, prior=None, ref=None, proposal=None, latex=None, drop=False, saved=True):
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
        if isinstance(basename, ParameterConfig):
            self.__dict__.update(basename.init().__dict__)
            return
        self.namespace = namespace
        names = str(basename).split(base.namespace_delimiter)
        self.basename, namespace = names[-1], base.namespace_delimiter.join(names[:-1])
        if self.namespace:
            if namespace:
                self.namespace = base.namespace_delimiter.join([self.namespace, namespace])
        else:
            if namespace:
                self.namespace = namespace
        self.value = float(value) if value is not None else None
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
        self.proposal = proposal
        if proposal is None:
            if (ref is not None or prior is not None):
                if hasattr(self.ref, 'scale'):
                    self.proposal = self.ref.scale
                elif self.ref.is_proper():
                    self.proposal = (self.ref.limits[1] - self.ref.limits[0]) / 2.
        self._derived = derived
        self.depends = {}
        if isinstance(derived, str):
            if self.solved:
                allowed_dists = ['norm', 'uniform']
                if self.prior.dist not in allowed_dists or self.prior.is_limited():
                    raise ParameterError('Prior must be one of {}, with no limits, to use analytic marginalisation for {}'.format(allowed_dists, self))
            else:
                placeholders = re.finditer(r'\{.*?\}', derived)
                for placeholder in placeholders:
                    placeholder = placeholder.group()
                    key = '_' * len(derived) + '{:d}_'.format(len(self.depends) + 1)
                    assert key not in derived
                    derived = derived.replace(placeholder, key)
                    self.depends[key] = placeholder[1:-1]
                self._derived = derived
        else:
            self._derived = bool(self._derived)
        if fixed is None:
            fixed = prior is None and ref is None
        self.fixed = bool(fixed)
        self.saved = bool(saved)
        self.drop = bool(drop)

    def eval(self, **values):
        if isinstance(self._derived, str) and not self.solved:
            try:
                values = {k: values[n] for k, n in self.depends.items()}
            except KeyError:
                raise ParameterError('Parameter {} is derived from {}, following {}'.format(self, list(self.depends.values()), self.derived))
            return utils.evaluate(self._derived, locals=values)
        return values[self.name]

    @property
    def derived(self):
        if isinstance(self._derived, str) and not self.solved:
            toret = self._derived
            for k, v in self.depends.items():
                toret = toret.replace(k, '{{{}}}'.format(v))
            return toret
        return self._derived

    @property
    def solved(self):
        return self._derived in self._allowed_solved

    @property
    def name(self):
        if self.namespace:
            return base.namespace_delimiter.join([self.namespace, self.basename])
        return self.basename

    def update(self, *args, **kwargs):
        """Update parameter attributes with new arguments ``kwargs``."""
        state = self.__getstate__()
        if len(args) == 1 and isinstance(args[0], self.__class__):
            state.update(args[0].__getstate__())
        elif len(args):
            raise ValueError('Unrecognized arguments {}'.format(args))
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

    def __copy__(self):
        new = super(Parameter, self).__copy__()
        new.depends = copy.copy(new.depends)
        return new

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for key in self._attrs:
            state[key] = getattr(self, key)
            if hasattr(state[key], '__getstate__'):
                state[key] = state[key].__getstate__()
        state['latex'] = state.pop('_latex')
        state.pop('_derived')
        state['derived'] = self.derived
        state.pop('depends')
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

    def __hash__(self):
        return hash(str(self))


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
        return param.name

    @classmethod
    def _get_param(cls, item):
        return item

    def __init__(self, data=None, attrs=None):
        """
        Initialize :class:`BaseParameterCollection`.

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
            dd = data
            data = {}
            for item in dd:
                data[self._get_name(item)] = item  # only name is provided

        for name, item in data.items():
            self[name] = item

    def __setitem__(self, name, item):
        """
        Update parameter in collection.
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
            self.data[name] = item  # list index
        except TypeError:
            item_name = self._get_name(item)
            if self._get_name(name) != item_name:
                raise KeyError('Parameter {} must be indexed by name (incorrect {})'.format(item_name, name))
            self.set(item)

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

    def sort(self, key=None):
        if key is not None:
            self.data = [self[kk] for kk in key]
        else:
            self.data = data.copy()
        return self

    def pop(self, name, *args, **kwargs):
        toret = self.get(name, *args, **kwargs)
        try:
            del self[name]
        except (IndexError, KeyError):
            pass
        return toret

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
        except KeyError:
            if has_default:
                return default
            raise KeyError('Parameter {} not found'.format(name))

    def set(self, item):
        """
        Set parameter ``param`` in collection.
        If there is already a parameter with same name in collection, replace this stored parameter by the input one.
        Else, append parameter to collection.
        """
        try:
            self.data[self.index(item)] = item
        except KeyError:
            self.data.append(item)

    def setdefault(self, item):
        """Set parameter ``param`` in collection if not already in it."""
        if not isinstance(item, self._type):
            raise TypeError('{} is not a {} instance.'.format(item, self._type))
        if item not in self:
            self.set(item)

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
        raise KeyError('Parameter {} not found'.format(name))

    def __contains__(self, name):
        """Whether collection contains parameter ``name``."""
        return self._get_name(name) in (self._get_name(item) for item in self.data)

    def select(self, **kwargs):
        """
        Return new collection, after selection of parameters whose attribute match input values::

            collection.select(fixed=True)

        returns collection of fixed parameters.
        If 'name' is provided, consider all matching parameters, e.g.::

            collection.select(varied=True, name='a_[0:2]')

        returns a collection of varied parameters, with name in ``['a_0', 'a_1']``.
        """
        toret = self.copy()
        if not kwargs:
            return toret
        toret.clear()
        for item in self:
            param = self._get_param(item)
            match = True
            for key, value in kwargs.items():
                param_value = getattr(param, key)
                if key in ['name', 'basename', 'namespace']:
                    key_match = value is None or bool(find_names([param_value], value))
                else:
                    key_match = deep_eq(value, param_value)
                    if not key_match:
                        try:
                            key_match |= any(deep_eq(v, param_value) for v in value)
                        except TypeError:
                            pass
                match &= key_match
                if not key_match: break
            if match:
                toret.data.append(item)
        return toret

    def params(self, **kwargs):
        return ParameterCollection([self._get_param(item) for item in self.select(**kwargs)])

    def names(self, **kwargs):
        """Return parameter names in collection."""
        params = self.params(**kwargs)
        return [param.name for param in params]

    def basenames(self, **kwargs):
        """Return base parameter names in collection."""
        params = self.params(**kwargs)
        return [param.basename for param in params]

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
        state = {'data': [item.__getstate__() for item in self.data]}
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
        for name in ['data'] + self._attrs:
            # if hasattr(self, name):
            setattr(new, name, copy.copy(getattr(new, name)))
        return new

    def clear(self):
        """Empty collection."""
        self.data.clear()
        return self

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

    def items(self, **kwargs):
        return [(self._get_name(item), item) for item in self.select(**kwargs)]

    def deepcopy(self):
        return copy.deepcopy(self)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == type(self) and list(other.params()) == list(self.params()) and all(deep_eq(other_value, self_value) for other_value, self_value in zip(other, self))


class ParameterConfig(NamespaceDict):

    def __init__(self, conf=None, **kwargs):
        if isinstance(conf, Parameter):
            conf = conf.__getstate__()
            if conf['namespace'] is None:
                conf.pop('namespace')
        super(ParameterConfig, self).__init__(conf, **kwargs)

    def init(self):
        state = self.__getstate__()
        if not isinstance(self.get('namespace', None), str):
            state['namespace'] = None
        for name in ['prior', 'ref']:
            if name in state and 'rescale' in state[name]:
                value = copy.copy(state[name])
                rescale = value.pop('rescale')
                if 'scale' in value:
                    value['scale'] *= rescale
                if 'limits' in value:
                    limits = np.array(value['limits'])
                    if np.isfinite(limits).all():
                        center = np.mean(limits)
                        value['limits'] = (limits - center) * rescale + center
                state[name] = value
        return Parameter(**state)

    @property
    def param(self):
        return self.init()

    def update(self, *args, exclude=(), **kwargs):
        other = self.__class__(*args, **kwargs)
        for name, value in other.items():
            if name not in exclude:
                if name in ['prior', 'ref'] and name in self and 'rescale' in value and len(value) == 1:
                    value = {**self[name], 'rescale': value['rescale']}
                self[name] = copy.copy(value)

    def is_in_derived(self, name):
        if self.get('derived', False) and isinstance(self.derived, str) and self.derived not in Parameter._allowed_solved:
            return '{{{}}}'.format(str(name)) in self.derived
        return

    def update_derived(self, oldname, newname):
        if self.is_in_derived(oldname):
            self.derived = self.derived.replace('{{{}}}'.format(str(oldname)), '{{{}}}'.format(str(newname)))

    @property
    def name(self):
        namespace = self.get('namespace', None)
        if isinstance(namespace, str) and namespace:
            return base.namespace_delimiter.join([namespace, self.basename])
        return self.basename

    @name.setter
    def name(self, name):
        names = str(name).split(base.namespace_delimiter)
        if len(names) >= 2:
            self.basename, self.namespace = names[-1], base.namespace_delimiter.join(names[:-1])
        else:
            self.basename = names[0]


class ParameterCollectionConfig(BaseParameterCollection):

    _type = ParameterConfig
    _attrs = ['fixed', 'derived', 'namespace', 'delete', 'wildcard', 'identifier']
    #_keywords = {'fixed': [], 'derived': ['.fixed', '.varied'], 'namespace': []}

    def _get_name(self, item):
        if isinstance(item, str):
            return item.split(base.namespace_delimiter)[-1]
        return getattr(item, self.identifier)

    @classmethod
    def _get_param(cls, item):
        return item.param

    def __init__(self, data=None, identifier='basename', **kwargs):
        if isinstance(data, self.__class__):
            self.__dict__.update(data.copy().__dict__)
            self.identifier = identifier
            return
        self.identifier = identifier
        if isinstance(data, ParameterCollection):
            dd = data
            data = {getattr(param, self.identifier): ParameterConfig(param) for param in data}
            if len(data) != len(dd):
                raise ValueError('Found different parameters with same {} in {}'.format(self.identifier, dd))
        else:
            data = BaseConfig(data, **kwargs).data

        self.fixed, self.derived, self.namespace, self.delete, self.wildcard = {}, {}, {}, {}, []
        for meta_name in ['fixed', 'varied', 'derived', 'namespace', 'delete']:
            meta = data.pop('.{}'.format(meta_name), {})
            if utils.is_sequence(meta):
                meta = {name: True for name in meta}
            elif not isinstance(meta, dict):
                meta = {meta: True}
            if meta_name in ['fixed', 'varied']:
                self.fixed.update({name: (meta_name == 'fixed' and value) or (meta_name == 'varied' and not value) for name, value in meta.items()})
            else:
                getattr(self, meta_name).update({name: bool(value) for name, value in meta.items()})
        self.data = []
        for name, conf in data.items():
            if isinstance(conf, numbers.Number):
                conf = {'value': conf}
            conf = (conf or {}).copy()
            latex = conf.pop('latex', None)
            for name, latex in yield_names_latex(name, latex=latex):
                tmp = conf.__getstate__() if isinstance(conf, ParameterConfig) else conf
                tmp = ParameterConfig(name=name, **tmp)
                if latex is not None: tmp.latex = latex
                if '*' in tmp[self.identifier]:
                    self.wildcard.append(tmp)
                    continue
                self.set(tmp)
                #for meta_name in ['fixed', 'derived', 'namespace']:
                #    meta = getattr(self, meta_name)
                #    if meta_name in tmp:
                #        meta.pop(name, None)
                #        meta[name] = tmp[meta_name]
        self._set_meta()

    def _set_meta(self):
        for meta_name in ['fixed', 'derived', 'namespace']:
            meta = getattr(self, meta_name)
            for name in reversed(meta):
                for tmpconf in self.select(**{self.identifier: name}):
                    if meta_name not in tmpconf:
                        tmpconf[meta_name] = meta[name]
        # Wildcard
        for conf in reversed(self.wildcard):
            for tmpconf in self.select(**{self.identifier: conf[self.identifier]}):
                tmpconf.update(conf.clone(tmpconf))

        for conf in self:
            if 'namespace' not in conf:
                conf.namespace = True

    def updated(self, param):
        # Updated with meta?
        paramname = self._get_name(param)
        for meta_name in ['fixed', 'derived', 'namespace']:
            meta = getattr(self, meta_name)
            for name in meta:
                if find_names([paramname], name):
                    return True
        for conf in self.wildcard:
            if find_names([paramname], conf[self.identifier]):
                return True
        return False

    def select(self, **kwargs):
        """
        Return new collection, after selection of parameters whose attribute match input values::

            collection.select(fixed=True)

        returns collection of fixed parameters.
        If 'name' is provided, consider all matching parameters, e.g.::

            collection.select(varied=True, name='a_[0:2]')

        returns a collection of varied parameters, with name in ``['a_0', 'a_1']``.
        """
        toret = self.copy()
        if not kwargs:
            return toret
        toret.clear()
        for item in self:
            param = item
            match = True
            for key, value in kwargs.items():
                param_value = getattr(param, key)
                if key in ['name', 'basename', 'namespace']:
                    key_match = value is None or bool(find_names([param_value], value))
                else:
                    key_match = deep_eq(value, param_value)
                    if not key_match:
                        try:
                            key_match |= any(deep_eq(v, param_value) for v in value)
                        except TypeError:
                            pass
                match &= key_match
                if not key_match: break
            if match:
                toret.data.append(item)
        return toret

    def update(self, *args, **kwargs):
        other = self.__class__(*args, **kwargs)

        for name, b in other.delete.items():
            if b:
                for param in self.select(**{self.identifier: name}):
                    del self[param[self.identifier]]

        for meta_name in ['fixed', 'derived', 'namespace']:
            meta = getattr(other, meta_name)
            for name in meta:
                for tmpconf in self.select(**{self.identifier: name}):
                    tmpconf[meta_name] = meta[name]

        for conf in other.wildcard:
            for tmpconf in self.select(**{self.identifier: conf[self.identifier]}):
                self[tmpconf[self.identifier]] = tmpconf.clone(conf, exclude=(self.identifier,))

        for conf in other:
            new = self.pop(conf[self.identifier], ParameterConfig()).clone(conf)
            self.set(new)

        def update_order(d1, d2):
            toret = {name: value for name, value in d1.items() if name not in d2}
            for name, value in d2.items():
                toret[name] = value
            return toret

        #self.delete = {}
        for meta_name in ['fixed', 'derived', 'namespace', 'delete']:
            setattr(self, meta_name, update_order(getattr(self, meta_name), getattr(other, meta_name)))
        self.wildcard = self.wildcard + other.wildcard
        self._set_meta()

    def with_namespace(self, namespace=None):
        new = self.deepcopy()
        new._set_meta()
        for name, param in new.items():
            if not isinstance(param.namespace, str) and param.namespace:
                param.namespace = namespace
                for dparam in new:
                    dparam.update_derived(param.basename, param.name)
                #self.namespace.pop(name, None)
                #self.namespace[name] = namespace
        for name, param in new.items():
            if param.get('drop', None):
                if any(p.is_in_derived(param) for p in new):
                    param.drop = True
        return new

    def init(self, namespace=None):
        return ParameterCollection([conf.param for conf in self.with_namespace(namespace=namespace)])

    def set(self, item):
        if not isinstance(item, ParameterConfig):
            item = ParameterConfig(item)
        try:
            self.data[self.index(item)] = item
        except KeyError:
            self.data.append(item)

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
        if not isinstance(item, ParameterConfig):
            item = ParameterConfig(item)
            item.setdefault(self.identifier, name)
        try:
            self.data[name] = item
        except TypeError:
            item_name = str(self._get_name(item))
            if str(name) != item_name:
                raise KeyError('Parameter {} must be indexed by name (incorrect {})'.format(item_name, name))
            self.data[self._index_name(name)] = item


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
        if isinstance(data, str):
            data = ParameterCollectionConfig(data)

        if isinstance(data, ParameterCollectionConfig):
            data = data.init()

        if isinstance(data, self.__class__):
            self.__dict__.update(data.copy().__dict__)
            return

        self.attrs = dict(attrs or {})
        self.data = []
        if data is None:
            return

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
                oldnames = self.names()
                for index in indices:
                    self.data[index] = self.data[index].clone(namespace=namespace)
                newnames = self.names()
                for index in indices:
                    dparam = self.data[index]
                    for k, v in dparam.depends.items():
                        if v in oldnames: dparam.depends[k] = newnames[oldnames.index(v)]
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
        if other == 0: return self
        self.__dict__.update(self.__add__(other).__dict__)
        return self

    def params(self, **kwargs):
        return self.select(**kwargs)


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
            Tuple corresponding to lower, upper limits.
            ``None`` means :math:`-\infty` for lower bound and :math:`\infty` for upper bound.
            Defaults to :math:`-\infty, \infty`.

        kwargs : dict
            Arguments for :func:`scipy.stats.dist`, typically ``loc``, ``scale``
            (mean and standard deviation in case of a normal distribution ``'dist' == 'norm'``)
        """
        if isinstance(dist, ParameterPrior):
            self.__dict__.update(dist.__dict__)
            return

        if limits is None:
            limits = (-np.inf, np.inf)
        limits = list(limits)
        if limits[0] is None: limits[0] = -np.inf
        if limits[1] is None: limits[1] = np.inf
        limits = tuple(limits)
        if limits[1] <= limits[0]:
            raise ParameterPriorError('ParameterPrior range {} has min greater than max'.format(limits))
        self.limits = limits
        self.dist = dist.lower()
        self.attrs = kwargs

        # improper prior
        if self.dist == 'uniform' and np.isinf(self.limits).any():
            return

        if not np.isinf(self.limits).all():
            dist = getattr(stats, self.dist if self.dist.startswith('trunc') or self.dist == 'uniform' else 'trunc{}'.format(self.dist))
            if self.dist == 'uniform':
                self.rv = dist(self.limits[0], self.limits[1] - self.limits[0])
            else:
                self.rv = dist(*self.limits, **kwargs)
        else:
            self.rv = getattr(stats, self.dist)(**kwargs)
        #self.limits = self.rv.support()

    def isin(self, x):
        """Whether ``x`` is within prior, i.e. within limits - strictly positive probability."""
        x = np.asarray(x)
        return (self.limits[0] < x) & (x < self.limits[1])

    def __call__(self, x, remove_zerolag=True):
        """Return probability density at ``x``."""
        if not self.is_proper():
            toret = np.full_like(x, -np.inf)
            toret[self.isin(x)] = 0.
            return toret
        toret = self.logpdf(x)
        if remove_zerolag:
            loc = self.attrs.get('loc', None)
            if loc is None: loc = np.mean(self.limits)
            toret -= self.logpdf(loc)
        return toret

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

    def affine_transform(self, loc=0., scale=1.):
        state = self.__getstate__()
        try:
            center = self.loc
        except AttributeError:
            if self.is_limited():
                center = np.mean([lim for lim in self.limits if not np.isinf(lim)])
            else:
                center = 0.
        for name, value in state.items():
            if name in ['loc']:
                state[name] = value + loc
            elif name in ['limits']:
                state[name] = tuple((lim - center) * scale + loc for lim in value)
            elif name in ['scale']:
                state[name] = value * scale
        return self.__class__(**state)

    def __getattr__(self, name):
        """Make :attr:`rv` attributes directly available in :class:`ParameterPrior`."""
        try:
            return getattr(object.__getattribute__(self, 'rv'), name)
        except AttributeError as exc:
            attrs = object.__getattribute__(self, 'attrs')
            if name in attrs:
                return attrs[name]
            else:
                raise exc

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == type(self) and all(getattr(other, key) == getattr(self, key) for key in ['dist', 'limits', 'attrs'])

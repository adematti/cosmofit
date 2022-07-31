import os
import sys
import importlib
import inspect
import copy

import numpy as np
import mpytools as mpy
from mpytools import CurrentMPIComm
from mpi4py.MPI import COMM_SELF

from . import utils
from .utils import BaseClass, NamespaceDict, deep_eq
from .parameter import Parameter, ParameterArray, ParameterCollectionConfig, ParameterCollection, find_names
from .io import BaseConfig
from .samples import ParameterValues
from .samples.utils import outputs_to_latex


namespace_delimiter = '.'


class RegisteredCalculator(type(BaseClass)):

    """Metaclass registering :class:`BaseCalculator`-derived classes."""

    _registry = set()

    def __new__(meta, *args, **kwargs):
        cls = super().__new__(meta, *args, **kwargs)
        meta._registry.add(cls)

        from functools import wraps

        def run(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                """A wrapper function"""
                try:
                    toret = func(self, *args, **kwargs)
                except Exception as exc:
                    raise PipelineError('Error in run of {}'.format(cls)) from exc
                self.runtime_info._derived = None
                return toret
            return wrapper

        cls.run = run(cls.run)
        return cls


class BaseCalculator(BaseClass, metaclass=RegisteredCalculator):
    """
    Base class for calculators.
    A new calculator should minimally implement:

    * :meth:`run`: perform computation (in practice setting attributes) for a given set of input parameters

    Optionally:

    * :meth:`__init__`: method to initialize the class, will receive 'init' dictionary (below) as input.

    * :attr:`requires`: dictionary listing dependencies, in the format name: ``{'class': ..., 'init': {...}, 'params': ...}``:

        * 'class' is the class name, or ``module.ClassName``, or the actual type instance (see :func:`import_cls`) of the calculator

        * 'init' (optionally) is the dictionary of arguments to be passed to the calculator

        * 'params' (optionally, rarely useful) is the dict, :class:`ParameterCollectionConfig` or :class:`ParameterCollection` listing parameters

        These required calculators are then accessible as attributes, with their given names.

    * :attr:`params`: optionally, dictionary of parameters, e.g. ``{'bias': {'value': 2., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 2.}}, 'f': ...}``
      This should usually be better specified in a *yaml* file at :attr:`config_fn`.

    * :meth:`set_params`: optionally, method to change parameters that are received as input, after :meth:`__init__`.

    * :meth:`__getstate__`: return dictionary of attributes characterizing calculator state after any :meth:`run` call.
      :meth:`__getstate__` / :meth:`__setstate__` are necessary for emulation of the calculator.

    * :meth:`__setstate__`: set calculator state from :meth:`__getstate__` output.

    Attributes
    ----------
    config_fn : string
        Path to *yaml* file specifying e.g. :attr:`info`, :attr:`params`, default arguments for :meth:`__init__`.
        By default, the same path as Python file, with a .yaml extension instead.

    info : Info
        Static information on this calculator.

    runtime_info : RuntimeInfo
        Information about calculator name, requirements, parameters values at a given step, etc. in this pipeline.
    """
    def __setattr__(self, name, item):
        """Check no attribute is set with the name of a required calculator."""
        super(BaseCalculator, self).__setattr__(name, item)
        if name in self.runtime_info.requires:
            raise PipelineError('Attribute {} is reserved to a calculator, hence cannot be set'.format(name))

    def __getattr__(self, name):
        if name in ['requires', 'globals']:
            setattr(self, name, {})
            return getattr(self, name)
        if name == 'runtime_info':
            self.runtime_info = RuntimeInfo(self)
            return self.runtime_info
        if name in self.runtime_info.requires:
            toret = self.runtime_info.requires[name]
            if toret.runtime_info.torun:
                toret.run(**toret.runtime_info.params)
                # Makes sure run() is not called the second time this calculator is accessed
                toret.runtime_info.torun = False
            return toret
        return super(BaseCalculator, self).__getattribute__(name)

    @property
    def mpicomm(self):
        """Return :attr:`mpicomm` (set if if does not exist)."""
        mpicomm = getattr(self, '_mpicomm', None)
        if mpicomm is None: mpicomm = CurrentMPIComm.get()
        self._mpicomm = mpicomm
        return self._mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self._mpicomm = mpicomm

    def save(self, filename):
        """Save calculator to ``filename``."""
        if self.mpicomm.rank == 0:
            self.log_info('Saving {}.'.format(filename))
            utils.mkdir(os.path.dirname(filename))
            state = {}
            for name in ['requires', 'globals']:
                state[name] = getattr(self, name)
            np.save(filename, {**state, **self.__getstate__()}, allow_pickle=True)

    def run(self, **params):
        raise NotImplementedError('run(**params) must be implemented in your calculator; it takes parameter names and scalar values as input')

    def mpirun(self, **params):
        """Call :meth:`run` in a MPI-parallel way on input arrays."""
        size, cshape = 0, ()
        names = self.mpicomm.bcast(list(params.keys()) if self.mpicomm.rank == 0 else None, root=0)
        for name in names:
            array = None
            if self.mpicomm.rank == 0:
                array = np.asarray(params[name])
                cshape = array.shape
                array = array.ravel()
            params[name] = mpy.scatter(array, mpicomm=self.mpicomm, mpiroot=0)
            size = params[name].size
        cumsizes = np.cumsum([0] + self.mpicomm.allgather(size))
        if not cumsizes[-1]: return
        mpicomm = self.mpicomm
        states = {}
        for ivalue in range(size):
            self.mpicomm = COMM_SELF
            self.run(**{name: value[ivalue] for name, value in params.items()})
            states[ivalue + cumsizes[mpicomm.rank]] = self.runtime_info.derived
        self.mpicomm = mpicomm
        derived = None
        states = self.mpicomm.gather(states, root=0)
        if self.mpicomm.rank == 0:
            derived = {}
            for state in states:
                derived.update(state)
            derived = ParameterValues.concatenate([derived[i][None, ...] for i in range(cumsizes[-1])])
            derived.shape = cshape
        return derived

    def __getstate__(self):
        """Return this class state dictionary."""
        return {}

    def __repr__(self):
        """Return string representation of this calculator; including name if :attr:`runtime_info` is set."""
        if 'runtime_info' in self.__dict__:
            return '{}({})'.format(self.__class__.__name__, self.runtime_info.name)
        return super(BaseCalculator, self).__repr__()


class PipelineError(Exception):

    """Exception raised when issue with pipeline."""


def import_cls(clsname, pythonpath=None, registry=BaseCalculator._registry):
    """
    Import class from class name.

    Parameters
    ----------
    clsname : string, type
        Class name, as ``module.ClassName`` w.r.t. ``pythonpath``, or directly class type;
        in this case, other arguments are ignored.

    pythonpath : string, default=None
        Optionally, path where to find package/module where class is defined.

    registry : set, default=BaseCalculator._registry
        Optionally, a set of class types to look into.
    """
    if isinstance(clsname, type):
        return clsname
    tmp = clsname.rsplit('.', 1)
    if len(tmp) == 1:
        allcls = []
        for cls in registry:
            if cls.__name__ == tmp[0]: allcls.append(cls)
        if len(allcls) == 1:
            return allcls[0]
        if len(allcls) > 1:
            raise PipelineError('Multiple calculator classes are named {}'.format(clsname))
        raise PipelineError('No calculator class {} found'.format(clsname))
    modname, clsname = tmp
    if pythonpath is not None:
        sys.path.insert(0, pythonpath)
    else:
        sys.path.append(os.path.dirname(__file__))
    module = importlib.import_module(modname)
    return getattr(module, clsname)


class Info(NamespaceDict):

    """Namespace/dictionary holding calculator static attributes."""
    # TODO: add bibtex support


class RuntimeInfo(BaseClass):
    """
    Information about calculator name, requirements, parameters values at a given step, etc. in this pipeline.

    Attributes
    ----------
    namespace : string, default=None
        Calculator namespace (in this pipeline).

    basename : string, default=None
        This calculator base basename (in this pipeline).

    name : string
        Namespace + basename.

    params : dict
        Dictionary of parameter basename: value to be passed to :meth:`BaseCalculator.run`.

    torun : bool
        Whether this calculator should be run (i.e., if parameters habe been updated).

    full_params : ParameterCollection
        Parameters with full names (namespace + basename) for this calculator, in this pipeline.

    base_params : dict
        Dictionary of parameter base name to parameter (specified in :attr:`full_params`).

    derived_params : ParameterCollection
        Parameters of :attr:`full_params` which are derived, i.e. should be stored at each :meth:`BaseCalculator.run` call.

    derived : ParameterValues
        Actual values, for each :meth:`BaseCalculator.run` call, for each of :attr:`derived_params`.
    """
    def __init__(self, calculator, namespace=None, basename=None, requires=None, required_by=None, config=None, full_params=None, derived_auto=None):
        """
        initialize :class:`RuntimeInfo`.

        Parameters
        ----------
        calculator : BaseCalculator
            The calculator this :class:`RuntimeInfo` instance is attached to.

        namespace : string, default=None
            Calculator namespace (in this pipeline).

        basename : string, default=None
            This calculator base basename (in this pipeline).

        requires : dict, default=None
            Calculator requirements.

        required_by : set, default=None
            Set of calculators that requires this calculator.

        config : CalculatorConfig, default=None
            Optionally, calculator config used to initialize this calculator.

        full_params : ParameterCollection
            Parameters with full names (namespace + basename) for this calculator, in this pipeline.

        derived_auto : list
            List of '.varied', '.fixed', to dynamically decide on derived quantities from this calculator.
        """
        self.config = config
        self.basename = basename
        self.namespace = namespace
        self.required_by = set(required_by or [])
        self.requires = requires
        if requires is None:
            self.requires = {}
            for name, calcdict in calculator.requires.items():
                self.requires[name] = CalculatorConfig(calcdict).init(namespace=namespace)
        if full_params is not None:
            self.full_params = full_params
        self.torun = True
        self.calculator = calculator
        self.derived_auto = set(derived_auto or [])

    @property
    def full_params(self):
        if getattr(self, '_full_params', None) is None:
            self._full_params = self.calculator.params
        return self._full_params

    @full_params.setter
    def full_params(self, full_params):
        self._full_params = full_params
        self._base_params = self._derived_params = self._params = None

    @property
    def base_params(self):
        if getattr(self, '_base_params', None) is None:
            self._base_params = {param.basename: param for param in self.full_params}
        return self._base_params

    @property
    def derived_params(self):
        if getattr(self, '_derived_params', None) is None:
            self._derived_params = self.full_params.select(derived=True, fixed=True)
        return self._derived_params

    @property
    def derived(self):
        if getattr(self, '_derived', None) is None:
            self._derived = ParameterValues()
            if self.derived_params:
                state = self.calculator.__getstate__()
                for param in self.derived_params:
                    name = param.basename
                    if name in state: value = state[name]
                    else: value = getattr(self.calculator, name)
                    self._derived.set(ParameterArray(value, param=param), output=True)
        return self._derived

    @derived.setter
    def derived(self, derived):
        self._derived = derived

    @property
    def torun(self):
        return self._torun

    @torun.setter
    def torun(self, torun):
        self._torun = torun
        if torun:
            for inst in self.required_by:
                inst.runtime_info.torun = True

    @property
    def params(self):
        if getattr(self, '_params', None) is None:
            self.params = {param.basename: param.value for param in self.full_params if not param.derived}
        return self._params

    @params.setter
    def params(self, params):
        self.torun = True
        self._params = params

    @property
    def name(self):
        if self.namespace:
            return namespace_delimiter.join([self.namespace, self.basename])
        return self.basename

    def __getstate__(self):
        """Return this class state dictionary."""
        return self.__dict__.copy()

    def update(self, *args, **kwargs):
        """Update with provided :class:`RuntimeInfo` instance of dict."""
        state = self.__getstate__()
        if len(args) == 1 and isinstance(args[0], self.__class__):
            state.update(args[0].__getstate__())
        elif len(args):
            raise ValueError('Unrecognized arguments {}'.format(args))
        state.update(kwargs)
        self.__setstate__(state)

    def clone(self, *args, **kwargs):
        """Clone, i.e. copy and update."""
        new = self.copy()
        new.update(*args, **kwargs)
        return new

    def deepcopy(self):
        import copy
        new = self.copy()
        for name in ['required_by', 'requires', 'derived_auto']:
            setattr(new, name, getattr(self, name).copy())
        new.full_params = copy.deepcopy(self.full_params)
        return new


class SectionConfig(BaseConfig):

    """Base class for config with several sections, see e.g. :class:`CalculatorConfig`."""

    _sections = []

    def __init__(self, *args, **kwargs):
        super(SectionConfig, self).__init__(*args, **kwargs)
        for name in self._sections:
            value = self.data.get(name, {})
            if value is None: value = {}
            self.data[name] = value

    def update(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], self.__class__):
            self.__dict__.update({name: value for name, value in args[0].__dict__.items() if name != 'data'})
            kwargs = {**args[0].data, **kwargs}
        elif len(args):
            raise ValueError('Unrecognized arguments {}'.format(args))
        for name, value in kwargs.items():
            if name in self._sections:
                self[name].update(value)
            else:
                self[name] = value


def _best_match_parameter(namespace, basename, params, choice='max'):
    assert choice in ['min', 'max']
    splitnamespace = [] if not isinstance(namespace, str) else namespace.split(namespace_delimiter)
    params = [param for param in params if param.basename == basename]
    nnamespaces_in_common, ibestmatch = -1 if choice == 'max' else np.inf, -1
    for iparam, param in enumerate(params):
        namespaces = param.name.split(namespace_delimiter)[:-1]
        nm = len(namespaces)
        if nm > len(splitnamespace) or namespaces != splitnamespace[:nm]:
            continue
        if (choice == 'max' and nm >= nnamespaces_in_common) or (choice == 'min' and nm <= nnamespaces_in_common):
            nnamespaces_in_common = nm
            ibestmatch = iparam
    if ibestmatch == -1:
        return None
    return params[ibestmatch]


class CalculatorConfig(SectionConfig):

    _sections = ['info', 'init', 'params']
    _keywords = ['class', 'info', 'init', 'params', 'emulator', 'load', 'save']

    def __init__(self, data, **kwargs):
        # cls, init kwargs
        if isinstance(data, str):
            data = BaseConfig(data, **kwargs).data
        if isinstance(data, (list, tuple)):
            if len(data) == 1:
                data = (data[0], {})
            if len(data) == 2:
                data = {'class': data[0], 'info': {}, 'init': data[1]}
            else:
                raise PipelineError('Provide (ClassName, {}) or {class: ClassName, ...}')
        else:
            data = dict(data)
        super(CalculatorConfig, self).__init__(data)
        self['class'] = import_cls(data.get('class'), pythonpath=data.get('pythonpath', None), registry=BaseCalculator._registry)
        self['info'] = Info(**self['info'])
        self['params'] = ParameterCollectionConfig(self['params'])
        load_fn = data.get('load', None)
        save_fn = data.get('save', None)
        self.setdefault('config_fn', None)
        if not isinstance(load_fn, str):
            if load_fn and isinstance(save_fn, str):
                load_fn = save_fn
            else:
                load_fn = None
        self._loaded = None
        if load_fn is not None:
            self['init'] = {}
            self['class'].log_info('Loading {}.'.format(load_fn))
            state = np.load(load_fn, allow_pickle=True)[()]
            if '_emulator_cls' in state:  # TODO: better managment of cls names
                from cosmofit.emulators import BaseEmulator
                self._loaded = BaseEmulator.from_state(state)
            else:
                self._loaded = self['class'].from_state(state)
            self['class'] = type(self._loaded)
            cls_params = getattr(self._loaded, 'params', None)
            if self['config_fn'] is None: self['config_fn'] = getattr(self._loaded, 'config_fn', None)
        else:
            cls_params = getattr(self['class'], 'params', None)
            if self['config_fn'] is None: self['config_fn'] = getattr(self['class'], 'config_fn', None)
        if cls_params is not None:
            self['params'] = ParameterCollectionConfig(cls_params).clone(self['params'])
        self['load'], self['save'] = load_fn, save_fn

    def init(self, namespace=None, params=None, globals=None, **kwargs):
        self_params = self['params'].with_namespace(namespace=namespace)
        self_derived = [param for param, b in self_params.derived.items() if b]
        derived_auto = set()
        for param in self_derived:
            if param in ['.varied', '.fixed']:
                derived_auto.add(param)
            elif param not in self_params:
                param = Parameter(param, namespace=namespace, derived=True)
                self_params.set(param)
        self_params.derived = {}
        if params is None:
            params = self_params
        for iparam, param in enumerate(self_params):
            if param in params:
                self_params[iparam] = params[param]
        if self._loaded is None:
            new = self['class'].__new__(self['class'])
        else:
            new = self._loaded.copy()
        new.info = self['info']
        if globals is not None:
            globals.update(new.globals)
            new.globals = globals
        if self._loaded is None:
            try:
                new.__init__(**self['init'])
            except TypeError as exc:
                raise PipelineError('Error in {}'.format(new.__class__)) from exc
        if hasattr(new, 'set_params'):
            self_params = new.set_params(self_params)
            if isinstance(self_params, ParameterCollectionConfig):
                self_params = self_params.with_namespace(namespace=namespace)
        new.runtime_info = RuntimeInfo(new, namespace=namespace, config=self, full_params=ParameterCollection(self_params), derived_auto=derived_auto, **kwargs)
        save_fn = self['save']
        if save_fn is not None:
            new.save(save_fn)
        return new

    def other_items(self):
        return [(name, value) for name, value in self.items() if name not in self._keywords]


class PipelineConfig(BaseConfig):

    @CurrentMPIComm.enable
    def __init__(self, *args, params=None, mpicomm=None, **kwargs):
        self.mpicomm = mpicomm
        super(PipelineConfig, self).__init__(*args, **kwargs)

        params = ParameterCollectionConfig(params, identifier='name')
        self.namespaces_deepfirst = ['']
        self.calculators_by_namespace = {}

        def sort_by_namespace(calculators_in_namespace, namespace=''):
            for basename, calcdict in calculators_in_namespace.items():
                if not basename:
                    raise PipelineError('Give non-empty namespace / calculator name')
                if 'class' in calcdict:
                    self.calculators_by_namespace[namespace] = {**self.calculators_by_namespace.get(namespace, {}), basename: calcdict}
                else:
                    if not namespace:
                        newnamespace = basename
                    else:
                        newnamespace = namespace_delimiter.join([namespace, basename])
                    self.namespaces_deepfirst.append(newnamespace)
                    sort_by_namespace(calcdict, namespace=newnamespace)

        sort_by_namespace(self)

        def updated(new, old):
            # Is new different to old? (other than namespace)
            if old is None:
                return True
            return (ParameterConfig(new) != ParameterConfig(old))

        self.params = ParameterCollectionConfig(identifier='name')
        self.updated_params = set()
        for namespace in self.namespaces_deepfirst:
            calculators_in_namespace = self.calculators_by_namespace[namespace] = self.calculators_by_namespace.get(namespace, {})
            for basename, calcdict in calculators_in_namespace.items():
                config = CalculatorConfig(calcdict)
                full_config = self.clone_config_with_fn(config)
                calculators_in_namespace[basename] = full_config
                full_config_params = full_config['params'].with_namespace(namespace=namespace)
                for param in full_config_params:
                    self.params.set(param)
                    if config['params'].updated(param.basename):
                        self.updated_params.add(param.name)

        for param in params:
            self.params.set(param)
            self.updated_params.add(param.name)

    def clone_config_with_fn(self, config):
        cls = config['class']
        config_fn = config.get('config_fn', None)
        if config_fn is None:
            try:
                fn = inspect.getfile(cls)
                default_config_fn = os.path.splitext(fn)[0] + '.yaml'
                if os.path.isfile(default_config_fn):
                    config_fn = default_config_fn
            except TypeError:  # built-in
                pass
        if config_fn is not None:
            if self.mpicomm.rank == 0:
                self.log_info('Loading config file {}'.format(config_fn))
            try:
                tmpconfig = CalculatorConfig(config_fn, index={'class': cls.__name__})
            except IndexError:  # no config for this class in config_fn
                if self.mpicomm.rank == 0:
                    self.log_info('No config for {} found in config file {}'.format(cls.__name__, config_fn))
            else:
                config = tmpconfig.clone(config)
        return config

    def init(self):

        def search_parent_namespace(namespace):
            toret = []
            splitnamespace = namespace.split(namespace_delimiter) if namespace else []
            for tmpnamespace in reversed(self.namespaces_deepfirst):
                tmpsplitnamespace = tmpnamespace.split(namespace_delimiter) if tmpnamespace else []
                if tmpsplitnamespace == splitnamespace[:len(tmpsplitnamespace)]:
                    for basename, config in self.calculators_by_namespace[tmpnamespace].items():
                        toret.append((tmpnamespace, basename, config))
            return toret

        def callback_init(namespace, basename, config, required_by=None):
            if required_by is None:
                globals = None
                if (namespace, basename) in used: return
            else:
                globals = required_by.globals
                required_by = {required_by}
            new = config.init(namespace=namespace, params=self.params, globals=globals, requires={}, required_by=required_by, basename=basename)
            calculators.append(new)
            for requirementbasename, config in getattr(new, 'requires', {}).items():
                # Search for calc in config
                config = CalculatorConfig(config)
                key_requires = requirementbasename
                requirementnamespace = namespace
                match_first, match_name = None, None
                for tmpnamespace, tmpbasename, tmpconfig in search_parent_namespace(namespace):
                    if issubclass(tmpconfig['class'], config['class']):
                        tc = tmpconfig.clone(config)
                        tc['class'], tc._loaded = tmpconfig['class'], tmpconfig._loaded
                        #print(config['class'], tc['class'], list(config['params'].keys()), list(tmpconfig['params'].keys()))
                        #if tc['class'].__name__ == 'DampedBAOWigglesPowerSpectrumMultipoles':
                        #    print(id(tc['params']), id(tmpconfig['params']))
                        tmp = (tmpnamespace, tmpbasename, tc)
                        if match_first is None:
                            match_first = tmp
                        if tmpbasename == namespace_delimiter.join([basename, requirementbasename]):
                            match_name = tmp
                            break
                if match_name:
                    requirementnamespace, requirementbasename, config = match_name
                    used.add((requirementnamespace, requirementbasename))
                elif match_first:
                    requirementnamespace, requirementbasename, config = match_first
                    used.add((requirementnamespace, requirementbasename))
                else:
                    config = self.clone_config_with_fn(config)
                already_instantiated = False
                for calc in calculators:
                    #if calc.__class__.__name__ == 'DampedBAOWigglesPowerSpectrumMultipoles':
                    #    print(id(calc.runtime_info.config['params']))
                    already_instantiated = calc.__class__ == config['class'] and calc.runtime_info.namespace == requirementnamespace\
                                           and calc.runtime_info.basename == requirementbasename and calc.runtime_info.config == config
                    if already_instantiated: break
                if already_instantiated:
                    #print(new.__class__, config['class'], id(calc.runtime_info.config['params']), id(config['params']))
                    requirement = calc
                    requirement.runtime_info.required_by.add(new)
                else:
                    requirement = callback_init(requirementnamespace, requirementbasename, config, required_by=new)
                new.runtime_info.requires[key_requires] = requirement

            return new

        calculators, used = [], set()
        for namespace in reversed(self.namespaces_deepfirst):
            for basename, config in self.calculators_by_namespace[namespace].items():
                callback_init(namespace, basename, config, required_by=None)

        all_params, updated_params = ParameterCollection(), ParameterCollection()
        for calculator in calculators:
            for param in calculator.runtime_info.full_params:
                all_params.set(param)
                if param.name in self.updated_params:
                    updated_params.set(param)

        for calculator in calculators:
            for iparam, param in enumerate(calculator.runtime_info.full_params):
                if param.name not in self.updated_params:
                    match_param = _best_match_parameter(param.namespace, param.basename, updated_params, choice='max')
                    if match_param is None:
                        match_param = _best_match_parameter(param.namespace, param.basename, all_params, choice='min')
                    if match_param is not None:
                        calculator.runtime_info.full_params[iparam] = match_param
            calculator.runtime_info.full_params = calculator.runtime_info.full_params  # to reset all params

        return calculators


class BasePipeline(BaseClass):

    @CurrentMPIComm.enable
    def __init__(self, calculators, params=None, mpicomm=None, quiet=False):

        params = ParameterCollectionConfig(params, identifier='name')

        if isinstance(calculators, BaseCalculator):
            calculators = [calculators]
        if utils.is_sequence(calculators):
            if not len(calculators):
                raise PipelineError('Need at least one calculator')
        else:
            calculators = PipelineConfig(calculators, params=params, mpicomm=mpicomm).init()

        self.calculators, self.end_calculators = calculators, []
        for calculator in self.calculators:
            if not calculator.runtime_info.required_by:
                self.end_calculators.append(calculator)

        if self.mpicomm.rank == 0:
            self.log_info('Found calculators {}.'.format(self.calculators))
            self.log_info('Found end calculators {}.'.format(self.end_calculators))

        # Checks
        for param in params:
            if not any(param in calculator.runtime_info.full_params for calculator in self.calculators):
                raise PipelineError('Parameter {} is not used by any calculator'.format(param))

        def callback_dependency(calculator, required_by):
            for calc in required_by:
                if calc is calculator:
                    raise PipelineError('Circular dependency for calculator {}'.format(calc))
                callback_dependency(calculator, calc.runtime_info.required_by)

        for calculator in self.calculators:
            calculator.runtime_info.pipeline = self
            callback_dependency(calculator, calculator.runtime_info.required_by)

        self.mpicomm = mpicomm
        # Init run, e.g. for fixed parameters
        self.set_params(quiet=quiet)
        for calculator in self.end_calculators:
            calculator.run(**calculator.runtime_info.params)
        self._set_derived_auto()
        self.set_params(quiet=quiet)

    def set_params(self, quiet=False):
        params_from_calculator = {}
        self.params = ParameterCollection()
        for calculator in self.calculators:
            for param in calculator.runtime_info.full_params:
                if not quiet and param in self.params:
                    if param.derived and param.fixed:
                        msg = 'Derived parameter {} of {} is already derived in {}'.format(param, calculator, params_from_calculator[param.name])
                        if param.basename not in calculator.runtime_info.derived_auto and param.basename not in params_from_calculator[param.name].runtime_info.derived_auto:
                            raise PipelineError(msg)
                        elif self.mpicomm.rank == 0:
                            self.log_warning(msg)
                    elif param != self.params[param]:
                        raise PipelineError('Parameter {} of {} is different from that of {}'.format(param, calculator, params_from_calculator[param.name]))
                params_from_calculator[param.name] = calculator
                self.params.set(param)

    def run(self, **params):  # params with namespace
        for calculator in self.calculators:
            calculator.runtime_info.torun = False
        for calculator in self.calculators:
            for param in calculator.runtime_info.full_params:
                value = params.get(param.name, None)
                if value is not None and value != calculator.runtime_info.params[param.basename]:
                    calculator.runtime_info.params[param.basename] = value
                    calculator.runtime_info.torun = True  # set torun = True of all required_by instances
        for calculator in self.end_calculators:
            if calculator.runtime_info.torun:
                calculator.run(**calculator.runtime_info.params)
        self.derived = ParameterValues()
        for calculator in self.calculators:
            self.derived.update(calculator.runtime_info.derived)

    def mpirun(self, **params):
        size, cshape = 0, ()
        names = self.mpicomm.bcast(list(params.keys()) if self.mpicomm.rank == 0 else None, root=0)
        for name in names:
            array = None
            if self.mpicomm.rank == 0:
                array = np.asarray(params[name])
                cshape = array.shape
                array = array.ravel()
            params[name] = mpy.scatter(array, mpicomm=self.mpicomm, mpiroot=0)
            size = params[name].size
        cumsizes = np.cumsum([0] + self.mpicomm.allgather(size))
        if not cumsizes[-1]:
            try:
                self.derived = self.derived[:0]
            except (AttributeError, TypeError, IndexError):
                self.derived = ParameterValues()
            return
        mpicomm = self.mpicomm
        states = {}
        for ivalue in range(size):
            self.mpicomm = COMM_SELF
            self.run(**{name: value[ivalue] for name, value in params.items()})
            states[ivalue + cumsizes[mpicomm.rank]] = self.derived
        self.mpicomm = mpicomm
        derived = None
        states = self.mpicomm.gather(states, root=0)
        if self.mpicomm.rank == 0:
            derived = {}
            for state in states:
                derived.update(state)
            derived = ParameterValues.concatenate([derived[i][None, ...] for i in range(cumsizes[-1])])
            derived.shape = cshape
        self.derived = derived

    @property
    def mpicomm(self):
        mpicomm = getattr(self, '_mpicomm', None)
        if mpicomm is None: mpicomm = CurrentMPIComm.get()
        self._mpicomm = mpicomm
        return self._mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self._mpicomm = mpicomm
        for calculator in self.calculators:
            calculator.mpicomm = mpicomm

    def __copy__(self, type=None):
        if type is None: type = self.__class__
        new = type.__new__(type)
        new.__dict__.update(self.__dict__)
        new.calculators = []
        for calculator in self.calculators:
            calculator = calculator.copy()
            calculator.runtime_info = calculator.runtime_info.deepcopy()
            new.calculators.append(calculator)
        for calculator in new.calculators:
            calculator.runtime_info.required_by = set([new.calculators[self.calculators.index(calc)] for calc in calculator.runtime_info.required_by])
            calculator.runtime_info.requires = {name: new.calculators[self.calculators.index(calc)] for name, calc in calculator.runtime_info.requires.items()}
            calculator.runtime_info.calculator = calculator
            calculator.runtime_info.pipeline = new
        new.end_calculators = [new.calculators[self.calculators.index(calc)] for calc in self.end_calculators]
        new.params = self.params.deepcopy()
        return new

    def select(self, end_calculators, remove_namespace=False, type=None):

        if not utils.is_sequence(end_calculators):
            end_calculators = [end_calculators]
        end_calculators = list(end_calculators)

        for icalc, end_calculator in enumerate(end_calculators):
            if isinstance(end_calculator, str):
                for calculator in self.calculators:
                    if calculator.runtime_info.name == end_calculator:
                        end_calculators[icalc] = calculator
                        break

        new = self.copy(type=type)
        new.params = ParameterCollection()
        new.end_calculators = [new.calculators[self.calculators.index(calc)] for calc in end_calculators]
        new.calculators = []

        def callback(calculator):
            for calc in calculator.runtime_info.requires.values():
                if calc not in new.calculators:
                    new.calculators.append(calc)
                    callback(calc)

        for end_calculator in new.end_calculators:
            end_calculator.runtime_info.required_by = set()
            new.calculators.append(end_calculator)
            callback(end_calculator)

        new.set_params()

        return new

    def remove_namespace(self):
        self.params = self.params.clone(namespace=None)
        for calculator in self.calculators:
            calculator.runtime_info = calculator.runtime_info.clone(full_params=calculator.runtime_info.full_params.clone(namespace=None))

    def _classify_derived_auto(self, calculators=None, niterations=3):
        if calculators is None:
            calculators = []
            for calculator in self.calculators:
                if any(kw in getattr(calculator.runtime_info, 'derived_auto', {}) for kw in ['.varied', '.fixed']):
                    calculators.append(calculator)

        states = [{} for i in range(len(calculators))]
        rng = np.random.RandomState(seed=42)
        if calculators:
            for ii in range(niterations):
                params = {str(param): param.ref.sample(random_state=rng) for param in self.params.select(varied=True)}
                BasePipeline.run(self, **params)
                for calculator, state in zip(calculators, states):
                    calcstate = calculator.__getstate__()
                    for name, value in calcstate.items():
                        state[name] = state.get(name, []) + [value]
                    for param in calculator.runtime_info.derived_params:
                        name = param.basename
                        if name not in calcstate:
                            state[name] = state.get(name, []) + [getattr(calculator, name)]

        fixed, varied = [], []
        for calculator, state in zip(calculators, states):
            fixed.append({})
            varied.append([])
            for name, values in state.items():
                if all(deep_eq(value, values[0]) for value in values):
                    fixed[-1][name] = values[0]
                else:
                    varied[-1].append(name)
                    dtype = np.asarray(values[0]).dtype
                    if not np.issubdtype(dtype, np.inexact):
                        raise ValueError('Attribute {} is of type {}, which is not supported (only float and complex supported)'.format(name, dtype))
        return calculators, fixed, varied

    def _set_derived_auto(self, *args, **kwargs):
        calculators, fixed, varied = self._classify_derived_auto(*args, **kwargs)
        for calculator, fixed_names, varied_names in zip(calculators, fixed, varied):
            derived_names = set()
            for derived_name in calculator.runtime_info.derived_auto:
                if derived_name == '.fixed':
                    derived_names |= set(fixed_names)
                elif derived_name == '.varied':
                    derived_names |= set(varied_names)
                else:
                    derived_names.add(derived_name)
            calculator.runtime_info.derived_auto |= derived_names
            for name in derived_names:
                if name not in calculator.runtime_info.base_params:
                    param = Parameter(name, namespace=calculator.runtime_info.namespace, derived=True)
                    calculator.runtime_info.full_params.set(param)
                    calculator.runtime_info.full_params = calculator.runtime_info.full_params
        return calculators, fixed, varied


class LikelihoodPipeline(BasePipeline):

    _likelihood_name = 'loglikelihood'

    def __init__(self, *args, **kwargs):
        super(LikelihoodPipeline, self).__init__(*args, quiet=True, **kwargs)
        # Check end_calculators are likelihoods
        # for calculator in self.end_calculators:
        #     self._likelihood_name = 'loglikelihood'
        #     if not hasattr(calculator, self._likelihood_name):
        #         raise PipelineError('End calculator {} has no attribute {}'.format(calculator, self._likelihood_name))
        #     loglikelihood = getattr(calculator, self._likelihood_name)
        #     if not np.ndim(loglikelihood) == 0:
        #         raise PipelineError('End calculator {} attribute {} must be scalar'.format(calculator, self._likelihood_name))
        # Select end_calculators with loglikelihood
        end_calculators = []
        for calculator in self.end_calculators:
            if hasattr(calculator, self._likelihood_name):
                end_calculators.append(calculator)
                if self.mpicomm.rank == 0:
                    self.log_info('Found likelihood {}.'.format(calculator.runtime_info.name))
                loglikelihood = getattr(calculator, self._likelihood_name)
                if not np.ndim(loglikelihood) == 0:
                    raise PipelineError('End calculator {} attribute {} must be scalar'.format(calculator, self._likelihood_name))
        self.__dict__.update(self.select(end_calculators).__dict__)
        from .samples.utils import outputs_to_latex
        for calculator in self.end_calculators:
            if self._likelihood_name not in calculator.runtime_info.base_params:
                param = Parameter(self._likelihood_name, namespace=calculator.runtime_info.namespace, latex=outputs_to_latex(self._likelihood_name), derived=True)
                calculator.runtime_info.full_params.set(param)
                calculator.runtime_info.full_params = calculator.runtime_info.full_params
        self.set_params()

    def run(self, **params):
        super(LikelihoodPipeline, self).run(**params)
        from .samples.utils import outputs_to_latex
        loglikelihood = 0.
        params = []
        for calculator in self.end_calculators:
            params.append(calculator.runtime_info.base_params[self._likelihood_name])
            loglikelihood += calculator.runtime_info.derived[params[-1]]
        if len(params) == 1 and not params[0].namespace:
            return
        param = Parameter(self._likelihood_name, namespace=None, latex=outputs_to_latex(self._likelihood_name), derived=True)
        if param in self.derived:
            raise PipelineError('{} is a reserved parameter name, do not use it!'.format(self._likelihood_name))
        self.derived.set(ParameterArray(loglikelihood, param), output=True)

    @property
    def loglikelihood(self):
        return self.derived['loglikelihood']

    def logprior(self, **params):
        logprior = 0.
        for name, value in params.items():
            logprior += self.params[name].prior(value)
        return logprior


class DoConfig(SectionConfig):

    _sections = ['source']

    def __init__(self, *args, **kwargs):
        super(DoConfig, self).__init__(*args, **kwargs)
        if 'do' not in self:
            self['do'] = {}
            for name, value in self.items():
                if name != 'source':
                    self[name] = [value] if utils.is_sequence(value) and value is not None else value
            if not self['do']: self['do'] = None
        elif not isinstance(self['do'], dict):
            self['do'] = {self['do']: None}

    def run(self, pipeline):
        for calculator in pipeline.calculators:
            for name, value in calculator.runtime_info.config.other_items():
                if self['do'] is None or (name in self['do']) and (self['do'][name] is None or calculator.runtime_info.name in self['do'][name]):
                    func = getattr(calculator, name, None)
                    if func is None:
                        if calculator.mpicomm.rank == 0:
                            calculator.log_warning('{} has no method "{}".'.format(calculator, name))
                        continue
                    if isinstance(value, dict):
                        func(**value)
                    else:
                        func(value)

import os
import sys
import inspect
import copy

import numpy as np
import mpytools as mpy
from mpytools import CurrentMPIComm
from mpi4py.MPI import COMM_SELF

from . import utils
from .utils import BaseClass, serialize_class, import_class, OrderedSet, Monitor, NamespaceDict, deep_eq, jax
from .parameter import Parameter, ParameterArray, ParameterCollectionConfig, ParameterCollection, find_names
from .io import BaseConfig
from .samples import ParameterValues
from .samples.utils import outputs_to_latex
from .install import InstallerConfig


namespace_delimiter = '.'


class RegisteredCalculator(type(BaseClass)):

    """Metaclass registering :class:`BaseCalculator`-derived classes."""

    _registry = OrderedSet()

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

        * 'class' is the class name, or ``module.ClassName``, or the actual type instance (see :func:`import_class`) of the calculator

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
    def __getattr__(self, name):
        if name in ['requires', 'globals']:
            setattr(self, name, {})
            return getattr(self, name)
        if name == 'runtime_info':
            self.runtime_info = RuntimeInfo(self)
            return self.runtime_info
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
            state['__class__'] = serialize_class(self.__class__)
            np.save(filename, {**state, **self.__getstate__()}, allow_pickle=True)

    def run(self, **params):
        raise NotImplementedError('run(**params) must be implemented in your calculator; it takes parameter names and scalar values as input')

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
    def __init__(self, calculator, namespace=None, basename=None, requires=None, required_by=None, config=None, full_params=None, speed=None, derived_auto=None):
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
        self.required_by = OrderedSet(required_by or [])
        self.requires = requires
        if requires is None:
            self.requires = {}
            for name, calcdict in calculator.requires.items():
                self.requires[name] = CalculatorConfig(calcdict).init(namespace=namespace)
        if full_params is not None:
            self.full_params = full_params
        self.torun = True
        self.calculator = calculator
        self.derived_auto = OrderedSet(derived_auto or [])
        self.speed = speed
        self.monitor = Monitor()

    @property
    def full_params(self):
        if getattr(self, '_full_params', None) is None:
            self._full_params = self.calculator.params
        return self._full_params

    @full_params.setter
    def full_params(self, full_params):
        self._full_params = full_params
        self._base_params = self._solved_params = self._derived_params = self._param_values = None

    @property
    def base_params(self):
        if getattr(self, '_base_params', None) is None:
            self._base_params = {param.basename: param for param in self.full_params}
        return self._base_params

    @property
    def solved_params(self):
        if getattr(self, '_solved_params', None) is None:
            self._solved_params = self.full_params.select(solved=True)
        return self._solved_params

    @property
    def derived_params(self):
        if getattr(self, '_derived_params', None) is None:
            self._derived_params = self.full_params.select(derived=True, solved=False, depends={})
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
                    self._derived.set(ParameterArray(np.asarray(value), param=param), output=True)
        return self._derived

    #@derived.setter
    #def derived(self, derived):
    #    self._derived = derived

    @property
    def torun(self):
        return self._torun

    @torun.setter
    def torun(self, torun):
        self._torun = torun
        if torun:
            for inst in self.required_by:
                inst.runtime_info.torun = True

    def _run(self, **params):
        for name, calc in self.requires.items():
            calc.runtime_info.run()
            setattr(self.calculator, name, calc)
        self.monitor.start()
        self.calculator.run(**params)
        self.monitor.stop()

    def run(self):
        if self.torun:
            self._run(**self.param_values)
        self.torun = False

    @property
    def param_values(self):
        if getattr(self, '_param_values', None) is None:
            self._param_values = {param.basename: param.value for param in self.full_params if not param.drop and (not param.derived or param.solved)}
        return self._param_values

    @param_values.setter
    def param_values(self, param_values):
        self.torun = True
        self._param_values = param_values

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
        for name, value in state.items():
            setattr(self, name, value)  # this is to properly update properties with setters

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


class InstallableSectionConfig(SectionConfig):

    def __init__(self, *args, install=None, **kwargs):
        super(InstallableSectionConfig, self).__init__(*args, **kwargs)
        if install is not None:
            self['install'] = InstallerConfig(self.get('install', {})).clone(InstallerConfig(install))
        else:
            self['install'] = None


def clone_config_with_fn(config):
    mpicomm = getattr(config, 'mpicomm', CurrentMPIComm.get())
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
        if mpicomm.rank == 0:
            config.log_info('Loading config file {}'.format(config_fn))
        try:
            tmpconfig = config.__class__(config_fn, index={'class': cls.__name__})
        except IndexError:  # no config for this class in config_fn
            if mpicomm.rank == 0:
                config.log_info('No config for {} found in config file {}'.format(cls.__name__, config_fn))
        else:
            config = tmpconfig.clone(config)
    return config



def is_in_namespace(child, parent):

    def split(namespace):
        if isinstance(namespace, str):
            if namespace:
                return namespace.split(namespace_delimiter)
            return []
        return namespace

    child, parent = split(child), split(parent)
    return child == parent[:len(child)]


def _best_match_parameter(namespace, basename, params, choice='max'):
    assert choice in ['min', 'max']
    params = [param for param in params if param.basename == basename]
    nnamespaces_in_common, ibestmatch = -1 if choice == 'max' else np.inf, -1
    for iparam, param in enumerate(params):
        namespaces = param.name.split(namespace_delimiter)[:-1]
        if not is_in_namespace(namespaces, namespace):
            continue
        nm = len(namespaces)
        if (choice == 'max' and nm >= nnamespaces_in_common) or (choice == 'min' and nm <= nnamespaces_in_common):
            nnamespaces_in_common = nm
            ibestmatch = iparam
    if ibestmatch == -1:
        return None
    return params[ibestmatch]


class CalculatorConfig(InstallableSectionConfig):

    _sections = ['info', 'init', 'params']
    _keywords = ['class', 'info', 'init', 'params', 'emulator', 'load', 'save', 'config_fn', 'speed', 'install']

    def __init__(self, data, install=None, **kwargs):
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
        super(CalculatorConfig, self).__init__(data, install=install)
        self['class'] = import_class(data.get('class'), pythonpath=data.get('pythonpath', None), registry=BaseCalculator._registry, install=self['install'])
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
        self._drop_calculators = self.get('drop_calculators', []) or []
        if load_fn is not None:
            self['init'] = {}
            self._loaded = BaseClass.load(load_fn, fallback_class=self['class'])
            if hasattr(self._loaded, 'to_calculator'):
                self._loaded = self._loaded.to_calculator()
                self._drop_calculators = self.get('drop_calculators', True) or []
                if isinstance(self._drop_calculators, bool) and self._drop_calculators:
                    self._drop_calculators = getattr(self._loaded, 'calculators__class__', [])
            self['class'] = type(self._loaded)
            cls_params = getattr(self._loaded, 'params', None)
            if self['config_fn'] is None: self['config_fn'] = getattr(self._loaded, 'config_fn', None)
        else:
            cls_params = getattr(self['class'], 'params', None)
            if self['config_fn'] is None: self['config_fn'] = getattr(self['class'], 'config_fn', None)
        self._drop_calculators = list(self._drop_calculators)
        for icalc, calc in enumerate(self._drop_calculators):
            try:
                self._drop_calculators[icalc] = import_class(*calc, registry=BaseCalculator._registry)
            except ImportError:
                pass
        if cls_params is not None:
            self['params'] = ParameterCollectionConfig(cls_params).clone(self['params'])
        self['load'], self['save'] = load_fn, save_fn

    def init(self, namespace=None, params=None, globals=None, **kwargs):
        self_params = self['params'].with_namespace(namespace=namespace)
        self_derived = [param for param, b in self_params.derived.items() if b]
        derived_auto = OrderedSet()
        for param in self_derived:
            if param in ['.varied', '.fixed']:
                derived_auto.add(param)
            elif param not in self_params:
                param = Parameter(param, namespace=namespace, derived=True)
                self_params.set(param)
        self_params.derived = {}
        if params is None:
            params = self_params
        #for iparam, param in enumerate(self_params):
        #    if param in params:
        #        self_params[iparam] = params[param]
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
        new.runtime_info = RuntimeInfo(new, namespace=namespace, config=self, full_params=ParameterCollection(self_params),
                                       derived_auto=derived_auto, speed=self.get('speed', None), **kwargs)
        save_fn = self['save']
        if save_fn is not None:
            new.save(save_fn)
        return new

    def other_items(self):
        return [(name, value) for name, value in self.items() if name not in self._keywords]


class PipelineConfig(BaseConfig):

    @CurrentMPIComm.enable
    def __init__(self, *args, params=None, mpicomm=None, install=None, **kwargs):
        self.mpicomm = mpicomm
        super(PipelineConfig, self).__init__(*args, **kwargs)
        self._install = install

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
        self.updated_params = OrderedSet()
        for namespace in self.namespaces_deepfirst:
            calculators_in_namespace = self.calculators_by_namespace[namespace] = self.calculators_by_namespace.get(namespace, {})
            for basename, calcdict in calculators_in_namespace.items():
                config = CalculatorConfig(calcdict, install=self._install)
                full_config = clone_config_with_fn(config)
                calculators_in_namespace[basename] = full_config
                full_config_params = full_config['params'].with_namespace(namespace=namespace)
                for param in full_config_params:
                    self.params.set(param)
                    if config['params'].updated(param.basename):
                        self.updated_params.add(param.name)

        for param in params:
            self.params.set(param)
            self.updated_params.add(param.name)

    def init(self):

        def search_parent_namespace(namespace):
            toret = []
            for tmpnamespace in reversed(self.namespaces_deepfirst):
                if is_in_namespace(tmpnamespace, namespace):
                    for basename, config in self.calculators_by_namespace[tmpnamespace].items():
                        toret.append((tmpnamespace, basename, config))
            return toret

        def callback_init(namespace, basename, config, required_by=None):
            if required_by is None:
                globals = None
                if (namespace, basename) in used: return
                for ns in drop:
                    if is_in_namespace(namespace, ns) and config['class'] in drop[ns]: return
            else:
                globals = required_by.globals
                required_by = {required_by}
            drop[namespace] = drop.get(namespace, []) + config._drop_calculators
            new = config.init(namespace=namespace, params=self.params, globals=globals, requires={}, required_by=required_by, basename=basename)
            calculators.append(new)
            for requirementbasename, config in getattr(new, 'requires', {}).items():
                # Search for calc in config
                config = CalculatorConfig(config, install=self._install)
                key_requires = requirementbasename
                requirementnamespace = namespace
                match_first, match_name = None, None
                #print(new.__class__, config['class'], config['init'])
                for tmpnamespace, tmpbasename, tmpconfig in search_parent_namespace(namespace):
                    if issubclass(tmpconfig['class'], config['class']):
                        #for param in config['params']: print(config['class'], param)
                        tc = tmpconfig.clone(config)
                        #tc['params'] = tmpconfig['params'].deepcopy()  # do not change parameter dict, only update names
                        #for param in config['params']: tc['params'].set(param)
                        #print(tc['params'], tmpconfig['params'], config['params'].delete)
                        tc['class'], tc._loaded, tc._drop_calculators = tmpconfig['class'], tmpconfig._loaded, tmpconfig._drop_calculators
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
                    config = clone_config_with_fn(config)
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

        calculators, used, drop = [], OrderedSet(), {}
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
        #for param in params:
        #    if not any(param in calculator.runtime_info.full_params for calculator in self.calculators):
        #        raise PipelineError('Parameter {} is not used by any calculator'.format(param))

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
        #for calculator in self.calculators:
        #    print(calculator, calculator.runtime_info.full_params.select(varied=True))
        self.set_params(params=params, quiet=quiet)
        for calculator in self.end_calculators:
            calculator.runtime_info.torun = True
            calculator.runtime_info.run()
        self._set_derived_auto()
        self.set_params(params=params, quiet=quiet)

    def set_params(self, params=None, quiet=False):
        params_from_calculator = {}
        self.params = ParameterCollection()
        ref_params = ParameterCollection(params)
        self._param_values = {}
        for calculator in self.calculators:
            for iparam, param in enumerate(calculator.runtime_info.full_params):
                if param in ref_params:
                    calculator.runtime_info.full_params[iparam] = param = ref_params[param]
                if not quiet and param in self.params:
                    if param.derived and param.fixed:
                        msg = 'Derived parameter {} of {} is already derived in {}.'.format(param, calculator, params_from_calculator[param.name])
                        if param.basename not in calculator.runtime_info.derived_auto and param.basename not in params_from_calculator[param.name].runtime_info.derived_auto:
                            raise PipelineError(msg)
                        elif self.mpicomm.rank == 0:
                            self.log_warning(msg)
                    elif param != self.params[param]:
                        raise PipelineError('Parameter {} of {} is different from that of {}.'.format(param, calculator, params_from_calculator[param.name]))
                params_from_calculator[param.name] = calculator
                self._param_values[param.name] = param.value
                self.params.set(param)
        for param in ref_params:
            if param not in self.params:
                raise PipelineError('Parameter {} is not used by any calculator'.format(param))
        self._derived = None

    def eval_params(self, params):
        toret = {}
        all_params = {**self._param_values, **params}
        for param in all_params:
            try:
                toret[param] = self.params[param].eval(**all_params)
            except KeyError:
                pass
        return toret

    def run(self, **params):  # params with namespace
        torun = self._derived is None
        self._param_values.update(params)
        params = self.eval_params(params)
        for calculator in self.calculators:
            calculator.runtime_info.torun = torun
        for calculator in self.calculators:
            for param in calculator.runtime_info.full_params:
                value = params.get(param.name, None)
                if value is not None and param.basename in calculator.runtime_info.param_values and value != calculator.runtime_info.param_values[param.basename]:
                    calculator.runtime_info.param_values[param.basename] = value
                    torun = calculator.runtime_info.torun = True  # set torun = True of all required_by instances
        if torun:
            self.derived = self._derived = ParameterValues()
            for param in self.params:
                if param.depends: self.derived[param] = params[param.name]
            for calculator in self.end_calculators:
                calculator.runtime_info.run()
            for calculator in self.calculators:
                self.derived.update(calculator.runtime_info.derived)
        return torun

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
            states[ivalue + cumsizes[mpicomm.rank]] = self._derived
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

    def jac(self, calculator, name, params=None):

        if jax is None:
            raise PipelineError('jax is required to compute the Jacobian')

        def fun(params):
            params = self.eval_params(params)
            for calc in self.calculators:
                calc.runtime_info.torun = False
            for calc in self.calculators:
                for param in calc.runtime_info.full_params:
                    value = params.get(param.name, None)
                    if value is not None and param.basename in calc.runtime_info.param_values:
                        calc.runtime_info.param_values[param.basename] = value
                        calc.runtime_info.torun = True  # set torun = True of all required_by instances
            for calc in self.end_calculators:
                calc.runtime_info.run()
            return getattr(calculator, name)

        jac = jax.jacfwd(fun, argnums=0, has_aux=False, holomorphic=False)

        if params is None:
            params = self._param_values
        elif not isinstance(params, dict):
            params = {str(param): self._param_values[str(param)] for param in params}

        jac = jac(params)
        fun(params)
        return {k: np.asarray(v) for k, v in jac.items()}

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
            calculator.runtime_info.required_by = OrderedSet(new.calculators[self.calculators.index(calc)] for calc in calculator.runtime_info.required_by)
            calculator.runtime_info.requires = {name: new.calculators[self.calculators.index(calc)] for name, calc in calculator.runtime_info.requires.items()}
            calculator.runtime_info.calculator = calculator
            calculator.runtime_info.pipeline = new
        new.end_calculators = [new.calculators[self.calculators.index(calc)] for calc in self.end_calculators]
        #new.params = self.params.deepcopy()
        new.set_params()
        return new

    def select(self, end_calculators, type=None):

        if not utils.is_sequence(end_calculators):
            end_calculators = [end_calculators]
        end_calculators = list(end_calculators)

        for icalc, end_calculator in enumerate(end_calculators):
            if isinstance(end_calculator, str):
                found = False
                for calculator in self.calculators:
                    if calculator.runtime_info.name == end_calculator:
                        end_calculators[icalc] = calculator
                        found = True
                        break
                if not found:
                    raise ValueError('Calculator of name {} not found in {}'.format(end_calculator, self.calculators))

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
            end_calculator.runtime_info.required_by = OrderedSet()
            new.calculators.append(end_calculator)
            callback(end_calculator)

        new.set_params()

        return new

    def with_namespace(self, namespace=None):
        self.params = self.params.clone(namespace=namespace)
        for calculator in self.calculators:
            calculator.runtime_info = calculator.runtime_info.clone(namespace=namespace, full_params=calculator.runtime_info.full_params.clone(namespace=namespace))

    def _classify_derived_auto(self, calculators=None, niterations=3, seed=42):
        if calculators is None:
            calculators = []
            for calculator in self.calculators:
                if any(kw in getattr(calculator.runtime_info, 'derived_auto', OrderedSet()) for kw in ['.varied', '.fixed']):
                    calculators.append(calculator)

        states = [{} for i in range(len(calculators))]
        rng = np.random.RandomState(seed=seed)
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
            derived_names = OrderedSet()
            for derived_name in calculator.runtime_info.derived_auto:
                if derived_name == '.fixed':
                    derived_names |= OrderedSet(fixed_names)
                elif derived_name == '.varied':
                    derived_names |= OrderedSet(varied_names)
                else:
                    derived_names.add(derived_name)
            calculator.runtime_info.derived_auto |= derived_names
            for name in derived_names:
                if name not in calculator.runtime_info.base_params:
                    param = Parameter(name, namespace=calculator.runtime_info.namespace, derived=True)
                    calculator.runtime_info.full_params.set(param)
                    calculator.runtime_info.full_params = calculator.runtime_info.full_params
        return calculators, fixed, varied

    def _set_speed(self, niterations=10, override=False, seed=42):
        seed = mpy.random.bcast_seed(seed=seed, mpicomm=self.mpicomm, size=10000)[self.mpicomm.rank]  # to get different seeds on each rank
        rng = np.random.RandomState(seed=seed)
        BasePipeline.run(self)  # to set _derived
        for calculator in self.calculators:
            calculator.runtime_info.monitor.reset()
        for ii in range(niterations):
            params = {str(param): param.ref.sample(random_state=rng) for param in self.params.select(varied=True, solved=False)}
            BasePipeline.run(self, **params)
        if self.mpicomm.rank == 0:
            self.log_info('Found speeds:')
        for calculator in self.calculators:
            if calculator.runtime_info.speed is None or override:
                total_time = self.mpicomm.allreduce(calculator.runtime_info.monitor.get('time', average=False))
                counter = self.mpicomm.allreduce(calculator.runtime_info.monitor.counter)
                if counter == 0:
                    calculator.runtime_info.speed = 1e6
                else:
                    calculator.runtime_info.speed = counter / total_time
                if self.mpicomm.rank == 0:
                    self.log_info('- {}: {:.2f} iterations / second'.format(calculator, calculator.runtime_info.speed))

    def block_params(self, params=None, nblocks=None, oversample_power=0, **kwargs):
        from itertools import permutations, chain
        if params is None: params = self.params.select(varied=True)
        else: params = [self.params[param] for param in params]
        blocks = []
        # Using same algorithm as Cobaya
        speeds = [calculator.runtime_info.speed for calculator in self.calculators]
        if any(speed is None for speed in speeds) or kwargs:
            self._set_speed(**kwargs)
            speeds = [calculator.runtime_info.speed for calculator in self.calculators]

        footprints = [tuple(param in calculator.runtime_info.full_params for calculator in self.calculators) for param in params]
        #print(self.calculators)
        #print(params)
        #print(footprints)
        unique_footprints = list(set(row for row in footprints))
        param_blocks = [[p for ip, p in enumerate(params) if footprints[ip] == uf] for uf in unique_footprints]
        #print(param_blocks)
        param_block_sizes = [len(b) for b in param_blocks]

        def sort_parameter_blocks(footprints, block_sizes, speeds, oversample_power=oversample_power):
            footprints = np.array(footprints, dtype='i4')
            block_sizes = np.array(block_sizes, dtype='i4')
            costs = 1. / np.array(speeds, dtype='f8')
            tri_lower = np.tri(len(block_sizes))
            assert footprints.shape[0] == block_sizes.size

            def get_cost_per_param_per_block(ordering):
                return np.minimum(1, tri_lower.T.dot(footprints[ordering])).dot(costs)

            if oversample_power >= 1:
                orderings = [sort_parameter_blocks(footprints, block_sizes, speeds, oversample_power=1 - 1e-3)[0]]
            else:
                orderings = list(permutations(np.arange(len(block_sizes))))

            permuted_costs_per_param_per_block = np.array([get_cost_per_param_per_block(list(o)) for o in orderings])
            permuted_oversample_factors = (permuted_costs_per_param_per_block[..., [0]] / permuted_costs_per_param_per_block)**oversample_power
            total_costs = np.array([(block_sizes[list(o)] * permuted_oversample_factors[i]).dot(permuted_costs_per_param_per_block[i]) for i, o in enumerate(orderings)])
            argmin = np.argmin(total_costs)
            optimal_ordering = orderings[argmin]
            costs = permuted_costs_per_param_per_block[argmin]
            return optimal_ordering, costs, permuted_oversample_factors[argmin].astype('i4')

        # a) Multiple blocks
        if nblocks is None:
            i_optimal_ordering, costs, oversample_factors = sort_parameter_blocks(unique_footprints, param_block_sizes, speeds, oversample_power=oversample_power)
            sorted_blocks = [param_blocks[i] for i in i_optimal_ordering]
        # b) 2-block slow-fast separation
        else:
            if len(param_blocks) < nblocks:
                raise ValueError('Cannot build up {:d} parameter blocks, as we only have {:d}'.format(nblocks, len(param_blocks)))
            # First sort them optimally (w/o oversampling)
            i_optimal_ordering, costs, oversample_factors = sort_parameter_blocks(unique_footprints, param_block_sizes, speeds, oversample_power=0)
            sorted_blocks = [param_blocks[i] for i in i_optimal_ordering]
            sorted_footprints = np.array(unique_footprints)[list(i_optimal_ordering)]
            # Then, find the split that maxes cost LOG-differences.
            # Since costs are already "accumulated down",
            # we need to subtract those below each one
            costs_per_block = costs - np.append(costs[1:], 0)
            # Split them so that "adding the next block to the slow ones" has max cost
            log_differences = np.zeros(len(costs_per_block) - 1, dtype='f8')  # some blocks are costless (no more parameters)
            nonzero = (costs_per_block[:-1] != 0.) & (costs_per_block[1:] != 0.)
            log_differences[nonzero] = np.log(costs_per_block[:-1][nonzero]) - np.log(costs_per_block[1:][nonzero])
            split_block_indices = np.pad(np.sort(np.argsort(log_differences)[-(nblocks - 1):]) + 1, (1, 1), mode='constant', constant_values=(0, len(param_block_sizes)))
            split_block_slices = list(zip(split_block_indices[:-1], split_block_indices[1:]))
            split_blocks = [list(chain(*sorted_blocks[low:up])) for low, up in split_block_slices]
            split_footprints = np.clip(np.array([np.array(sorted_footprints[low:up]).sum(axis=0) for low, up in split_block_slices]), 0, 1)  # type: ignore
            # Recalculate oversampling factor with 2 blocks
            oversample_factors = sort_parameter_blocks(split_footprints, [len(block) for block in split_blocks], speeds,
                                                       oversample_power=oversample_power)[2]
            # Finally, unfold `oversampling_factors` to have the right number of elements,
            # taking into account that that of the fast blocks should be interpreted as a
            # global one for all of them.
            oversample_factors = np.concatenate([np.full(size, factor, dtype='f8') for factor, size in zip(oversample_factors, np.diff(split_block_slices, axis=-1))])
        return sorted_blocks, oversample_factors


class LikelihoodPipeline(BasePipeline):

    _loglikelihood_name = 'loglikelihood'
    _logprior_name = 'logprior'
    _logposterior_name = 'logposterior'

    def __init__(self, *args, **kwargs):
        super(LikelihoodPipeline, self).__init__(*args, quiet=True, **kwargs)
        # Check end_calculators are likelihoods
        # for calculator in self.end_calculators:
        #     self._loglikelihood_name = 'loglikelihood'
        #     if not hasattr(calculator, self._loglikelihood_name):
        #         raise PipelineError('End calculator {} has no attribute {}'.format(calculator, self._loglikelihood_name))
        #     loglikelihood = getattr(calculator, self._loglikelihood_name)
        #     if not np.ndim(loglikelihood) == 0:
        #         raise PipelineError('End calculator {} attribute {} must be scalar'.format(calculator, self._loglikelihood_name))
        # Select end_calculators with loglikelihood
        end_calculators = []
        for calculator in self.end_calculators:
            if hasattr(calculator, self._loglikelihood_name):
                end_calculators.append(calculator)
                if self.mpicomm.rank == 0:
                    self.log_info('Found likelihood {}.'.format(calculator.runtime_info.name))
                loglikelihood = getattr(calculator, self._loglikelihood_name)
                if not np.ndim(loglikelihood) == 0:
                    raise PipelineError('End calculator {} attribute {} must be scalar'.format(calculator, self._loglikelihood_name))
        self.__dict__.update(self.select(end_calculators).__dict__)
        if not self.end_calculators:
            raise PipelineError('No likelihood (= calculator with {} attribute set) found'.format(self._loglikelihood_name))
        from .samples.utils import outputs_to_latex
        for calculator in self.end_calculators:
            if self._loglikelihood_name not in calculator.runtime_info.base_params:
                param = Parameter(self._loglikelihood_name, namespace=calculator.runtime_info.namespace, latex=outputs_to_latex(self._loglikelihood_name), derived=True)
                calculator.runtime_info.full_params.set(param)
                calculator.runtime_info.full_params = calculator.runtime_info.full_params
        self.set_params()
        #self.stop_at_inf_prior = False
        self.solved_default = '.best'

    def set_params(self, *args, **kwargs):
        super(LikelihoodPipeline, self).set_params(*args, **kwargs)
        self.solved_params_by_calculator = {}

        def callback(calculator, solved_params):
            solved_params += calculator.runtime_info.solved_params
            for calc in calculator.runtime_info.requires.values():
                callback(calc, solved_params)

        for calculator in self.end_calculators:
            self.solved_params_by_calculator[calculator] = ParameterCollection()
            callback(calculator, self.solved_params_by_calculator[calculator])

        self.solved_params = sum(self.solved_params_by_calculator.values())

    def run(self, **params):
        #if self.stop_at_inf_prior and not np.isfinite(sum_logprior): return
        #pipeline_params = params.copy()
        #for param in self.solved_params:
        #    pipeline_params[param.name] = 0.
        torun = super(LikelihoodPipeline, self).run(**params)
        if not torun:
            return torun

        sum_logprior = 0.
        for param in self.params:
            if param.varied and not param.solved:
                if param.derived and not param.drop:
                    array = self.derived[param]
                    sum_logprior += array.param.prior(array)
                else:
                    sum_logprior += param.prior(params.get(param.name, self._param_values[param.name]))

        #if self.stop_at_inf_prior and not np.isfinite(sum_logprior): return
        indices_best, indices_marg = [], []
        for iparam, param in enumerate(self.solved_params):
            solved = param.derived
            if solved == '.auto': solved = self.solved_default
            if solved == '.best':
                indices_best.append(iparam)
            elif solved == '.marg':  # marg
                indices_marg.append(iparam)
            else:
                raise PipelineError('Unknown option for solved = {}'.format(solved))
        loglikelihoods, solve_calculators, projections, inverse_fishers = [], [], [], []
        for calculator in self.end_calculators:
            loglikelihood_param = calculator.runtime_info.base_params[self._loglikelihood_name]
            loglikelihoods.append(calculator.runtime_info.derived[loglikelihood_param].copy())
            solved_params = self.solved_params_by_calculator[calculator]
            if solved_params:
                flatdiff = calculator.flatdiff  # flatdiff is model - data
                jac = self.jac(calculator, 'flatdiff', solved_params)
                solve_calculators.append(calculator)
                zeros = np.zeros_like(calculator.precision, shape=calculator.precision.shape[0])
                jac = np.column_stack([jac[param.name] if param.name in jac else zeros for param in self.solved_params])
                projector = calculator.precision.dot(jac)
                projection = projector.T.dot(flatdiff)
                invfisher = jac.T.dot(projector)
                projections.append(projection)
                inverse_fishers.append(invfisher)
        dx, x = [], []
        if solve_calculators:
            inverse_priors, x0 = [], []
            for param in self.solved_params:
                scale = getattr(param.prior, 'scale', None)
                inverse_priors.append(0. if scale is None or param.fixed else scale**(-2))
                x0.append(self._param_values[param.name])
            inverse_priors = np.array(inverse_priors)
            sum_inverse_fishers = sum(inverse_fishers + [np.diag(inverse_priors)])
            dx = - np.linalg.solve(sum_inverse_fishers, sum(projections))
            x = x0 + dx
        for param, xx in zip(self.solved_params, x):
            sum_logprior += self.params[param].prior(xx)
            if param.derived:
                self.derived.set(ParameterArray(xx, param), output=True)
        #if self.stop_at_inf_prior and not np.isfinite(sum_logprior): return
        from .samples.utils import outputs_to_latex
        param = Parameter(self._logprior_name, namespace=None, latex=outputs_to_latex(self._logprior_name), derived=True)
        if param in self.derived:
            raise PipelineError('{} is a reserved parameter name, do not use it!'.format(self._logprior_name))
        self.derived.set(ParameterArray(sum_logprior, param), output=True)

        sum_loglikelihood = 0.
        for calculator, loglikelihood in zip(self.end_calculators, loglikelihoods):
            if calculator in solve_calculators:
                index = solve_calculators.index(calculator)
                # Note: priors of solved params have already been added
                if indices_best:
                    loglikelihood -= 1. / 2. * dx[indices_best].dot(inverse_fishers[index][np.ix_(indices_best, indices_best)]).dot(dx[indices_best])
                    loglikelihood -= projections[index][indices_best].dot(dx[indices_best])
                if indices_marg:
                    loglikelihood += 1. / 2. * dx[indices_marg].dot(inverse_fishers[index][np.ix_(indices_marg, indices_marg)]).dot(dx[indices_marg])
            sum_loglikelihood += loglikelihood
        if indices_marg:
            sum_loglikelihood -= 1. / 2. * np.linalg.slogdet(sum_inverse_fishers[np.ix_(indices_marg, indices_marg)])[1]
            #sum_loglikelihood += 1. / 2. * len(indices_marg) * np.log(2. * np.pi)
            # Convention: in the limit of no likelihood constraint on dx, no change to the loglikelihood
            # This allows to ~ keep the interpretation in terms of -1./2. chi2
            ip = inverse_priors[indices_marg]
            sum_loglikelihood += 1. / 2. * np.sum(np.log(ip[ip > 0.]))  # logdet
            #sum_loglikelihood -= 1. / 2. * len(indices_marg) * np.log(2. * np.pi)
        if len(self.end_calculators) == 1 and not loglikelihood_param.namespace:  # loglikelihood already set in self.derived
            self.derived[loglikelihood_param] = sum_loglikelihood
        else:
            param = Parameter(self._loglikelihood_name, namespace=None, latex=outputs_to_latex(self._loglikelihood_name), derived=True)
            if param in self.derived:
                raise PipelineError('{} is a reserved parameter name, do not use it!'.format(self._loglikelihood_name))
            self.derived.set(ParameterArray(sum_loglikelihood, param), output=True)

        param = Parameter(self._logposterior_name, namespace=None, latex=outputs_to_latex(self._logposterior_name), derived=True)
        if param in self.derived:
            raise PipelineError('{} is a reserved parameter name, do not use it!'.format(self._logposterior_name))
        #print(sum_logprior, sum_loglikelihood, indices_marg, indices_best)
        self.derived.set(ParameterArray(sum_logprior + sum_loglikelihood, param), output=True)

        return torun

    #def logprior(self, **params):
    #    logprior = 0.
    #    for name, value in params.items():
    #        logprior += self.params[name].prior(value)
    #    return logprior

    @property
    def loglikelihood(self):
        return self.derived['loglikelihood']

    @property
    def logprior(self):
        return self.derived['logprior']


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
        from .samples import SourceConfig
        values = SourceConfig(self['source']).choice(params=pipeline.params)
        pipeline.run(**dict(zip(pipeline.params.names(), values)))
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

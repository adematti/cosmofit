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
from .utils import BaseClass
from .parameter import Parameter, ParameterArray, ParameterCollection, ParameterConfig
from .io import BaseConfig
from .samples import ParameterValues
from .samples.utils import outputs_to_latex


namespace_delimiter = '.'


class RegisteredCalculator(type(BaseClass)):

    _registry = set()

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        fn = inspect.getfile(cls)
        cls.config_fn = os.path.splitext(fn)[0] + '.yaml'
        meta._registry.add(cls)
        return cls


class BaseCalculator(BaseClass, metaclass=RegisteredCalculator):

    def __setattr__(self, name, item):
        super(BaseCalculator, self).__setattr__(name, item)
        if name in self.runtime_info.requires:
            raise PipelineError('Attribute {} is reserved to a calculator, hence cannot be set!'.format(name))

    def __getattr__(self, name):
        if name == 'requires':
            return {}
        if name == 'params':
            return ParameterCollection()
        if name == 'runtime_info':
            self.runtime_info = RuntimeInfo(self)
            return self.runtime_info
        if name in self.runtime_info.requires:
            toret = self.runtime_info.requires[name]
            if toret.runtime_info.torun:
                toret.run(**toret.runtime_info.params)
            return toret
        return super(BaseCalculator, self).__getattribute__(name)

    def __repr__(self):
        return '{}(namespace={}, basename={})'.format(self.__class__.__name__, self.runtime_info.namespace, self.runtime_info.basename)

    @property
    def mpicomm(self):
        mpicomm = getattr(self, '_mpicomm', None)
        if mpicomm is None: mpicomm = CurrentMPIComm.get()
        self._mpicomm = mpicomm
        return self._mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self._mpicomm = mpicomm

    def save(self, filename):
        if self.mpicomm.rank == 0:
            self.log_info('Saving {}.'.format(filename))
            utils.mkdir(os.path.dirname(filename))
            state = {}
            for name in ['requires']:
                state[name] = getattr(self, name)
            np.save(filename, {**state, **self.__getstate__()}, allow_pickle=True)

    def mpirun(self, **params):
        value = []
        names = self.mpicomm.bcast(params.keys() if self.mpicomm.rank == 0 else None, root=0)
        for name in names:
            params[name] = value = mpy.scatter(np.ravel(params[name]) if self.mpicomm.rank == 0 else None, mpicomm=self.mpicomm, mpiroot=0)
        size = len(value)
        cumsize = np.cumsum([0] + self.mpicomm.allgather(size))
        if not cumsize[-1]: return
        mpicomm = self.mpicomm
        states = {}
        for ivalue in range(size):
            self.mpicomm = COMM_SELF
            self.run(**{name: value[ivalue] for name, value in params.items()})
            states[ivalue + cumsize[mpicomm.rank]] = self.runtime_info.derived
        self.mpicomm = mpicomm
        derived = None
        states = self.mpicomm.gather(states, root=0)
        if self.mpicomm.rank == 0:
            derived = {}
            for st in states:
                derived.update(state)
            derived = ParameterValues.concatenate([derived[i][None, ...] for i in range(cumsize[-1])])
        self.runtime_info.derived = derived

    def __getstate__(self):
        return {}


class PipelineError(Exception):

    pass


def import_cls(clsname, pythonpath=None, registry=BaseCalculator._registry):

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
    module = importlib.import_module(modname)
    return getattr(module, clsname)


class SectionConfig(BaseConfig):

    _sections = []

    def __init__(self, *args, **kwargs):
        super(SectionConfig, self).__init__(*args, **kwargs)
        for name in self._sections:
            self.data[name] = self.data.get(name, None) or {}

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
                self.data[name] = value


class CalculatorConfig(SectionConfig):

    _sections = ['info', 'init', 'params']
    _attrs = ['class', 'info', 'init', 'params', 'emulator', 'load_fn', 'save_fn']

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
        self['params'] = ParameterConfig(self['params'])
        load_fn = data.get('load_fn', None)
        save_fn = data.get('save_fn', None)
        if not isinstance(load_fn, str):
            if load_fn and isinstance(save_fn, str):
                load_fn = save_fn
            else:
                load_fn = None
        if load_fn is not None:
            self['init'] = {}
            self['class'].log_info('Loading {}.'.format(load_fn))
            state = np.load(load_fn, allow_pickle=True)[()]
            if '_engine_cls' in state:  # TODO: better managment of cls names
                from cosmofit.emulators import BaseEmulator
                self._loaded = BaseEmulator.from_state(state)
            else:
                self._loaded = self['class'].from_state(state)
            self['class'] = type(self._loaded)
        self['load_fn'], self['save_fn'] = load_fn, save_fn

    def init(self, namespace=None, params=None, **kwargs):
        initparams = self['params']
        if params is not None:
            if isinstance(params, ParameterCollection):
                initparams = initparams.init(namespace=namespace)
                initparams.update(params)
            else:
                initparams = initparams.with_namespace(namespace)
                initparams.update(params)
                initparams = initparams.init()
        else:
            initparams = initparams.init(namespace=namespace)
        if self['load_fn'] is not None:
            new = self._loaded.copy()
        else:
            new = self['class'].__new__(self['class'])
        params = ParameterCollection(getattr(new, 'params', None)).clone(namespace=namespace, name=[name for name, value in self['params'].namespace.items() if value])
        this_params = getattr(new, 'this_params', None)
        if this_params is not None:
            for iparam, param in enumerate(params):
                if param.basename not in this_params:
                    params[iparam].namespace = None
        params.update(initparams)
        new.info = self['info']
        if self['load_fn'] is None:
            new.params = params.clone(namespace=None)
            try:
                new.__init__(**self['init'])
            except TypeError as exc:
                raise PipelineError('Error in {}'.format(new.__class__)) from exc
        # Propagate possible update to params by __init__
        full_params = ParameterCollection()
        for param in new.params:
            this_namespace = namespace
            for full_param in params:
                if full_param.basename == param.basename:
                    this_namespace = full_param.namespace
            full_params.set(param.clone(namespace=this_namespace))
        new.runtime_info = RuntimeInfo(new, namespace=namespace, config=self, full_params=full_params, **kwargs)
        save_fn = self['save_fn']
        if save_fn is not None:
            new.save(save_fn)
        return new

    def other_items(self):
        return [(name, value) for name, value in self.items() if name not in self._attrs]


class Info(BaseClass):

    # TODO: add bibtex support

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def __getstate__(self):
        return self.__dict__

    def update(self, *args, **kwargs):
        state = self.__getstate__()
        if len(args) == 1 and isinstance(args[0], self.__class__):
            state.update(args[0].__getstate__())
        elif len(args):
            raise ValueError('Unrecognized arguments {}'.format(args))
        state.update(kwargs)
        self.__setstate__(state)

    def clone(self, *args, **kwargs):
        new = self.copy()
        new.update(*args, **kwargs)
        return new

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.__dict__)

    def __eq__(self, other):
        return type(other) == type(self) and other.__dict__ == self.__dict__


class RuntimeInfo(BaseClass):

    def __init__(self, calculator, namespace=None, requires=None, required_by=None, config=None, full_params=None, basename=None):
        self.config = config
        self.basename = basename
        self.namespace = namespace
        self.required_by = set(required_by or [])
        self.requires = requires
        if requires is None:
            self.requires = {}
            for name, clsdict in calculator.requires.items():
                self.requires[name] = CalculatorConfig(clsdict).init(namespace=namespace)
        if full_params is not None:
            self.full_params = full_params
        self.torun = True
        self.calculator = calculator

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
            self._derived_params = self.full_params.select(derived=True)
        return self._derived_params

    @property
    def derived(self):
        if self.torun or getattr(self, '_derived', None) is None:
            self._derived = ParameterValues()
            if self.derived_params:
                state = self.calculator.__getstate__()
                for param in self.derived_params:
                    name = param.basename
                    if name in state: value = state[name]
                    else: value = getattr(self.calculator, name)
                    self._derived.set(ParameterArray(value, param=param))
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
                inst.runtime_info.torun = torun

    @property
    def params(self):
        if getattr(self, '_params', None) is None:
            self._params = {param.basename: param.value for param in self.full_params if not param.derived}
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

    def update(self, *args, **kwargs):
        state = {}
        state.update(*args, **kwargs)
        for name, value in state.items():
            setattr(self, name, value)

    def clone(self, *args, **kwargs):
        new = self.copy()
        new.update(*args, **kwargs)
        return new


class BasePipeline(BaseClass):

    @CurrentMPIComm.enable
    def __init__(self, calculators, params=None, mpicomm=None):

        if isinstance(calculators, BaseCalculator):
            calculators = [calculators]
        is_config = True
        if utils.is_sequence(calculators):
            if not len(calculators):
                raise PipelineError('Need at least one calculator')
            is_config = not isinstance(calculators[0], BaseCalculator)

        if is_config:
            self.params = params
            calculators = BaseConfig(copy.deepcopy(calculators))

            instantiated = []

            def callback_namespace(calculators_in_namespace, todo, namespace=None, replace=False):

                if namespace is not None and namespace_delimiter in namespace:
                    raise PipelineError('Do not use {} in namespace'.format(namespace_delimiter))

                for basename, calcdict in calculators_in_namespace.items():
                    if 'class' in calcdict:
                        tmp = todo(namespace, basename, calcdict)
                        if replace:
                            calculators_in_namespace[basename] = tmp
                    else:  # check for another namespace
                        if namespace is None:
                            newnamespace = basename
                        else:
                            newnamespace = namespace_delimiter.join([namespace, basename])
                        callback_namespace(calcdict, todo, namespace=newnamespace, replace=replace)

            def search_parent_namespace(calculators, namespace):
                # First in this namespace, then go back
                if namespace is None:
                    splitnamespace = None
                else:
                    splitnamespace = namespace.split(namespace_delimiter)

                toret = []
                def callback_parent(tmpnamespace, basename, calcdict):
                    if tmpnamespace is None:
                        toret.append((tmpnamespace, basename, calcdict))
                    else:
                        tmpsplitnamespace = tmpnamespace.split(namespace_delimiter)
                        if tmpsplitnamespace == splitnamespace[:len(tmpsplitnamespace)]:
                            toret.append((tmpnamespace, basename, calcdict))

                callback_namespace(calculators, callback_parent, namespace=None)
                return toret

            def callback_config(namespace, basename, calcdict):
                return CalculatorConfig(calcdict)

            callback_namespace(calculators, callback_config, replace=True)

            def callback_instantiate(namespace, basename, config, required_by=None):
                if required_by is None and any((inst.runtime_info.namespace, inst.runtime_info.basename) == (namespace, basename) for inst in instantiated):
                    return

                def update_from_config_fn(config, cls, config_fn=None):
                    default_config_fn = cls.config_fn
                    if config_fn is None and os.path.isfile(default_config_fn):
                        config_fn = default_config_fn
                    if config_fn is not None:
                        if self.mpicomm.rank == 0:
                            self.log_info('Loading config file {}'.format(config_fn))
                        try:
                            tmpconfig = CalculatorConfig(config_fn, index={'class': cls.__name__})
                        except IndexError:  # no config for this class in config_fn
                            if self.mpicomm.rank == 0:
                                self.log_info('No config for {} found in config file {}'.format(cls.__name__, config_fn))
                            config = config.deepcopy()
                        else:
                            config = tmpconfig.clone(config)
                    else:
                        config = config.deepcopy()
                    return config

                if required_by is None:
                    config = update_from_config_fn(config, config['class'], config_fn=config.get('config_fn', None)).clone(config)
                new = config.init(namespace=namespace, params=self.params, requires={}, required_by=required_by, basename=basename)
                instantiated.append(new)
                for requirementbasename, config in getattr(new, 'requires', {}).items():
                    # Search for parameter
                    config = CalculatorConfig(config)
                    key_requires = requirementbasename

                    requirementnamespace = namespace
                    match_first, match_name = None, None
                    for tmpnamespace, tmpbasename, tmpconfig in search_parent_namespace(calculators, namespace):
                        if issubclass(tmpconfig['class'], config['class']):
                            tmpconfig = update_from_config_fn(config, tmpconfig['class'], config_fn=tmpconfig.get('config_fn', None)).clone(tmpconfig)
                            tmp = (tmpnamespace, tmpbasename, config.clone(tmpconfig))
                            if match_first is None:
                                match_first = tmp
                            if tmpbasename == requirementbasename:
                                match_name = tmp
                                break
                    if match_name:
                        requirementnamespace, requirementbasename, config = match_name
                    elif match_first:
                        requirementnamespace, requirementbasename, config = match_first
                    else:
                        config = update_from_config_fn(config, config['class'], config_fn=config.get('config_fn', None))
                    already_instantiated = False
                    for inst in instantiated:
                        already_instantiated = inst.__class__ == config['class'] and inst.runtime_info.namespace == requirementnamespace\
                                               and inst.runtime_info.basename == requirementbasename and inst.runtime_info.config == config
                        if already_instantiated: break
                    if already_instantiated:
                        requirement = inst
                        requirement.runtime_info.required_by.add(new)
                    else:
                        requirement = callback_instantiate(requirementnamespace, requirementbasename, config, required_by={new})
                    new.runtime_info.requires[key_requires] = requirement

                return new

            callback_namespace(calculators, callback_instantiate)
            self.calculators = instantiated

        self.params = ParameterCollection()
        self.end_calculators = []

        for calculator in self.calculators:
            namespace, full_params = calculator.runtime_info.namespace, calculator.runtime_info.full_params
            namespaces = []
            if namespace is not None:
                namespaces = namespace.split(namespace_delimiter)
                namespaces = [namespace_delimiter.join(namespaces[:stop]) for stop in range(1, len(namespaces) + 1)][::-1]
            namespaces += [None]
            for namespace in namespaces:
                for calc in self.calculators:
                    if calc.runtime_info.namespace == namespace:
                        calc_base_params = [param.basename for param in calc.runtime_info.full_params]
                        for iparam, param in enumerate(full_params):
                            if param.basename in calc_base_params:
                                calc_iparam = calc_base_params.index(param.basename)
                                calc_param = calc.runtime_info.full_params[calc_iparam]
                                if param.namespace is None:
                                    full_params[iparam].namespace = calc_param.namespace
                                elif param.namespace == calc_param.namespace:
                                    full_params[iparam].update(calc_param)
                                calc.runtime_info.full_params[calc_iparam].update(full_params[iparam])

        for calculator in self.calculators:
            self.params.update(calculator.runtime_info.full_params)
            if not calculator.runtime_info.required_by:
                self.end_calculators.append(calculator)
        # Checks
        for param in self.params:
            if not any(param in calculator.runtime_info.full_params for calculator in self.calculators):
                raise PipelineError('Parameter {} is not used by any calculator'.format(param))

        def callback_dependency(calculator, required_by):
            for calc in required_by:
                if calc is calculator:
                    raise PipelineError('Circular dependency for calculator {}'.format(calc))
                callback_dependency(calculator, calc.runtime_info.required_by)

        for calculator in self.calculators:
            callback_dependency(calculator, calculator.runtime_info.required_by)

        self.mpicomm = mpicomm
        # Init run, e.g. for fixed parameters
        # for calculator in self.calculators:
        #     print(calculator.runtime_info.params)

        for calculator in self.end_calculators:
            calculator.run(**calculator.runtime_info.params)

    def run(self, **params):  # params with namespace
        for calculator in self.calculators:
            calculator.runtime_info.torun = False
        for calculator in self.calculators:
            for param in calculator.runtime_info.full_params:
                value = params.get(param.name, None)
                if value is not None and value != calculator.runtime_info.params[param.basename]:
                    calculator.runtime_info.params[param.basename] = value
                    calculator.runtime_info.torun = True
        for calculator in self.end_calculators:
            calculator.run(**calculator.runtime_info.params)
        self.derived = ParameterValues()
        for calculator in self.calculators:
            self.derived.update(calculator.runtime_info.derived)

    def mpirun(self, **params):
        value = []
        for name, value in params.items():
            value = mpy.scatter(np.ravel(value) if self.mpicomm.rank == 0 else None, mpicomm=self.mpicomm, mpiroot=0)
            params[name] = value
        size = len(value)
        cumsize = np.cumsum([0] + self.mpicomm.allgather(size))
        if not cumsize[-1]: return
        mpicomm = self.mpicomm
        states = {}
        for ivalue in range(size):
            self.mpicomm = COMM_SELF
            self.run(**{name: value[ivalue] for name, value in params.items()})
            states[ivalue + cumsize[mpicomm.rank]] = self.derived
        self.mpicomm = mpicomm
        derived = None
        states = self.mpicomm.gather(states, root=0)
        if self.mpicomm.rank == 0:
            derived = {}
            for state in states:
                derived.update(state)
            derived = ParameterValues.concatenate([derived[i][None, ...] for i in range(cumsize[-1])])
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

    def __copy__(self):
        new = super(BasePipeline, self).__copy__()
        new.calculators = []
        for calculator in self.calculators:
            calculator = calculator.copy()
            calculator.runtime_info = calculator.runtime_info.copy()
            new.calculators.append(calculator)
        for calculator in new.calculators:
            calculator.runtime_info.required_by = set([new.calculators[self.calculators.index(calc)] for calc in calculator.runtime_info.required_by])
            calculator.runtime_info.requires = {name: new.calculators[self.calculators.index(calc)] for name, calc in calculator.runtime_info.requires.items()}
            calculator.runtime_info.calculator = calculator
        new.end_calculators = [new.calculators[self.calculators.index(calc)] for calc in self.end_calculators]
        new.params = self.params.copy()
        return new

    def select(self, end_calculators, remove_namespace=False):

        if not utils.is_sequence(end_calculators):
            end_calculators = [end_calculators]
        end_calculators = list(end_calculators)

        for icalc, end_calculator in enumerate(end_calculators):
            if isinstance(end_calculator, str):
                for calculator in self.calculators:
                    if calculator.runtime_info.name == end_calculator:
                        end_calculators[icalc] = calculator
                        break

        new = self.copy()
        new.params = ParameterCollection()
        new.end_calculators = [new.calculators[self.calculators.index(calc)] for calc in end_calculators]
        new.calculators = []

        def callback(calculator):
            for name, calc in calculator.runtime_info.requires.items():
                if name not in [calc.runtime_info.name for calc in new.calculators]:
                    new.calculators.append(calc)
                    callback(calc)

        for end_calculator in new.end_calculators:
            end_calculator.runtime_info.required_by = set([])
            new.calculators.append(end_calculator)
            callback(end_calculator)

        for calculator in new.calculators:
            new.params.update(calculator.runtime_info.full_params)

        return new

    def remove_namespace(self):
        self.params = self.params.clone(namespace=None)
        for calculator in self.calculators:
            calculator.runtime_info = calculator.runtime_info.clone(full_params=calculator.runtime_info.full_params.clone(namespace=None))


class LikelihoodPipeline(BasePipeline):

    def __init__(self, *args, **kwargs):
        super(LikelihoodPipeline, self).__init__(*args, **kwargs)
        # Check end_calculators are likelihoods
        # for calculator in self.end_calculators:
        #     likelihood_name = 'loglikelihood'
        #     if not hasattr(calculator, likelihood_name):
        #         raise PipelineError('End calculator {} has no attribute {}'.format(calculator, likelihood_name))
        #     loglikelihood = getattr(calculator, likelihood_name)
        #     if not np.ndim(loglikelihood) == 0:
        #         raise PipelineError('End calculator {} attribute {} must be scalar'.format(calculator, likelihood_name))
        # Select end_calculators with loglikelihood
        likelihood_name = 'loglikelihood'
        end_calculators = []
        for calculator in self.end_calculators:
            if hasattr(calculator, likelihood_name):
                end_calculators.append(calculator)
                if self.mpicomm.rank == 0:
                    self.log_info('Found likelihood {}.'.format(calculator.runtime_info.name))
                loglikelihood = getattr(calculator, likelihood_name)
                if not np.ndim(loglikelihood) == 0:
                    raise PipelineError('End calculator {} attribute {} must be scalar'.format(calculator, likelihood_name))
        self.__dict__.update(self.select(end_calculators).__dict__)
        from .samples.utils import outputs_to_latex
        for calculator in self.end_calculators:
            if likelihood_name not in calculator.runtime_info.base_params:
                param = Parameter('loglikelihood', namespace=calculator.runtime_info.namespace, latex=outputs_to_latex('loglikelihood'), derived=True)
                calculator.runtime_info.full_params.set(param)
                calculator.runtime_info.full_params = calculator.runtime_info.full_params
                self.params.set(param)

    def run(self, **params):
        super(LikelihoodPipeline, self).run(**params)
        from .samples.utils import outputs_to_latex
        loglikelihood = 0.
        for calculator in self.end_calculators:
            loglikelihood += calculator.runtime_info.derived[calculator.runtime_info.base_params['loglikelihood']]
        param = Parameter('loglikelihood', namespace=None, latex=outputs_to_latex('loglikelihood'), derived=True)
        if param in self.derived: raise PipelineError('loglikelihood is a reserved parameter name, do not use it!')
        self.derived.set(ParameterArray(loglikelihood, param))

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
                    if func is None: continue
                    if isinstance(value, dict):
                        func(**value)
                    else:
                        func(value)

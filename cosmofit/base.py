import os
import sys
import importlib
import inspect
import copy

import numpy as np
from mpytools import CurrentMPIComm, COMM_SELF

from . import utils
from .utils import BaseClass
from .parameter import ParameterCollection, ParameterConfig
from .io import BaseConfig


namespace_delimiter = '.'


def source_sample_params(source='params', fn=None, params=None):
    params = params or {}
    if source == 'params':
        return {param.name: param.value for param in ParameterCollection(params)}
    if source == 'profiles':
        from cosmofit.samples import Profiles
        profiles = Profiles.load(fn)
        argmax = profiles.bestfit.logposterior.argmax()
        return {param.name: profiles.bestfit[param][argmax] for param in profiles.bestfit.params()}
    if source == 'chain':
        from cosmofit.samples import Chain
        chain = Chain.load(fn)
        argmax = chain.logposterior.argmax()
        return {param.name: chain[param].flat[argmax] for param in chain.params()}
    raise ValueError('source must be one of ["params", "profiles", "chain"]')


class RegisteredCalculator(type(BaseClass)):

    _registry = set()

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry.add(cls)
        return cls

    @property
    def config_fn(cls):
        fn = inspect.getfile(cls)
        return os.path.splitext(fn)[0] + '.yaml'


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
            np.save(filename, {'requires': self.requires, **self.__getstate__()}, allow_pickle=True)

    def mpirun(self, **params):
        self.allstates = []
        value = []
        for name, value in params.items():
            params[name] = np.ravel(value)
        csize = len(value)
        if not csize: return
        mpicomm = self.mpicomm
        states = {}
        for ivalue in range(self.mpicomm.rank * csize // self.mpicomm.size, (self.mpicomm.rank + 1) * csize // self.mpicomm.size):
            self.mpicomm = COMM_SELF
            self.run(**{name: value[ivalue] for name, value in params.items()})
            states[ivalue] = self.__getstate__()
        self.mpicomm = mpicomm
        allstates = {}
        for state in self.mpicomm.allgather(states): allstates.update(state)
        self.allstates = [allstates[i] for i in range(csize)]


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
            for name in self.data:
                if name in self._sections and name in args[0]:
                    self.data[name].update(args[0][name])
                else:
                    self.data[name] = args[0][name]
        elif len(args):
            raise ValueError('Unrecognized arguments {}'.format(args))
        for name, value in kwargs.items():
            if name in self._sections:
                self[name].update(value)
            else:
                self.data[name] = value


class CalculatorConfig(SectionConfig):

    _sections = ['info', 'init', 'params']

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
        self['load_fn'], self['save_fn'] = load_fn, save_fn

    def init(self, namespace=None, params=None, **kwargs):
        cls = self['class']
        new = cls.__new__(cls)
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
        new.params = ParameterCollection(getattr(new, 'params', None)).clone(namespace=self['params'].namespace, name=self['params'].namespace)
        new.params.update(initparams)
        new.info = self['info']
        runtime_info = RuntimeInfo(new, namespace=namespace, config=self, **kwargs)
        load_fn = self['load_fn']
        save_fn = self['save_fn']
        if load_fn:
            new = cls.load(save_fn)
        else:
            try:
                new.__init__(**self['init'])
            except TypeError as exc:
                raise PipelineError('Error in {}'.format(new.__class__)) from exc
        new.runtime_info = runtime_info
        if save_fn is not None:
            new.save(save_fn)
        return new


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

    def __init__(self, calculator, namespace=None, requires=None, required_by=None, config=None, basename=None):
        self.config = config
        self.basename = basename
        self.namespace = namespace
        self.required_by = set(required_by or [])
        self.requires = requires
        if requires is None:
            self.requires = {}
            for name, clsdict in calculator.requires:
                self.requires.update(CalculatorConfig(clsdict).init(namespace=namespace))
        self.params = {param.basename: param.value for param in calculator.params}
        self.torun = True

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


class BasePipeline(BaseClass):

    @CurrentMPIComm.enable
    def __init__(self, calculators, params=None, mpicomm=None):
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
            # first in this namespace, then go back
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
            new = config.init(namespace=namespace, params=self.params, requires={}, required_by=required_by, basename=basename)
            instantiated.append(new)
            for requirementbasename, config in getattr(new, 'requires', {}).items():
                # Search for parameter
                config = CalculatorConfig(config)

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
                            self.log_info('No config for {} found in config file {}'.format(cls.__name__, config_fn))
                            config = config.deepcopy()
                        else:
                            config = tmpconfig.clone(config)
                    else:
                        config = config.deepcopy()
                    return config

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
                new.runtime_info.requires[requirementbasename] = requirement

            return new

        callback_namespace(calculators, callback_instantiate)
        self.params = ParameterCollection(params)
        self.end_calculators = []
        self.calculators = instantiated
        for calculator in self.calculators:
            self.params.update(calculator.params)
            if not calculator.runtime_info.required_by:
                self.end_calculators.append(calculator)

        # Checks
        for param in self.params:
            if not any(param in calculator.params for calculator in self.calculators):
                raise PipelineError('Parameter {} is not used by any calculator'.format(param))

        def callback(calculator, required_by):
            for calc in required_by:
                if calc is calculator:
                    raise PipelineError('Circular dependency for calculator {}'.format(calc))
                callback(calculator, calc.runtime_info.required_by)

        for calculator in self.calculators:
            callback(calculator, calculator.runtime_info.required_by)

        self.mpicomm = mpicomm
        # Init run, e.g. for fixed parameters
        # for calculator in self.calculators:
        #     print(calculator.runtime_info.params)

        for calculator in self.end_calculators:
            calculator.run(**calculator.runtime_info.params)

    def run(self, **params):  # params with namespace
        for calculator in self.calculators:
            calculator.runtime_info.torun = False
            for param in calculator.params:
                value = params.get(param.name, None)
                if value is not None and value != calculator.runtime_info.params[param.basename]:
                    calculator.runtime_info.params[param.basename] = value
                    calculator.runtime_info.torun = True
        # for calculator in self.calculators:
        #     if calculator.runtime_info.torun:
        #         print(calculator, calculator.runtime_info.torun)
        for calculator in self.end_calculators:
            calculator.run(**calculator.runtime_info.params)

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


class LikelihoodPipeline(BasePipeline):

    def __init__(self, *args, **kwargs):
        super(LikelihoodPipeline, self).__init__(*args, **kwargs)
        # Check end_calculators are likelihoods
        for calculator in self.end_calculators:
            likelihood_name = 'loglikelihood'
            if not hasattr(calculator, likelihood_name):
                raise PipelineError('End calculator {} has no attribute {}'.format(calculator, likelihood_name))
            loglikelihood = getattr(calculator, likelihood_name)
            if not np.ndim(loglikelihood) == 0:
                raise PipelineError('End calculator {} attribute {} must be scalar'.format(calculator, likelihood_name))

    def run(self, **params):
        super(LikelihoodPipeline, self).run(**params)
        self.loglikelihoods = {}
        for calculator in self.end_calculators:
            self.loglikelihoods[calculator.runtime_info.name] = calculator.loglikelihood
        self.loglikelihood = sum(loglike for loglike in self.loglikelihoods.values())

    def mpirun(self, **params):
        BaseCalculator.mpirun(self, **params)
        self.loglikelihoods = {calculator.runtime_info.name: np.array([state['loglikelihoods'][calculator.runtime_info.name] for state in self.allstates]) for calculator in self.end_calculators}
        self.loglikelihood = np.array([state['loglikelihood'] for state in self.allstates])

    def __getstate__(self):
        state = {}
        for name in ['loglikelihoods', 'loglikelihood']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def logprior(self, **params):
        logprior = 0.
        for name, value in params.items():
            logprior += self.params[name].prior(value)
        return logprior


class RunnerConfig(SectionConfig):

    _sections = ['source']

    def __init__(self, *args, **kwargs):
        super(RunnerConfig, self).__init__(*args, **kwargs)
        if 'run' not in self:
            self['run'] = {section: value for section, value in self.items() if section != 'source'}

    def params(self, params=None):
        from cosmofit import ParameterCollection
        params = {param.name: param.value for param in ParameterCollection(params)}
        for source, value in self['source'].items():
            if isinstance(value, str):
                value = {'fn': value}
            value = dict(value)
            fn = value.pop('fn', None)
            params.update(source_sample_params(source=source, fn=fn, params=value))
        return params

    def run(self, pipeline):
        calculators = {calculator.runtime_info.name: calculator for calculator in pipeline.calculators}
        for section in self['run']:
            for name, value in self['run'][section].items():
                func = getattr(calculators[name], section)
                if isinstance(value, dict):
                    func(**value)
                else:
                    func(value)

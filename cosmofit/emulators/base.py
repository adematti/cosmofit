import os
import sys

import numpy as np

from cosmofit.samples import ParameterValues
from cosmofit.base import PipelineError, SectionConfig, import_cls
from cosmofit.utils import BaseClass
from cosmofit.parameter import Parameter, ParameterArray, ParameterPriorError


class EmulatorConfig(SectionConfig):

    _sections = ['source', 'init', 'fit']

    def run(self, pipeline):
        for calculator in pipeline.calculators:
            calcdict = calculator.runtime_info.config
            if 'emulator' in calcdict:
                emudict = calcdict['emulator']
                if not isinstance(emudict, dict):
                    emudict = {'save_fn': emudict}
                emudict = self.clone(EmulatorConfig(emudict))
                save_fn = emudict.get('save_fn', calcdict.get('save_fn', None))
                cls = import_cls(emudict['class'], pythonpath=emudict.get('pythonpath', None), registry=BaseEmulatorEngine._registry)
                emulator = cls(pipeline.select(calculator), **emudict['init'])
                sample = emudict.get('sample', {})
                if not isinstance(sample, dict):
                    sample = {'samples': sample}
                elif 'class' in sample:
                    from cosmofit.samplers import SamplerConfig
                    config_sampler = SamplerConfig(sample)
                    sample = {'samples': config_sampler.run(emulator.pipeline).samples}
                emulator.set_samples(**sample)
                emulator.fit(**emudict['fit'])
                if save_fn is not None and emulator.mpicomm.rank == 0:
                    emulator.save(save_fn)


class RegisteredEmulatorEngine(type(BaseClass)):

    _registry = set()

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry.add(cls)
        return cls


class BaseEmulatorEngine(BaseClass, metaclass=RegisteredEmulatorEngine):

    def __init__(self, pipeline, mpicomm=None):
        if mpicomm is None:
            mpicomm = pipeline.mpicomm
        self.mpicomm = mpicomm
        self.pipeline = pipeline.copy()
        self.pipeline.mpicomm = mpicomm
        if len(self.pipeline.end_calculators) > 1:
            raise PipelineError('For emulator, pipeline must have a single end calculator; use pipeline.select()')
        calculator = self.pipeline.end_calculators[0]

        states = {}
        rng = np.random.RandomState(seed=42)
        for ii in range(3):
            params = {str(param): param.ref.sample(random_state=rng) for param in self.pipeline.params.select(varied=True)}
            self.pipeline.run(**params)
            for name, value in calculator.__getstate__().items():
                states[name] = states.get(name, []) + [value]

        self.fixed, self.varied = {}, []
        for name, values in states.items():
            if all(np.all(value == values[0]) for value in values):
                self.fixed[name] = values[0]
            else:
                self.varied.append(name)
                dtype = np.asarray(values[0]).dtype
                if not np.issubdtype(dtype, np.inexact):
                    raise ValueError('Attribute {} is of type {}, which is not supported (only float and complex supported)'.format(name, dtype))

        self.params = self.pipeline.params.clone(namespace=None)
        self.varied_params = self.params.select(varied=True, derived=False)
        self.this_params = calculator.runtime_info.full_params.clone(namespace=None)

        for name in self.varied:
            if name not in calculator.runtime_info.base_params:
                param = Parameter(name, namespace=calculator.runtime_info.namespace, derived=True)
                calculator.runtime_info.full_params.set(param)
                calculator.runtime_info.full_params = calculator.runtime_info.full_params

        def serialize_cls(self):
            return ('.'.join([self.__module__, self.__class__.__name__]), os.path.dirname(sys.modules[self.__module__].__file__))

        self._engine_cls = {'emulator': serialize_cls(self), 'calculator': serialize_cls(calculator)}

    def set_samples(self, samples=None, **kwargs):
        if self.mpicomm.bcast(samples is None, root=0):
            samples = self.get_default_samples(**kwargs)
        elif self.mpicomm.rank == 0:
            samples if isinstance(samples, ParameterValues) else ParameterValues.load(samples)
        if self.mpicomm.rank == 0:
            self.samples = ParameterValues()
            for param in self.pipeline.params.select(varied=True, derived=False):
                self.samples.set(ParameterArray(samples[param], param=param.clone(namespace=None)))
            for name in self.varied:
                param = self.pipeline.end_calculators[0].runtime_info.base_params[name]
                self.samples.set(ParameterArray(samples[param], param=param.clone(namespace=None)), output=True)

    def get_default_samples(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def predict(self, **params):
        raise NotImplementedError

    def __getstate__(self):
        state = {}
        for name in ['params', 'this_params', 'fixed', 'varied', '_engine_cls']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def to_calculator(self):
        return BaseEmulator.from_state(self.__getstate__())



class PointEmulatorEngine(BaseClass, metaclass=RegisteredEmulatorEngine):

    def get_default_samples(self):
        from cosmofit.samples import GridSampler
        sampler = GridSampler(self.pipeline, ngrid=1).run()
        return sampler.samples

    def fit(self):
        self.point = {name: np.asarray(self.samples[name][0]) for name in self.samples.outputs}

    def predict(self, **params):
        # Dumb prediction
        return {**self.fixed, **self.point}

    def __getstate__(self):
        state = super(PointEmulatorEngine, self).__getstate__()
        for name in ['varied']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def to_calculator(self):
        return BaseEmulator.from_state(self.__getstate__())


class BaseEmulator(BaseClass):

    @classmethod
    def from_state(cls, state):
        EmulatorEngine = import_cls(*state['_engine_cls']['emulator'])
        Calculator = import_cls(*state['_engine_cls']['calculator'])
        new_name = Calculator.__name__

        clsdict = {}

        def new_run(self, **params):
            Calculator.__setstate__(self, EmulatorEngine.predict(self, **params))

        clsdict = {'run': new_run}

        new_meta = type('MetaEmulatorCalculator', (type(EmulatorEngine), type(Calculator)), {})
        new_cls = new_meta(new_name, (EmulatorEngine, Calculator), clsdict)
        new_cls.config_fn = Calculator.config_fn

        def new_from_state(cls, *args, **kwargs):
            new = cls.__new__(cls)
            new.__dict__.update(EmulatorEngine.from_state(*args, **kwargs).__dict__)
            return new

        def new_load(cls, *args, **kwargs):
            new = cls.__new__(cls)
            new.__dict__.update(EmulatorEngine.load(*args, **kwargs).__dict__)
            return new

        new_cls.from_state = classmethod(new_from_state)
        new_cls.load = classmethod(new_load)
        return new_cls.from_state(state)

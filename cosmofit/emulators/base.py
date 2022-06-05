import os
import sys

import numpy as np

from cosmofit.base import PipelineError, SectionConfig, import_cls
from cosmofit.utils import BaseClass
from cosmofit.parameter import ParameterPriorError


class EmulatorConfig(SectionConfig):

    _sections = ['source', 'fit']

    def run(self, pipeline):
        for calculator in pipeline.calculators:
            calcdict = calculator.runtime_info.config
            if 'emulator' in calcdict:
                emudict = calcdict['emulator']
                if not isinstance(emudict, dict):
                    emudict = {'save_fn': emudict}
                emudict = {**self, **emudict}
                fitdict = {**self['fit'], **emudict.get('fit', {})}
                save_fn = emudict.get('save_fn', self.get('save_fn', None))
                if save_fn is None:
                    save_fn = calcdict.get('save_fn', None)
                cls = import_cls(emudict.get('class', None), pythonpath=emudict.get('pythonpath', None), registry=BaseEmulatorEngine._registry)
                emulator = cls(pipeline.select(calculator, remove_namespace=True), **fitdict)
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
        self.pipeline = pipeline
        if len(self.pipeline.end_calculators) > 1:
            raise PipelineError('For emulator, pipeline must have a single end calculator')
        self.pipeline.mpicomm = mpicomm
        self.centers, self.limits = {}, {}
        self.params = self.pipeline.params
        self.this_params = self.pipeline.end_calculators[0].runtime_info.full_params
        self.varied = self.params.select(varied=True)
        self.varied_names = self.varied.names()
        for param in self.varied:
            name = str(param)
            self.centers[name] = param.value
            if hasattr(param.ref, 'scale'):
                self.limits[name] = (param.value - param.ref.scale, param.value + param.ref.scale)
            elif param.ref.is_proper():
                self.limits[name] = list(param.ref.limits)
            else:
                raise ParameterPriorError('Provide parameter limits')

        def serialize_cls(self):
            return ('.'.join([self.__module__, self.__class__.__name__]), os.path.dirname(sys.modules[self.__module__].__file__))

        self._engine_cls = {'emulator': serialize_cls(self), 'calculator': serialize_cls(self.pipeline.end_calculators[0])}
        self.fit()

    def mpirun_pipeline(self, **params):
        self.pipeline.mpirun(**params)
        allstates = self.pipeline.allstates
        for calculator_name in allstates[0]: break
        fixed, varied = {}, {}
        for name in allstates[0][calculator_name]:
            values = [s[calculator_name][name] for s in allstates]
            if all(np.all(value == values[0]) for value in values):
                fixed[name] = values[0]
            else:
                varied[name] = np.asarray(values)
                dtype = varied[name].dtype
                if not np.issubdtype(dtype, np.inexact):
                    raise ValueError('Attribute {} is of type {}, which is not supported (only float and complex supported)'.format(name, dtype))
        return varied, fixed

    def fit(self):
        # Dumb fit
        self.varied_values, self.fixed_values = self.mpirun_pipeline(**self.centers)

    def predict(self, **params):
        # Dumb prediction
        return {**self.fixed_values, **self.varied_values}

    def __getstate__(self):
        state = {}
        for name in ['params', 'this_params', 'centers', 'varied_names', 'varied_values', 'fixed_values', '_engine_cls']:
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
            self.__dict__.update(EmulatorEngine.predict(self, **params))

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

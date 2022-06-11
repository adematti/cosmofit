import os
import sys

import numpy as np

from cosmofit.base import PipelineError, SectionConfig, import_cls
from cosmofit.utils import BaseClass
from cosmofit.parameter import ParameterPriorError


class EmulatorConfig(SectionConfig):

    _sections = ['source', 'sample', 'fit']

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
        self.varied_params = self.params.select(varied=True, derived=False)
        self.varied_names = self.varied_params.names()
        for param in self.varied_params:
            name = str(param)
            self.centers[name] = param.value
            if param.ref.is_proper():
                self.limits[name] = (param.value - param.proposal, param.value + param.proposal)
            else:
                raise ParameterPriorError('Provide parameter limits or proposal')

        def serialize_cls(self):
            return ('.'.join([self.__module__, self.__class__.__name__]), os.path.dirname(sys.modules[self.__module__].__file__))

        self._engine_cls = {'emulator': serialize_cls(self), 'calculator': serialize_cls(self.pipeline.end_calculators[0])}

    def sample(self):
        # Dumb sampling
        self.samples = self._mpirun_pipeline(self.centers)

    def _mpirun_pipeline(self, values):
        samples = ParameterValues(values, params=self.varied_params)
        self.pipeline.mpirun(**samples.to_dict())
        allstates = self.pipeline.allstates
        for calculator_name in allstates[0]: break
        for name in allstates[0][calculator_name]:
            param = Parameter(name)
            if param in samples:
                raise ValueError('Name {} already in samples'.format(param))
            values = np.asarray([s[calculator_name][name] for s in allstates])
            if all(np.all(value == values[0]) for value in values):
                samples.attrs[name] = values[0]
            else:
                samples[name] = values = np.asarray(values)
                dtype = values.dtype
                if not np.issubdtype(dtype, np.inexact):
                    raise ValueError('Attribute {} is of type {}, which is not supported (only float and complex supported)'.format(name, dtype))
        return samples

    def fit(self):
        pass

    def predict(self, **params):
        # Dumb prediction
        return {**self.fixed_Y, **self.varied_Y}

    def __getstate__(self):
        state = {}
        for name in ['params', 'this_params', 'centers', 'varied_names', 'varied_X', 'varied_Y', 'fixed_Y', '_engine_cls']:
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

import os
import sys

import numpy as np

from cosmofit.samples import ParameterValues
from cosmofit.base import PipelineError, SectionConfig, import_cls
from cosmofit.utils import BaseClass
from cosmofit.parameter import Parameter, ParameterArray, ParameterPriorError, ParameterCollection


class EmulatorConfig(SectionConfig):

    _sections = ['source', 'init', 'fit', 'check']

    def run(self, pipeline):
        for calculator in pipeline.calculators:
            calcdict = calculator.runtime_info.config
            if 'emulator' in calcdict:
                emudict = calcdict['emulator']
                if not isinstance(emudict, dict):
                    emudict = {'save': emudict}
                emudict = self.clone(EmulatorConfig(emudict))
                save_fn = emudict.get('save', calcdict.get('save', None))
                cls = import_cls(emudict['class'], pythonpath=emudict.get('pythonpath', None), registry=BaseEmulatorEngine._registry)
                emulator = cls(pipeline.select(calculator), **emudict['init'])
                sample = emudict.get('sample', {})
                save_samples_fn = None
                if not isinstance(sample, dict):
                    sample = {'samples': sample}
                elif 'class' in sample:
                    from cosmofit.samplers import SamplerConfig
                    config_sampler = SamplerConfig(sample)
                    sampler = config_sampler.run(emulator.pipeline)
                    sample = {'samples': ParameterValues.concatenate(sampler.chains) if hasattr(sampler, 'chains') else sampler.samples}
                emulator.set_samples(**sample)
                if save_samples_fn is not None:
                    emulator.samples.save(save_samples_fn)
                emulator.fit(**emudict['fit'])
                emulator.check(**emudict['check'])
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
        self.params = self.pipeline.params.clone(namespace=None).select(derived=False)
        self.varied_params = self.params.names(varied=True)

        calculator.runtime_info._derived_names = {'fixed': True, 'varied': True}
        fixed, varied = self.pipeline._set_auto_derived([calculator])[1:]
        self.fixed, self.varied = fixed[0], varied[0]

        if self.mpicomm.rank == 0:
            self.log_info('Varied parameters: {}.'.format(self.varied_params))
            self.log_info('Found varying {} and fixed {} outputs.'.format(self.varied, list(self.fixed.keys())))

        def serialize_cls(self):
            return ('.'.join([self.__module__, self.__class__.__name__]), os.path.dirname(sys.modules[self.__module__].__file__))

        self._emulator_cls = serialize_cls(self)
        self._calculator_cls = serialize_cls(calculator)
        self.diagnostics = {}

    def set_samples(self, samples=None, save_fn=None, **kwargs):
        if self.mpicomm.bcast(samples is None, root=0):
            samples = self.get_default_samples(**kwargs)
        elif self.mpicomm.rank == 0:
            samples = samples if isinstance(samples, ParameterValues) else ParameterValues.load(samples)
        if self.mpicomm.rank == 0:
            if save_fn is not None:
                samples.save(save_fn)
            self.samples = ParameterValues(attrs=samples.attrs)
            for param in self.pipeline.params.select(varied=True, derived=False):
                self.samples.set(ParameterArray(samples[param], param=param.clone(namespace=None)))
            for name in self.varied:
                param = self.pipeline.end_calculators[0].runtime_info.base_params[name]
                self.samples.set(ParameterArray(samples[param], param=param.clone(namespace=None)), output=True)
            self.samples = self.samples.ravel()

    def get_default_samples(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def predict(self, **params):
        raise NotImplementedError

    def check(self, mse_stop=None, validation_frac=1.):

        def add_diagnostics(name, value):
            if name not in self.diagnostics:
                self.diagnostics[name] = [value]
            else:
                self.diagnostics[name].append(value)
            return value

        if self.mpicomm.rank == 0:
            self.log_info('Diagnostics:')
        item = '- '
        toret = True
        nsamples = self.mpicomm.bcast(self.samples.size if self.mpicomm.rank == 0 else None)
        nvalidation = int(nsamples * validation_frac + 0.5)
        if nvalidation > nsamples:
            raise ValueError('Cannot use {:d} validation samples (> {:d} total samples)'.format(nvalidation, nsamples))
        rng = np.random.RandomState(seed=42)
        if self.mpicomm.rank == 0:
            samples = self.samples[rng.choice(nsamples, size=nvalidation, replace=False)]
        calculator = self.to_calculator()
        for name in self.varied:
            if name not in calculator.runtime_info.base_params:
                param = Parameter(name, namespace=None, derived=True)
                calculator.runtime_info.full_params.set(param)
                calculator.runtime_info.full_params = calculator.runtime_info.full_params
        derived = calculator.mpirun(**{name: samples[name] if self.mpicomm.rank == 0 else None for name in self.varied_params})

        if self.mpicomm.rank == 0:
            mse = {}
            for name in self.varied:
                mse[name] = np.mean((derived[name] - samples[name]) ** 2)
                msg = '{}mse of {} is {:.3g} (square root = {:.3g})'.format(item, name, mse[name], np.sqrt(mse[name]))
                if mse_stop is not None:
                    test = mse[name] < mse_stop
                    self.log_info('{} {} {:.3g}.'.format(msg, '<' if test else '>', mse_stop))
                    add_diagnostics('mse', mse[name])
                    toret &= test
                else:
                    self.log_info('{}.'.format(msg))
            add_diagnostics('mse', mse)

        self.diagnostics = self.mpicomm.bcast(self.diagnostics, root=0)

        return self.mpicomm.bcast(toret, root=0)

    def to_calculator(self):
        return BaseEmulator.from_state(self.__getstate__())

    def __getstate__(self):
        state = {}
        for name in ['varied_params', 'fixed', 'varied', '_emulator_cls', '_calculator_cls']:
            state[name] = getattr(self, name)
        state['params'] = self.params.__getstate__()
        return state

    def __setstate__(self, state):
        super(BaseEmulatorEngine, self).__setstate__(state)
        self.params = ParameterCollection.from_state(state['params'])


class PointEmulatorEngine(BaseEmulatorEngine):

    def get_default_samples(self):
        from cosmofit.samplers import GridSampler
        sampler = GridSampler(self.pipeline, ngrid=1)
        sampler.run()
        return sampler.samples

    def fit(self):
        self.point = {name: np.asarray(self.samples[name][0]) for name in self.samples.outputs}

    def predict(self, **params):
        # Dumb prediction
        return self.point

    def __getstate__(self):
        state = super(PointEmulatorEngine, self).__getstate__()
        for name in ['point']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


class BaseEmulator(BaseClass):

    @classmethod
    def from_state(cls, state):
        EmulatorEngine = import_cls(*state['_emulator_cls'])
        Calculator = import_cls(*state['_calculator_cls'])
        new_name = Calculator.__name__

        clsdict = {}

        def new_set_params(self, params):
            return params

        def new_run(self, **params):
            Calculator.__setstate__(self, {**self.fixed, **EmulatorEngine.predict(self, **params)})

        def new_getstate(self):
            return Calculator.__getstate__(self)

        clsdict = {'set_params': new_set_params, 'run': new_run, '__getstate__': new_getstate, '__module__': Calculator.__module__}

        new_meta = type('MetaEmulatorCalculator', (type(EmulatorEngine), type(Calculator)), {})
        new_cls = new_meta(new_name, (EmulatorEngine, Calculator), clsdict)
        new_cls.config_fn = Calculator.config_fn

        def from_state(cls, *args, **kwargs):
            new = cls.__new__(cls)
            new.__dict__.update(EmulatorEngine.from_state(*args, **kwargs).__dict__)
            return new

        return from_state(new_cls, state)

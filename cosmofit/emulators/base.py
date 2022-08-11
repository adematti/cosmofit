import os
import sys

import numpy as np

from cosmofit.samples import ParameterValues
from cosmofit.base import PipelineError, SectionConfig, import_cls
from cosmofit import utils, plotting
from cosmofit.utils import BaseClass
from cosmofit.parameter import Parameter, ParameterArray, ParameterPriorError, ParameterCollection, ParameterConfig


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
                plot = emudict.get('plot', None)
                if plot is not None:
                    if not isinstance(plot, dict):
                        plot = {'fn': plot}
                    emulator.plot(**plot)
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
        self.pipeline.remove_namespace()
        self.pipeline.mpicomm = mpicomm
        if len(self.pipeline.end_calculators) > 1:
            raise PipelineError('For emulator, pipeline must have a single end calculator; use pipeline.select()')

        self.params = self.pipeline.params.clone(namespace=None)
        self.varied_params = self.params.names(varied=True, derived=False)

        calculators = []
        for calculator in self.pipeline.calculators:
            if calculator.runtime_info.derived_params and calculator is not self.pipeline.end_calculators[0]:
                calculators.append(calculator)

        calculator = self.pipeline.end_calculators[0]
        calculator.runtime_info.derived_auto = {'.fixed', '.varied'}
        calculators.append(calculator)
        calculators, fixed, varied = self.pipeline._set_derived_auto(calculators)
        self.pipeline.set_params()
        self.fixed, self.varied = {}, set()

        for cc, ff, vv in zip(calculators, fixed, varied):
            bp = cc.runtime_info.base_params
            self.fixed.update({k: v for k, v in ff.items() if k in bp and bp[k].derived})
            self.varied |= {k for k in vv if k in bp and bp[k].derived}

        if self.mpicomm.rank == 0:
            self.log_info('Varied parameters: {}.'.format(self.varied_params))
            self.log_info('Found varying {} and fixed {} outputs.'.format(self.varied, list(self.fixed.keys())))

        def serialize_cls(self):
            clsname = '.'.join([self.__module__, self.__class__.__name__])
            try:
                pythonpath = os.path.dirname(sys.modules[self.__module__].__file__)
            except AttributeError:
                pythonpath = None
            return (clsname, pythonpath)

        self._emulator_cls = serialize_cls(self)
        self._calculator_cls = serialize_cls(calculator)
        self.yaml_data = {}
        self.yaml_data['class'] = calculator.__class__.__name__
        self.yaml_data['info'] = dict(calculator.info)
        self.yaml_data['init'] = {}
        if calculator.runtime_info.config is not None:
            self.yaml_data['init'].update(calculator.runtime_info.config['init'])
        params = {}
        for param in self.params:
            params[param.name] = dict(ParameterConfig(param))
            params[param.name].pop('basename')
        self.yaml_data['params'] = params
        self.yaml_data = utils.dict_to_yaml(self.yaml_data)
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
            for param in self.pipeline.params.select(derived=True):
                if param.basename in self.varied:
                    self.samples.set(ParameterArray(samples[param], param=param.clone(namespace=None)), output=True)
            #self.samples = self.samples.ravel()

    def get_default_samples(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def predict(self, **params):
        raise NotImplementedError

    def subsamples(self, frac=1., nmax=np.inf, seed=42):
        nsamples = self.mpicomm.bcast(self.samples.size if self.mpicomm.rank == 0 else None)
        size = min(int(nsamples * frac + 0.5), nmax)
        if size > nsamples:
            raise ValueError('Cannot use {:d} subsamples (> {:d} total samples)'.format(size, nsamples))
        rng = np.random.RandomState(seed=seed)
        samples = None
        if self.mpicomm.rank == 0:
            samples = self.samples.ravel()[rng.choice(nsamples, size=size, replace=False)]
        return samples

    def check(self, mse_stop=None, **kwargs):

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
        samples = self.subsamples(**kwargs)
        calculator = self.to_calculator(derived=self.varied)
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

    def plot(self, fn=None, name=None, kw_save=None, nmax=100, **kwargs):
        names = name
        if names is None:
            names = self.varied
        if utils.is_sequence(names):
            if fn is None:
                fn = [None] * len(names)
            elif not utils.is_sequence(fn):
                fn = [fn.replace('*', '{}').format(name) for name in names]
        else:
            names = [names]
            fn = [fn]
        samples = self.subsamples(nmax=nmax, **kwargs)
        calculator = self.to_calculator(derived=names)
        derived = calculator.mpirun(**{name: samples[name] if self.mpicomm.rank == 0 else None for name in self.varied_params})

        from matplotlib import pyplot as plt
        if self.mpicomm.rank == 0:
            for name, fn in zip(names, fn):
                plt.close(plt.gcf())
                fig, lax = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'height_ratios': (2, 1)}, figsize=(6, 6), squeeze=True)
                fig.subplots_adjust(hspace=0)
                for d, s in zip(derived[name], samples[name]):
                    lax[0].plot(d.ravel(), color='k', marker='+', markersize=1, alpha=0.2)
                    lax[1].plot((d - s).ravel(), color='k', marker='+',  markersize=1, alpha=0.2)
                lax[0].set_ylabel(name)
                lax[1].set_ylabel(r'$\Delta$ {}'.format(name))
                for ax in lax: ax.grid(True)
                if fn is not None:
                    plotting.savefig(fn, fig=fig, **(kw_save or {}))
        return lax

    def to_calculator(self, derived=None):
        calculator = BaseEmulator.from_state(self.__getstate__())
        if derived is not None:
            for name in derived:
                if name not in calculator.runtime_info.base_params:
                    param = Parameter(name, namespace=None, derived=True)
                    calculator.runtime_info.full_params.set(param)
                    calculator.runtime_info.full_params = calculator.runtime_info.full_params
        return calculator

    def __getstate__(self):
        state = {}
        for name in ['varied_params', 'fixed', 'varied', 'yaml_data', '_emulator_cls', '_calculator_cls']:
            state[name] = getattr(self, name)
        state['params'] = self.params.__getstate__()
        return state

    def save(self, filename, yaml=True):
        self.log_info('Saving {}.'.format(filename))
        utils.mkdir(os.path.dirname(filename))
        state = self.__getstate__()
        if yaml:
            state['config_fn'] = fn = os.path.splitext(filename)[0] + '.yaml'
            self.log_info('Saving {}.'.format(fn))
            self.save_yaml(fn)
        np.save(filename, state, allow_pickle=True)

    def save_yaml(self, fn):
        import yaml
        def list_rep(dumper, data):
            return dumper.represent_sequence(u'tag:yaml.org,2002:seq', data, flow_style=True)
        yaml.add_representer(list, list_rep)
        utils.mkdir(os.path.dirname(fn))
        with open(fn, 'w') as file:
            yaml.dump(self.yaml_data, file, default_flow_style=False)

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
        self.point = {name: np.asarray(self.samples[name].flat[0]) for name in self.samples.outputs}

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
        try:
            new_cls.config_fn = Calculator.config_fn
        except AttributeError:
            pass

        def from_state(cls, *args, **kwargs):
            new = cls.__new__(cls)
            new.__dict__.update(EmulatorEngine.from_state(*args, **kwargs).__dict__)  # should update config_fn
            return new

        return from_state(new_cls, state)

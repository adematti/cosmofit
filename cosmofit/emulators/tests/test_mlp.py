import numpy as np

from cosmofit.emulators import MLPEmulatorEngine
from cosmofit.base import BaseCalculator, BasePipeline
from cosmofit import setup_logging


class LinearModel(BaseCalculator):

    #params = {'a': {'value': 0.5, 'ref': {'limits': [-2., 2.]}}, 'b': {'value': 0.5, 'ref': {'limits': [-2., 2.]}}}
    #params = {'a': {'value': 0.5, 'ref': {'limits': [-2., 2.]}}}
    params = {'b': {'value': 0.5, 'ref': {'limits': [-2., 2.]}}}

    def __init__(self):
        self.x = np.linspace(0.1, 1.1, 11)

    def run(self, a=0., b=0.):
        self.model = a * self.x + b

    def __getstate__(self):
        return {name: getattr(self, name) for name in ['x', 'model']}


def test_mlp_linear(plot=False):
    pipeline = {}
    pipeline['model'] = {'class': 'LinearModel'}
    pipeline = BasePipeline(pipeline)
    calculator = pipeline.end_calculators[0]
    emulator = MLPEmulatorEngine(pipeline, nhidden=(), npcs=3)
    emulator.set_samples(niterations=int(1e5))

    #emulator.fit()
    emulator.fit(batch_sizes=(10000,), epochs=1000, learning_rates=None)
    emulator.check(validation_frac=0.5)

    emulated_calculator = emulator.to_calculator()

    if plot:
        from matplotlib import pyplot as plt
        ax = plt.gca()
        for i, dx in enumerate(np.linspace(-1., 1., 5)):
            calculator.run(**{str(param): param.value + dx for param in calculator.runtime_info.full_params if param.varied})
            emulated_calculator.run(**{str(param): param.value + dx for param in emulated_calculator.runtime_info.full_params if param.varied})
            color = 'C{:d}'.format(i)
            ax.plot(calculator.x, calculator.model, color=color, linestyle='--')
            ax.plot(emulated_calculator.x, emulated_calculator.model, color=color, linestyle='-')
        plt.show()


def test_mlp(plot=False):
    pipeline = {}
    pipeline['theory'] = {'class': 'cosmofit.theories.galaxy_clustering.DampedBAOWigglesTracerPowerSpectrumMultipoles', 'init': {'fiducial': 'DESI'}, 'params': {'.fixed': 'al*'}}
    pipeline['param'] = {'class': 'cosmofit.theories.galaxy_clustering.BAOPowerSpectrumParameterization'}
    pipeline['cosmo'] = {'class': 'cosmofit.theories.primordial_cosmology.Cosmoprimo', 'params': {'.fixed': '*'}}
    pipeline = BasePipeline(pipeline)
    power_bak = pipeline.end_calculators[0].power.copy()
    emulator = MLPEmulatorEngine(pipeline)
    emulator.set_samples()
    emulator.fit()
    emulator.check(validation_frac=1.)

    calculator = emulator.to_calculator()
    calculator.run(**{str(param): param.value for param in calculator.params if param.varied})
    calculator.run(**{str(param): param.value * 1.1 for param in calculator.params if param.varied})
    assert not np.allclose(calculator.power, power_bak)

    if plot:
        from matplotlib import pyplot as plt
        ax = plt.gca()
        for ill, ell in enumerate(calculator.ells):
            color = 'C{:d}'.format(ill)
            ax.plot(calculator.k, calculator.k * power_bak[ill], color=color, label=r'$\ell = {:d}$'.format(ell))
            ax.plot(calculator.k, calculator.k * calculator.power[ill], color=color, linestyle='--')
        plt.show()


if __name__ == '__main__':

    setup_logging()
    test_mlp_linear(plot=True)
    #test_mlp(plot=True)

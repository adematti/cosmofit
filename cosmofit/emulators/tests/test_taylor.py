import numpy as np

from cosmofit.emulators import TaylorEmulatorEngine
from cosmofit.base import BaseCalculator, BasePipeline
from cosmofit import setup_logging


class PowerModel(BaseCalculator):

    def __init__(self, order=4):
        self.x = np.linspace(0.1, 1.1, 11)
        self.order = order

    def set_params(self, params):
        for i in range(self.order):
            params.set({'basename': 'a{:d}'.format(i), 'value': 0.5, 'ref': {'limits': [-2., 2.]}})
        return params

    def run(self, **kwargs):
        self.model = sum(kwargs['a{:d}'.format(i)] * self.x**i for i in range(self.order))

    def __getstate__(self):
        return {name: getattr(self, name) for name in ['x', 'model']}


def test_taylor_power(plot=False):
    for order in range(3, 5):
        pipeline = {}
        pipeline['model'] = {'class': 'PowerModel', 'init': {'order': order}}
        pipeline = BasePipeline(pipeline)
        calculator = pipeline.end_calculators[0]
        emulator = TaylorEmulatorEngine(pipeline, order=order)
        emulator.set_samples()
        emulator.fit()
        emulator.check()

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


def test_taylor(plot=False):
    pipeline = {}
    pipeline['theory'] = {'class': 'cosmofit.theories.bao.DampedBAOWigglesTracerPowerSpectrumMultipoles', 'init': {'fiducial': 'DESI'}, 'params': {'.fixed': 'al*'}}
    pipeline['param'] = {'class': 'cosmofit.theories.power_template.BAOPowerSpectrumParameterization'}
    pipeline['cosmo'] = {'class': 'cosmofit.theories.primordial_cosmology.Cosmoprimo', 'params': {'.fixed': '*'}}
    pipeline = BasePipeline(pipeline)
    power_bak = pipeline.end_calculators[0].power.copy()
    emulator = TaylorEmulatorEngine(pipeline, order=1)
    emulator.set_samples()
    emulator.fit()

    calculator = emulator.to_calculator()
    calculator.run(**{str(param): param.value for param in calculator.params if param.varied})
    assert np.allclose(calculator.power, power_bak)
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
    test_taylor_power(plot=True)
    #test_taylor(plot=True)

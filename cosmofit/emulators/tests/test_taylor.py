import numpy as np

from cosmofit.emulators import TaylorEmulatorEngine
from cosmofit.base import BasePipeline
from cosmofit import setup_logging


def test_taylor(plot=False):
    pipeline = {}
    pipeline['theory'] = {'class': 'cosmofit.theories.bao.Beutler2017BAOGalaxyPowerSpectrum', 'params': {'fixed': 'al*'}}
    pipeline['effectap'] = {'class': 'cosmofit.theories.base.EffectAP', 'init': {'mode': 'qparqper', 'fiducial': 'DESI'}}
    pipeline['cosmo'] = {'class': 'cosmofit.theories.primordial_cosmology.Cosmoprimo', 'params': {'fixed': '*'}}
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
    test_taylor(plot=True)

import os

from cosmofit.emulators.base import BaseEmulatorEngine, BaseEmulator
from cosmofit.base import BasePipeline
from cosmofit import setup_logging


def test_base():
    emulator_dir = '_tests'
    pipeline = {}
    pipeline['theory'] = {'class': 'cosmofit.theories.bao.Beutler2017BAOGalaxyPowerSpectrum'}
    pipeline['effectap'] = {'class': 'cosmofit.theories.base.EffectAP', 'init': {'mode': 'qparqper', 'fiducial': 'DESI'}}
    pipeline['cosmo'] = {'class': 'cosmofit.theories.primordial_cosmology.Cosmoprimo', 'params': {'fixed': '*'}}
    pipeline = BasePipeline(pipeline)
    emulator = PointEmulatorEngine(pipeline)
    assert 'power' in emulator.predict()
    fn = os.path.join(emulator_dir, 'emu.npy')
    emulator.save(fn)
    emulator.to_calculator().run()

    emulator = BaseEmulator.load(fn)
    assert isinstance(emulator, pipeline.end_calculators[0].__class__)
    emulator.run()
    emulator.save(fn)
    emulator = BaseEmulator.load(fn)
    emulator.save(fn)
    print(emulator.__class__.config_fn)


if __name__ == '__main__':

    setup_logging()
    test_base()

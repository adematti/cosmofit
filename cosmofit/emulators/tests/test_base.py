import os

from cosmofit.emulators.base import PointEmulatorEngine, BaseEmulator
from cosmofit.base import BasePipeline
from cosmofit import setup_logging


def test_base():
    emulator_dir = '_tests'
    fn = os.path.join(emulator_dir, 'emu.npy')
    config = {}
    config['theory'] = {'class': 'cosmofit.theories.bao.DampedBAOWigglesTracerPowerSpectrumMultipoles', 'init': {'fiducial': 'DESI'}}
    config['param'] = {'class': 'cosmofit.theories.power_template.BAOPowerSpectrumParameterization'}
    config['cosmo'] = {'class': 'cosmofit.theories.primordial_cosmology.Cosmoprimo', 'params': {'.fixed': '*'}}

    pipeline = BasePipeline(config)
    emulator = PointEmulatorEngine(pipeline)
    emulator.set_samples()
    emulator.fit()
    assert 'power' in emulator.predict()
    emulator.save(fn)
    emulator.to_calculator().run()

    emulator = BaseClass.load(fn)
    assert isinstance(emulator, pipeline.end_calculators[0].__class__)
    emulator.run()

    config['theory']['load_fn'] = fn
    pipeline = BasePipeline(config).select('theory')
    emulator = PointEmulatorEngine(pipeline)
    emulator.set_samples()
    emulator.fit()


if __name__ == '__main__':

    setup_logging()
    test_base()

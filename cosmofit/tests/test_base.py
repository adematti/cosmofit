import numpy as np

from cosmofit import setup_logging
from cosmofit.base import BaseConfig, BasePipeline, ParameterCollection


def test_config():

    config = BaseConfig('test_config.yaml')
    assert np.allclose(config['pipeline']['namespace1']['theory1']['init']['k'], np.linspace(0., 10., 11))


def test_params():

    config = BaseConfig('/local/home/adematti/Bureau/DESI/NERSC/cosmodesi/cosmofit/cosmofit/theory/bao.yaml', index={'class': 'Beutler2017BAOGalaxyPowerSpectrum'})
    params = ParameterCollection(config['params'])


def test_pipeline():

    config = BaseConfig('bao_pipeline.yaml')
    pipeline = BasePipeline(config['pipeline'], params=config.get('params', None))


if __name__ == '__main__':

    setup_logging('warning')

    # test_config()
    # test_params()
    test_pipeline()

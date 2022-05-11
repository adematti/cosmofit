import numpy as np

from cosmofit import setup_logging
from cosmofit.base import BaseConfig, BasePipeline, LikelihoodPipeline, ParameterCollection


def test_config():

    config = BaseConfig('test_config.yaml')
    assert np.allclose(config['pipeline']['namespace1']['theory1']['init']['k'], np.linspace(0., 10., 11))


def test_params():

    config = BaseConfig('/local/home/adematti/Bureau/DESI/NERSC/cosmodesi/cosmofit/cosmofit/theories/bao.yaml', index={'class': 'Beutler2017BAOGalaxyPowerSpectrum'})
    params = ParameterCollection(config['params'])
    assert params.names(name='sigmar') == ['sigmar']
    assert params.names(name=['sigmar', 'al[:5:2]_[-3:2]']) == ['sigmar', 'al0_-3', 'al0_-2', 'al0_-1', 'al0_0', 'al0_1', 'al2_-3', 'al2_-2', 'al2_-1', 'al2_0', 'al2_1', 'al4_-3', 'al4_-2', 'al4_-1', 'al4_0', 'al4_1']


def test_pipeline():

    config = BaseConfig('bao_pipeline.yaml')
    pipeline = BasePipeline(config['pipeline'], params=config.get('params', None))
    assert len(pipeline.end_calculators) == 1 and pipeline.end_calculators[0].runtime_info.basename == 'like'
    """
    assert len(pipeline.calculators) == 6
    assert len(pipeline.params.select(fixed=True)) == 14
    assert pipeline.params.names() == ['QSO.bias', 'QSO.sigmar', 'QSO.sigmas', 'QSO.sigmapar', 'QSO.sigmaper', 'QSO.al0_-3', 'QSO.al0_-2', 'QSO.al0_-1', 'QSO.al0_0', 'QSO.al0_1', 'QSO.al2_-3', 'QSO.al2_-2', 'QSO.al2_-1', 'QSO.al2_0', 'QSO.al2_1', 'QSO.al4_-3', 'QSO.al4_-2', 'QSO.al4_-1', 'QSO.al4_0', 'QSO.al4_1',
                                       'h', 'omega_cdm', 'omega_b', 'A_s', 'k_pivot', 'n_s', 'omega_ncdm', 'N_ur', 'tau_reio', 'w0_fld', 'wa_fld']
    pipeline.run()
    """


def test_likelihood():

    config = BaseConfig('bao_pipeline.yaml')
    pipeline = LikelihoodPipeline(config['pipeline'], params=config.get('params', None))
    pipeline.run(**{'QSO.sigmar': 2.})
    pipeline.mpirun(**{'QSO.sigmar': [1., 2.]})
    print(pipeline.loglikelihood)
    assert len(pipeline.loglikelihood) == 2


def test_sampler():
    from cosmofit.main import sample_from_config
    sample_from_config('bao_pipeline.yaml')


def test_profiler():
    from cosmofit.main import profile_from_config
    profile_from_config('bao_pipeline.yaml')


if __name__ == '__main__':

    setup_logging('info')

    # test_config()
    # test_params()
    # test_pipeline()
    # test_likelihood()
    test_sampler()
    # test_profiler()

import os
import copy

import pytest
import numpy as np

from cosmofit import setup_logging
from cosmofit.base import BaseConfig, BasePipeline, PipelineError, LikelihoodPipeline
from cosmofit.parameter import ParameterConfig, ParameterCollectionConfig, Parameter, ParameterCollection, ParameterPrior, decode_name, find_names, yield_names_latex


def test_config():

    config = BaseConfig('test_config.yaml')
    assert config['pipeline']['namespace1']['like1']['init']['answer_str'] == '42 or 43.00 is 42? {test}'
    assert config['pipeline']['namespace1']['like1']['init']['answer_str_test'] == '42/42'
    assert config['pipeline']['namespace1']['like1']['init']['answer_int'] == 42
    assert config['pipeline']['namespace1']['like1']['init']['answer_int_p2'] == 44
    assert np.allclose(config['pipeline']['namespace1']['theory1']['init']['k'], np.linspace(0., 10., 11))


def test_params():
    config = BaseConfig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'theories', 'bao.yaml'),
                        index={'class': 'DampedBAOWigglesTracerPowerSpectrumMultipoles'})
    params = ParameterCollection(config['params'])
    assert params.params() == params.params()

    assert params.names(name='sigmas') == ['sigmas']
    assert params.names(name=['sigmas', 'al[:5:2]_[-2:2]']) == ['sigmas', 'al0_-2', 'al0_-1', 'al0_0', 'al0_1', 'al2_-2', 'al2_-1', 'al2_0', 'al2_1', 'al4_-2', 'al4_-1', 'al4_0', 'al4_1']

    ref_config = {'al[:5:2]_[-3:2]': {'prior': {'limits': [0., 1]}}, 'sigma': {'latex': r'\sigma'}, 'bias': {'latex': 'b', 'fixed': False}}
    config = copy.deepcopy(ref_config)
    params = ParameterCollection(config)
    assert Parameter(**params['al0_-1'].__getstate__()) == params['al0_-1']
    assert ParameterPrior(**params['al0_-1'].prior.__getstate__()) == params['al0_-1'].prior
    assert (not params['al0_-1'].fixed) and (params['sigma'].fixed) and not (params['bias'].fixed)
    config['.fixed'] = 'al[:5:2]_[-3:2]'
    params = ParameterCollection(ParameterCollectionConfig(config))
    assert params['al0_-1'].fixed and (params['sigma'].fixed) and not (params['bias'].fixed)
    config['.fixed'] = '*'
    params = ParameterCollectionConfig(config).init()
    assert params['sigma'].fixed
    config['.varied'] = 'sigma'
    params = ParameterCollectionConfig(config).init()
    assert not params['sigma'].fixed
    config['.namespace'] = ['bias']
    params = ParameterCollectionConfig(config).init(namespace='test')
    assert params['test.bias'].namespace == 'test'
    config2 = copy.deepcopy(ref_config)
    config2['bias']['latex'] = 'bias'
    config2['.fixed'] = '*'
    config = ParameterCollectionConfig(config)
    config.update(config2)
    params = config.init()
    assert params['sigma'].fixed
    assert params['bias'].latex() == 'bias'
    config2['.namespace'] = '*'
    config.update(config2)
    params = config.init(namespace='test')
    assert params['test.sigma'].namespace == 'test'
    assert decode_name('a*b') == (['a*b'], [])
    assert find_names(['a1', 'a2'], 'a*b') == []
    with pytest.raises(ValueError):
        for name in yield_names_latex('a[:]'):
            print(name)
    config = ParameterCollectionConfig({'a*b': {'value': 1.}})
    assert len(config.init()) == 0
    config = config.clone({'a2b': {'latex': 'latex'}})
    params = config.init()
    assert len(params) == 1
    assert params['a2b'].value == 1. and params['a2b'].latex() == 'latex'
    config = config.clone({'a*b': {'value': 2.}})
    params = config.init()
    assert params['a2b'].value == 2 and params['a2b'].latex() == 'latex'
    config = config.clone({'.delete': '*'})
    assert not len(config.init())
    params = ParameterCollection({'a': 0.2, 'n.a': 1., 'n.b': 2.})
    from cosmofit.base import _best_match_parameter
    assert _best_match_parameter('n.m', 'a', params, choice='max').name == 'n.a'
    assert _best_match_parameter('n.m', 'a', params, choice='min').name == 'a'
    assert _best_match_parameter('m', 'a', params, choice='min').name == 'a'
    assert _best_match_parameter('m', 'b', params, choice='min') is None

    prior = ParameterPrior(dist='norm', loc=0., scale=1.)
    assert np.allclose(prior(0.), 0.)

def test_pipeline():
    config = BaseConfig('bao_power_pipeline.yaml')
    pipeline = BasePipeline(config['pipeline'], params=config.get('params', None))
    # import pytest
    # with pytest.raises(PipelineError):
    #     pipeline.end_calculators[0].power = None
    assert len(pipeline.end_calculators) == 1 and pipeline.end_calculators[0].runtime_info.basename == 'like'
    assert len(pipeline.calculators) == 7
    varied = pipeline.params.select(varied=True)
    assert len(varied) == 13
    assert pipeline.params['QSO.sigmas'].latex() == r'\Sigma_{s}'
    assert len(pipeline.params.select(fixed=True)) == 19
    assert pipeline.params.names() == ['qpar', 'qper', 'bias', 'sigmas', 'sigmapar', 'sigmaper', 'al0_-3', 'al0_-2', 'al0_-1', 'al0_0', 'al0_1',
                                       'al2_-3', 'al2_-2', 'al2_-1', 'al2_0', 'al2_1', 'al4_-3', 'al4_-2', 'al4_-1', 'al4_0', 'al4_1', 'h',
                                       'omega_cdm', 'omega_b', 'A_s', 'k_pivot', 'n_s', 'omega_ncdm', 'N_ur', 'tau_reio', 'w0_fld', 'wa_fld']
    pipeline.run()

    config = BaseConfig('fs_pipeline.yaml')
    pipeline = BasePipeline(config['pipeline'], params=config.get('params', None))
    pipeline.run()


def test_likelihood():

    config = BaseConfig('bao_power_pipeline.yaml')
    pipeline = LikelihoodPipeline(config['pipeline'], params=config.get('params', None))
    print(pipeline.params.select(varied=True))
    pipeline.run(**{'qpar': 1.2})
    likelihood = pipeline.loglikelihood
    pipeline.run(**{'qpar': 1.})
    assert not np.allclose(pipeline.loglikelihood, likelihood)
    pipeline.mpirun(**{'sigmas': [1., 2.]})
    assert len(pipeline.loglikelihood) == 2
    pipeline.mpirun(**{'sigmas': []})
    assert len(pipeline.loglikelihood) == 0


def test_sample(config_fn='bao_power_pipeline.yaml'):
    from cosmofit.main import sample_from_config
    sample_from_config(config_fn)


def test_profile(config_fn='bao_power_pipeline.yaml'):
    from cosmofit.main import profile_from_config
    profiler = profile_from_config(config_fn)
    assert profiler.likelihood.loglikelihood < 0.


def test_do(config_fn='bao_power_pipeline.yaml'):
    from cosmofit.main import do_from_config
    do_from_config(config_fn)


def test_summarize(config_fn='bao_power_pipeline.yaml'):
    from cosmofit.main import summarize_from_config
    summarize_from_config(config_fn)


def test_emulate(config_fn='bao_power_pipeline.yaml'):
    from cosmofit.main import emulate_from_config
    emulate_from_config(config_fn)


if __name__ == '__main__':

    setup_logging('info')

    # test_config()
    # test_params()
    # test_pipeline()
    test_likelihood()
    # test_sample()
    # test_profile()
    # test_do()
    # test_summarize()
    # test_emulate()
    # test_emulate(config_fn='fs_pipeline.yaml')
    # test_profile(config_fn='fs_pipeline.yaml')
    # test_sample(config_fn='fs_pipeline.yaml')
    # test_summarize(config_fn='fs_pipeline.yaml')
    # test_sample(config_fn='png_pipeline.yaml')
    # test_profile(config_fn='png_pipeline.yaml')
    # test_do(config_fn='png_pipeline.yaml')

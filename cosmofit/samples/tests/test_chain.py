import os

import numpy as np
from cosmofit import Chain, Parameter, ParameterPrior, diagnostics, setup_logging


def get_chain(params, nwalkers=4, size=4000, seed=42):
    rng = np.random.RandomState(seed=seed)
    ndim = len(params)
    mean = np.zeros(ndim, dtype='f8')
    cov = np.eye(ndim, dtype='f8')
    cov += 0.1  # off-diagonal
    invcov = np.linalg.inv(cov)
    array = rng.multivariate_normal(mean, cov, size=size)
    diff = array - mean
    logposterior = -0.5 * np.sum(diff.dot(invcov) * diff, axis=-1)
    chain = Chain(list(array.T) + [logposterior], params=params + ['logposterior'])
    for iparam, param in enumerate(chain.params()):
        param.fixed = False
        param.value = mean[iparam]
    return mean, cov, chain


def test_misc():

    chain_dir = '_chain'
    params = ['like.a', 'like.b', 'like.c', 'like.d']
    mean, cov, chain = get_chain(params, nwalkers=10)

    chain['like.a'].param.latex = 'a'
    chain['like.a'].param.prior = ParameterPrior(limits=(-10., 10.))
    pb = chain['like.b'].param
    pb.prior = ParameterPrior(dist='norm', loc=1., limits=(-10., 10.))
    pb = Parameter.from_state(pb.__getstate__())
    chain['logposterior'] = np.zeros(chain.shape, dtype='f8')
    fn = os.path.join(chain_dir, 'chain.npy')
    chain.save(fn)
    base_fn = os.path.join(chain_dir, 'chain')
    chain.write_cosmomc(base_fn, ichain=0)
    chain2 = Chain.read_cosmomc(base_fn)
    chain.to_getdist()
    chain.interval('like.a')
    chain2 = chain.deepcopy()
    chain['like.a'] += 1
    chain2['like.a'].param.latex = 'answer'
    assert np.allclose(chain2['like.a'], chain['like.a'] - 1)
    assert chain2['like.a'].param.latex() != chain['like.a'].param.latex()
    size = chain2.size * 2
    chain2.extend(chain2)
    assert chain2.size == size
    assert chain == chain
    chain.bcast(chain)
    chain['like.a'].param.fixed = False
    assert not chain[4:10]['like.a'].param.fixed
    assert not chain.concatenate(chain, chain)['like.a'].param.fixed


def test_stats():
    params = ['like.a', 'like.b', 'like.c', 'like.d']
    mean, cov, chain = get_chain(params)

    try:
        from emcee import autocorr
        ref = autocorr.integrated_time(chain['like.a'].T, quiet=True)
        assert np.allclose(diagnostics.integrated_autocorrelation_time(chain, params='like.a'), ref)
        assert len(diagnostics.integrated_autocorrelation_time(chain, params=['like.a'] * 2)) == 2
    except ImportError:
        pass

    chains = [chain] + [get_chain(params, seed=seed)[-1] for seed in range(44, 54)]
    assert np.allclose(diagnostics.gelman_rubin(chains, 'like.a', method='diag'), diagnostics.gelman_rubin(chains, 'like.a', method='eigen'))
    assert np.ndim(diagnostics.gelman_rubin(chains, 'like.a', method='eigen')) == 0
    assert diagnostics.gelman_rubin(chains, ['like.a'], method='eigen').shape == (1,)
    assert np.ndim(diagnostics.integrated_autocorrelation_time(chains, 'like.a')) == 0
    assert diagnostics.geweke(chains, params=['like.a'] * 2, first=0.25, last=0.75).shape == (2, len(chains))
    print(chain.to_stats(tablefmt='latex_raw'))


def test_bcast():
    from cosmofit.parameter import ParameterArray
    import mpytools as mpy

    mpicomm = mpy.COMM_WORLD

    array = mpy.array(np.ones(5))
    print(mpicomm.rank, type(mpy.bcast(array, mpiroot=0, mpicomm=mpicomm)))

    array = ParameterArray(np.ones(5), Parameter('a'))
    print(mpicomm.rank, type(mpy.bcast(array, mpiroot=0, mpicomm=mpicomm)))


if __name__ == '__main__':

    setup_logging()

    # test_bcast()
    test_misc()
    test_stats()

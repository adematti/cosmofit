import numpy as np
import dynesty

from cosmofit.samples import Chain
from cosmofit.parameter import ParameterError
from .base import BaseSampler


class FakePool(object):

    def __init__(self, size=1):
        self.size = size

    def map(self, func, values):
        return func(values)


class DynestySampler(BaseSampler):

    def __init__(self, *args, mode='static', nlive=500, bound='multi', sample='auto', update_interval=None, dlogz=0.01, **kwargs):
        self.mode = mode
        self.nlive = int(nlive)
        self.bound = bound
        self.sample = sample
        self.update_interval = update_interval
        self.dlogz = float(dlogz)
        super(DynestySampler, self).__init__(*args, **kwargs)
        for param in self.varied:
            if not param.prior.is_proper():
                raise ParameterError('Prior for {} is improper, Dynesty requires proper priors'.format(param.name))

    def mpiprior_transform(self, values):
        toret = np.empty_like(values)
        for iparam, (value, param) in enumerate(zip(values.T, self.varied)):
            toret[..., iparam] = param.prior.ppf(value)
        return toret

    def _set_sampler(self):
        ndim = len(self.varied)
        self.pool = FakePool(size=self.mpicomm.size)
        use_pool = {'prior_transform': True, 'loglikelihood': True, 'propose_point': False, 'update_bound': False}
        if self.mode == 'static':
            self.sampler = dynesty.NestedSampler(self.loglikelihood, self.mpiprior_transform, ndim, nlive=self.nlive, bound=self.bound, sample=self.sample, update_interval=self.update_interval, pool=self.pool, use_pool=use_pool)
        else:
            self.sampler = dynesty.DynamicNestedSampler(self.loglikelihood, self.mpiprior_transform, ndim, bound=self.bound, sample=self.sample, update_interval=self.update_interval, pool=self.pool, use_pool=use_pool)

    def _run_one(self, start, niterations=300, thin_by=1):
        if self.mode == 'static':
            self.sampler.run_nested(maxiter=niterations + self.sampler.it, dlogz=self.dlogz)
        else:
            self.sampler.run_nested(nlive_init=self.nlive, maxiter=niterations + self.sampler.it, dlogz_init=self.dlogz)
        results = self.sampler.results
        chain = [results['samples'][..., iparam] for iparam, param in enumerate(self.varied)]
        logprior = sum(param.prior(value) for param, value in zip(self.varied, chain))
        chain.append(logprior)
        chain.append(results['logl'] + logprior)
        chain.append(results['logwt'])
        chain.append(np.exp(results.logwt - results.logz[-1]))
        for ivalue, value in enumerate(chain): chain[ivalue] = value[..., None]
        return Chain(chain, params=self.varied + ['logprior', 'logposterior', 'logweight', 'aweight'])

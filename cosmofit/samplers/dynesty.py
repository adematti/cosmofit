import numpy as np
import dynesty

from cosmofit.samples import Chain
from cosmofit.parameter import ParameterError
from .base import BaseSampler


class DynestySampler(BaseSampler):

    def __init__(self, *args, mode='static', nlive=500, bound='multi', sample='auto', update_interval=None, dlogz=0.01, **kwargs):
        super(DynestySampler, self).__init__(*args, **kwargs)
        for param in self.varied:
            if not param.prior.is_proper():
                raise ParameterError('Prior for {} is improper, Dynesty requires proper priors'.format(param.name))

    def _set_sampler(self):
        ndim = len(self.varied)

        if self.mode == 'static':
            self.sampler = dynesty.NestedSampler(self.likelihood.loglkl, self.prior_transform, ndim, nlive=self.nlive, bound=self.bound, sample=self.sample, update_interval=self.update_interval)
        else:
            self.sampler = dynesty.DynamicNestedSampler(self.likelihood.loglkl, self.prior_transform, ndim, bound=self.bound, sample=self.sample, update_interval=self.update_interval)

    def prior_transform(self, values):
        toret = np.empty_like(values)
        for iparam, (value, param) in enumerate(zip(values, self.varied)):
            toret[iparam] = param.prior.ppf(value)
        return toret

    def _sample_single_chain(self, start, nsteps=300, thin_by=1):
        if self.mode == 'static':
            self.sampler.run_nested(nlive_init=self.nlive, maxiter=nsteps, dlogz_init=self.dlogz)
        else:
            self.sampler.run_nested(nlive_init=self.nlive, maxiter=nsteps, dlogz_init=self.dlogz)
        results = self.sampler.results
        chain = [results['samples'][..., iparam] for iparam, param in enumerate(self.varied)]
        logprior = sum(param.prior(value) for param, value in zip(self.varied, chain))
        chain.append(logprior)
        chain.append(results['logl'] + logprior)
        chain.append(results['logwt'])
        chain.append(np.exp(results.logwt - results.logz[-1]))
        return Chain(chain, parameters=self.varied + ['logprior', 'logposterior', 'logweight', 'fweight'])

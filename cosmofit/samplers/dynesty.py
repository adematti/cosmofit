import os

import numpy as np

from cosmofit.samples import Chain
from cosmofit.parameter import ParameterError
from .base import BasePosteriorSampler, load_source, ParameterValues


class FakePool(object):

    def __init__(self, size=1):
        self.size = size

    def map(self, func, values):
        return func(values)


class DynestySampler(BasePosteriorSampler):

    check = None

    def __init__(self, *args, mode='static', nlive=500, bound='multi', sample='auto', update_interval=None, **kwargs):
        self.mode = mode
        self.nlive = int(nlive)
        self.attrs = {'bound': bound, 'sample': sample, 'update_interval': update_interval}
        super(DynestySampler, self).__init__(*args, **kwargs)
        if self.save_fn is None:
            raise ValueError('save_fn must be provided to save dynesty state')
        self.state_fn = [os.path.splitext(fn)[0] + 'dynesty.state' for fn in self.save_fn]

    def prior_transform(self, values):
        toret = np.empty_like(values)
        for iparam, (value, param) in enumerate(zip(values.T, self.varied_params)):
            toret[..., iparam] = param.prior.ppf(value)
        return toret

    def _prepare(self):
        self.resume = self.mpicomm.bcast(self.chains[0] is not None, root=0)

    def _run_one(self, start, max_iterations=100000, **kwargs):
        import dynesty
        from dynesty import utils

        # Instantiation already runs somes samples
        if not hasattr(self, 'sampler'):
            ndim = len(self.varied_params)
            use_pool = {'prior_transform': True, 'loglikelihood': True, 'propose_point': False, 'update_bound': False}
            pool = FakePool(size=self.mpicomm.size)
            if self.mode == 'dynamic':
                self.sampler = dynesty.DynamicNestedSampler(self.loglikelihood, self.prior_transform, ndim, pool=pool, use_pool=use_pool, **self.attrs)
            else:
                self.sampler = dynesty.NestedSampler(self.loglikelihood, self.prior_transform, ndim, nlive=self.nlive, pool=pool, use_pool=use_pool, **self.attrs)

        if self.resume:
            sampler = utils.restore_sampler(self.state_fn[self._ichain])
            del sampler.loglikelihood, sampler.prior_transform, sampler.pool, sampler.M
            if type(sampler) is not type(self.sampler):
                raise ValueError('Previous run used {}, not {}.'.format(type(sampler), type(self.sampler)))
            self.sampler.__dict__.update(sampler.__dict__)

        if self.mode == 'dynamic':
            self.sampler.run_nested(nlive_init=self.nlive, maxiter=max_iterations, **kwargs)
        else:
            self.sampler.run_nested(maxiter=max_iterations, **kwargs)

        results = self.sampler.results
        chain = [results['samples'][..., iparam] for iparam, param in enumerate(self.varied_params)]
        logprior = sum(param.prior(value) for param, value in zip(self.varied_params, chain))
        chain.append(logprior)
        chain.append(results['logl'] + logprior)
        chain.append(results['logwt'])
        chain.append(np.exp(results.logwt - results.logz[-1]))
        if self.mpicomm.rank == 0:
            utils.save_sampler(self.sampler, self.state_fn[self._ichain])
            if self.resume:
                derived = load_source(self.save_fn[self._ichain])[0]
                points = {}
                for param in self.varied_params:
                    points[param.name] = derived.pop(param)
                derived = derived.select(derived=True)
                if self.derived is None:
                    self.derived = [derived, points]
                else:
                    self.derived = [ParameterValues.concatenate([self.derived[0], derived]), {name: np.concatenate([self.derived[1][name], points[name]], axis=0) for name in points}]
        return Chain(chain, params=self.varied_params + ['logprior', 'logposterior', 'logweight', 'aweight'])

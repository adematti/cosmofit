import os

import numpy as np

from cosmofit.samples import Chain
from cosmofit import utils
from .base import BasePosteriorSampler


class PocoMCSampler(BasePosteriorSampler):

    def __init__(self, *args, nwalkers=None, threshold=1.0, scale=True, rescale=False, diagonal=True, flow_config=None, train_config=None, **kwargs):
        super(PocoMCSampler, self).__init__(*args, **kwargs)
        ndim = len(self.varied_params)
        if nwalkers is None:
            nwalkers = 2 * max((int(2.5 * ndim) + 1) // 2, 2)
        self.nwalkers = utils.evaluate(nwalkers, type=int, locals={'ndim': ndim})
        bounds = np.array([tuple(None if np.isinf(lim) else lim for lim in param.prior.limits) for param in self.varied_params], dtype='f8')
        import pocomc
        self.sampler = pocomc.Sampler(self.nwalkers, ndim, self.loglikelihood, self.logprior, bounds=bounds, threshold=threshold, scale=scale,
                                      rescale=rescale, diagonal=diagonal, flow_config=flow_config, train_config=train_config,
                                      vectorize_likelihood=True, vectorize_prior=True, infer_vectorization=False,
                                      output_dir=None, output_label=None)
        if self.save_fn is None:
            raise ValueError('save_fn must be provided to save pocomc state')
        self.state_fn = [os.path.splitext(fn)[0] + 'pocomc.state' for fn in self.save_fn]

    def logprior(self, params, bounds=None):
        return super(PocoMCSampler, self).logprior(params)

    def _prepare(self):
        self.resume = self.mpicomm.bcast(self.chains[0] is not None, root=0)

    def _run_one(self, start, niterations=300, progress=False, **kwargs):
        if self.resume:
            self.sampler.load_state(self.state_fn[self._ichain])
            #self.derived = self.sampler.derived
            #del self.sampler.derived
            from pocomc.tools import FunctionWrapper
            # Because dill is unable to cope with our loglikelihood and logprior
            self.sampler.log_likelihood = FunctionWrapper(self.loglikelihood, args=None, kwargs=None)
            self.sampler.log_prior = FunctionWrapper(self.logprior, args=None, kwargs=None)
            self.sampler.log_likelihood(self.sampler.x)  # to set derived parameters
        else:
            self.sampler.run(prior_samples=start, progress=progress, **kwargs)
        self.sampler.add_samples(n=niterations)
        result = self.sampler.results
        data = [result['samples'][..., iparam] for iparam, param in enumerate(self.varied_params)] + [result['logprior'], result['loglikelihood']]
        # This is not picklable
        del self.sampler.log_likelihood, self.sampler.log_prior
        # Clear saved quantities to save space
        for name in self.sampler.__dict__:
            if name.startswith('saved_'): setattr(self.sampler, name, [])
        # Save last parameters, which be reused in the next run
        #self.sampler.derived = [self.derived[0][:-1], {name: self.derived[1][name][:-1] for name in self.derived[1]}]
        if self.mpicomm.rank == 0:
            self.sampler.save_state(self.state_fn[self._ichain])
        return Chain(data=data, params=self.varied_params + ['logprior', 'loglikelihood'])

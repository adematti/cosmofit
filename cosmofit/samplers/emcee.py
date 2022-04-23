import emcee

from cosmofit.samples import Chain
from .base import BaseSampler


class EmceeSampler(BaseSampler):

    def __init__(self, *args, nwalkers=None, **kwargs):
        super(EmceeSampler, self).__init__(*args, **kwargs)
        if nwalkers is None:
            nwalkers = 2 * ((int(2.5 * len(self.varied)) + 1) // 2)
        self.nwalkers = int(nwalkers)

    def _set_sampler(self):
        self.sampler = emcee.EnsembleSampler(self.nwalkers, len(self.varied), self.likelihood.logposterior, vectorize=True)

    def _sample_single_chain(self, start, nsteps=300, thin_by=1):
        for _ in self.sampler.sample(initial_state=start, iterations=nsteps, progress=False, store=True, thin_by=thin_by):
            pass
        chain = self.sampler.get_chain()
        data = [chain[..., iparam] for iparam, param in enumerate(self.varied)] + [self.sampler.get_log_prob()]
        self.sampler.reset()
        return Chain(data=data, parameters=self.varied + ['logposterior'])

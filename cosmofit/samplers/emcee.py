from cosmofit.samples import Chain
from .base import BasePosteriorSampler


class EmceeSampler(BasePosteriorSampler):

    def __init__(self, *args, nwalkers=None, **kwargs):
        super(EmceeSampler, self).__init__(*args, **kwargs)
        if nwalkers is None:
            nwalkers = 2 * max((int(2.5 * len(self.varied_params)) + 1) // 2, 2)
        self.nwalkers = int(nwalkers)

    def _set_sampler(self):
        import emcee
        self.sampler = emcee.EnsembleSampler(self.nwalkers, len(self.varied_params), self.logposterior, vectorize=True)

    def _run_one(self, start, niterations=300, thin_by=1):
        for _ in self.sampler.sample(initial_state=start, iterations=niterations, progress=False, store=True, thin_by=thin_by, skip_initial_state_check=False):
            pass
        chain = self.sampler.get_chain()
        data = [chain[..., iparam] for iparam, param in enumerate(self.varied_params)] + [self.sampler.get_log_prob()]
        self.sampler.reset()
        return Chain(data=data, params=self.varied_params + ['logposterior'])

import logging

from cosmofit.samples import Chain
from .base import BasePosteriorSampler


class ZeusSampler(BasePosteriorSampler):

    def __init__(self, *args, nwalkers=None, light_mode=False, **kwargs):
        super(ZeusSampler, self).__init__(*args, **kwargs)
        if nwalkers is None:
            nwalkers = 2 * max((int(2.5 * len(self.varied_params)) + 1) // 2, 2)
        self.nwalkers = int(nwalkers)
        self.light_mode = bool(light_mode)

    def _set_sampler(self):
        import zeus
        handlers = logging.root.handlers.copy()
        level = logging.root.level
        self.sampler = zeus.EnsembleSampler(self.nwalkers, len(self.varied_params), self.logposterior, verbose=False, light_mode=self.light_mode, vectorize=False)
        logging.root.handlers = handlers
        logging.root.level = level

    def _run_one(self, start, niterations=300, thin_by=1):
        for _ in self.sampler.sample(start=start, iterations=niterations, progress=False, thin_by=thin_by):
            pass
        chain = self.sampler.get_chain()
        data = [chain[..., iparam] for iparam, param in enumerate(self.varied_params)] + [self.sampler.get_log_prob()]
        self.sampler.reset()
        return Chain(data=data, params=self.varied_params + ['logposterior'])

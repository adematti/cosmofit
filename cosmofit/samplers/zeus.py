import logging

import zeus

from cosmofit.samples import Chain
from .base import BaseSampler


class ZeusSampler(BaseSampler):

    def __init__(self, *args, nwalkers=None, light_mode=False, **kwargs):
        super(ZeusSampler, self).__init__(*args, **kwargs)
        if nwalkers is None:
            nwalkers = 2 * ((int(2.5 * len(self.varied)) + 1) // 2)
        self.nwalkers = int(nwalkers)
        self.light_mode = bool(light_mode)

    def init_sampler(self):
        handlers = logging.root.handlers.copy()
        level = logging.root.level
        self.sampler = zeus.EnsembleSampler(self.nwalkers, len(self.varied), self.likelihood.logposterior, verbose=False, light_mode=self.light_mode)
        logging.root.handlers = handlers
        logging.root.level = level

    def _sample_single_chain(self, start, nsteps=300, thin_by=1):
        for _ in self.sampler.sample(start=start, iterations=nsteps, progress=False, thin_by=self.thin_by):
            pass
        chain = self.sampler.get_chain()
        data = [chain[..., iparam] for iparam, param in enumerate(self.varied)] + [self.sampler.get_log_prob()]
        self.sampler.reset()
        return Chain(data=data, parameters=self.varied + ['logposterior'])

import logging
import random

import numpy as np

from cosmofit.samples import Chain
from cosmofit import utils
from .base import BaseBatchPosteriorSampler


class ZeusSampler(BaseBatchPosteriorSampler):

    def __init__(self, *args, nwalkers=None, light_mode=False, **kwargs):
        super(ZeusSampler, self).__init__(*args, **kwargs)
        ndim = len(self.varied_params)
        if nwalkers is None:
            nwalkers = 2 * max((int(2.5 * ndim) + 1) // 2, 2)
        self.nwalkers = utils.evaluate(nwalkers, type=int, locals={'ndim': ndim})
        import zeus
        handlers = logging.root.handlers.copy()
        level = logging.root.level
        self.sampler = zeus.EnsembleSampler(self.nwalkers, ndim, self.logposterior, verbose=False, light_mode=bool(light_mode), vectorize=True)
        logging.root.handlers = handlers
        logging.root.level = level

    def _run_one(self, start, niterations=300, thin_by=1, progress=False):
        py_random_state_bak, np_random_state_bak = random.getstate(), np.random.get_state()
        random.setstate(utils.numpy_to_python_random_state(self.rng.get_state()))  # self.rng is same for all ranks
        np.random.set_state(self.rng.get_state())
        for _ in self.sampler.sample(start=start, iterations=niterations, progress=progress, thin_by=thin_by):
            pass
        chain = self.sampler.get_chain()
        data = [chain[..., iparam] for iparam, param in enumerate(self.varied_params)] + [self.sampler.get_log_prob()]
        self.sampler.reset()
        random.setstate(py_random_state_bak)
        np.random.set_state(np_random_state_bak)
        return Chain(data=data, params=self.varied_params + ['logposterior'])

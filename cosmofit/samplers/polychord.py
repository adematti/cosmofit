import os
import logging
import itertools

import numpy as np
from mpi4py import MPI

from cosmofit.samples import Chain, ParameterValues, load_source
from cosmofit import utils
from .base import BasePosteriorSampler


class PolychordSampler(BasePosteriorSampler):

    check = None

    def __init__(self, *args, blocks=None, nlive='25*ndim', max_ndead=None, nprior='10*nlive', nfail='1*nlive',
                 nrepeats='2*ndim', nlives=None, do_clustering=True, boost_posterior=0, compression_factor=np.exp(-1),
                 synchronous=True, seed=None, **kwargs):

        super(PolychordSampler, self).__init__(*args, seed=seed, **kwargs)
        logzero = np.nan_to_num(-np.inf)
        di = {'ndim': len(self.varied_params)}
        di['nlive'] = nlive = utils.evaluate(nlive, type=int, locals=di)
        max_ndead = utils.evaluate(max_ndead if max_ndead is not None else -1, type=int, locals=di)
        nprior = utils.evaluate(nprior, type=int, locals=di)
        nfail = utils.evaluate(nfail, type=int, locals=di)
        feedback = {logging.CRITICAL: 0, logging.ERROR: 0, logging.WARNING: 0,
                    logging.INFO: 1, logging.DEBUG: 2}[logging.root.level]
        from .mcmc import _format_blocks
        if blocks is None:
            blocks, oversample_factors = self.likelihood.block_params(params=self.varied_params)
        else:
            blocks, oversample_factors = _format_blocks(blocks, self.varied_params)
        self.varied_params.sort(itertools.chain(*blocks))
        grade_dims = [len(block) for block in blocks]
        grade_frac = [int(o * utils.evaluate(nrepeats, type=int, locals={'ndim': block_size}))
                      for o, block_size in zip(oversample_factors, grade_dims)]

        if self.save_fn is None:
            raise ValueError('save_fn must be provided to save samples in polychord format\
                              alternatively one may update https://github.com/PolyChord/PolyChordLite\
                              to export samples directly as arrays')
        self.base_dirs = [os.path.dirname(fn) for fn in self.save_fn]
        self.file_roots = [os.path.splitext(os.path.basename(fn))[0] + '.polychord' for fn in self.save_fn]
        kwargs = {'nlive': nlive, 'nprior': nprior, 'nfail': nfail,
                  'do_clustering': do_clustering, 'feedback': feedback, 'precision_criterion': 1e-3,
                  'logzero': logzero, 'max_ndead': max_ndead, 'boost_posterior': boost_posterior,
                  'posteriors': True, 'equals': True, 'cluster_posteriors': True,
                  'write_resume': True, 'read_resume': False, 'write_stats': False,
                  'write_live': True, 'write_dead': True, 'write_prior': True,
                  'maximise': False, 'compression_factor': compression_factor, 'synchronous': synchronous,
                  'grade_dims': grade_dims, 'grade_frac': grade_frac, 'nlives': nlives or {}}

        from pypolychord import settings
        self.settings = settings.PolyChordSettings(di['ndim'], 0, seed=(seed if seed is not None else -1), **kwargs)

    def prior_transform(self, values):
        toret = np.empty_like(values)
        for iparam, (value, param) in enumerate(zip(values, self.varied_params)):
            toret[iparam] = param.prior.ppf(value)
        return toret

    def _prepare(self):
        self.settings.read_resume = self.mpicomm.bcast(self.chains[0] is not None, root=0)

    def _run_one(self, start, precision_criterion=None):
        import pypolychord

        def logposterior(values):
            return (max(self.logposterior(values), self.settings.logzero), [])

        def dumper(live, dead, logweights, logZ, logZerr):
            # Periodically save samples
            nlive = len(live)
            samples = list(np.loadtxt(prefix + '.txt', unpack=True))
            aweight, logposterior = [np.concatenate([sample, np.full(nlive, value, dtype='f8')]) for sample, value in zip(samples[:2], [0., np.nan])]
            points = [np.concatenate([sample, live[:, iparam]]) for iparam, sample in enumerate(samples[2: 2 + ndim])]
            logposterior[logposterior <= self.settings.logzero] = -np.inf
            chain = Chain(points + [aweight, logposterior], params=self.varied_params + ['aweight', 'logposterior'])
            # derived is different on each process
            derived = self.mpicomm.gather(self.derived, root=0)
            if self.mpicomm.rank == 0:
                self.derived = [ParameterValues.concatenate([dd[0] for dd in derived]),
                                {name: np.concatenate([dd[1][name] for dd in derived], axis=0) for name in derived[0][1]}]
                if self.resume_derived is not None:
                    self.derived = [ParameterValues.concatenate([self.resume_derived[0], self.derived[0]]), {name: np.concatenate([self.resume_derived[1][name], self.derived[1][name]], axis=0) for name in self.resume_derived[1]}]
                chain = self._set_derived(chain)
                chain.save(self.save_fn[self._ichain])
                self.resume_chain = chain[:-nlive]
                chain[-nlive].save(prefix + '.state.npy')

            self.resume_derived = self.derived
            self.derived = None

        self.likelihood.mpicomm = MPI.COMM_SELF
        self.settings.base_dir = self.base_dirs[self._ichain]
        self.settings.file_root = self.file_roots[self._ichain]
        prefix = os.path.join(self.settings.base_dir, self.settings.file_root)
        if precision_criterion is not None:
            self.settings.precision_criterion = float(precision_criterion)

        self.resume_derived, self.resume_chain = None, None
        if self.settings.read_resume:
            source = load_source([self.save_fn[self._ichain], prefix + '.state.npy'])
            source = source[0].concatenate(source)
            points = {}
            for param in self.varied_params:
                points[param.name] = source.pop(param)
            self.resume_derived = [source.select(derived=True), points]

        ndim = len(self.varied_params)
        kwargs = {}
        if self.mpicomm is not MPI.COMM_WORLD:
            kwargs['comm'] = self.mpicomm
        try:
            pypolychord.run_polychord(logposterior, ndim, 0, self.settings,
                                      prior=self.prior_transform, dumper=dumper,
                                      **kwargs)
        except TypeError as exc:
            raise ImportError('To use polychord in parallel, please use version at https://github.com/adematti/PolyChordLite')
        # derived is different on each process
        self.derived = self.resume_derived
        return self.resume_chain

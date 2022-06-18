import sys
import numbers

import numpy as np
import mpytools as mpy

from cosmofit import utils
from cosmofit.io import ConfigError
from cosmofit.base import SectionConfig, import_cls
from cosmofit.utils import BaseClass, TaskManager
from cosmofit.samples import diagnostics, Chain, ParameterValues
from cosmofit.parameter import ParameterPriorError


class SamplerConfig(SectionConfig):

    _sections = ['init', 'run', 'check']

    def __init__(self, *args, **kwargs):
        # cls, init kwargs
        super(SamplerConfig, self).__init__(*args, **kwargs)
        self['class'] = import_cls(self['class'], pythonpath=self.pop('pythonpath', None), registry=BaseSampler._registry)

    def run(self, likelihood):
        sampler = self['class'](likelihood, **self['init'])
        save_fn = self.get('save_fn', None)
        check = self.get('check', {})
        run_check = bool(check)
        if isinstance(check, bool): check = {}
        min_iterations = self['run'].get('min_iterations', 0)
        max_iterations = self['run'].get('max_iterations', int(1e5) if run_check else sys.maxsize)
        check_every = self['run'].get('check_every', 200)

        run_kwargs = {}
        for name in ['thin_by']:
            if name in self['run']: run_kwargs = self['run'].get(name)

        if save_fn is not None:
            if isinstance(save_fn, str):
                save_fn = [save_fn.replace('*', '{}').format(i) for i in range(sampler.nchains)]
            else:
                if len(save_fn) != sampler.nchains:
                    raise ConfigError('Provide {:d} chain file names'.format(sampler.nchains))

        count_iterations = 0
        is_converged = False
        while not is_converged:
            niter = min(max_iterations - count_iterations, check_every)
            count_iterations += niter
            sampler.run(niterations=niter, **run_kwargs)
            if save_fn is not None and sampler.mpicomm.rank == 0:
                for ichain in range(sampler.nchains):
                    sampler.chains[ichain].save(save_fn[ichain])
            is_converged = sampler.check(**check) if run_check else False
            if count_iterations < min_iterations:
                is_converged = False
            if count_iterations >= max_iterations:
                is_converged = True


class RegisteredSampler(type(BaseClass)):

    _registry = set()

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry.add(cls)
        return cls


class BasePosteriorSampler(BaseClass, metaclass=RegisteredSampler):

    nwalkers = 1

    def __init__(self, likelihood, rng=None, seed=None, max_tries=1000, chains=None, mpicomm=None):
        if mpicomm is None:
            mpicomm = likelihood.mpicomm
        self.mpicomm = mpicomm
        self.likelihood = likelihood
        self.varied_params = self.likelihood.params.select(varied=True, derived=False)
        if self.mpicomm.rank == 0:
            if chains is None: chains = max(self.mpicomm.size - 1, 1)
            if isinstance(chains, numbers.Number):
                self.chains = [None] * int(chains)
            else:
                if not utils.is_sequence(chains):
                    chains = [chains]
                self.chains = [chain if isinstance(chain, Chain) else Chain.load(chain) for chain in chains]
        nchains = self.mpicomm.bcast(len(self.chains), root=0)
        if self.mpicomm.rank != 0:
            self.chains = [None] * len(chains)
        self.max_tries = int(max_tries)
        self._set_rng(rng=rng, seed=seed)
        self.diagnostics = {}

    def loglikelihood(self, values):
        values = np.asarray(values)
        isscalar = values.ndim == 1
        values = np.atleast_2d(values)
        di = {str(param): [value[iparam] for value in values] for iparam, param in enumerate(self.varied_params)}
        self.likelihood.mpirun(**di)
        toret = None
        if self.likelihood.mpicomm.rank == 0:
            if self.derived is None:
                self.derived = [self.likelihood.derived, di]
            else:
                self.derived = [ParameterValues.concatenate([self.derived[0], self.likelihood.derived]), {name: self.derived[1][name] + di[name] for name in di}]
            toret = self.likelihood.loglikelihood
            for array in self.likelihood.derived:
                if array.param.varied_params:
                    toret += array.param.prior(array)
            if isscalar: toret = toret[0]
        return self.likelihood.mpicomm.bcast(toret, root=0)

    def logposterior(self, values):
        values = np.asarray(values)
        isscalar = values.ndim == 1
        values = np.atleast_2d(values)
        params = {str(param): np.array([value[iparam] for value in values]) for iparam, param in enumerate(self.varied_params)}
        toret = self.likelihood.logprior(**params)
        mask = ~np.isinf(toret)
        toret[mask] += self.loglikelihood(values[mask])
        if isscalar: toret = toret[0]
        return toret

    def __getstate__(self):
        state = {}
        for name in ['max_tries', 'diagnostics']:
            state[name] = getattr(self, name)
        return state

    def _set_rng(self, rng=None, seed=None):
        self.rng = self.mpicomm.bcast(rng, root=0)
        if self.rng is None:
            seed = mpy.random.bcast_seed(seed=seed, mpicomm=self.mpicomm, size=None)
        self.rng = np.random.RandomState(seed=seed)

    def _set_sampler(self):
        raise NotImplementedError

    @property
    def nchains(self):
        return len(self.chains)

    @property
    def start(self):
        if getattr(self, '_start', None) is None:
            self._start = self._get_start()
        return self._start

    @start.setter
    def start(self, start):
        if start is not None:
            self._start = np.asarray(start).astype(dtype='f8', copy=False).reshape(self.chains, self.nwalkers, len(self.varied_params))
        self._start = None

    def _get_start(self, max_tries=None):
        if max_tries is None:
            max_tries = self.max_tries

        def get_start(size=1):
            toret = []
            for param in self.varied_params:
                try:
                    toret.append(param.ref.sample(size=size, random_state=self.rng))
                except ParameterPriorError as exc:
                    raise ParameterPriorError('Error in ref/prior distribution of parameter {}'.format(param)) from exc
            return np.array(toret).T

        start = np.full((self.nchains * self.nwalkers, len(self.varied_params)), np.nan)
        logposterior = np.full(self.nchains * self.nwalkers, np.inf)
        for itry in range(max_tries):
            mask = np.isfinite(logposterior)
            if mask.all(): break
            mask = ~mask
            values = get_start(size=mask.sum())
            start[mask] = values
            logposterior[mask] = self.logposterior(values)

        if not np.isfinite(logposterior).all():
            raise ValueError('Could not find finite log posterior after {:d} tries'.format(itry))
        start.shape = (self.nchains, self.nwalkers, -1)
        return start

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def run(self, niterations=300, thin_by=1, **kwargs):
        self.derived = None
        if niterations <= 0:
            return
        if getattr(self, 'sampler', None) is None:
            self._set_sampler()
        nprocs_per_chain = max((self.mpicomm.size - 1) // self.nchains, 1)
        chains = [None] * self.nchains
        if self.mpicomm.bcast(self.chains[0] is not None, root=0):
            start = self.mpicomm.bcast([np.array([chain[param][-1] for param in self.varied_params]).T for chain in self.chains] if self.mpicomm.rank == 0 else None, root=0)
        else:
            start = self.start

        with TaskManager(nprocs_per_task=nprocs_per_chain, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.likelihood.mpicomm = tm.mpicomm
            for ichain in tm.iterate(range(self.nchains)):
                self.derived = None
                chain = self._run_one(start[ichain], niterations=niterations, thin_by=thin_by, **kwargs)
                if self.likelihood.mpicomm.rank == 0:
                    indices_in_chain, indices = ParameterValues(self.derived[1]).match(chain, name=self.varied_params.names())
                    assert indices_in_chain[0].size == chain.size
                    for array in self.derived[0]:
                        chain.set(array[indices].reshape(chain.shape), output=True)
                else:
                    chain = None
                chains[ichain] = chain
        for ichain, chain in enumerate(chains):
            mpiroot_worker = self.mpicomm.rank if chain is not None else None
            for mpiroot_worker in self.mpicomm.allgather(mpiroot_worker):
                if mpiroot_worker is not None: break
            assert mpiroot_worker is not None
            chains[ichain] = Chain.sendrecv(chain, source=mpiroot_worker, dest=0, mpicomm=self.mpicomm)

        if self.mpicomm.rank == 0:
            if self.chains[0] is None:
                self.chains = chains
            else:
                for ichain, (chain, new_chain) in enumerate(zip(chains, self.chains)):
                    self.chains[ichain] = Chain.concatenate(chain, new_chain)
        self.likelihood.mpicomm = self.mpicomm

    def check(self, nsplits=4, stable_over=2, burnin=0.3, eigen_gr_stop=0.03, diag_gr_stop=None,
              cl_diag_gr_stop=None, nsigmas_cl_diag_gr_stop=1., geweke_stop=None, geweke_pvalue_stop=None,
              iact_stop=None, iact_reliable=50, dact_stop=None):

        toret = None

        if self.mpicomm.rank == 0:

            def add_diagnostics(name, value):
                if name not in self.diagnostics:
                    self.diagnostics[name] = [value]
                else:
                    self.diagnostics[name].append(value)
                return value

            def is_stable(name):
                if len(self.diagnostics[name]) < stable_over:
                    return False
                return all(self.diagnostics[name][-stable_over:])

            if 0 < burnin < 1:
                burnin = int(burnin * self.chains[0].shape[0] + 0.5)

            lensplits = (self.chains[0].shape[0] - burnin) // nsplits

            if lensplits * self.nwalkers < 2:
                return False

            split_samples = [chain[burnin + islab * lensplits:burnin + (islab + 1) * lensplits] for islab in range(nsplits) for chain in self.chains]

            self.log_info('Diagnostics:')
            item = '- '
            toret = True
            eigen_gr = diagnostics.gelman_rubin(split_samples, self.varied_params, method='eigen', check_valid='ignore').max() - 1
            msg = '{}max eigen Gelman-Rubin - 1 is {:.3g}'.format(item, eigen_gr)
            if eigen_gr_stop is not None:
                test = eigen_gr < eigen_gr_stop
                self.log_info('{} {} {:.3g}.'.format(msg, '<' if test else '>', eigen_gr_stop))
                add_diagnostics('eigen_gr', test)
                toret = is_stable('eigen_gr')
            else:
                self.log_info('{}.'.format(msg))

            diag_gr = diagnostics.gelman_rubin(split_samples, self.varied_params, method='diag').max() - 1
            msg = '{}max diag Gelman-Rubin - 1 is {:.3g}'.format(item, diag_gr)
            if diag_gr_stop is not None:
                test = diag_gr < diag_gr_stop
                self.log_info('{} {} {:.3g}.'.format(msg, '<' if test else '>', diag_gr_stop))
                add_diagnostics('diag_gr', test)
                toret = is_stable('diag_gr')
            else:
                self.log_info('{}.'.format(msg))

            def cl_lower(samples, params):
                return np.array([samples.interval(param, nsigmas=nsigmas_cl_diag_gr_stop)[0] for param in params])

            def cl_upper(samples, params):
                return np.array([samples.interval(param, nsigmas=nsigmas_cl_diag_gr_stop)[1] for param in params])

            cl_diag_gr = np.max([diagnostics.gelman_rubin(split_samples, self.varied_params, statistic=cl_lower, method='diag'),
                                 diagnostics.gelman_rubin(split_samples, self.varied_params, statistic=cl_upper, method='diag')]) - 1
            msg = '{}max diag Gelman-Rubin - 1 at {:.1f} sigmas is {:.3g}'.format(item, nsigmas_cl_diag_gr_stop, cl_diag_gr)
            if cl_diag_gr_stop is not None:
                test = cl_diag_gr - 1 < cl_diag_gr_stop
                self.log_info('{} {} {:.3g}'.format(msg, '<' if test else '>', cl_diag_gr_stop))
                add_diagnostics('cl_diag_gr', test)
                toret = is_stable('cl_diag_gr')
            else:
                self.log_info('{}.'.format(msg))

            # source: https://github.com/JohannesBuchner/autoemcee/blob/38feff48ae524280c8ea235def1f29e1649bb1b6/autoemcee.py#L337
            geweke = diagnostics.geweke(split_samples, self.varied_params, first=0.25, last=0.75)
            geweke_max = np.max(geweke)
            msg = '{}max Geweke is {:.3g}'.format(item, geweke_max)
            if geweke_stop is not None:
                test = geweke_max < geweke_stop
                self.log_info('{} {} {:.3g}.'.format(msg, '<' if test else '>', geweke_stop))
                add_diagnostics('geweke', test)
                toret = is_stable('geweke')
            else:
                self.log_info('{}.'.format(msg))

            if geweke_pvalue_stop is not None:
                from scipy import stats
                geweke_pvalue = stats.normaltest(toret).pvalue
                test = geweke_pvalue < geweke_pvalue_stop
                self.log_info('{} {} {:.3g}.'.format(msg, '<' if test else '>', geweke_pvalue_stop))
                add_diagnostics('geweke_pvalue', test)
                toret = is_stable('geweke_pvalue')

            split_samples = [chain[burnin:, iwalker] for iwalker in range(self.nwalkers) for chain in self.chains]
            iact = diagnostics.integrated_autocorrelation_time(split_samples, self.varied_params, check_valid='ignore')
            add_diagnostics('tau', iact)

            iact = iact.max()
            msg = '{}max integrated autocorrelation time is {:.3g}'.format(item, iact)
            niterations = len(split_samples[0])
            if iact_reliable * iact < niterations:
                msg = '{} (reliable)'.format(msg)
            if iact_stop is not None:
                test = iact * iact_stop < niterations
                self.log_info('{} {} {:d}/{:.1f} = {:.3g}'.format(msg, '<' if test else '>', niterations, iact_stop, niterations / iact_stop))
                add_diagnostics('iact', test)
                toret = is_stable('iact')
            else:
                self.log_info('{}.'.format(msg))

            tau = self.diagnostics['tau']
            if len(tau) >= 2:
                rel = np.abs(tau[-2] / tau[-1] - 1).max()
                msg = '{}max variation of integrated autocorrelation time is {:.3g}'.format(item, rel)
                if dact_stop is not None:
                    test = rel < dact_stop
                    self.log_info('{} {} {:.3g}'.format(msg, '<' if test else '>', dact_stop))
                    add_diagnostics('dact', test)
                    toret = is_stable('dact')
                else:
                    self.log_info('{}.'.format(msg))

        self.diagnostics = self.mpicomm.bcast(self.diagnostics, root=0)

        return self.mpicomm.bcast(toret, root=0)

import sys
import numbers

import numpy as np
import mpytools as mpy

from cosmofit import utils
from cosmofit.io import ConfigError
from cosmofit.base import SectionConfig, import_class
from cosmofit.utils import BaseClass, TaskManager
from cosmofit.samples import Chain, ParameterValues, load_source
from cosmofit.samples import diagnostics as sample_diagnostics
from cosmofit.parameter import ParameterPriorError, ParameterArray


class SamplerConfig(SectionConfig):

    _sections = ['source', 'init', 'run', 'check']

    def __init__(self, *args, **kwargs):
        # cls, init kwargs
        super(SamplerConfig, self).__init__(*args, **kwargs)
        self['class'] = import_class(self['class'], pythonpath=self.pop('pythonpath', None), registry=BasePosteriorSampler._registry)
        self.is_posterior_sampler = issubclass(self['class'], BasePosteriorSampler)

    def run(self, pipeline):
        save_fn = self.get('save', None)
        if 'save_fn' in self['init'] and 'save' in self:
            raise ConfigError('Provide either init: save_fn or save, not both')

        from cosmofit.samples import SourceConfig
        values = SourceConfig(self['source']).choice(params=pipeline.params)
        pipeline = pipeline.copy()
        params = pipeline.params.deepcopy()
        for param, value in zip(params, values): param.value = value
        pipeline.set_params(params)

        sampler = self['class'](pipeline, **{'save_fn': save_fn, **self['init']})

        can_check = getattr(sampler, 'check', None) is not None
        check = self.get('check', {})
        run_check = bool(check)
        if isinstance(check, bool): check = {}

        if can_check:
            run_kwargs = dict(self['run'])
            min_iterations = run_kwargs.pop('min_iterations', 0)
            max_iterations = run_kwargs.pop('max_iterations', sys.maxsize if run_check else int(1e5))
            check_every = run_kwargs.pop('check_every', 200)
            count_iterations = 0
            is_converged = False
            while not is_converged:
                niter = min(max_iterations - count_iterations, check_every)
                count_iterations += niter
                sampler.run(niterations=niter, **run_kwargs)
                is_converged = sampler.check(**check) if run_check else False
                if count_iterations < min_iterations:
                    is_converged = False
                if count_iterations >= max_iterations:
                    is_converged = True
        else:
            sampler.run(**{**check, **self['run']})

        return sampler


def iterate(func, min_iterations=0, max_iterations=sys.maxsize, check_every=200, **kwargs):
    count_iterations = 0
    is_converged = False
    while not is_converged:
        niter = min(max_iterations - count_iterations, check_every)
        count_iterations += niter
        is_converged = func(niterations=niter, **kwargs)
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

    def __init__(self, likelihood, rng=None, seed=None, max_tries=1000, chains=None, save_fn=None, mpicomm=None):
        if mpicomm is None:
            mpicomm = likelihood.mpicomm
        self.likelihood = BaseClass.copy(likelihood)
        self.mpicomm = mpicomm
        self.likelihood.solved_default = '.marg'
        self.varied_params = self.likelihood.params.select(varied=True, derived=False, solved=False)
        if self.mpicomm.rank == 0:
            self.log_info('Varied parameters: {}.'.format(self.varied_params.names()))
        if not self.varied_params:
            raise ValueError('No parameters to be varied!')
        if self.mpicomm.rank == 0:
            if chains is None: chains = 1
            if isinstance(chains, numbers.Number):
                self.chains = [None] * int(chains)
            else:
                self.chains = load_source(chains)
        nchains = self.mpicomm.bcast(len(self.chains) if self.mpicomm.rank == 0 else None, root=0)
        if self.mpicomm.rank != 0:
            self.chains = [None] * nchains
        self.save_fn = save_fn
        if save_fn is not None:
            if isinstance(save_fn, str):
                self.save_fn = [save_fn.replace('*', '{}').format(i) for i in range(self.nchains)]
            else:
                if len(save_fn) != self.nchains:
                    raise ValueError('Provide {:d} chain file names'.format(self.nchains))
        self.max_tries = int(max_tries)
        self._set_rng(rng=rng, seed=seed)
        self.diagnostics = {}
        self.derived = None

    def loglikelihood(self, values):
        values = self.likelihood.mpicomm.bcast(np.asarray(values), root=0)
        if not values.size:
            return -np.inf
        isscalar = values.ndim == 1
        values = np.atleast_2d(values)
        points = {str(param): values[:, iparam] for iparam, param in enumerate(self.varied_params)}
        self.likelihood.mpirun(**points)
        toret = None
        if self.likelihood.mpicomm.rank == 0:
            if self.derived is None:
                self.derived = [self.likelihood.derived, points]
            else:
                self.derived = [ParameterValues.concatenate([self.derived[0], self.likelihood.derived]), {name: np.concatenate([self.derived[1][name], points[name]], axis=0) for name in points}]
            toret = self.likelihood.loglikelihood + self.likelihood.logprior
        toret = self.likelihood.mpicomm.bcast(toret, root=0)
        mask = np.isnan(toret)
        toret[mask] = -np.inf
        if mask.any() and self.mpicomm.rank == 0:
            self.log_warning('loglikelihood is NaN for {}'.format({k: v[mask] for k, v in points.items()}))
        if isscalar: toret = toret[0]
        return toret

    def logprior(self, values):
        logprior = 0.
        for param, value in zip(self.varied_params, values.T):
            logprior += param.prior(value)
        return logprior

    def logposterior(self, values):
        values = self.likelihood.mpicomm.bcast(np.asarray(values), root=0)
        isscalar = values.ndim == 1
        values = np.atleast_2d(values)
        toret = self.logprior(values)
        mask = ~np.isinf(toret)
        toret[mask] = self.loglikelihood(values[mask])
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

    def _prepare(self):
        pass

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
            raise ValueError('Could not find finite log posterior after {:d} tries'.format(max_tries))
        start.shape = (self.nchains, self.nwalkers, -1)
        return start

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    @property
    def mpicomm(self):
        return self._mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self._mpicomm = self.likelihood.mpicomm = mpicomm

    def _set_derived(self, chain):
        for param in self.likelihood.params.select(fixed=True, derived=False):
            chain.set(ParameterArray(np.full(chain.shape, param.value, dtype='f8'), param))
        values = ParameterValues(self.derived[1])
        indices_in_chain, indices = values.match(chain, params=self.varied_params)
        assert indices_in_chain[0].size == chain.size
        for array in self.derived[0]:
            chain.set(array[indices].reshape(chain.shape + array.shape[1:]), output=True)
        return chain

    def run(self, **kwargs):
        #self.derived = None
        nprocs_per_chain = max((self.mpicomm.size - 1) // self.nchains, 1)
        chains = [None] * self.nchains
        ncalls = [None] * self.nchains
        if self.mpicomm.bcast(self.chains[0] is not None, root=0):
            start = self.mpicomm.bcast([np.array([chain[param][-1] for param in self.varied_params]).T for chain in self.chains] if self.mpicomm.rank == 0 else None, root=0)
        else:
            start = self.start

        mpicomm_bak = self.mpicomm
        self._prepare()
        with TaskManager(nprocs_per_task=nprocs_per_chain, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.mpicomm = tm.mpicomm
            for ichain in tm.iterate(range(self.nchains)):
                self.derived = None
                self._ichain = ichain
                chain = self._run_one(start[ichain], **kwargs)
                if self.mpicomm.rank == 0:
                    ncall = (self.derived[0]['loglikelihood'].size, chain.size)
                    chain = self._set_derived(chain)
                else:
                    chain = ncall = None
                chains[ichain] = chain
                ncalls[ichain] = ncall
        self.mpicomm = mpicomm_bak

        for ichain, chain in enumerate(chains):
            mpiroot_worker = self.mpicomm.rank if chain is not None else None
            for mpiroot_worker in self.mpicomm.allgather(mpiroot_worker):
                if mpiroot_worker is not None: break
            assert mpiroot_worker is not None
            chains[ichain] = Chain.sendrecv(chain, source=mpiroot_worker, dest=0, mpicomm=self.mpicomm)
            ncalls[ichain] = self.mpicomm.bcast(ncalls[ichain], root=mpiroot_worker)

        self.diagnostics['ncall'] = [ncall[0] for ncall in ncalls]
        self.diagnostics['naccepted'] = [ncall[1] for ncall in ncalls]
        if self.mpicomm.rank == 0:
            if self.chains[0] is None:
                self.chains = chains
            else:
                for ichain, (chain, new_chain) in enumerate(zip(chains, self.chains)):
                    self.chains[ichain] = Chain.concatenate(chain, new_chain)
            if self.save_fn is not None:
                for ichain, chain in enumerate(self.chains):
                    chain.save(self.save_fn[ichain])

    def check(self, nsplits=4, stable_over=2, burnin=0.5,
              eigen_gr_max=0.03, diag_gr_max=None, cl_diag_gr_max=None, nsigmas_cl_diag_gr_max=1., geweke_max=None, geweke_pvalue_max=None,
              iterations_per_iact_min=None, iterations_per_iact_reliable=50, dact_max=None,
              eigen_gr_min=None, diag_gr_min=None, cl_diag_gr_min=None, nsigmas_cl_diag_gr_min=None, geweke_min=None, geweke_pvalue_min=None,
              iterations_per_iact_max=None,  dact_min=None,
              diagnostics=None, quiet=False):

        toret = None
        if diagnostics is None:
            diagnostics = self.diagnostics
        verbose = not quiet

        if self.mpicomm.rank == 0:

            def add_diagnostics(name, value):
                if name not in diagnostics:
                    diagnostics[name] = [value]
                else:
                    diagnostics[name].append(value)
                return value

            def is_stable(name):
                if len(diagnostics[name]) < stable_over:
                    return False
                return all(diagnostics[name][-stable_over:])

            def bool_test(value, low=None, up=None):
                test = True
                if low is not None:
                    test &= value > low
                if up is not None:
                    test &= value < up
                return test

            def log_test(msg, test, low=None, up=None):
                if verbose:
                    if low is None: low = ''
                    else: low = '{:.3g}'.format(low)
                    if up is None: up = ''
                    else: up = '{:.3g}'.format(up)
                    isnot = '' if test else 'not '
                    if not (low or up):
                        msg = '{}.'.format(msg)
                    elif (low and up):
                        msg = '{}; {}in [{}, {}].'.format(msg, isnot, low, up)
                    elif low:
                        msg = '{}; {}> {}.'.format(msg, isnot, low)
                    elif up:
                        msg = '{}; {}< {}.'.format(msg, isnot, up)
                    self.log_info(msg)

            def full_test(key, name, value, low=None, up=None):
                add_diagnostics(key, value)
                key = '{}_test'.format(key)
                msg = '{}{} is {:.3g}'.format(item, name, value)
                if any(lu is not None for lu in (low, up)):
                    test = bool_test(value, low=low, up=up)
                    log_test(msg, test, low=low, up=up)
                    add_diagnostics(key, test)
                    return is_stable(key)
                if verbose:
                    self.log_info('{}.'.format(msg))
                    return True

            if 0 < burnin < 1:
                burnin = int(burnin * self.chains[0].shape[0] + 0.5)

            lensplits = (self.chains[0].shape[0] - burnin) // nsplits

            if lensplits * self.nwalkers < 2:
                return False

            split_samples = [chain[burnin + islab * lensplits:burnin + (islab + 1) * lensplits] for islab in range(nsplits) for chain in self.chains]

            if verbose: self.log_info('Diagnostics:')
            item = '- '
            toret = True

            eigen_gr = sample_diagnostics.gelman_rubin(split_samples, self.varied_params, method='eigen', check_valid='ignore').max() - 1
            toret &= full_test('eigen_gr','max eigen Gelman-Rubin - 1', eigen_gr, eigen_gr_min, eigen_gr_max)

            diag_gr = sample_diagnostics.gelman_rubin(split_samples, self.varied_params, method='diag').max() - 1
            toret &= full_test('diag_gr', 'max diag Gelman-Rubin - 1', diag_gr, diag_gr_min, diag_gr_max)

            def cl_lower(samples, params):
                return np.array([samples.interval(param, nsigmas=nsigmas_cl_diag_gr_max)[0] for param in params])

            def cl_upper(samples, params):
                return np.array([samples.interval(param, nsigmas=nsigmas_cl_diag_gr_max)[1] for param in params])

            cl_diag_gr = np.max([sample_diagnostics.gelman_rubin(split_samples, self.varied_params, statistic=cl_lower, method='diag'),
                                 sample_diagnostics.gelman_rubin(split_samples, self.varied_params, statistic=cl_upper, method='diag')]) - 1
            toret &= full_test('cl_diag_gr', 'max diag Gelman-Rubin - 1 at {:.1f} sigmas'.format(nsigmas_cl_diag_gr_max), cl_diag_gr, cl_diag_gr_min, cl_diag_gr_max)

            # source: https://github.com/JohannesBuchner/autoemcee/blob/38feff48ae524280c8ea235def1f29e1649bb1b6/autoemcee.py#L337
            all_geweke = sample_diagnostics.geweke(split_samples, self.varied_params, first=0.1, last=0.5)
            geweke = np.max(all_geweke)
            toret &= full_test('geweke', 'max Geweke', geweke, geweke_min, geweke_max)

            if any(lu is not None for lu in (geweke_pvalue_min, geweke_pvalue_max)):
                # normaltest raises an error for less than 6 samples
                from scipy import stats
                geweke_pvalue = stats.normaltest(all_geweke).pvalue
                toret &= full_test('geweke_pvalue', 'Geweke p-value', geweke_pvalue, geweke_pvalue_min, geweke_pvalue_max)

            split_samples = []
            for chain in self.chains:
                chain = chain[burnin:]
                chain = chain.reshape(len(chain), -1)
                for iwalker in range(chain.shape[1]):
                    split_samples.append(chain[..., iwalker])
            iact = sample_diagnostics.integrated_autocorrelation_time(split_samples, self.varied_params, check_valid='ignore')
            add_diagnostics('iact', iact)
            niterations = len(split_samples[0])
            iact = iact.max()
            name = '{:d} iterations / integrated autocorrelation time'.format(niterations)
            if iterations_per_iact_reliable * iact < niterations:
                name = '{} (reliable)'.format(name)
            toret &= full_test('iterations_per_iact', name, niterations / iact, iterations_per_iact_min, iterations_per_iact_max)

            iact = diagnostics['iact']
            if len(iact) >= 2:
                rel = np.abs(iact[-2] / iact[-1] - 1).max()
                toret &= full_test('dact', 'max variation of integrated autocorrelation time', rel, dact_min, dact_max)

        diagnostics.update(self.mpicomm.bcast(diagnostics, root=0))

        return self.mpicomm.bcast(toret, root=0)

import numbers

import numpy as np
import mpytools as mpy
from mpytools import CurrentMPIComm

from cosmofit import utils
from cosmofit.utils import BaseClass, TaskManager
from cosmofit.chains import diagnostics, Chain


class BaseSampler(BaseClass):

    nwalkers = 1

    @CurrentMPIComm.enable
    def __init__(self, likelihood, rng=None, seed=None, max_tries=1000, chains=None, mpicomm=None):
        self.mpicomm = mpicomm
        self.likelihood = likelihood
        self.varied = self.likelihood.parameters(varied=True)
        if chains is None: chains = max(self.mpicomm.size - 1, 1)
        if isinstance(chains, numbers.Number):
            self.chains = [None] * int(chains)
        else:
            if not utils.is_sequence(chains):
                chains = [chains]
            self.chains = [chain if isinstance(chain, Chain) else Chain.load(chain) for chain in chains]
        self.max_tries = int(max_tries)
        self._set_rng(rng=rng, seed=seed)
        self._set_sampler()
        self.diagnostics = {}

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
        if getattr(self, '_start') is None:
            self._start = self._get_start()
        return self._start

    @start.setter
    def start(self, start):
        if start is not None:
            self._start = np.asarray(start).astype(dtype='f8', copy=False).reshape(self.chains, self.nwalkers, len(self.varied))
        self._start = None

    def _get_start(self, max_tries=1000):
        if max_tries is None:
            max_tries = self.max_tries

        def get_start(size=1):
            toret = []
            for param in self.varied:
                if param.ref.is_proper():
                    toret.append(param.ref.sample(size=size, random_state=self.rng))
                else:
                    toret.append([param.value] * size)
            return np.array(toret).T

        itry = 0
        start = np.full(self.nwalkers * self.nchains * len(self.varied), np.nan)
        while itry < max_tries:
            mask = np.isnan(start)
            values = get_start(size=mask.sum())
            itry += 1
            start[mask] = self.likelihood.logposterior(values)

        if np.isnan(start).any():
            raise ValueError('Could not find finite log posterior after {:d} tries'.format(max_tries))

        return self.mpicomm.bcast(np.array(start), root=0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def run(self, niterations=300, thin_by=1, **kwargs):
        chains = [None] * self.nchains
        nprocs_per_chain = max((self.mpicomm.size - 1) // self.nchains, 1)
        if self.chains[0] is not None: start = [np.array([chain[param][-1] for param in self.varied]).T for chain in self.chains]
        else: start = self.start

        with TaskManager(nprocs_per_task=nprocs_per_chain, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.likelihood.mpicomm = tm.mpicomm
            for ichain in tm.iterate(range(self.nchains)):
                chains[ichain] = self._run_one(start[ichain], niterations=niterations, thin_by=thin_by, **kwargs)
        for ichain, chain in enumerate(chains):
            mpiroot_worker = self.mpicomm.rank if chain is not None else None
            for mpiroot_worker in self.mpicomm.allgather(mpiroot_worker):
                if mpiroot_worker is not None: break
            assert mpiroot_worker is not None
            chains[ichain] = Chain.bcast(chain, mpicomm=self.mpicomm, mpiroot=mpiroot_worker)

        if self.chains[0] is None:
            self.chains = chains
        else:
            for ichain, (chain, new_chain) in enumerate(zip(chains, self.chains)):
                self.chains[ichain] = Chain.concatenate(chain, new_chain)

    def diagnose(self, nsplits=4, stable_over=2, burnin=0.3, eigen_gr_stop=0.03, diag_gr_stop=None,
                 cl_diag_gr_stop=None, nsigmas_cl_diag_gr_stop=1., geweke_stop=None,
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
                burnin = int(burnin * len(self.chains[0]) + 0.5)

            lensplits = (len(self.chains[0]) - burnin) // nsplits

            if lensplits * self.nwalkers < 2:
                return False

            split_samples = [chain[burnin + islab * lensplits:burnin + (islab + 1) * lensplits] for islab in range(nsplits) for chain in self.chains]

            self.log_info('Diagnostics:')
            item = '- '
            toret = True

            eigen_gr = diagnostics.gelman_rubin(split_samples, self.varied, method='eigen', check=False).max() - 1
            msg = '{}max eigen Gelman-Rubin - 1 is {:.3g}'.format(item, eigen_gr)
            if eigen_gr_stop is not None:
                test = eigen_gr < eigen_gr_stop
                self.log_info('{} {} {:.3g}.'.format(msg, '<' if test else '>', eigen_gr_stop))
                add_diagnostics('eigen_gr', test)
                toret = is_stable('eigen_gr')
            else:
                self.log_info('{}.'.format(msg))

            varied = split_samples[0].parameters(varied=True)

            diag_gr = diagnostics.gelman_rubin(split_samples, varied, method='diag').max() - 1
            msg = '{}max diag Gelman-Rubin - 1 is {:.3g}'.format(item, diag_gr)
            if diag_gr_stop is not None:
                test = diag_gr < diag_gr_stop
                self.log_info('{} {} {:.3g}.'.format(msg, '<' if test else '>', diag_gr_stop))
                add_diagnostics('diag_gr', test)
                toret = is_stable('diag_gr')
            else:
                self.log_info('{}.'.format(msg))

            def cl_lower(samples, parameters):
                return samples.interval(parameters, nsigmas=nsigmas_cl_diag_gr_stop)[:, 0]

            def cl_upper(samples, parameters):
                return samples.interval(parameters, nsigmas=nsigmas_cl_diag_gr_stop)[:, 1]

            cl_diag_gr = np.max([diagnostics.gelman_rubin(split_samples, varied, statistic=cl_lower, method='diag'),
                                 diagnostics.gelman_rubin(split_samples, varied, statistic=cl_upper, method='diag')]) - 1
            msg = '{}max diag Gelman-Rubin - 1 at {:.1f} sigmas is {:.3g}'.format(item, nsigmas_cl_diag_gr_stop, cl_diag_gr)
            if cl_diag_gr_stop is not None:
                test = cl_diag_gr - 1 < cl_diag_gr_stop
                self.log_info('{} {} {:.3g}'.format(msg, '<' if test else '>', cl_diag_gr_stop))
                add_diagnostics('cl_diag_gr', test)
                toret = is_stable('cl_diag_gr')
            else:
                self.log_info('{}.'.format(msg))

            # source: https://github.com/JohannesBuchner/autoemcee/blob/38feff48ae524280c8ea235def1f29e1649bb1b6/autoemcee.py#L337
            geweke = diagnostics.geweke(self.chains, varied, first=0.25, last=0.75).max()
            msg = '{}Max Geweke p-value is {:.3g}'.format(item, geweke)
            if geweke_stop is not None:
                test = geweke < geweke_stop
                self.log_info('{} {} {:.3g}.'.format(msg, '<' if test else '>', geweke_stop))
                add_diagnostics('geweke', test)
                toret = is_stable('geweke')
            else:
                self.log_info('{}.'.format(msg))

            split_samples = [chain[burnin:, iwalker] for iwalker in range(self.nwalkers) for chain in self.chains]
            iact = diagnostics.integrated_autocorrelation_time(split_samples, varied)
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

        return self.mpicomm.bcast(toret, root=0)

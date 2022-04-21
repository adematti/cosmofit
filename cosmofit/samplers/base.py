import numpy as np
import mpytools as mpy

from cosmofit.utils import BaseClass
from cosmofit.chains import diagnostics


class BaseSampler(BaseClass):

    def __init__(self, likelihood, rng=None, seed=None, max_tries=1000, nprocs_per_chain=None):
        self.likelihood = likelihood
        self.mpicomm = self.likelihood.mpicomm
        self._set_rng(rng=rng, seed=seed)
        self._set_sampler()
        self._set_start(max_tries=max_tries)
        self._current_diagnostics = {}

    def _set_rng(self, rng=None, seed=None):
        self.rng = self.mpicomm.bcast(rng, root=0)
        if self.rng is None:
            seed = mpy.random.bcast_seed(seed=seed, mpicomm=self.mpicomm, size=None)
        self.rng = np.random.RandomState(seed=seed)

    def _set_sampler(self):
        raise NotImplementedError

    def _set_start(self, max_tries=1000):
        # TODO: in parallel

        def get_start():
            toret = []
            for param in self.varied:
                if param.ref.is_proper():
                    toret.append(param.ref.sample(random_state=self.rng))
                else:
                    toret.append(param.value)
            return toret

        start = []
        for iwalker in range(self.nwalkers * self.nchains):
            correct_init = False
            itry = 0
            while itry < max_tries:
                values = get_start()
                itry += 1
                if np.isfinite(self.likelihood.logposterior(values)):
                    correct_init = True
                    break
            if not correct_init:
                raise ValueError('Could not find finite log posterior after {:d} tries'.format(max_tries))
            start.append(values)
        self.start = self.mpicomm.bcast(np.array(start), root=0).reshape(self.nchains, self.nwalkers)

    def __enter__(self):
        return self

    def sample(self, nsteps=300, thin_by=1):
        raise NotImplementedError

    def diagnose(self, nsplits=4, stable_over=2, burnin=0.3, eigen_gr_stop=0.03, diag_gr_stop=None,
                 cl_diag_gr_stop=None, nsigmas_cl_diag_gr_stop=1., geweke_stop=None,
                 iact_stop=None, iact_reliable=50, dact_stop=None):

        def add_diagnostics(name, value):
            if name not in self._current_diagnostics:
                self._current_diagnostics[name] = [value]
            else:
                self._current_diagnostics[name].append(value)
            return value

        def is_stable(name):
            if len(self._current_diagnostics[name]) < stable_over:
                return False
            return all(self._current_diagnostics[name][-stable_over:])

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
        geweke = diagnostics.geweke(self.chains, varied, first=0.25, last=0.75)
        msg = '{}Geweke p-value is {:.3g}'.format(item, geweke)
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
        n_iterations = len(split_samples[0])
        if iact_reliable * iact < n_iterations:
            msg = '{} (reliable)'.format(msg)
        if iact_stop is not None:
            test = iact * iact_stop < n_iterations
            self.log_info('{} {} {:d}/{:.1f} = {:.3g}'.format(msg, '<' if test else '>', n_iterations, iact_stop, n_iterations / iact_stop))
            add_diagnostics('iact', test)
            toret = is_stable('iact')
        else:
            self.log_info('{}.'.format(msg))

        tau = self._current_diagnostics['tau']
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

        return toret

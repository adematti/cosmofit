import numpy as np
import mpytools as mpy

from cosmofit.base import SectionConfig, import_cls
from cosmofit.utils import BaseClass, TaskManager
from cosmofit.samples import Profiles, ParameterValues
from cosmofit.parameter import ParameterArray


class ProfilerConfig(SectionConfig):

    _sections = ['init']

    def __init__(self, *args, **kwargs):
        # cls, init kwargs
        super(ProfilerConfig, self).__init__(*args, **kwargs)
        self['class'] = import_cls(self['class'], pythonpath=self.pop('pythonpath', None), registry=BaseProfiler._registry)

    def run(self, likelihood):
        profiler = self['class'](likelihood, **self['init'])
        save_fn = self.get('save', None)

        for name in ['maximize', 'interval', 'profile', 'contour']:
            if name in self:
                tmp = self[name]
                if tmp is None: tmp = {}
                getattr(profiler, name)(**tmp)

        if save_fn is not None and profiler.mpicomm.rank == 0:
            profiler.profiles.save(save_fn)

        return profiler


class RegisteredProfiler(type(BaseClass)):

    _registry = set()

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry.add(cls)
        return cls


class BaseProfiler(BaseClass, metaclass=RegisteredProfiler):

    def __init__(self, likelihood, rng=None, seed=None, max_tries=1000, profiles=None, mpicomm=None):
        if mpicomm is None:
            mpicomm = likelihood.mpicomm
        self.mpicomm = mpicomm
        self.likelihood = likelihood
        self.varied_params = self.likelihood.params.select(varied=True, derived=False)
        if self.mpicomm.rank == 0:
            self.log_info('Varied parameters: {}.'.format(self.varied_params.names()))
        self.max_tries = int(max_tries)
        self.profiles = profiles
        if profiles is not None and not isinstance(profiles, Profiles):
            self.profiles = Profiles.load(profiles)
        self._set_rng(rng=rng, seed=seed)
        self._set_profiler()

    def loglikelihood(self, values):
        values = self.likelihood.mpicomm.bcast(np.asarray(values), root=0)
        isscalar = values.ndim == 1
        values = np.atleast_2d(values)
        di = {str(param): values[:, iparam] for iparam, param in enumerate(self.varied_params)}
        self.likelihood.mpirun(**di)
        toret = None
        if self.likelihood.mpicomm.rank == 0:
            if self.derived is None:
                self.derived = [self.likelihood.derived, di]
            else:
                self.derived = [ParameterValues.concatenate([self.derived[0], self.likelihood.derived]), {name: np.concatenate([self.derived[1][name], di[name]], axis=0) for name in di}]
            toret = self.likelihood.loglikelihood
            for array in self.likelihood.derived:
                if array.param.varied:
                    toret += array.param.prior(array)
            if isscalar: toret = toret[0]
        toret = self.likelihood.mpicomm.bcast(toret, root=0)
        mask = np.isnan(toret)
        toret[mask] = -np.inf
        if mask.any() and self.mpicomm.rank == 0:
            self.log_warning('loglikelihood is NaN for {}'.format({k: v[mask] for k, v in di.items()}))
        return toret

    def logposterior(self, values):
        values = self.likelihood.mpicomm.bcast(np.asarray(values), root=0)
        isscalar = values.ndim == 1
        values = np.atleast_2d(values)
        params = {str(param): values[:, iparam] for iparam, param in enumerate(self.varied_params)}
        toret = self.likelihood.logprior(**params)
        mask = ~np.isinf(toret)
        toret[mask] += self.loglikelihood(values[mask])
        if isscalar: toret = toret[0]
        return toret

    def chi2(self, values):
        return -2. * self.logposterior(values)

    def __getstate__(self):
        state = {}
        for name in ['max_tries']:
            state[name] = getattr(self, name)
        return state

    def _set_rng(self, rng=None, seed=None):
        self.rng = self.mpicomm.bcast(rng, root=0)
        if self.rng is None:
            seed = mpy.random.bcast_seed(seed=seed, mpicomm=self.mpicomm, size=None)
        self.rng = np.random.RandomState(seed=seed)

    def _set_profiler(self):
        raise NotImplementedError

    def _get_start(self, max_tries=1000):
        if max_tries is None:
            max_tries = self.max_tries

        def get_start(size=1):
            toret = []
            for param in self.varied_params:
                if param.ref.is_proper():
                    toret.append(param.ref.sample(size=size, random_state=self.rng))
                else:
                    toret.append([param.value] * size)
            return np.array(toret).T

        logposterior = np.inf
        for itry in range(max_tries):
            if np.isfinite(logposterior): break
            start = np.ravel(get_start(size=1))
            logposterior = self.logposterior(start)

        if np.isnan(logposterior):
            raise ValueError('Could not find finite log posterior after {:d} tries'.format(itry))
        return start

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def maximize(self, niterations=None, **kwargs):
        if niterations is None: niterations = max(self.mpicomm.size - 1, 1)
        niterations = int(niterations)
        nprocs_per_iteration = max((self.mpicomm.size - 1) // niterations, 1)
        list_profiles = [None] * niterations
        with TaskManager(nprocs_per_task=nprocs_per_iteration, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.likelihood.mpicomm = tm.mpicomm
            for ii in tm.iterate(range(niterations)):
                self.derived = None
                start = self._get_start()
                profile = self._maximize_one(start, **kwargs)
                if self.likelihood.mpicomm.rank == 0:
                    for param in self.likelihood.params.select(fixed=True, derived=False):
                        profile.bestfit.set(ParameterArray(np.array(param.value, dtype='f8'), param))
                    index_in_profile, index = ParameterValues(self.derived[1]).match(profile.bestfit, name=self.varied_params.names())
                    assert index_in_profile[0].size == 1
                    for array in self.derived[0]:
                        profile.bestfit.set(array[index], output=True)
                else:
                    profile = None
                list_profiles[ii] = profile
        for iprofile, profile in enumerate(list_profiles):
            mpiroot_worker = self.mpicomm.rank if profile is not None else None
            for mpiroot_worker in self.mpicomm.allgather(mpiroot_worker):
                if mpiroot_worker is not None: break
            assert mpiroot_worker is not None
            list_profiles[iprofile] = Profiles.bcast(profile, mpicomm=self.mpicomm, mpiroot=mpiroot_worker)
        profiles = Profiles.concatenate(list_profiles)

        if self.profiles is None:
            self.profiles = profiles
        else:
            self.profiles = Profiles.concatenate(self.profiles, profiles)

    def _iterate_over_params(self, params, method, **kwargs):
        nparams = len(params)
        nprocs_per_param = max((self.mpicomm.size - 1) // nparams, 1)
        if self.profiles is None:
            start = self._get_start()
        else:
            argmax = self.profiles.bestfit.logposterior.argmax()
            start = [self.profiles.bestfit[param][argmax] for param in self.varied_params]
        list_profiles = [None] * nparams
        with TaskManager(nprocs_per_task=nprocs_per_param, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.likelihood.mpicomm = tm.mpicomm
            for iparam, param in tm.iterate(enumerate(params)):
                self.derived = None
                list_profiles[iparam] = method(start, param, **kwargs)
        profiles = Profiles()
        for iprofile, profile in enumerate(list_profiles):
            mpiroot_worker = self.mpicomm.rank if profile is not None else None
            for mpiroot_worker in self.mpicomm.allgather(mpiroot_worker):
                if mpiroot_worker is not None: break
            assert mpiroot_worker is not None
            profiles.update(Profiles.bcast(profile, mpicomm=self.mpicomm, mpiroot=mpiroot_worker))

        if self.profiles is None:
            self.profiles = profiles
        else:
            self.profiles.update(profiles)

    def interval(self, params=None, **kwargs):
        if params is None:
            params = self.varied_params
        else:
            params = ParameterCollection([self.varied_params[param] for param in params])
        self._iterate_over_params(params, self._interval_one, **kwargs)

    def profile(self, params=None, **kwargs):
        if params is None:
            params = self.varied_params
        else:
            params = ParameterCollection([self.varied_params[param] for param in params])
        self._iterate_over_params(params, self._profile_one, **kwargs)

    def contour(self, params=None, **kwargs):
        if params is None:
            params = self.varied_params
        params = list(params)
        if not utils.is_sequence(params[0]):
            params = [(param1, param2) for param2 in params[iparam1 + 1:] for iparam1, param1 in enumerate(params1)]
        params = [(self.varied_params[param1], self.varied_params[param2]) for param1, param2 in params]
        self._iterate_over_params(params, self._contour_one, **kwargs)

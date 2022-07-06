import numpy as np
import mpytools as mpy

from cosmofit.base import SectionConfig, import_cls
from cosmofit.utils import BaseClass, TaskManager
from cosmofit.samples import Profiles,ParameterValues


class ProfilerConfig(SectionConfig):

    _sections = ['init', 'run']

    def __init__(self, *args, **kwargs):
        # cls, init kwargs
        super(ProfilerConfig, self).__init__(*args, **kwargs)
        self['class'] = import_cls(self['class'], pythonpath=self.pop('pythonpath', None), registry=BaseProfiler._registry)

    def run(self, likelihood):
        profiler = self['class'](likelihood, **self['init'])
        save_fn = self.get('save_fn', None)

        profiler.run(**self['run'])
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
                if array.param.varied:
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

    def run(self, niterations=10, **kwargs):
        self.derived = None
        if niterations is None: niterations = max(self.mpicomm.size - 1, 1)
        niterations = int(niterations)
        nprocs_per_iteration = max((self.mpicomm.size - 1) // niterations, 1)
        profiles = [None] * niterations
        with TaskManager(nprocs_per_task=nprocs_per_iteration, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.likelihood.mpicomm = tm.mpicomm
            for ii in tm.iterate(range(niterations)):
                start = self._get_start()
                self.derived = None
                profile = self._run_one(start, **kwargs)
                if self.likelihood.mpicomm.rank == 0:
                    if profile.has('bestfit'):
                        index_in_profile, index = ParameterValues(self.derived[1]).match(profile.bestfit, name=self.varied_params.names())
                        assert index_in_profile[0].size == 1
                        for array in self.derived[0]:
                            profile.set(array[index], output=True)
                else:
                    profile = None
                profiles[ii] = profile
        for iprofile, profile in enumerate(profiles):
            mpiroot_worker = self.mpicomm.rank if profile is not None else None
            for mpiroot_worker in self.mpicomm.allgather(mpiroot_worker):
                if mpiroot_worker is not None: break
            assert mpiroot_worker is not None
            profiles[iprofile] = Profiles.bcast(profile, mpicomm=self.mpicomm, mpiroot=mpiroot_worker)
        profiles = Profiles.concatenate(profiles)

        if self.profiles is None:
            self.profiles = profiles
        else:
            self.profiles = Profiles.concatenate(self.profiles, profiles)

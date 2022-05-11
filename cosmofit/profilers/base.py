import numpy as np
import mpytools as mpy
from mpytools import CurrentMPIComm

from cosmofit.base import SectionConfig, import_cls
from cosmofit.utils import BaseClass, TaskManager
from cosmofit.samples import Profiles


class ProfilerConfig(SectionConfig):

    _sections = ['init', 'run']

    def __init__(self, *args, **kwargs):
        # cls, init kwargs
        super(ProfilerConfig, self).__init__(*args, **kwargs)
        self['class'] = import_cls(self['class'], pythonpath=self.pop('pythonpath', None), registry=BaseProfiler._registry)

    def init(self, *args, **kwargs):
        kwargs = {**self['init'], **kwargs}
        return self['class'](*args, **kwargs)


class RegisteredProfiler(type(BaseClass)):

    _registry = set()

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry.add(cls)
        return cls


class BaseProfiler(BaseClass, metaclass=RegisteredProfiler):

    @CurrentMPIComm.enable
    def __init__(self, likelihood, rng=None, seed=None, max_tries=1000, profiles=None, mpicomm=None):
        self.mpicomm = mpicomm
        self.likelihood = likelihood
        self.varied = self.likelihood.params.select(varied=True)
        self.max_tries = int(max_tries)
        self.profiles = profiles
        if profiles is not None and not isinstance(profiles, Profiles):
            self.profiles = Profiles.load(profiles)
        self._set_rng(rng=rng, seed=seed)
        self._set_profiler()

    def mpiloglikelihood(self, values):
        self.likelihood.mpirun(**{str(param): [value[iparam] for value in values] for iparam, param in enumerate(self.varied)})
        return self.likelihood.loglikelihood

    def mpilogposterior(self, values):
        params = {str(param): np.array([value[iparam] for value in values]) for iparam, param in enumerate(self.varied)}
        toret = self.likelihood.logprior(**params)
        mask = ~np.isinf(toret)
        self.likelihood.mpirun(**{param: value[mask] for param, value in params.items()})
        toret[mask] += self.likelihood.loglikelihood
        return toret

    def loglikelihood(self, values):
        self.likelihood.run(**{str(param): value for param, value in zip(self.varied, values)})
        return self.likelihood.loglikelihood

    def logposterior(self, values):
        params = {str(param): value for param, value in zip(self.varied, values)}
        toret = self.likelihood.logprior(**params)
        if np.isinf(toret):
            return toret
        self.likelihood.run(**params)
        toret += self.likelihood.loglikelihood
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
            for param in self.varied:
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
        if niterations is None: niterations = max(self.mpicomm.size - 1, 1)
        niterations = int(niterations)
        nprocs_per_iteration = max((self.mpicomm.size - 1) // niterations, 1)
        profiles = [None] * niterations
        with TaskManager(nprocs_per_task=nprocs_per_iteration, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.likelihood.mpicomm = tm.mpicomm
            for ii in tm.iterate(range(niterations)):
                start = self._get_start()
                profiles[ii] = self._run_one(start, **kwargs)
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

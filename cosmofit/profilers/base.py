import numpy as np
import mpytools as mpy
from mpytools import CurrentMPIComm

from cosmofit.samples import Profiles
from cosmofit.utils import BaseClass, TaskManager


class BaseProfiler(BaseClass):

    @CurrentMPIComm.enable
    def __init__(self, likelihood, rng=None, seed=None, max_tries=1000, niterations=None, mpicomm=None):
        self.mpicomm = mpicomm
        self.likelihood = likelihood
        self.varied = self.likelihood.params(varied=True)
        self.max_tries = int(max_tries)
        self.profiles = None
        self._set_rng(rng=rng, seed=seed)
        self._set_profiler()

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

        itry = 0
        start = np.full(len(self.varied), np.nan)
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

    def run(self, niterations=10, **kwargs):
        if niterations is None: niterations = max(self.mpicomm.size - 1, 1)
        niterations = int(niterations)
        nprocs_per_iteration = max((self.mpicomm.size - 1) // niterations, 1)
        profiles = [None] * niterations
        with TaskManager(nprocs_per_task=nprocs_per_iteration, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.likelihood.mpicomm = tm.mpicomm
            for ii in tm.iterate(range(niterations)):
                start = self._get_start()
                profiles[ii] = self._run_single_iteration(start, **kwargs)
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

import numpy as np
import mpytools as mpy

from cosmofit.base import SectionConfig, import_class
from cosmofit.utils import BaseClass, TaskManager
from cosmofit.samples import SourceConfig
from cosmofit.samples.profiles import Profiles, ParameterValues, ParameterBestFit
from cosmofit.parameter import ParameterArray, ParameterCollection


class ProfilerConfig(SectionConfig):

    _sections = ['source', 'init']

    def __init__(self, *args, **kwargs):
        # cls, init kwargs
        super(ProfilerConfig, self).__init__(*args, **kwargs)
        self['class'] = import_class(self['class'], pythonpath=self.pop('pythonpath', None), registry=BaseProfiler._registry)

    def run(self, likelihood):
        from cosmofit.samples import SourceConfig
        values = SourceConfig(self['source']).choice(params=likelihood.params)
        likelihood = likelihood.copy()
        params = likelihood.params.deepcopy()
        for param, value in zip(params, values): param.value = value
        likelihood.set_params(params)

        save_fn = self.get('save', None)
        if 'save_fn' in self['init'] and 'save' in self:
            raise ConfigError('Provide either init: save_fn or save, not both')

        profiler = self['class'](likelihood, **{'save_fn': save_fn, **self['init']})
        save_fn = self.get('save', None)

        for name in ['maximize', 'interval', 'profile', 'contour']:
            if name in self:
                tmp = self[name]
                if tmp is None: tmp = {}
                getattr(profiler, name)(**tmp)

        return profiler


class RegisteredProfiler(type(BaseClass)):

    _registry = set()

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry.add(cls)
        return cls


class BaseProfiler(BaseClass, metaclass=RegisteredProfiler):

    _check_same_input = False

    def __init__(self, likelihood, rng=None, seed=None, max_tries=1000, profiles=None, covariance=None, rescale=False, save_fn=None, mpicomm=None):
        if mpicomm is None:
            mpicomm = likelihood.mpicomm
        self.likelihood = BaseClass.copy(likelihood)
        self.mpicomm = mpicomm
        self.likelihood.solved_default = '.best'
        self.varied_params = self.likelihood.params.select(varied=True, derived=False, solved=False)
        if self.mpicomm.rank == 0:
            self.log_info('Varied parameters: {}.'.format(self.varied_params.names()))
        if not self.varied_params:
            raise ValueError('No parameters to be varied!')
        self.max_tries = int(max_tries)
        self.profiles = profiles
        if profiles is not None and not isinstance(profiles, Profiles):
            self.profiles = Profiles.load(profiles)
        self._set_rng(rng=rng, seed=seed)
        covariance = SourceConfig(covariance)
        if covariance.source is not None:
            for param in self.varied_params:
                if param in covariance.source.params():
                    param.ref = ParameterPrior(loc=param.value, scale=np.sqrt(covariance.cov(params=param)), dist='norm')
        # Get the covariance for all parameters
        covariance = covariance.cov(params=self.varied_params)
        self._original_params = self.varied_params

        if rescale:

            self._params_transform_loc = np.array([param.value for param in self.varied_params], dtype='f8')
            self._params_transform_scale = np.diag(covariance)**0.5

            def _params_forward_transform(values):
                return values * self._params_transform_scale + self._params_transform_loc

            def _params_backward_transform(values):
                return (values - self._params_transform_loc) / self._params_transform_scale

            self.varied_params = ParameterCollection()
            for param, loc, scale in zip(self._original_params, self._params_transform_loc, self._params_transform_scale):
                loc, scale = - loc, 1. / scale
                param = param.clone(prior=param.prior.affine_transform(loc=loc, scale=scale),
                                    ref=param.ref.affine_transform(loc=loc, scale=scale),
                                    proposal=param.proposal * scale)
                self.varied_params.set(param)

        else:

            self._params_transform_loc = np.zeros(len(self.varied_params), dtype='f8')
            self._params_transform_scale = np.ones(len(self.varied_params), dtype='f8')

            def _params_forward_transform(values):
                return values

            def _params_backward_transform(values):
                return values

            self.varied_params = self.varied_params.deepcopy()

        self._params_forward_transform = _params_forward_transform
        self._params_backward_transform = _params_backward_transform

        self.save_fn = save_fn

    def _bcast_values(self, values):
        values = np.asarray(values)
        if self._check_same_input:
            all_values = self.likelihood.mpicomm.allgather(values)
            if not all(np.allclose(values, all_values[0], atol=0., rtol=1e-7) for values in all_values if values is not None):
                raise ValueError('Input values different on all ranks: {}'.format(all_values))
        return self.likelihood.mpicomm.bcast(values, root=0)

    def loglikelihood(self, values):
        values = self._bcast_values(values)
        if not values.size:
            return -np.inf
        isscalar = values.ndim == 1
        values = np.atleast_2d(values)
        values = self._params_forward_transform(values)
        points = ParameterValues(values.T, params=self.varied_params)
        self.likelihood.mpirun(**points.to_dict())
        toret = None
        if self.likelihood.mpicomm.rank == 0:
            if self.derived is None:
                self.derived = [points, self.likelihood.derived]
            else:
                self.derived = [ParameterValues.concatenate([self.derived[0], points]),
                                ParameterValues.concatenate([self.derived[1], self.likelihood.derived])]
            toret = self.likelihood.loglikelihood + self.likelihood.logprior
        else:
            self.derived = None
        toret = self.likelihood.mpicomm.bcast(toret, root=0)
        mask = np.isnan(toret)
        toret[mask] = -np.inf
        if mask.any() and self.mpicomm.rank == 0:
            self.log_warning('loglikelihood is NaN for {}'.format({k: v[mask] for k, v in points.items()}))
        if isscalar: toret = toret[0]
        return toret

    def logprior(self, values):
        values = self._bcast_values(values)
        logprior = 0.
        values = self._params_forward_transform(values)
        for param, value in zip(self.varied_params, values.T):
            logprior += param.prior(value)
        return logprior

    def logposterior(self, values):
        values = self._bcast_values(values)
        isscalar = values.ndim == 1
        values = np.atleast_2d(values)
        toret = self.logprior(values)
        mask = ~np.isinf(toret)
        toret[mask] = self.loglikelihood(values[mask])
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
            self.derived = None
            start = np.ravel(get_start(size=1))
            logposterior = self.logposterior(start)

        if np.isnan(logposterior):
            raise ValueError('Could not find finite log posterior after {:d} tries'.format(itry))
        return start, logposterior

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

    def _profiles_transform(self, profiles):
        toret = profiles.deepcopy()

        def transform_array(array, scale_only=False):
            try:
                iparam = self._original_params.index(array.param)
            except KeyError:
                return array
            array.param = self._original_params[iparam]
            array = array * self._params_transform_scale[iparam]
            if not scale_only: array += self._params_transform_loc[iparam]
            return array

        for name, item in toret.items():
            if name == 'covariance':
                iparams = [self._original_params.index(param) for param in item._params]
                item._params = self._original_params.sort(key=iparams)
                item._value = item._value * (self._params_transform_scale[iparams, None] * self._params_transform_scale[iparams])
            elif name == 'contour':
                item.data = [tuple(transform_array(array) for array in arrays) for arrays in item.data]
            else: # 'start', 'bestfit', 'error', 'interval', 'profile'
                item.data = [transform_array(array, scale_only= name == 'error') for array in item.data]
            toret.set(name=item)
        return toret

    def maximize(self, niterations=None, **kwargs):
        if niterations is None: niterations = max(self.mpicomm.size - 1, 1)
        niterations = int(niterations)
        nprocs_per_iteration = max((self.mpicomm.size - 1) // niterations, 1)
        list_profiles = [None] * niterations
        mpicomm_bak = self.mpicomm
        with TaskManager(nprocs_per_task=nprocs_per_iteration, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.mpicomm = tm.mpicomm
            for ii in tm.iterate(range(niterations)):
                start, logposterior = self._get_start()
                p = self._maximize_one(start, **kwargs)
                if self.mpicomm.rank == 0:
                    profiles = Profiles(start=ParameterValues(start, params=self.varied_params),
                                        bestfit=ParameterBestFit(list(start) + [logposterior], params=self.varied_params + ['logposterior']))
                    profiles.update(p)
                    profiles = self._profiles_transform(profiles)
                    for param in self.likelihood.params.select(fixed=True, derived=False):
                        profiles.bestfit.set(ParameterArray(np.array(param.value, dtype='f8'), param))
                    index_in_profile, index = self.derived[0].match(profiles.bestfit, params=profiles.start.params())
                    assert index_in_profile[0].size == 1
                    for array in self.derived[1]:
                        profiles.bestfit.set(array[index], output=True)
                else:
                    profiles = None
                list_profiles[ii] = profiles
        self.mpicomm = mpicomm_bak
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

        if self.mpicomm.rank == 0 and self.save_fn is not None:
            self.profiles.save(self.save_fn)

    def _iterate_over_params(self, params, method, **kwargs):
        nparams = len(params)
        nprocs_per_param = max((self.mpicomm.size - 1) // nparams, 1)
        if self.profiles is None:
            start = self._get_start()
        else:
            argmax = self.profiles.bestfit.logposterior.argmax()
            start = self._params_backward_transform([self.profiles.bestfit[param][argmax] for param in self.varied_params])
        list_profiles = [None] * nparams
        mpicomm_bak = self.mpicomm
        with TaskManager(nprocs_per_task=nprocs_per_param, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:
            self.likelihood.mpicomm = self.mpicomm = tm.mpicomm
            for iparam, param in tm.iterate(enumerate(params)):
                self.derived = None
                profiles = method(start, param, **kwargs)
                list_profiles[iparam] = self._profiles_transform(profiles) if self.mpicomm.rank == 0 else None
        self.mpicomm = mpicomm_bak
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

        if self.mpicomm.rank == 0 and self.save_fn is not None:
            self.profiles.save(self.save_fn)

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

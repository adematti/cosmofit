import numpy as np

from cosmofit import utils
from cosmofit.samples.profile import Profiles, ParameterValues, ParameterBestFit, ParameterCovariance

from .base import BaseProfiler


class MinuitProfiler(BaseProfiler):

    def __init__(self, *args, migrad=None, minos=None, **kwargs):
        self.migrad_params = dict(migrad or {})
        self.minos_params = dict(minos or {})
        super(MinuitProfiler, self).__init__(*args, **kwargs)

    def chi2(self, *values):
        return super(MinuitProfiler, self).chi2(values)

    def _set_profiler(self):
        import iminuit
        minuit_params = {}
        minuit_params['name'] = parameter_names = [str(param) for param in self.varied_params]
        self.minuit = iminuit.Minuit(self.chi2, **dict(zip(parameter_names, [param.value for param in self.varied_params])), **minuit_params)
        self.minuit.errordef = 1.0
        for param in self.varied_params:
            self.minuit.limits[str(param)] = tuple(None if np.isinf(lim) else lim for lim in param.prior.limits)
            if param.ref.is_proper():
                self.minuit.errors[str(param)] = param.proposal

    def _run_one(self, start, algorithms=('migrad',)):
        if not utils.is_sequence(algorithms): algorithms = [algorithms]
        algorithms = list(algorithms)

        profiles = Profiles()
        if 'migrad' in algorithms:
            for param, value in zip(self.varied_params, start):
                self.minuit.values[str(param)] = value
            profiles.set(start=ParameterValues(start, params=self.varied_params))
            self.minuit.migrad(**self.migrad_params)
            profiles.set(bestfit=ParameterBestFit([self.minuit.values[str(param)] for param in self.varied_params] + [self.minuit.fval], params=self.varied_params + ['logposterior']))
            profiles.set(parabolic_errors=ParameterValues([self.minuit.errors[str(param)] for param in self.varied_params], params=self.varied_params))
            profiles.set(covariance=ParameterCovariance(np.array(self.minuit.covariance), params=self.varied_params))

        if 'minos' in algorithms:
            errors = []
            for param in self.varied_params:
                param = str(param)
                self.minuit.minos(param, **self.minos_params)
                errors.append((self.minuit.merrors[param].lower, self.minuit.merrors[param].upper))
            profiles.set(deltachi2_errors=ParameterValues(errors, params=self.varied_params))

        return profiles

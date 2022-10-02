import numpy as np

from cosmofit import utils
from cosmofit.samples.profiles import Profiles, ParameterValues, ParameterBestFit, ParameterCovariance

from .base import BaseProfiler


class ScipyProfiler(BaseProfiler):

    def __init__(self, *args, **kwargs):
        super(ScipyProfiler, self).__init__(*args, **kwargs)

    def _maximize_one(self, start, tol=None, **kwargs):
        from scipy import optimize
        bounds = [tuple(None if np.isinf(lim) else lim for lim in param.prior.limits) for param in self.varied_params]
        result = optimize.minimize(fun=self.chi2, x0=start, bounds=bounds, tol=tol, options=kwargs)
        if not result.success and self.mpicomm.rank == 0:
            self.log_error('Finished unsuccessfully.')
        profiles = Profiles()
        profiles.set(bestfit=ParameterBestFit(list(result.x) + [- 0.5 * result.fun], params=self.varied_params + ['logposterior']))
        if getattr(result, 'hess_inv', None) is not None:
            cov = result.hess_inv.todense()
            profiles.set(error=ParameterValues(np.diag(cov)**0.5, params=self.varied_params))
            profiles.set(covariance=ParameterCovariance(cov, params=self.varied_params))
        return profiles

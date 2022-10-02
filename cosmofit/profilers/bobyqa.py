import numpy as np

from cosmofit import utils
from cosmofit.samples.profiles import Profiles, ParameterValues, ParameterBestFit, ParameterCovariance

from .base import BaseProfiler


class BOBYQAProfiler(BaseProfiler):

    def __init__(self, *args, **kwargs):
        super(BOBYQAProfiler, self).__init__(*args, **kwargs)

    def _maximize_one(self, start, max_iterations=int(1e5), **kwargs):
        import pybobyqa
        infs = [- 1e20, 1e20]  # pybobyqa defaults
        bounds = np.array([[inf if np.isinf(lim) else lim for lim, inf in zip(param.prior.limits, infs)] for param in self.varied_params]).T
        result = pybobyqa.solve(objfun=self.chi2, x0=start, bounds=bounds, maxfun=max_iterations, **kwargs)
        success = result.flag == result.EXIT_SUCCESS
        profiles = Profiles()
        if not success and self.mpicomm.rank == 0:
            self.log_error('Finished unsuccessfully.')
            return profiles
        profiles.set(bestfit=ParameterBestFit(list(result.x) + [- 0.5 * result.f], params=self.varied_params + ['logposterior']))
        cov = utils.inv(result.hessian)
        profiles.set(error=ParameterValues(np.diag(cov)**0.5, params=self.varied_params))
        profiles.set(covariance=ParameterCovariance(cov, params=self.varied_params))
        return profiles

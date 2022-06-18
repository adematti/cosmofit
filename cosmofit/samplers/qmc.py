from scipy.stats import qmc
from scipy.stats.qmc import Sobol, Halton, LatinHypercube

from cosmofit.utils import BaseClass
from .base import RegisteredSampler


class RQuasiRandomSequence(qmc.QMCEngine):

    def __init__(self, d, seed=0.5):
        super().__init__(d=d)
        self.seed = float(seed)
        phi = 1.0
        # This is the Newton's method, solving phi**(d+1) - phi - 1 = 0
        while (np.abs(eq_check) > 1e-15):
            phi -= (phi**(self.d + 1) - phi - 1) / ((self.d + 1) * phi**self.d - 1)
            eq_check = phi**(self.d + 1) - phi - 1
        self.inv_phi = [phi**(-(1 + d)) for d in range(self.d)]

    def random(self, n=1):
        self.num_generated += n
        return (self.seed + np.arange(self.num_generated + 1, self.num_generated + n + 1)[:, None] * self.inv_phi) % 1.

    def reset(self):
        self.num_generated = 0
        return self

    def fast_forward(self, n):
        self.num_generated += n
        return self


def get_qmc_engine(engine):

    return {'sobol': Sobol, 'halton': Halton, 'lhs': LatinHypercube, 'rqrs': RQuasiRandomSequence}.get(engine, engine)


class QMCSampler(BaseClass, metaclass=RegisteredSampler):

    def __init__(self, pipeline, samples=None, mpicomm=None, engine='rqrs', **kwargs):
        if mpicomm is None:
            mpicomm = pipeline.mpicomm
        self.mpicomm = mpicomm
        self.pipeline = pipeline
        self.varied_params = self.pipeline.params.select(varied=True, derived=False)
        self.engine = get_qmc_engine(engine)(d=len(self.varied_params), **kwargs)
        self.samples = samples

    def run(self, niterations=300):
        lower, upper = [], []
        for param in self.varied_params:
            if param.ref.is_proper():
                lower.append(param.value - param.proposal)
                upper.append(param.value + param.proposal)
            else:
                raise ParameterPriorError('Provide parameter limits or proposal')
        if self.mpicomm.rank == 0:
            self.engine.reset()
            nsamples = len(self.samples) if self.samples is not None else 0
            self.engine.fast_forward(nsamples)
            samples = self.engine.scale(self.engine.random(n=niterations), lower, upper)
            samples = ParameterValues(samples, params=self.varied_params)
        mpicomm = self.pipeline.mpicomm
        self.pipeline.mpicomm = self.mpicomm
        self.pipeline.mpirun(**(samples.to_dict() if self.mpicomm.rank == 0 else {}))
        self.pipeline.mpicomm = mpicomm
        if self.mpicomm.rank == 0:
            samples.update(self.pipeline.derived)
            if self.samples is None:
                self.samples = samples
            else:
                self.samples = ParameterValues.concatenate(self.samples, samples)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

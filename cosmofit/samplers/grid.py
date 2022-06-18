import numpy as np

from cosmofit.samples import ParameterValues
from cosmofit.utils import BaseClass
from .base import RegisteredSampler


class GridSampler(BaseClass, metaclass=RegisteredSampler):

    def __init__(self, pipeline, mpicomm=None, ngrid=3, scale=1.):
        if mpicomm is None:
            mpicomm = pipeline.mpicomm
        self.mpicomm = mpicomm
        self.pipeline = pipeline
        self.varied_params = self.pipeline.params.select(varied=True, derived=False)
        self.scale = float(scale)
        if not isinstance(ngrid, dict):
            ngrid = {str(param): ngrid for param in self.varied_params}
        self.ngrid = [max(ngrid[str(param)], 1) for param in self.varied_params]

    def run(self):
        grid = []
        for iparam, (param, ngrid) in enumerate(zip(self.varied_params, self.ngrid)):
            ngrid = self.ngrid[iparam]
            if ngrid == 1:
                grid.append(np.array(param.value))
            elif param.ref.is_proper():
                grid.append(np.linspace(param.value - self.scale * param.proposal, param.value + self.scale * param.proposal, ngrid))
            else:
                raise ParameterPriorError('Provide parameter limits or proposal')
        if self.mpicomm.rank == 0:
            samples = ParameterValues([value.ravel() for value in np.meshgrid(*grid, indexing='ij')], params=self.varied_params)
            samples.attrs['ngrid'] = self.ngrid
        mpicomm = self.pipeline.mpicomm
        self.pipeline.mpicomm = self.mpicomm
        self.pipeline.mpirun(**(samples.to_dict() if self.mpicomm.rank == 0 else {}))
        self.pipeline.mpicomm = mpicomm
        if self.mpicomm.rank == 0:
            samples.update(self.pipeline.derived)
            self.samples = samples

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

import numpy as np

from cosmofit.base import BaseLikelihood
from cosmofit import utils


class GalaxyPowerSpectrumLikelihood(BaseLikelihood):

    def __init__(self, data, covariance):
        self.data = np.asarray(data, dtype='f8')
        self.covariance = np.asarray(covariance, dtype='f8')
        self.precision = utils.inv(self.covariance)

    def loglikelihood(self, inputs):
        diff = self.data - self.model(inputs)
        return -0.5 * diff.dot(self.precision).T.dot(diff)

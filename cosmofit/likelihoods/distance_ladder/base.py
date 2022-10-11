import os

import numpy as np

from cosmofit.likelihoods.base import BaseCalculator
from cosmofit.theories.primordial_cosmology import BasePrimordialCosmology


class H0Likelihood(BaseCalculator):

    def __init__(self, mean, std):
        self.mean = float(mean)
        self.std = float(std)
        self.requires = {'cosmo': {'class': BasePrimordialCosmology}}

    def run(self):
        H0 = self.cosmo.H0
        return -0.5 * (H0 - self.mean)**2 / self.std**2


class MbLikelihood(BaseCalculator):

    def __init__(self, mean, std):
        self.mean = float(mean)
        self.std = float(std)
        self.requires = {'cosmo': {'class': BasePrimordialCosmology}}

    def run(self, Mb):
        return -0.5 * (Mb - self.mean)**2 / self.std**2

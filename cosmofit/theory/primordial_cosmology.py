import cosmoprimo

from cosmofit.base import BaseCalculator


class BasePrimordialCosmology(BaseCalculator):

    pass


class Cosmoprimo(cosmoprimo.Cosmology, BasePrimordialCosmology):

    def __init__(self, engine='class', extra_params=None):
        self.engine = engine
        self.extra_params = extra_params or {}

    def run(self, **params):
        super(Cosmoprimo, self).__init__(**params, extra_params=self.extra_params, engine=self.engine)
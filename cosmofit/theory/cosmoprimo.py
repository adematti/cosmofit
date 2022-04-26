import cosmoprimo

from cosmofit.base import BaseCalculator


class Cosmoprimo(BaseCalculator):

    def __init__(self, engine='class', extra_parameters=None):
        self.engine = engine
        self.extra_parameters = extra_parameters or {}

    def get_output(self, inputs):
        return {'cosmoprimo': cosmoprimo.Cosmology(**inputs, extra_parameters=self.extra_parameters, engine=self.engine)}

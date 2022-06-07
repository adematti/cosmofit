from cosmoprimo import Cosmology, CosmologyError

from cosmofit.base import BaseCalculator
from .base import get_cosmo


class BasePrimordialCosmology(BaseCalculator):

    pass


class Cosmoprimo(BasePrimordialCosmology):

    def __init__(self, fiducial=None, engine='class', extra_params=None):
        self.engine = engine
        self.extra_params = extra_params or {}
        if fiducial is not None:
            cosmo = get_cosmo(fiducial)
            for param in self.params:
                name = param.basename
                if name.lower().startswith('omega_'):
                    name = name[:5] + '0' + name[5:]
                if name.startswith('omega'):
                    param.value = getattr(cosmo, 'O' + name[1:]) * cosmo.h ** 2
                else:
                    param.value = getattr(cosmo, name)
        self.requires = {}

    def run(self, **params):
        self.cosmo = Cosmology(**params, extra_params=self.extra_params, engine=self.engine)

    def __getattr__(self, name):
        try:
            return super(Cosmoprimo, self).__getattr__(name)
        except AttributeError as exc:
            if 'cosmo' in self.__dict__:
                try:
                    return getattr(self.cosmo, name)
                except CosmologyError as exc:  # TODO: remove once cosmoprimo is updated
                    raise AttributeError from exc
            raise exc

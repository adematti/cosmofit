from cosmoprimo import Cosmology, CosmologyError

from cosmofit.base import BaseCalculator


class BasePrimordialCosmology(BaseCalculator):

    pass


def get_from_cosmo(cosmo, name):
    if name.lower().startswith('omega_'):
        name = name[:5] + '0' + name[5:]
    if name.startswith('omega'):
        Omega = getattr(cosmo, 'O' + name[1:], None)
        if Omega is not None:
            return Omega * cosmo.h ** 2
    return getattr(cosmo, name)


class Cosmoprimo(BasePrimordialCosmology):

    def __init__(self, fiducial=None, engine='class', extra_params=None):
        self.engine = engine
        self.extra_params = extra_params or {}
        self.fiducial = fiducial
        self.requires = {}

    def set_params(self, params):
        if self.fiducial is not None:
            from .base import get_cosmo
            cosmo = get_cosmo(self.fiducial)
            for param in params:
                #print(param)
                param.value = get_from_cosmo(cosmo, param.basename)
                param.fixed = param.get('fixed', True)
        return params

    def run(self, **params):
        self.cosmo = Cosmology(**params, extra_params=self.extra_params, engine=self.engine)

    def __getattr__(self, name):
        try:
            return super(Cosmoprimo, self).__getattr__(name)
        except AttributeError as exc:
            if 'cosmo' in self.__dict__:
                try:
                    return get_from_cosmo(self.cosmo, name)
                except CosmologyError as exc:  # TODO: remove once cosmoprimo is updated
                    raise AttributeError from exc
            raise exc

from cosmoprimo import Cosmology, CosmologyError

from cosmofit.base import BaseCalculator


class BasePrimordialCosmology(BaseCalculator):

    pass


def get_cosmo(cosmo):
    import cosmoprimo
    if isinstance(cosmo, str):
        cosmo = (cosmo, {})
    if isinstance(cosmo, tuple):
        return getattr(cosmoprimo.fiducial, cosmo[0])(**cosmo[1])
    return cosmoprimo.Cosmology(**cosmo)


def get_from_cosmo(cosmo, name):
    if name.lower().startswith('omega_'):
        name = name[:5] + '0' + name[5:]
    if name.startswith('omega'):
        Omega = getattr(cosmo, 'O' + name[1:], None)
        if Omega is not None:
            return Omega * cosmo.h ** 2
    if name == 'k_pivot':
        return cosmo.k_pivot * cosmo.h
    return getattr(cosmo, name)


class Cosmoprimo(BasePrimordialCosmology):

    def __init__(self, fiducial=None, engine='class', params=None, extra_params=None):
        self.engine = engine
        self.extra_params = extra_params or {}
        if fiducial is not None:
            fiducial = get_cosmo(fiducial)
        else:
            fiducial = Cosmology()
        self.fiducial = fiducial.clone(**(params or {}), extra_params=self.extra_params, engine=self.engine)
        self.requires = {}

    def set_params(self, params):
        if self.fiducial is not None:
            for param in params:
                if not param.get('drop', False):
                    param.value = get_from_cosmo(self.fiducial, param.basename)
                    param.fixed = param.get('fixed', True)
        return params

    def run(self, **params):
        print(params)
        self.cosmo = self.fiducial.clone(**params)

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

import numpy as np
from scipy import special

import cosmoprimo

from cosmofit.base import BaseCalculator
from . import utils


class BasePowerSpectrumMultipoles(BaseCalculator):

    def __init__(self, k, zeff=1., ells=(0, 2, 4)):
        self.k = np.asarray(k, dtype='f8')
        self.zeff = float(zeff)
        self.ells = tuple(ells)

    def __getstate__(self):
        return {'k': self.k, 'zeff': self.zeff, 'ells': self.ells, 'power': self.power}


class TrapzPowerSpectrumMultipoles(BasePowerSpectrumMultipoles):

    def __init__(self, *args, mu=200, **kwargs):
        super(TrapzPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        self.set_k_mu(k=self.k, mu=mu, ells=self.ells)

    def set_k_mu(self, k, mu=200, ells=(0, 2, 4)):
        self.k = np.asarray(k, dtype='f8')
        if np.ndim(mu) == 0:
            self.mu = np.linspace(0., 1., mu)
        else:
            self.mu = np.asarray(mu)
        muw = utils.weights_trapz(mu)
        self.muweights = np.array([muw * (2 * ell + 1) * special.legendre(ell)(self.mu) for ell in ells]) / (self.mu[-1] - self.mu[0])

    def to_poles(self, pkmu):
        return np.sum(pkmu * self.muweights[:, None, :], axis=-1)


def get_cosmo(cosmo):
    if isinstance(cosmo, str):
        cosmo = (cosmo, {})
    if isinstance(cosmo, tuple):
        return getattr(cosmoprimo.fiducial, cosmo[0])(**cosmo[1])
    return cosmoprimo.Cosmology(**cosmo)


class EffectAP(BaseCalculator):

    name = 'effectap'

    def __init__(self, zeff=1., fiducial='DESI'):
        self.zeff = float(zeff)
        fiducial = get_cosmo(fiducial)
        self.efunc_fid = fiducial.efunc(self.zeff)
        self.comoving_angular_distance_fid = fiducial.comoving_angular_distance(self.zeff)

        self.parameterization = 'distances'
        if 'aiso' in self.parameters:
            self.parameterization = 'aiso'
        elif 'apar' in self.parameters and 'aper' in self.parameters:
            self.parameterization = 'aparaper'

    def requires(self):
        return ['cosmoprimo']

    def run(self, **params):
        if self.parameterization == 'distances':
            qpar, qper = self.efunc_fid / self.cosmoprimo.efunc(self.zeff), self.cosmoprimo.comoving_angular_distance(self.zeff) / self.comoving_angular_distance_fid
        elif self.parameterization == 'aiso':
            qpar = qper = params['aiso']
        else:
            qpar, qper = params['apar'], params['aper']
        self.qpar, self.qper = qpar, qper

    def ap_k_mu(self, k, mu):
        jac = 1. / (self.qpar * self.qper ** 2)
        F = self.qpar / self.qper
        factor_ap = np.sqrt(1 + mu**2 * (1. / F**2 - 1))
        # Beutler 2016 (arXiv: 1607.03150v1) eq 44
        kap = k / self.qper * factor_ap
        # Beutler 2016 (arXiv: 1607.03150v1) eq 45
        muap = mu / F / factor_ap
        return jac, kap, muap

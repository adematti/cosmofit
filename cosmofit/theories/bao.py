import re

import numpy as np

from cosmoprimo import PowerSpectrumBAOFilter

from cosmofit.parameter import ParameterCollection
from cosmofit.base import BaseCalculator
from .base import BaseTheoryPowerSpectrumMultipoles, TrapzTheoryPowerSpectrumMultipoles


class PowerSpectrumNoWiggles(BaseCalculator):

    def __init__(self, zeff=1., engine='wallish2018'):
        self.engine = engine
        self.zeff = float(zeff)
        self.requires = {'cosmo': ('BasePrimordialCosmology', {})}

    def run(self):
        self.power = self.cosmo.get_fourier().pk_interpolator().to_1d(z=self.zeff)
        self.power_now = PowerSpectrumBAOFilter(self.power, engine=self.engine).smooth_pk_interpolator()


class BaseBAOPowerSpectrum(BaseTheoryPowerSpectrumMultipoles):

    def __init__(self, *args, **kwargs):
        super(BaseBAOPowerSpectrum, self).__init__(*args, **kwargs)
        params = ParameterCollection()
        for param in self.params:
            ellpow = self.decode_broadband_param(param)
            if ellpow is not None and ellpow[0] not in self.ells:
                continue
            params.set(param)
        self.params = params
        self.set_broadband_coeffs()
        self.requires = {'effectap': ('EffectAP', {'zeff': self.zeff, 'fiducial': self.fiducial}), 'powernowiggles': ('PowerSpectrumNoWiggles', {'zeff': self.zeff})}

    @staticmethod
    def decode_broadband_param(param):
        match = re.match('al(.*)_(.*)', str(param))
        if match:
            ell = int(match.group(1))
            pow = int(match.group(2))
            return ell, pow
        return None

    def set_broadband_coeffs(self):
        self.broadband_coeffs = {}
        for ell in self.ells:
            self.broadband_coeffs[ell] = {}
        for param in self.params:
            name = param.basename
            ellpow = self.decode_broadband_param(param)
            if ellpow is not None:
                self.broadband_coeffs[ellpow[0]][name] = ellpow[1]

    def broadband_poles(self, **params):
        toret = []
        for ell in self.ells:
            tmp = np.zeros_like(self.k)
            for name, ii in self.broadband_coeffs[ell].items():
                tmp += params[name] * self.k**ii
            toret.append(tmp)
        return np.array(toret)

    def beta(self, bias=1., **kwargs):
        if 'beta' in kwargs:
            beta = kwargs['beta']
        elif 'growth_rate' in kwargs:
            beta = kwargs['growth_rate'] / bias
        else:
            beta = self.powernowiggles.cosmo.growth_rate(self.zeff) / bias
        return beta


class Beutler2017BAOGalaxyPowerSpectrum(BaseBAOPowerSpectrum, TrapzTheoryPowerSpectrumMultipoles):

    def __init__(self, *args, mu=200, **kwargs):
        super(Beutler2017BAOGalaxyPowerSpectrum, self).__init__(*args, **kwargs)
        self.set_k_mu(k=self.k, mu=mu, ells=self.ells)

    def run(self, bias=1., sigmar=0., sigmas=5., sigmapar=8., sigmaper=4., **kwargs):
        beta = self.beta(bias=bias, **kwargs)
        jac, kap, muap = self.effectap.ap_k_mu(self.k, self.mu)
        pk = self.powernowiggles.power(kap)
        wiggles = pk / self.powernowiggles.power_now(kap)
        sigmanl2 = kap**2 * (sigmapar**2 * muap**2 + sigmaper**2 * (1. - muap**2))
        fog = 1. / (1. + (sigmas * kap * muap)**2 / 2.)**2.
        r = 1. - np.exp(-(sigmar * kap)**2 / 2.)
        pkmu = jac * fog * bias**2 * (1 + beta * muap**2 * r)**2 * pk * (1. + (wiggles - 1.) * np.exp(-sigmanl2 / 2.))
        self.power = self.to_poles(pkmu) + self.broadband_poles(**kwargs)

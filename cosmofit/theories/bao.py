import re

import numpy as np

from cosmoprimo import PowerSpectrumBAOFilter

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
        self.set_broadband_coeffs()
        self.requires = {'effectap': ('EffectAP', {'zeff': self.zeff, 'fiducial': self.fiducial}), 'powernowiggles': ('PowerSpectrumNoWiggles', {'zeff': self.zeff})}

    def set_broadband_coeffs(self):
        self.broadband_coeffs = {}
        for ell in self.ells:
            self.broadband_coeffs[ell] = {}
            for param in self.params:
                name = param.basename
                match = re.match('al{:d}_(.*)'.format(ell), name)
                if match:
                    pow = int(match.group(1))
                    self.broadband_coeffs[ell][name] = pow

    def beta(self, bias=1., **kwargs):
        if 'beta' in kwargs:
            beta = kwargs['beta']
        elif 'growth_rate' in kwargs:
            beta = kwargs['growth_rate'] / bias
        else:
            beta = self.powernowiggles.cosmo.growth_rate(self.zeff) / bias
        return beta


class Beutler2017BAOGalaxyPowerSpectrum(TrapzTheoryPowerSpectrumMultipoles, BaseBAOPowerSpectrum):

    def run(self, bias=1., sigmar=0., sigmas=5., sigmapar=8., sigmaper=4., **kwargs):
        beta = self.beta(bias=bias, **kwargs)
        jac, kap, muap = self.effectap.ap_k_mu(self.k, self.mu)
        pk = self.powernowiggles.power(kap)
        wiggles = pk / self.powernowiggles.power_now(kap)
        sigmanl2 = kap**2 * (sigmapar**2 * muap**2 + sigmaper**2 * (1. - muap**2))
        fog = 1. / (1. + (sigmas * kap * muap)**2 / 2.)**2.
        r = 1. - np.exp(-(sigmar * kap)**2 / 2.)
        pkmu = jac * fog * bias**2 * (1 + beta * muap**2 * r)**2 * pk * (1. + (wiggles - 1.) * np.exp(-sigmanl2 / 2.))
        self.power = self.to_poles(pkmu)

    def broadband_poles(self, inputs):
        toret = []
        for ell in self.ells:
            tmp = 0.
            for name, ii in self.broadband_coeffs[ell].items():
                tmp += inputs[name] * self.k**ii
            toret.append(tmp)
        return np.array(toret)

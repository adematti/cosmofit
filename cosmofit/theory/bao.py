import re

import numpy as np

from cosmoprimo import PowerSpectrumBAOFilter

from cosmofit.base import BaseCalculator
from .base import BasePowerSpectrumMultipoles


class PowerSpectrumNoWiggles(BaseCalculator):

    name = 'powernowiggles'

    def __init__(self, zeff=1., engine='wallish2018', **kwargs):
        self.engine = engine
        self.zeff = float(zeff)

    def requires(self):
        return ['cosmoprimo']

    def run(self):
        self.pk = self.cosmoprimo.get_gourier().pk_interpolator().to_1d(z=self.zeff)
        self.pknow = PowerSpectrumBAOFilter(self.pk, engine=self.engine).smooth_pk_interpolator()
        self.efunc = self.cosmoprimo.efunc(self.zeff)
        self.comoving_angular_distance = self.cosmoprimo.comoving_angular_distance(self.zeff)


class BaseBAOPowerSpectrum(BasePowerSpectrumMultipoles):

    name = 'baopower'

    def __init__(self, *args, zeff=1., **kwargs):
        self.zeff = float(zeff)
        super(BaseBAOPowerSpectrum, self).__init__(*args, **kwargs)
        self.set_broadband_coeffs()

    def requires(self):
        return [('effectap', {'zeff': self.zeff}), ('powernowiggles', {'zeff': self.zeff})]

    def set_broadband_coeffs(self):
        self.broadband_coeffs = {}
        for ell in self.ells:
            self.broadband_coeffs[ell] = {}
            for name in self.parameters:
                match = re.match('a(*.)_l{:d}'.format(ell), name)
                if match:
                    s = match.group(1)
                    pow = {'m': -1, 'p': 1}[s[:1]] * int(s[1:])
                    self.broadband_coeffs[ell][name] = pow

    def beta(self, bias=1., **kwargs):
        if 'beta' in kwargs:
            beta = kwargs['beta']
        elif 'growth_rate' in kwargs:
            beta = kwargs['growth_rate'] / bias
        else:
            beta = self.cosmoprimo.growth_rate() / bias
        return beta


class Beutler2017BAOPowerSpectrum(BaseBAOPowerSpectrum):

    def run(self, bias=1., sigmar=0., sigmas=5., sigmapar=8., sigmaper=4., **kwargs):
        beta = self.beta(bias=bias, **kwargs)
        jac, kap, muap = self.effectap(self.k, self.mu)
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

import re

import numpy as np
from scipy import special, integrate

from cosmofit.parameter import ParameterCollection
from .power_template import PowerSpectrumNoWiggles
from .base import BaseTheoryPowerSpectrumMultipoles, TrapzTheoryPowerSpectrumMultipoles


class BaseBAOWigglesPowerSpectrum(BaseTheoryPowerSpectrumMultipoles):

    def __init__(self, *args, mode='', smoothing_radius=15., **kwargs):
        super(BaseBAOWigglesPowerSpectrum, self).__init__(*args, **kwargs)
        self.mode = str(mode).lower()
        available_modes = ['', 'recsym', 'reciso']
        if self.mode in available_modes:
            raise ValueError('reconstruction mode must be one of {}'.format(available_modes))
        self.smoothing_radius = float(smoothing_radius)
        params = ParameterCollection()
        for param in self.params:
            ellpow = self.decode_broadband_param(param)
            if ellpow is None or ellpow[0] in self.ells:
                params.set(param)
        self.params = params
        self.set_broadband_coeffs()
        self.requires = {'effectap': ('EffectAP', {'zeff': self.zeff, 'fiducial': self.fiducial}), 'wiggles': ('BasePowerSpectrumWiggles', {'zeff': self.zeff})}

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
        elif 'f' in kwargs:
            beta = kwargs['f'] / bias
        else:
            beta = self.wiggles.cosmo.growth_rate(self.zeff) / bias
        return beta


class EmpiricalBAOWigglesGalaxyPowerSpectrum(BaseBAOWigglesPowerSpectrum, TrapzTheoryPowerSpectrumMultipoles):

    def __init__(self, *args, mu=200, **kwargs):
        super(EmpiricalBAOWigglesGalaxyPowerSpectrum, self).__init__(*args, **kwargs)
        self.set_k_mu(k=self.k, mu=mu, ells=self.ells)

    def run(self, bias=1., sigmas=0., sigmapar=8., sigmaper=4., **kwargs):
        f = self.beta(bias=bias, **kwargs) * bias
        jac, kap, muap = self.effectap.ap_k_mu(self.k, self.mu)
        pk = self.powernowiggles.power(kap)
        pknow = self.powernowiggles.power_now(kap)
        sigmanl2 = kap**2 * (sigmapar**2 * muap**2 + sigmaper**2 * (1. - muap**2))
        damped_wiggles = (pk - pknow) * np.exp(-sigmanl2 / 2.)
        fog = 1. / (1. + (sigmas * kap * muap)**2 / 2.)**2.
        sk = 0.
        if self.mode == 'reciso': sk = np.exp(-1. / 2. * (kap * self.smoothing_radius)**2)
        pkmu = jac * fog * (bias + f * muap**2 * (1 - sk))**2 * (pknow + damped_wiggles)
        self.power = self.to_poles(pkmu) + self.broadband_poles(**kwargs)


class ResummedPowerSpectrumWiggles(BasePowerSpectrumWiggles):

    def __init__(self, *args, mode='', smoothing_radius=15., **kwargs):
        super(ResummedPowerSpectrumWiggles, self).__init__(*args, **kwargs)
        self.mode = str(mode).lower()
        available_modes = ['', 'recsym', 'reciso']
        if self.mode in available_modes:
            raise ValueError('reconstruction mode must be one of {}'.format(available_modes))
        self.smoothing_radius = float(smoothing_radius)

    def run(self):
        super(ResummedPowerSpectrumWiggles, self).run()
        k = self.power_now.k
        pklin = self.power_now.pk
        q = self.cosmo.rs_drag
        j0 = special.jn(0, q * k)
        sk = 0.
        if self.mode: sk = np.exp(-1. / 2. * (k * self.smoothing_radius)**2)
        self.sigma_dd = 1. / (3. * np.pi**2) * integrate.simps((1. - j0) * (1. - sk)**2 * pklin, k)
        if self.mode:
            self.sigma_ss = 1. / (3. * np.pi**2) * integrate.simps((1. - j0) * sk**2 * pklin, k)
            if self.mode == 'recsym':
                self.sigma_ds = 1. / (3. * np.pi**2) * integrate.simps((1. / 2. * ((1. - sk)**2 + sk**2) + j0 * sk * (1. - sk)) * pklin, k)
            else:
                self.sigma_ds_dd = 1. / (6. * np.pi**2) * integrate.simps((1. - sk)**2 * pklin, k)
                self.sigma_ds_ds = - 1. / (6. * np.pi**2) * integrate.simps(j0 * sk * (1. - sk) * pklin, k)
                self.sigma_ds_ss = 1. / (6. * np.pi**2) * integrate.simps(sk**2 * pklin, k)

    def wiggles(self, k, mu, bias=1., f=0.):
        wiggles = super(ResummedPowerSpectrumWiggles, self).wiggles(k)
        b1 = bias - 1.  # lagrangian bias
        sk = 0.
        if self.mode: sk = np.exp(-1. / 2. * (k * self.smoothing_radius)**2)
        ksq = (1 + f * (f + 2) * mu**2) * k**2
        damping_dd = np.exp(-1. / 2. * ksq * self.sigma_dd)
        resummed_wiggles = damping_dd * ((1 + f * mu**2) * (1 - sk) + b1)**2 * wiggles
        if self.mode == 'reciso':
            damping_ds = np.exp(-1. / 2. * (ksq * self.sigma_ds_dd + k**2 * (self.sigma_ds_ss - 2. * (1 + f * mu**2) * self.sigma_ds_dd)))
            resummed_wiggles -= 2. * damping_ds * ((1 + f * mu**2) * (1 - sk) + b1) * sk * wiggles
            damping_ss = np.exp(-1. / 2. * ksq**2 * self.sigma_ss)
            resummed_wiggles += damping_ss * (1 + f * mu**2)**2 * sk**2 * wiggles
        else:
            damping_ds = np.exp(-1. / 2. * (ksq * self.sigma_ds_dd + k**2 * (self.sigma_ds_ss - 2. * (1 + f * mu**2) * self.sigma_ds_dd)))
            resummed_wiggles -= 2. * damping_ds * ((1 + f * mu**2) * (1 - sk) + b1) * (1 + f * mu**2) * sk * wiggles
            damping_ss = np.exp(-1. / 2. * k**2 * self.sigma_ss)
            resummed_wiggles += damping_ss * sk**2 * wiggles
        return resummed_wiggles


class ResummedBAOWigglesGalaxyPowerSpectrum(BaseBAOWigglesPowerSpectrum, TrapzTheoryPowerSpectrumMultipoles):

    def __init__(self, *args, mu=200, **kwargs):
        super(ResummedBAOWigglesGalaxyPowerSpectrum, self).__init__(*args, **kwargs)
        self.set_k_mu(k=self.k, mu=mu, ells=self.ells)
        self.requires['wiggles'] = ('ResummedPowerSpectrumWiggles', {'zeff': self.zeff, 'mode': self.mode, 'smoothing_radius': self.smoothing_radius})

    def run(self, bias=1., sigmas=0., **kwargs):
        f = self.beta(bias=bias, **kwargs) * bias
        jac, kap, muap = self.effectap.ap_k_mu(self.k, self.mu)
        pknow = self.wiggles.power_now(kap)
        wiggles = self.wiggles.wiggles(kap, muap, bias=bias, **kwargs)
        fog = 1. / (1. + (sigmas * kap * muap)**2 / 2.)**2.
        pkmu = jac * (wiggles + fog * (bias + f * muap**2 * (1 - sk))**2 * pknow)
        self.power = self.to_poles(pkmu) + self.broadband_poles(**kwargs)

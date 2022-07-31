import re

import numpy as np
from scipy import special, integrate

from cosmofit.parameter import ParameterCollection
from .power_template import BasePowerSpectrumWiggles, BAOPowerSpectrumParameterization
from .base import (BaseTheoryPowerSpectrumMultipoles, TrapzTheoryPowerSpectrumMultipoles,
                   BaseTheoryCorrelationFunctionMultipoles, BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles)


class BaseBAOWigglesPowerSpectrumMultipoles(BaseTheoryPowerSpectrumMultipoles):

    def __init__(self, *args, mode='', nowiggle=False, smoothing_radius=15., **kwargs):
        super(BaseBAOWigglesPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        self.nowiggle = bool(nowiggle)
        self.mode = str(mode).lower()
        available_modes = ['', 'recsym', 'reciso']
        if self.mode not in available_modes:
            raise ValueError('Reconstruction mode {} must be one of {}'.format(self.mode, available_modes))
        self.smoothing_radius = float(smoothing_radius)
        self.requires = {'template': {'class': BAOPowerSpectrumParameterization,
                                      'init': {'wiggles': BasePowerSpectrumWiggles, 'zeff': self.zeff, 'fiducial': self.fiducial}}}


class DampedBAOWigglesPowerSpectrumMultipoles(BaseBAOWigglesPowerSpectrumMultipoles, TrapzTheoryPowerSpectrumMultipoles):

    def __init__(self, *args, mu=200, **kwargs):
        super(DampedBAOWigglesPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        self.set_k_mu(k=self.k, mu=mu, ells=self.ells)

    def run(self, bias=1., sigmas=0., sigmapar=8., sigmaper=4., **kwargs):
        f = self.template.f
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        pknow = self.template.power_now(kap)
        pk = pknow if self.nowiggle else self.template.power(kap)
        sigmanl2 = kap**2 * (sigmapar**2 * muap**2 + sigmaper**2 * (1. - muap**2))
        damped_wiggles = (pk - pknow) * np.exp(-sigmanl2 / 2.)
        fog = 1. / (1. + (sigmas * kap * muap)**2 / 2.)**2.
        sk = 0.
        if self.mode == 'reciso': sk = np.exp(-1. / 2. * (kap * self.smoothing_radius)**2)
        pkmu = jac * fog * (bias + f * muap**2 * (1 - sk))**2 * (pknow + damped_wiggles)
        self.power = self.to_poles(pkmu)


class ResummedPowerSpectrumWiggles(BasePowerSpectrumWiggles):

    def __init__(self, *args, mode='', smoothing_radius=15., **kwargs):
        super(ResummedPowerSpectrumWiggles, self).__init__(*args, **kwargs)
        self.mode = str(mode).lower()
        available_modes = ['', 'recsym', 'reciso']
        if self.mode not in available_modes:
            raise ValueError('reconstruction mode {} must be one of {}'.format(self.mode, available_modes))
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


class ResummedBAOWigglesPowerSpectrumMultipoles(BaseBAOWigglesPowerSpectrumMultipoles, TrapzTheoryPowerSpectrumMultipoles):

    def __init__(self, *args, mu=200, **kwargs):
        super(ResummedBAOWigglesTracerPowerSpectrum, self).__init__(*args, **kwargs)
        self.set_k_mu(k=self.k, mu=mu, ells=self.ells)
        if not self.nowiggle:
            self.requires['template']['init'].update({'wiggles': ResummedPowerSpectrumWiggles, 'mode': self.mode, 'smoothing_radius': self.smoothing_radius})

    def run(self, bias=1., sigmas=0., **kwargs):
        f = self.template.f
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        pknow = self.template.power_now(kap)
        wiggles = 0. if self.nowiggle else self.template.wiggles(kap, muap, bias=bias, **kwargs)
        fog = 1. / (1. + (sigmas * kap * muap)**2 / 2.)**2.
        sk = 0.
        if self.mode == 'reciso': sk = np.exp(-1. / 2. * (kap * self.smoothing_radius)**2)
        pkmu = jac * fog * (wiggles + (bias + f * muap**2 * (1 - sk))**2 * pknow)
        self.power = self.to_poles(pkmu)


class BaseBAOWigglesTracerPowerSpectrumMultipoles(BaseTheoryPowerSpectrumMultipoles):

    def __init__(self, k=None, zeff=1., ells=(0, 2, 4), fiducial=None, **kwargs):
        super(BaseBAOWigglesTracerPowerSpectrumMultipoles, self).__init__(k=k, zeff=zeff, ells=ells, fiducial=fiducial)
        self.requires = {'bao': {'class': self.__class__.__name__.replace('Tracer', ''),
                                 'init': {'k': self.k, 'zeff': self.zeff, 'ells': self.ells, 'fiducial': self.fiducial, **kwargs}}}

    def set_params(self, params):
        self_params = params.select(basename='al*_*')
        bao_params = params.copy()
        for param in bao_params.names():
            if param in self_params: del bao_params[param]
        self.requires['bao']['params'] = bao_params
        self.broadband_coeffs = {}
        for ell in self.ells:
            self.broadband_coeffs[ell] = {}
        for param in self_params.params():
            name = param.basename
            match = re.match('al(.*)_(.*)', name)
            if match:
                ell = int(match.group(1))
                pow = int(match.group(2))
                if ell in self.ells:
                    self.broadband_coeffs[ell][name] = pow
                else:
                    del self_params[param]
            else:
                raise ValueError('Unrecognized parameter {}'.format(param))
        return self_params

    def run(self, **params):
        self.power = self.bao.power.copy()
        for ill, ell in enumerate(self.ells):
            for name, ii in self.broadband_coeffs[ell].items():
                self.power[ill] += params[name] * self.k**ii

    @property
    def nowiggle(self):
        return self.bao.nowiggle

    @nowiggle.setter
    def nowiggle(self, nowiggle):
        self.bao.nowiggle = nowiggle


class DampedBAOWigglesTracerPowerSpectrumMultipoles(BaseBAOWigglesTracerPowerSpectrumMultipoles):

    pass


class ResummedBAOWigglesTracerPowerSpectrumMultipoles(BaseBAOWigglesTracerPowerSpectrumMultipoles):

    pass


class BaseBAOWigglesCorrelationFunctionMultipoles(BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles):

    def __init__(self, s=None, zeff=1., ells=(0, 2, 4), fiducial=None, **kwargs):
        super(BaseBAOWigglesCorrelationFunctionMultipoles, self).__init__(s=s, zeff=zeff, ells=ells, fiducial=fiducial)
        self.requires['power']['class'] = self.__class__.__name__.replace('CorrelationFunction', 'PowerSpectrum')


class DampedBAOWigglesCorrelationFunctionMultipoles(BaseBAOWigglesCorrelationFunctionMultipoles):

    pass


class ResummedBAOWigglesCorrelationFunctionMultipoles(BaseBAOWigglesCorrelationFunctionMultipoles):

    pass


class BaseBAOWigglesTracerCorrelationFunctionMultipoles(BaseTheoryCorrelationFunctionMultipoles):

    def __init__(self, s=None, zeff=1., ells=(0, 2, 4), fiducial=None, **kwargs):
        super(BaseBAOWigglesTracerCorrelationFunctionMultipoles, self).__init__(s=s, zeff=zeff, ells=ells, fiducial=fiducial)
        self.requires = {'bao': {'class': self.__class__.__name__.replace('Tracer', ''),
                                 'init': {'s': self.s, 'zeff': self.zeff, 'ells': self.ells, 'fiducial': self.fiducial, **kwargs}}}

    def run(self, **params):
        self.corr = self.bao.corr.copy()
        for ill, ell in enumerate(self.ells):
            for name, ii in self.broadband_coeffs[ell].items():
                self.corr[ill] += params[name] * self.s**ii

    @property
    def nowiggle(self):
        return self.bao.nowiggle

    @nowiggle.setter
    def nowiggle(self, nowiggle):
        self.bao.nowiggle = nowiggle


BaseBAOWigglesTracerCorrelationFunctionMultipoles.set_params = BaseBAOWigglesTracerPowerSpectrumMultipoles.set_params


class DampedBAOWigglesTracerCorrelationFunctionMultipoles(BaseBAOWigglesTracerCorrelationFunctionMultipoles):

    pass


class ResummedBAOWigglesTracerCorrelationFunctionMultipoles(BaseBAOWigglesTracerCorrelationFunctionMultipoles):

    pass

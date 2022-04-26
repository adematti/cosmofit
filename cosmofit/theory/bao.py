import re

import numpy as np
from scipy import special

import cosmoprimo
from cosmoprimo import PowerSpectrumBAOFilter

from cosmofit.base import BaseCalculator
from . import utils


def get_cosmo(cosmo):
    if isinstance(cosmo, str):
        cosmo = (cosmo, {})
    if isinstance(cosmo, tuple):
        return getattr(cosmoprimo.fiducial, cosmo[0])(**cosmo[1])
    return cosmoprimo.Cosmology(**cosmo)


class PowerSpectrumNoWiggles(BaseCalculator):

    def __init__(self, zeff=1., engine='wallish2018', **kwargs):
        self.engine = engine
        self.zeff = float(zeff)

    def get_output(self, inputs):
        cosmo = inputs['Cosmoprimo']
        self.pk = cosmo.get_gourier().pk_interpolator().to_1d(z=self.zeff)
        self.pknow = PowerSpectrumBAOFilter(self.pk, engine=self.engine).smooth_pk_interpolator()
        self.efunc = cosmo.efunc(self.zeff)
        self.comoving_angular_distance = cosmo.comoving_angular_distance(self.zeff)
        return {'{}_z={}'.format(self.__class__.__name__, self.zeff): self}


class BasePowerSpectrumMultipoles(BaseCalculator):

    def set_k_mu(self, k, mu=200, ells=None):
        self.k = np.asarray(k, dtype='f8')
        if np.ndim(mu) == 0:
            self.mu = np.linspace(0., 1., mu)
        else:
            self.mu = np.asarray(mu)
        muw = utils.weights_trapz(mu)
        self.muweights = np.array([muw * (2 * ell + 1) * special.legendre(ell)(self.mu) for ell in ells]) / (self.mu[-1] - self.mu[0])

    def pk_ell(self, inputs):
        pkmu = self.pk_mu(inputs)
        return np.sum(pkmu * self.muweights[:, None, :], axis=-1)

    def get_output(self, inputs):
        self.poles = self.pk_ell(inputs)
        return {'{}_z={}'.format(self.__class__.__name__, self.zeff): self}


class BaseBAOPowerSpectrum(BasePowerSpectrumMultipoles):

    def __init__(self, k, zeff=1., mu=101, ells=(0, 2, 4), fiducial='DESI', **kwargs):
        self.set_k_mu(k, mu, ells=ells)
        fiducial = get_cosmo(fiducial)
        self.efunc_fid = fiducial.efunc(self.zeff)
        self.comoving_angular_distance_fid = fiducial.comoving_angular_distance(self.zeff)

        self.parameterization = 'distances'
        if 'aiso' in self.parameters:
            self.parameterization = 'aiso'
        elif 'apar' in self.parameters and 'aper' in self.parameters:
            self.parameterization = 'aparaper'

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

    def set_ap_k_mu(self):
        self.jac = 1. / (self.qpar * self.qper ** 2)
        F = self.qpar / self.qper
        factor_ap = np.sqrt(1 + self.mu**2 * (1. / F**2 - 1))
        # Beutler 2016 (arXiv: 1607.03150v1) eq 44
        self.kap = self.k / self.qper * factor_ap
        # Beutler 2016 (arXiv: 1607.03150v1) eq 45
        self.muap = self.mu / F / factor_ap

    def prepare(self, inputs):
        pknow = inputs['PowerSpectrumNoWiggles_z={}'.format(self.zeff)]
        if self.parameterization == 'distances':
            qpar, qper = self.efunc_fid / pknow.efunc, pknow.comoving_angular_distance / self.comoving_angular_distance_fid
        elif self.parameterization == 'aiso':
            qpar = qper = inputs['aiso']
        else:
            qpar, qper = inputs['apar'], inputs['aper']
        self.qpar, self.qper = qpar, qper
        tmp = inputs['power_no_wiggle_z={}'.format(self.zeff)]
        self.pk, self.pknow = tmp.pk, tmp.pknow
        self.bias = inputs['bias']
        if 'beta' in self.parameters:
            self.beta = inputs['beta']
        elif 'growth_rate' in self.parameters:
            self.beta = inputs['growth_rate'] / self.bias
        else:
            self.beta = inputs['Cosmoprimo'].growth_rate() / self.bias
        self.set_ap_k_mu()


class Beutler2017BAOPowerSpectrum(BaseBAOPowerSpectrum):

    def pk_kmu(self, inputs):
        self.prepare(inputs)
        sigmar, sigmas, sigmapar, sigmaper = inputs['sigmar'], inputs['sigmas'], inputs['sigmapar'], inputs['sigmaper']
        pk = self.pk(self.kap)
        wiggles = pk / self.pknow(self.kap)
        sigmanl2 = self.kap**2 * (sigmapar**2 * self.muap**2 + sigmaper**2 * (1. - self.muap**2))
        fog = 1. / (1. + (sigmas * self.kap * self.muap)**2 / 2.)**2.
        r = 1. - np.exp(-(sigmar * self.kap)**2 / 2.)
        return fog * self.bias**2 * (1 + self.beta * self.muap**2 * r)**2 * pk * (1. + (wiggles - 1.) * np.exp(-sigmanl2 / 2.))

    def broadband_poles(self, inputs):
        toret = []
        for ell in self.ells:
            tmp = 0.
            for name, ii in self.broadband_coeffs[ell].items():
                tmp += inputs[name] * self.k**ii
            toret.append(tmp)
        return np.array(toret)

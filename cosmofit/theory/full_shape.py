import numpy as np

from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD

from cosmofit.base import BaseCalculator
from .bao import get_cosmo, BaseTheoryPowerSpectrumMultipoles


class BasePT(BaseCalculator):

    def __init__(self, k, zeff=1., ells=(0, 2, 4), fiducial='DESI'):
        self.k = np.asarray(k, dtype='f8')
        self.ells = tuple(ells)
        self.zeff = float(zeff)
        fiducial = get_cosmo(fiducial)
        self.efunc_fid = fiducial.efunc(self.zeff)
        self.comoving_angular_distance_fid = fiducial.comoving_angular_distance(self.zeff)
        self.requires = {'cosmoprimo': 'BasePrimordialCosmology', 'effectap': ('EffectAP', {'zeff': self.zeff})}

    def run(self):
        fo = self.cosmoprimo.get_fourier()
        self.growth_rate = fo.sigma8_z(self.zeff, of='theta_cb') / fo.sigma8_z(self.zeff, of='delta_cb')
        self.sigma8 = fo.sigma8_z(self.zeff, of='delta_cb')
        self.pklin = fo.pk_interpolator(of='delta_cb').to_1d(self.zeff)
        efunc = self.cosmoprimo.efunc(self.zeff)
        comoving_angular_distance = self.cosmoprimo.comoving_angular_distance(self.zeff)
        self.qpar, self.qper = self.efunc_fid / efunc, comoving_angular_distance / self.comoving_angular_distance_fid


class LPT(BasePT):

    calculator_type = 'pt'

    def run(self, **params):
        ki = np.logspace(-3., 1., 200)
        self.lpt = LPT_RSD(ki, self.pklin(ki), kIR=0.2, cutoff=10, extrap_min=-4, extrap_max=3, N=2000, threads=1, jn=5)
        self.lpt.make_pltable(self.growth_rate, kv=self.k, apar=self.qpar, aperp=self.qper, ngauss=2)

    def combine_bias_terms_power_poles(self, b1=1.69, b2=-1.17, bs=-0.71, b3=0., alpha0=0., alpha2=0., alpha4=0., alpha6=0., sn0=0., sn2=0., sn4=0.):
        bias = [b1, b2, bs, b3]
        bias += [alpha0, alpha2, alpha4, alpha6]
        bias += [sn0, sn2, sn4]
        pkells = self.combine_bias_terms_pkell(bias)[1:]
        return [pkells[self.ells.index(ell)] for ell in self.ells]


class PTPowerSpectrum(BaseTheoryPowerSpectrumMultipoles):

    def __init__(self, k, zeff=1., ells=(0, 2, 4), fiducial='DESI'):
        self.k = np.asarray(k, dtype='f8')
        self.zeff = float(zeff)
        self.ells = tuple(ells)
        self.fiducial = fiducial
        self.requires = {'pt': ('BasePT', {'k': self.k, 'zeff': self.zeff, 'ells': self.ells})}

    def run(self, **params):
        params['b1'] = params.pop('bsigma8') / self.pt.sigma8 - 1.
        self.power = self.pt.combine_bias_terms_power_poles(**params)

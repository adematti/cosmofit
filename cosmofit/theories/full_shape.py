import numpy as np

from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD

from .bao import BaseTheoryPowerSpectrumMultipoles
from .power_template import BasePowerSpectrumTemplate  # to add calculator in the registry


class BasePTPowerSpectrum(BaseTheoryPowerSpectrumMultipoles):

    def __init__(self, *args, **kwargs):
        super(BasePTPowerSpectrum, self).__init__(*args, **kwargs)
        self.requires = {'effectap': ('EffectAP', {'zeff': self.zeff, 'fiducial': self.fiducial}), 'pklin': ('BasePowerSpectrumTemplate', {'k': self.kin})}


class LPTPowerSpectrum(BasePTPowerSpectrum):

    kin = np.logspace(-3., 1., 200)

    def run(self, **params):
        self.sigma8 = self.pklin.sigma8
        if 'fsigma8' in params:
            growth_rate = params['fsigma8'] / self.sigma8
        else:
            growth_rate = self.pklin.growth_rate
        self.lpt = LPT_RSD(self.kin, self.pklin.power_dd, kIR=0.2, cutoff=10, extrap_min=-4, extrap_max=3, N=2000, threads=1, jn=5)
        self.lpt.make_pltable(growth_rate, kv=self.k, apar=self.effectap.qpar, aperp=self.effectap.qper, ngauss=3)
        self.lpttable = [self.lpt.p0ktable, self.lpt.p2ktable, self.lpt.p4ktable]

    def combine_bias_terms_power_poles(self, b1=1.69, b2=-1.17, bs=-0.71, b3=0., alpha0=0., alpha2=0., alpha4=0., alpha6=0., sn0=0., sn2=0., sn4=0.):
        bias = [b1, b2, bs, b3]
        bias += [alpha0, alpha2, alpha4, alpha6]
        bias += [sn0, sn2, sn4]
        # pkells = self.lpt.combine_bias_terms_pkell(bias)[1:]
        # return np.array([pkells[[0, 2, 4].index(ell)] for ell in self.ells])
        bias_monomials = np.array([1, b1, b1**2, b2, b1 * b2, b2**2, bs, b1 * bs, b2 * bs, bs**2, b3, b1 * b3, alpha0, alpha2, alpha4, alpha6, sn0, sn2, sn4])
        return np.array([np.sum(self.lpttable[[0, 2, 4].index(ell)] * bias_monomials, axis=-1) for ell in self.ells])

    def __getstate__(self):
        return {'k': self.k, 'zeff': self.zeff, 'ells': self.ells, 'lpttable': self.lpttable}


class LPTGalaxyPowerSpectrum(BaseTheoryPowerSpectrumMultipoles):

    def __init__(self, *args, **kwargs):
        super(LPTGalaxyPowerSpectrum, self).__init__(*args, **kwargs)
        self.requires = {'pt': ('LPTPowerSpectrum', {'k': self.k, 'zeff': self.zeff, 'ells': self.ells, 'fiducial': self.fiducial})}

    def run(self, **params):
        if 'b1sigma8' in params:
            params['b1'] = params.pop('b1sigma8') / self.pt.sigma8
        self.power = self.pt.combine_bias_terms_power_poles(**params)
        #print(params)
        #if np.isnan(self.power).any():
        #    exit()

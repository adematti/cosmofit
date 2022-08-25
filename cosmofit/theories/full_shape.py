import numpy as np

from .base import TrapzTheoryPowerSpectrumMultipoles
from .bao import BaseTheoryPowerSpectrumMultipoles, BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles
from .power_template import BasePowerSpectrumParameterization  # to add calculator in the registry


class BasePTPowerSpectrumMultipoles(BaseTheoryPowerSpectrumMultipoles):

    def __init__(self, *args, **kwargs):
        super(BasePTPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        self.kin = np.geomspace(min(1e-3, self.k[0] / 2), max(10., self.k[0] * 2), 200)  # margin for AP effect
        self.requires = {'template': (BasePowerSpectrumParameterization, {'k': self.kin, 'zeff': self.zeff, 'fiducial': self.fiducial})}


class LPTPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles):

    # Slow, ~ 2 sec per iteration

    def run(self):
        from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
        self.lpt = LPT_RSD(self.kin, self.template.power_dd, kIR=0.2, cutoff=10, extrap_min=-4, extrap_max=3, N=2000, threads=1, jn=5)
        self.lpt.make_pltable(self.template.f, kv=self.k, apar=self.template.qpar, aperp=self.template.qper, ngauss=3)
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
        state = {}
        for name in ['k', 'zeff', 'ells', 'lpttable']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


class LPTTracerPowerSpectrumMultipoles(BaseTheoryPowerSpectrumMultipoles):

    def __init__(self, *args, **kwargs):
        super(LPTTracerPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        self.requires = {'pt': (self.__class__.__name__.replace('Tracer', ''), {'k': self.k, 'zeff': self.zeff, 'ells': self.ells, 'fiducial': self.fiducial})}

    def run(self, **params):
        self.power = self.pt.combine_bias_terms_power_poles(**params)


class KaiserTracerPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles, TrapzTheoryPowerSpectrumMultipoles):

    def __init__(self, *args, mu=200, **kwargs):
        super(KaiserTracerPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        self.set_k_mu(k=self.k, mu=mu, ells=self.ells)

    def run(self, b1=1., sn0=0.):
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        f = self.template.f
        pkmu = (b1 + f * muap**2)**2 * np.interp(np.log10(kap), np.log10(self.kin), self.template.power_dd) + sn0
        self.power = self.to_poles(pkmu)


class BaseTracerCorrelationFunctionMultipoles(BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles):

    def __init__(self, s=None, zeff=1., ells=(0, 2, 4), fiducial=None, **kwargs):
        super(BaseTracerCorrelationFunctionMultipoles, self).__init__(s=s, zeff=zeff, ells=ells, fiducial=fiducial)
        self.requires['power']['class'] = self.__class__.__name__.replace('CorrelationFunction', 'PowerSpectrum')


class LPTTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionMultipoles):

    pass


class KaiserTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionMultipoles):

    pass

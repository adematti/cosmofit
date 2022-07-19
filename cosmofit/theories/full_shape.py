import numpy as np

from .bao import BaseTheoryPowerSpectrumMultipoles, BaseTheoryCorrelationFunctionMultipoles
from .power_template import BasePowerSpectrumParameterization  # to add calculator in the registry


class BasePTPowerSpectrumMultipoles(BaseTheoryPowerSpectrumMultipoles):

    def __init__(self, *args, **kwargs):
        super(BasePTPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        self.requires = {'template': (BasePowerSpectrumParameterization, {'k': self.kin, 'zeff': self.zeff, 'fiducial': self.fiducial})}


class LPTPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles):

    kin = np.logspace(-3., 1., 200)

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


class KaiserTracerPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles):

    kin = np.logspace(-3., 1., 200)

    def run(self, b1=1.):
        f = self.template.f
        pdd = self.template.power_dd
        beta = f / b1
        factors = {0: (b1**2 + 2. / 3. * b1 * f + 1. / 5. * f**2),
                   2: (4. / 3. * b1 * f + 4. / 7. * f**2),
                   4: 8. / 35. * f**2}
        self.power = np.array([factors.get(ell, 0) for ell in self.ells])[:, None] * pdd


class BaseTracerCorrelationFunctionMultipoles(BaseTheoryCorrelationFunctionMultipoles):

    def __init__(self, s=None, zeff=1., ells=(0, 2, 4), fiducial=None, **kwargs):
        super(LPTTracerCorrelationFunctionMultipoles, self).__init__(s=s, zeff=zeff, ells=ells, fiducial=fiducial)
        self.k = np.logspace(min(-3, - np.log10(self.s[-1]) - 0.1), max(2, - np.log10(self.s[0]) + 0.1), 2000)
        from cosmoprimo import PowerToCorrelation
        self.fftlog = PowerToCorrelation(self.k, ell=self.ells, q=0, lowring=False)
        kv = np.geomspace(self.k[0], 0.5, 200)
        mask = self.k > kv[-1]
        self.pad = np.ones((len(self.ells), mask.sum()), dtype='f8')
        self.pad *= np.exp(-(self.k[mask] - kv)**2 / (2. * (0.5)**2))
        self.requires = {'power': (self.__class__.__name__.replace('CorrelationFunction', 'PowerSpectrum'),
                                  {'k': kv, 'zeff': self.zeff, 'ells': self.ells, 'fiducial': self.fiducial, **kwargs})}

    def run(self):
        k, power = self.power.k, self.power.power
        power = np.concatenate([power, power[:, -1:] * self.pad], axis=-1)
        s, self.corr = self.fftlog([np.interp(np.log(self.k), np.log(kv), power)])
        self.s = s[0]


class LPTTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionMultipoles):

    pass


class KaiserTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionMultipoles):

    pass

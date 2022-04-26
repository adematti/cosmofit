import numpy as np

from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD

from cosmofit.base import BaseCalculator
from .bao import get_cosmo


class BasePT(BaseCalculator):

    def __init__(self, k, zeff=1., mu=101, ells=(0, 2, 4), fiducial='DESI', engine='LPT'):
        self.engine = engine
        self.zeff = float(zeff)
        fiducial = get_cosmo(fiducial)
        self.efunc_fid = fiducial.efunc(self.zeff)
        self.comoving_angular_distance_fid = fiducial.comoving_angular_distance(self.zeff)

    def prepare(self, inputs):
        cosmo = inputs['cosmoprimo']
        fo = cosmo.get_fourier()
        self.growth_rate = fo.sigma8_z(self.zeff, of='theta_cb') / fo.sigma8_z(self.zeff, of='delta_cb')
        self.pklin = fo.pk_interpolator(of='delta_cb').to_1d(self.zeff)
        efunc = cosmo.efunc(self.zeff)
        comoving_angular_distance = cosmo.comoving_angular_distance(self.zeff)
        self.qpar, self.qper = self.efunc_fid / efunc, comoving_angular_distance / self.comoving_angular_distance_fid


class LPT(BasePT):

    def get_output(self, inputs):
        ki = np.logspace(-3, 1, 200)
        lpt = LPT_RSD(ki, self.pklin(ki), kIR=0.2, cutoff=10, extrap_min=-4, extrap_max=3, N=2000, threads=1, jn=5)
        lpt.make_pltable(self.growth_rate, kv=self.k, apar=self.qpar, aperp=self.qper, ngauss=2)
        return {'{}_z={}'.format(self.__class__.__name__, self.zeff): lpt}


class LPTPowerSpectrum(BaseCalculator):

    def __init__(self, zeff=1., ells=(0, 2, 4)):
        self.zeff = float(zeff)
        self.ells = tuple(ells)

    def pk_ell(self, inputs):
        lpt = inputs['LPT_z={}'.format(self.zeff)]
        sigma8 = inputs['cosmoprimo'].sigma8_m
        b1 = inputs['bsigma8'] / sigma8 - 1.
        bias = [b1, inputs['b2'], inputs['bs'], 0.]
        bias += [inputs['alpha0'], inputs['alpha2'], 0., 0.]
        bias += [inputs['sn0'], inputs['sn2'], 0.]
        self.k = lpt.kv
        pkells = lpt.combine_bias_terms_pkell(bias)[1:]
        return [pkells[self.ells.index(ell)] for ell in self.ells]

    def get_output(self, inputs):
        self.poles = self.pk_ell(inputs)
        return {'{}_z={}'.format(self.__class__.__name__, self.zeff): self}

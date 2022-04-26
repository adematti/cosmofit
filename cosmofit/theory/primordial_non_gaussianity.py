import numpy as np
from scipy import constants

from .bao import BasePowerSpectrumMultipoles


class PrimordialNonGaussianityPowerSpectrum(BasePowerSpectrumMultipoles):

    def __init__(self, k, zeff=1., mu=101, ells=(0, 2, 4)):
        self.set_k_mu(k, mu, ells=ells)
        self.zeff = float(zeff)

    def pk_mu(self, inputs):
        cosmo = inputs['cosmoprimo']
        pk = cosmo.pk_interpolator().to_1d(self.k, z=self.zeff)
        tk = pk / self.k**cosmo.n_s
        tk /= tk[0]
        # https://arxiv.org/pdf/1904.08859.pdf eq. 2.3
        alpha = 3. * cosmo.Omega0_m * 100**2 * 1.686 / ((constants.c / 1e3) ** 2 * self.k**2 * tk * cosmo.growth_factor(self.zeff))
        bias = inputs['bias'] + inputs['fnl_loc'] * (inputs['bias'] - inputs['p']) * alpha
        fog = 1. / (1. + inputs['sigmas']**2 * self.k**2 * self.mu[None, :]**2 / 2.)**2.
        return fog * (bias + inputs['growth_rate'] * self.mu[None, :]**2)**2 * pk + inputs['sn0']

    def pk_ell(self, inputs):
        pkmu = self.pk_mu(inputs)
        return np.sum(pkmu * self.muweights[:, None, :], axis=-1)

    def get_output(self, inputs):
        self.poles = self.pk_ell(inputs)
        return {'{}_z={}'.format(self.__class__.__name__, self.zeff): self}

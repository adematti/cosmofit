from scipy import constants

from .base import TrapzTheoryPowerSpectrumMultipoles


class PrimordialNonGaussianityPowerSpectrumMultipoles(TrapzTheoryPowerSpectrumMultipoles):

    def __init__(self, *args, **kwargs):
        super(PrimordialNonGaussianityPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        self.requires = {'cosmo': ('BasePrimordialCosmology', {})}

    def run(self, bfnl_loc=0., bias=2., sigmas=0., sn0=0.):
        power = self.cosmo.get_fourier().pk_interpolator().to_1d(self.zeff)
        power_prim = self.cosmo.get_primordial().pk_interpolator()  # power_prim is ~ k^(n_s - 1)
        pk = power(self.k)
        k0 = power.k[0]
        tk = (pk / power_prim(self.k) / self.k / (power(k0) / power_prim(k0) / k0))**0.5
        # https://arxiv.org/pdf/1904.08859.pdf eq. 2.3
        znorm = 10.
        normalized_growth_factor = self.cosmo.growth_factor(self.zeff) / self.cosmo.growth_factor(znorm) / (1 + znorm)
        alpha = 3. * self.cosmo.Omega0_m * 100**2 / (2. * (constants.c / 1e3)**2 * self.k**2 * tk * normalized_growth_factor)
        growth_rate = self.cosmo.growth_rate(self.zeff)
        bias = bias + bfnl_loc * alpha
        fog = 1. / (1. + sigmas**2 * self.k[:, None]**2 * self.mu**2 / 2.)**2.
        pkmu = fog * (bias[:, None] + growth_rate * self.mu**2)**2 * pk[:, None] + sn0
        self.power = self.to_poles(pkmu)

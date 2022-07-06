from scipy import constants

from .bao import BaseTheoryPowerSpectrumMultipoles


class PrimordialNonGaussianityPowerSpectrum(BaseTheoryPowerSpectrumMultipoles):

    def __init__(self, *args, **kwargs):
        super(PrimordialNonGaussianityPowerSpectrum, self).__init__(*args, **kwargs)
        self.requires = {'cosmo': ('BasePrimordialCosmology', {})}

    def run(self, fnl_loc=0., p=1., bias=2., sigmas=0., sn0=0.):
        pk = self.cosmo.get_fourier().pk_interpolator().to_1d(self.k, z=self.zeff)
        tk = pk / self.cosmo.get_primordial().pk_interpolator()(self.k)
        tk /= tk[0]
        # https://arxiv.org/pdf/1904.08859.pdf eq. 2.3
        znorm = 10.
        normalized_growth_factor = self.cosmo.growth_factor(self.zeff) / self.cosmo.growth_factor(znorm) * (1 + znorm)
        alpha = 3. * self.cosmo.Omega0_m * 100**2 * 1.686 / ((constants.c / 1e3) ** 2 * self.k**2 * tk * normalized_growth_factor)
        growth_rate = self.cosmo.growth_rate(self.zeff)
        bias = bias + fnl_loc * (bias - p) * alpha
        fog = 1. / (1. + sigmas**2 * self.k**2 * self.mu[None, :]**2 / 2.)**2.
        pkmu = fog * (bias + growth_rate * self.mu[None, :]**2)**2 * pk + sn0
        self.power = self.to_poles(pkmu)

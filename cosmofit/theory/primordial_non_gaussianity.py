from scipy import constants

from .bao import BasePowerSpectrumMultipoles


class PrimordialNonGaussianityPowerSpectrum(BasePowerSpectrumMultipoles):

    name = 'pngpower'

    def requires(self):
        return ['cosmoprimo']

    def run(self, fnl_loc=0., p=1., bias=2., sigmas=0., sn0=0.):
        pk = self.cosmoprimo.get_fourier().pk_interpolator().to_1d(self.k, z=self.zeff)
        tk = pk / self.cosmoprimo.get_primordial().pk_interpolator()(self.k)
        tk /= tk[0]
        # https://arxiv.org/pdf/1904.08859.pdf eq. 2.3
        alpha = 3. * self.cosmoprimo.Omega0_m * 100**2 * 1.686 / ((constants.c / 1e3) ** 2 * self.k**2 * tk * self.cosmoprimo.growth_factor(self.zeff))
        growth_rate = self.cosmoprimo.growth_rate(self.zeff)
        bias = bias + fnl_loc * (bias - p) * alpha
        fog = 1. / (1. + sigmas**2 * self.k**2 * self.mu[None, :]**2 / 2.)**2.
        pkmu = fog * (bias + growth_rate * self.mu[None, :]**2)**2 * pk + sn0
        self.power = self.to_poles(pkmu)

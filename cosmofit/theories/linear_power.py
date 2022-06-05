import numpy as np

from cosmofit.base import BaseCalculator


class BasePrimordialPowerSpectrum(BaseCalculator):

    def __init__(self, k=None, zeff=1.):
        if k is None:
            k = np.logspace(-3., 1., 200)
        self.k = np.array(k, dtype='f8')
        self.zeff = float(zeff)
        self.requires = {'cosmo': ('BasePrimordialCosmology', {})}

    def run(self):
        fo = self.cosmo.get_fourier()
        self.growth_rate = fo.sigma8_z(self.zeff, of='theta_cb') / fo.sigma8_z(self.zeff, of='delta_cb')
        self.sigma8 = fo.sigma8_z(self.zeff, of='delta_cb')
        self.pkdd = fo.pk_interpolator(of='delta_cb')(self.k, z=self.zeff)

    def __getstate__(self):
        state = {}
        for name in ['k', 'zeff', 'growth_rate', 'sigma8', 'pkdd']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


class ShapeFitPrimordialPowerSpectrum(BasePrimordialPowerSpectrum):

    def __init__(self, *args, a=0.6, kpivot=0.03, **kwargs):
        super(ShapeFitPrimordialPowerSpectrum, self).__init__(*args, **kwargs)
        self.a = float(a)
        self.kpivot = float(kpivot)

    def run(self, m=0., n=0.):
        super(ShapeFitPrimordialPowerSpectrum, self).run()
        factor = m / self.a * np.tanh(self.a * np.log(self.k / self.kpivot)) + n * np.log(self.k / self.kpivot)
        self.pkdd *= np.exp(factor)

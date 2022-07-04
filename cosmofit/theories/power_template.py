import numpy as np

from cosmoprimo import PowerSpectrumBAOFilter

from cosmofit.base import BaseCalculator


class BasePowerSpectrumWiggles(BaseCalculator):

    def __init__(self, zeff=1., engine='wallish2018'):
        self.engine = engine
        self.zeff = float(zeff)
        self.requires = {'cosmo': ('BasePrimordialCosmology', {})}

    def run(self):
        self.power = self.cosmo.get_fourier().pk_interpolator().to_1d(z=self.zeff)
        self.power_now = PowerSpectrumBAOFilter(self.power, engine=self.engine).smooth_pk_interpolator()

    def wiggles(self, k):
        return self.power(k) - self.power_now(k)


class BasePowerSpectrumTemplate(BaseCalculator):

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
        self.power_dd = fo.pk_interpolator(of='delta_cb')(self.k, z=self.zeff)

    def __getstate__(self):
        state = {}
        for name in ['k', 'zeff', 'growth_rate', 'sigma8', 'power_dd']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


class ShapeFitPowerSpectrumTemplate(BasePowerSpectrumTemplate):

    def __init__(self, *args, a=0.6, k_pivot=0.03, **kwargs):
        super(ShapeFitPowerSpectrumTemplate, self).__init__(*args, **kwargs)
        self.a = float(a)
        self.k_pivot = float(k_pivot)
        self.requires['wiggles'] = ('PowerSpectrumWiggles', {'zeff': self.zeff})

    def run(self, m=0., n=0.):
        super(ShapeFitPowerSpectrumTemplate, self).run()
        factor = m / self.a * np.tanh(self.a * np.log(self.k / self.k_pivot)) + n * np.log(self.k / self.k_pivot)
        self.power_dd *= np.exp(factor)
        self.A_p_ref = self.wiggles.power_now(self.k_pivot)
        self.n_s_ref = n + self.cosmo.n_s
        dk = 1e-3
        k = self.k_pivot * np.array([1. - dk, 1. + dk])
        if self.params['n'].varied:
            pk_prim = self.cosmo.get_primordial().pk_interpolator()(k)
        else:
            pk_prim = 1.
        self.m_ref = m + (np.diff(np.log(self.wiggles.power_now(k) / pk_prim)) / np.diff(np.log(k)))[0]

    def __getstate__(self):
        state = super(ShapeFitPowerSpectrumTemplate, self).__getstate__()
        for name in ['A_p_ref', 'n_s_ref', 'm_ref']:
            state[name] = getattr(self, name)
        return state

import numpy as np

from cosmofit.base import BaseCalculator
from .primordial_cosmology import BasePrimordialCosmology
from .base import EffectAP


class BasePowerSpectrumWiggles(BaseCalculator):

    def __init__(self, zeff=1., engine='wallish2018', **kwargs):
        self.engine = engine
        self.zeff = float(zeff)
        self.requires = {'cosmo': {'class': BasePrimordialCosmology, 'init': kwargs}}

    def run(self):
        fo = self.cosmo.get_fourier()
        self.sigma8 = fo.sigma8_z(self.zeff, of='delta_cb')
        self.fsigma8 = fo.sigma8_z(self.zeff, of='theta_cb')
        self.power = fo.pk_interpolator().to_1d(z=self.zeff)
        from cosmoprimo import PowerSpectrumBAOFilter
        self.power_now = PowerSpectrumBAOFilter(self.power, engine=self.engine).smooth_pk_interpolator()

    def wiggles(self, k):
        return self.power(k) - self.power_now(k)


class BasePowerSpectrumTemplate(BaseCalculator):

    def __init__(self, k=None, zeff=1., **kwargs):
        if k is None:
            k = np.logspace(-3., 1., 200)
        self.k = np.array(k, dtype='f8')
        self.zeff = float(zeff)
        self.requires = {'cosmo': {'class': BasePrimordialCosmology, 'init': kwargs}}

    def run(self):
        fo = self.cosmo.get_fourier()
        self.sigma8 = fo.sigma8_z(self.zeff, of='delta_cb')
        self.fsigma8 = fo.sigma8_z(self.zeff, of='theta_cb')
        self.power_dd = fo.pk_interpolator(of='delta_cb')(self.k, z=self.zeff)

    def __getstate__(self):
        state = {}
        for name in ['k', 'zeff', 'sigma8', 'fsigma8', 'power_dd']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


class FullPowerSpectrumTemplate(BasePowerSpectrumTemplate):

    def __init__(self, k=None, zeff=1.):
        super(FullPowerSpectrumTemplate, self).__init__(k=k, zeff=zeff, fiducial=None)


class ShapeFitPowerSpectrumTemplate(BasePowerSpectrumTemplate):

    def __init__(self, *args, a=0.6, k_pivot=0.03, **kwargs):
        super(ShapeFitPowerSpectrumTemplate, self).__init__(*args, **kwargs)
        self.a = float(a)
        self.k_pivot = float(k_pivot)
        self.requires['wiggles'] = (BasePowerSpectrumWiggles, {'zeff': self.zeff, **self.requires['cosmo']['init']})

    def run(self, dm=0., dn=0.):
        super(ShapeFitPowerSpectrumTemplate, self).run()
        factor = dm / self.a * np.tanh(self.a * np.log(self.k / self.k_pivot)) + dn * np.log(self.k / self.k_pivot)
        self.power_dd *= np.exp(factor)
        self.A_p = self.wiggles.power_now(self.k_pivot)
        self.n = dn + self.cosmo.n_s
        dk = 1e-3
        k = self.k_pivot * np.array([1. - dk, 1. + dk])
        if self.runtime_info.base_params['dn'].varied:
            pk_prim = self.cosmo.get_primordial().pk_interpolator()(k)
        else:
            pk_prim = 1.
        self.m = dm + (np.diff(np.log(self.wiggles.power_now(k) / pk_prim)) / np.diff(np.log(k)))[0]

    def __getstate__(self):
        state = super(ShapeFitPowerSpectrumTemplate, self).__getstate__()
        for name in ['A_p', 'n', 'm']:
            state[name] = getattr(self, name)
        return state


class BandVelocityPowerSpectrumTemplate(BasePowerSpectrumTemplate):

    _baseparamname = 'rptt'

    def __init__(self, *args, kpoints=None, **kwargs):
        super(BandVelocityPowerSpectrumTemplate, self).__init__(*args, **kwargs)
        self.kpoints = kpoints

    def set_params(self, params):
        params = params.select(basename=['{}*'.format(self._baseparamname)])
        npoints = len(params)
        if not npoints:
            raise ValueError('No parameter {}* found'.format(self._baseparamname))
        if self.kpoints is None:
            step = (self.k[-1] - self.k[0]) / npoints
            self.kpoints = (self.k[0] + step / 2., self.k[-1] - step / 2.)
        if len(self.kpoints) == 2:
            self.kpoints = np.linspace(*self.kpoints, num=npoints)
        self.kpoints = np.array(self.kpoints)
        if self.kpoints.size != npoints:
            raise ValueError('{:d} (!= {:d} parameters {}*) points have been provided'.format(self.kpoints.size, npoints, self._baseparamname))

        if self.kpoints is not None:
            for ikp, kp in enumerate(self.kpoints):
                basename = '{}{:d}'.format(self._baseparamname, ikp)
                param = params.select(basename=[basename])[0]
                if param.latex is None:
                    param.latex = r'P_{\{{0}\{0}}}(k={1:.3f})'.format('theta', kp)

        zeros = (self.kpoints[:-1] + self.kpoints[1:]) / 2.
        if self.kpoints[0] < self.k[0]:
            raise ValueError('Theory k starts at {0:.2e} but first point is {1:.2e} < {0:.2e}'.format(self.k[0], self.kpoints[0]))
        if self.kpoints[-1] > self.k[-1]:
            raise ValueError('Theory k ends at {0:.2e} but last point is {1:.2e} > {0:.2e}'.format(self.k[-1], self.kpoints[-1]))
        zeros = np.concatenate([[self.k[0]], zeros, [self.k[-1]]], axis=0)
        self.templates = []
        for ip, kp in enumerate(self.kpoints):
            diff = self.k - kp
            mask_neg = diff < 0
            diff[mask_neg] /= (zeros[ip] - kp)
            diff[~mask_neg] /= (zeros[ip + 1] - kp)
            self.templates.append(np.maximum(1. - diff, 0.))
        self.templates = np.array(self.templates)
        return params

    def run(self, f=None, **params):
        fo = self.cosmo.get_fourier()
        self.power_tt = fo.pk_interpolator(of='theta_cb')(self.k, z=self.zeff)
        ppoints = fo.pk_interpolator(of='theta_cb')(self.kpoints, z=self.zeff)
        self.ptt = np.empty(len(self.kpoints), dtype='f8')
        for ii, template in enumerate(self.templates):
            rptt = params['{}{:d}'.format(self._baseparamname, ii)]
            self.ptt[ii] = ppoints[ii] * (1. + rptt)
            self.power_tt += rptt * template


class BasePowerSpectrumParameterization(BaseCalculator):

    _parambasenames = ()

    def __init__(self, k=None, zeff=1., fiducial=None, **kwargs):
        self.requires = {'template': {'class': self.__class__.__name__.replace('Parameterization', 'Template'), 'init': {'k': k, 'zeff': zeff, **kwargs}},
                         'effectap': {'class': EffectAP, 'init': {'zeff': zeff, 'fiducial': fiducial}}}

    def set_params(self, params):
        self_params = params.select(basename=self._parambasenames)
        effectap_params = params.select(basename=['qpar', 'qper', 'qiso', 'qap'])
        self.requires['effectap']['params'] = effectap_params
        template_params = params.copy()
        for param in self_params: del template_params[param]
        for param in effectap_params: del template_params[param]
        self.requires['template']['params'] = template_params
        return self_params

    def run(self):
        self.power_dd = self.template.power_dd
        self.qpar, self.qper = self.effectap.qpar, self.effectap.qper

    def ap_k_mu(self, k, mu):
        return self.effectap.ap_k_mu(k, mu)


class FullPowerSpectrumParameterization(BasePowerSpectrumParameterization):

    def __init__(self, *args, **kwargs):
        super(FullPowerSpectrumParameterization, self).__init__(*args, **kwargs)
        self.requires['effectap']['init']['mode'] = 'distances'


class ShapeFitPowerSpectrumParameterization(BasePowerSpectrumParameterization):

    _parambasenames = ('f', 'A_p', 'f_sqrt_A_p', 'n', 'm')

    def __init__(self, *args, fiducial=None, **kwargs):
        super(ShapeFitPowerSpectrumParameterization, self).__init__(*args, fiducial=fiducial, **kwargs)
        if fiducial is None:
            raise ValueError('Give fiducial cosmology for power spectrum template')
        self.requires['template']['init']['fiducial'] = fiducial
        self.requires['effectap']['init']['mode'] = 'qparqper'

    def run(self, f=None):
        super(ShapeFitPowerSpectrumParameterization, self).run()
        self.f = f
        for name in ['A_p', 'n', 'm']:
            setattr(self, name, getattr(self.template, name))
        self.f_sqrt_A_p = self.f * self.A_p**0.5


class BandVelocityPowerSpectrumParameterization(BasePowerSpectrumParameterization):

    _parambasenames = ('f',)

    def __init__(self, *args, fiducial=None, **kwargs):
        super(BandVelocityPowerSpectrumParameterization, self).__init__(*args, fiducial=fiducial, **kwargs)
        if fiducial is None:
            raise ValueError('Give fiducial cosmology for power spectrum template')
        self.requires['template']['init']['fiducial'] = fiducial
        self.requires['effectap']['init']['mode'] = 'qap'

    def run(self, f=None):
        self.f = f
        if f is None: self.f = self.template.fsigma8 / self.template.sigma8
        self.power_dd = self.template.power_tt / f**2
        self.ptt = self.template.ptt
        self.qpar, self.qper = self.effectap.qpar, self.effectap.qper


class BAOWigglesPowerSpectrumParameterization(BasePowerSpectrumParameterization):

    _parambasenames = ('f',)

    def __init__(self, zeff=1., fiducial=None, wiggles='BasePowerSpectrumWiggles', **kwargs):
        if fiducial is None:
            raise ValueError('Give fiducial cosmology for power spectrum template')
        self.requires = {'template': {'class': wiggles, 'init': {'zeff': zeff, 'fiducial': fiducial, **kwargs}},
                         'effectap': {'class': EffectAP, 'init': {'zeff': zeff, 'fiducial': fiducial, 'mode': 'qparqper'}}}

    def run(self, f=None):
        for name in ['power', 'power_now', 'wiggles']:
            setattr(self, name, getattr(self.template, name))
        self.f = f
        if f is None: self.f = self.template.fsigma8 / self.template.sigma8
        self.qpar, self.qper = self.effectap.qpar, self.effectap.qper

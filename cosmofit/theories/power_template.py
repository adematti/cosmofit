import numpy as np
from scipy import constants

from cosmofit.base import BaseCalculator
from .primordial_cosmology import BasePrimordialCosmology
from .base import EffectAP


class BasePowerSpectrumWiggles(BaseCalculator):

    def __init__(self, zeff=1., engine='wallish2018', of='delta_cb', **kwargs):
        self.engine = engine
        self.zeff = float(zeff)
        self.of = of
        self.requires = {'cosmo': {'class': BasePrimordialCosmology, 'init': kwargs}}

    def run(self):
        fo = self.cosmo.get_fourier()
        self.sigma8 = fo.sigma8_z(self.zeff, of='delta_cb')
        self.fsigma8 = fo.sigma8_z(self.zeff, of='theta_cb')
        self.power = fo.pk_interpolator(of=self.of).to_1d(z=self.zeff, extrap_kmin=1e-6, extrap_kmax=1e3)
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


class BAOExtractor(BaseCalculator):

    def __init__(self, zeff=1., **kwargs):
        self.zeff = float(zeff)
        self.requires = {'cosmo': (BasePrimordialCosmology, kwargs)}

    def run(self, qpar=1., qper=1.):
        rd = self.cosmo.rs_drag
        self.DH_over_rd = qpar * constants.c / 1e3 / self.cosmo.hubble_function(self.zeff) / rd
        self.DM_over_rd = qper * self.cosmo.comoving_angular_distance(self.zeff) / rd
        self.DH_over_DM = self.DH_over_rd / self.DM_over_rd
        self.DV_over_rd = (self.DH_over_rd * self.DM_over_rd**2. * self.zeff)**(1. / 3.)


class ShapeFitPowerSpectrumExtractor(BaseCalculator):

    def __init__(self, zeff=1., kpivot=0.03, n_varied=True, **kwargs):
        self.zeff = float(zeff)
        self.kpivot = float(kpivot)
        self.n_varied = bool(n_varied)
        self.requires = {'cosmo': {'class': BasePrimordialCosmology, 'init': kwargs},
                         'wiggles': {'class': BasePowerSpectrumWiggles, 'init': {'zeff': self.zeff, **kwargs}}}

    def run(self):
        self.A_p = self.wiggles.power_now(self.kpivot)
        self.n = self.cosmo.n_s
        dk = 1e-3
        k = self.kpivot * np.array([1. - dk, 1. + dk])
        if self.n_varied:
            pk_prim = self.cosmo.get_primordial().pk_interpolator()(k) * k
        else:
            pk_prim = 1.
        self.m = (np.diff(np.log(self.wiggles.power_now(k) / pk_prim)) / np.diff(np.log(k)))[0]

    def __getstate__(self):
        state = {}
        for name in ['A_p', 'n', 'm', 'n_varied']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


class ShapeFitPowerSpectrumTemplate(BasePowerSpectrumTemplate):

    def __init__(self, a=0.6, kpivot=0.03, k=None, **kwargs):
        super(ShapeFitPowerSpectrumTemplate, self).__init__(k=k, **kwargs)
        ShapeFitPowerSpectrumExtractor.__init__(self, kpivot=kpivot, **kwargs)
        self.a = float(a)

    def run(self, dm=0., dn=0.):
        self.n_varied = self.runtime_info.base_params['dn'].varied  # for ShapeFitPowerSpectrumExtractor
        super(ShapeFitPowerSpectrumTemplate, self).run()
        ShapeFitPowerSpectrumExtractor.run(self)
        factor = dm / self.a * np.tanh(self.a * np.log(self.k / self.kpivot)) + dn * np.log(self.k / self.kpivot)
        self.power_dd *= np.exp(factor)
        self.n += dn
        self.m += dm

    def __getstate__(self):
        state = super(ShapeFitPowerSpectrumTemplate, self).__getstate__()
        state.update(ShapeFitPowerSpectrumExtractor.__getstate__(self))
        return state


class WiggleSplitPowerSpectrumTemplate(BasePowerSpectrumTemplate):

    def __init__(self, k=None, r=8., **kwargs):
        super(WiggleSplitPowerSpectrumTemplate, self).__init__(k=k, **kwargs)
        self.requires = {'wiggles': {'class': BasePowerSpectrumWiggles, 'init': {'zeff': self.zeff, 'of': 'theta_cb', **kwargs}}}
        self.r = float(r)

    def run(self, qbao=1.):
        self.power_tt = self.wiggles.power_now(self.k) + self.wiggles.wiggles(self.k / qbao)
        fo = self.wiggles.cosmo.get_fourier()
        self.sigmar = fo.sigma_rz(self.r, self.zeff, of='delta_cb')
        self.fsigmar = fo.sigma_rz(self.r, self.zeff, of='theta_cb')
        self.f = self.fsigmar / self.sigmar
        self.power_tt /= self.fsigmar**2

    def __getstate__(self):
        state = {}
        for name in ['power_tt']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


class BandVelocityPowerSpectrumExtractor(BaseCalculator):

    def __init__(self, kptt=None, zeff=1., **kwargs):
        if kptt is None:
            kptt = self.globals.get('kdata', None)
            if kptt is not None and np.ndim(kptt[0]) != 0:  # multipoles
                kptt = kptt[0]
        self.kptt = kptt
        self.zeff = float(zeff)
        self.requires = {'cosmo': {'class': BasePrimordialCosmology, 'init': kwargs}}

    def run(self):
        self.power_tt = self.cosmo.get_fourier().pk_interpolator(of='theta_cb').to_1d(z=self.zeff)
        self.ptt = self.power_tt(self.kptt)

    def __getstate__(self):
        state = {}
        for name in ['kptt', 'ptt']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


class BandVelocityPowerSpectrumTemplate(BasePowerSpectrumTemplate):

    _baseparamname = 'rptt'

    def __init__(self, kptt=None, k=None, zeff=1., **kwargs):
        super(BandVelocityPowerSpectrumTemplate, self).__init__(k=k, zeff=zeff, **kwargs)
        BandVelocityPowerSpectrumExtractor.__init__(self, kptt=kptt, **kwargs)

    def set_params(self, params):
        import re
        rparams = params.select(basename=re.compile(r'{}(-?\d+)'.format(self._baseparamname)))
        npoints = len(rparams)
        if self.kptt is None:
            if not npoints:
                raise ValueError('No parameter {}* found'.format(self._baseparamname))
            step = (self.k[-1] - self.k[0]) / npoints
            self.kptt = (self.k[0] + step / 2., self.k[-1] - step / 2.)
        if npoints:
            if len(self.kptt) == 2:
                self.kptt = np.linspace(*self.kptt, num=npoints)
            self.kptt = np.array(self.kptt)
            if self.kptt.size != npoints:
                raise ValueError('{:d} (!= {:d} parameters {}*) points have been provided'.format(self.kptt.size, npoints, self._baseparamname))
        self_params = params.__class__()
        for ikp, kp in enumerate(self.kptt):
            basename = '{}{:d}'.format(self._baseparamname, ikp)
            self_params.set(dict(value=0., basename=basename,
                                 prior={'dist': 'norm', 'loc': 0., 'scale': 1.},
                                 latex=r'(\Delta P / P)_{{\{0}\{0}}}(k={1:.3f})'.format('theta', kp)))
        params = self_params.clone(params)

        zeros = (self.kptt[:-1] + self.kptt[1:]) / 2.
        if self.kptt[0] < self.k[0]:
            raise ValueError('Theory k starts at {0:.2e} but first point is {1:.2e} < {0:.2e}'.format(self.k[0], self.kptt[0]))
        if self.kptt[-1] > self.k[-1]:
            raise ValueError('Theory k ends at {0:.2e} but last point is {1:.2e} > {0:.2e}'.format(self.k[-1], self.kptt[-1]))
        zeros = np.concatenate([[self.k[0]], zeros, [self.k[-1]]], axis=0)
        self.templates = []
        for ip, kp in enumerate(self.kptt):
            diff = self.k - kp
            mask_neg = diff < 0
            diff[mask_neg] /= (zeros[ip] - kp)
            diff[~mask_neg] /= (zeros[ip + 1] - kp)
            self.templates.append(np.maximum(1. - diff, 0.))
        self.templates = np.array(self.templates)
        return params

    def run(self, **params):
        BandVelocityPowerSpectrumExtractor.run(self)
        rptt = []
        for ii, template in enumerate(self.templates):
            r = params['{}{:d}'.format(self._baseparamname, ii)]
            self.ptt[ii] *= (1. + r)
            rptt.append(r)
        self.power_tt = self.power_tt(self.k)
        self.power_tt *= 1. + np.dot(rptt, self.templates)
        fo = self.cosmo.get_fourier()
        self.f = fo.sigma8_z(z=self.zeff, of='theta_cb') / fo.sigma8_z(z=self.zeff, of='delta_cb')

    def __getstate__(self):
        state = BandVelocityPowerSpectrumExtractor.__getstate__(self)
        for name in ['power_tt']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


class BasePowerSpectrumParameterization(BaseCalculator):

    _parambasenames = ()

    def __init__(self, k=None, zeff=1., fiducial=None, **kwargs):
        self.zeff = float(zeff)
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
        mode = self.requires['effectap']['init'].get('mode', None)
        if mode is None:
            mode = self.requires['effectap']['init']['mode'] = EffectAP.guess_mode(effectap_params, default='qparqper')
        if mode == 'qiso':
            for param in self_params:
                if param.basename in ['DM_over_rd', 'DH_over_rd', 'DH_over_DM']: del self_params[param]
        if mode == 'qap':
             for param in self_params:
                 if param.basename in ['DM_over_rd', 'DH_over_rd', 'DV_over_rd']: del self_params[param]
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

    _parambasenames = ('f', 'A_p', 'f_sqrt_A_p', 'n', 'm', 'DM_over_rd', 'DH_over_rd', 'DH_over_DM', 'DV_over_rd')

    def __init__(self, *args, mode=None, **kwargs):
        super(ShapeFitPowerSpectrumParameterization, self).__init__(*args, **kwargs)
        self.requires['template']['init']['fiducial'] = self.requires['effectap']['init']['fiducial']
        self.requires['effectap']['init']['mode'] = mode

    def run(self, f=None):
        super(ShapeFitPowerSpectrumParameterization, self).run()
        self.f = f
        for name in ['A_p', 'n', 'm']:
            setattr(self, name, getattr(self.template, name))
        self.f_sqrt_A_p = self.f * self.A_p**0.5
        self.qpar, self.qper = self.effectap.qpar, self.effectap.qper
        self.cosmo = self.template.cosmo
        BAOExtractor.run(self, qpar=self.qpar, qper=self.qper)


class WiggleSplitPowerSpectrumParameterization(BasePowerSpectrumParameterization):

    _parambasenames = ('fsigmar',)

    def __init__(self, *args, **kwargs):
        super(WiggleSplitPowerSpectrumParameterization, self).__init__(*args, **kwargs)
        self.requires['template']['init']['fiducial'] = self.requires['effectap']['init']['fiducial']
        self.requires['effectap']['init']['mode'] = 'qap'

    def run(self, fsigmar):
        self.fsigmar = fsigmar
        self.f = self.fsigmar / self.template.sigmar
        self.power_dd = self.template.power_tt / self.template.sigmar**2
        self.qpar, self.qper = self.effectap.qpar, self.effectap.qper


class BandVelocityPowerSpectrumParameterization(BasePowerSpectrumParameterization):

    _parambasenames = ('f', 'ptt')

    def __init__(self, *args, **kwargs):
        super(BandVelocityPowerSpectrumParameterization, self).__init__(*args, **kwargs)
        self.requires['template']['init']['fiducial'] = self.requires['effectap']['init']['fiducial']
        self.requires['effectap']['init']['mode'] = 'qap'

    def run(self, f=None):
        self.f = f
        if f is None: self.f = self.template.f
        self.power_dd = self.template.power_tt / self.template.f**2
        self.ptt = self.template.ptt
        self.qpar, self.qper = self.effectap.qpar, self.effectap.qper


class BAOPowerSpectrumParameterization(BasePowerSpectrumParameterization):

    _parambasenames = ('f', 'DM_over_rd', 'DH_over_rd', 'DH_over_DM', 'DV_over_rd')

    def __init__(self, zeff=1., fiducial=None, mode=None, wiggles='BasePowerSpectrumWiggles', **kwargs):
        self.zeff = float(zeff)
        self.requires = {'template': {'class': wiggles, 'init': {'zeff': zeff, 'fiducial': fiducial, **kwargs}},
                         'effectap': {'class': EffectAP, 'init': {'zeff': zeff, 'fiducial': fiducial, 'mode': mode}}}

    def run(self, f=None):
        for name in ['power', 'power_now', 'wiggles']:
            setattr(self, name, getattr(self.template, name))
        self.f = f
        if f is None: self.f = self.template.fsigma8 / self.template.sigma8
        self.qpar, self.qper = self.effectap.qpar, self.effectap.qper
        self.cosmo = self.template.cosmo
        BAOExtractor.run(self, qpar=self.qpar, qper=self.qper)

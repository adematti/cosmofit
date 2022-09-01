import numpy as np

from .base import TrapzTheoryPowerSpectrumMultipoles
from .bao import BaseTheoryPowerSpectrumMultipoles, BaseTheoryCorrelationFunctionMultipoles, BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles
from .power_template import BasePowerSpectrumParameterization  # to add calculator in the registry


class BasePTPowerSpectrumMultipoles(BaseTheoryPowerSpectrumMultipoles):

    def __init__(self, *args, **kwargs):
        super(BasePTPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        self.kin = np.geomspace(min(1e-3, self.k[0] / 2), max(10., self.k[0] * 2), 200)  # margin for AP effect
        self.requires = {'template': (BasePowerSpectrumParameterization, {'k': self.kin, 'zeff': self.zeff, 'fiducial': self.fiducial})}


class BasePTCorrelationFunctionMultipoles(BaseTheoryCorrelationFunctionMultipoles):

    def __init__(self, s=None, zeff=1., ells=(0, 2, 4), fiducial=None):
        super(BasePTCorrelationFunctionMultipoles, self).__init__(s=s, zeff=zeff, ells=ells, fiducial=fiducial)
        self.kin = np.geomspace(min(1e-3, 1 / self.s[-1] / 2), max(10., 1 / self.s[0] * 2), 200)  # margin for AP effect
        self.requires = {'template': (BasePowerSpectrumParameterization, {'k': self.kin, 'zeff': self.zeff, 'fiducial': self.fiducial})}


class BaseTracerCorrelationFunctionMultipoles(BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles):

    def __init__(self, s=None, zeff=1., ells=(0, 2, 4), fiducial=None):
        super(BaseTracerCorrelationFunctionMultipoles, self).__init__(s=s, zeff=zeff, ells=ells, fiducial=fiducial)
        self.requires['power']['class'] = self.__class__.__name__.replace('CorrelationFunction', 'PowerSpectrum')


class KaiserTracerPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles, TrapzTheoryPowerSpectrumMultipoles):

    def __init__(self, *args, mu=200, **kwargs):
        super(KaiserTracerPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        self.set_k_mu(k=self.k, mu=mu, ells=self.ells)

    def run(self, b1=1., sn0=0.):
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        f = self.template.f
        pkmu = (b1 + f * muap**2)**2 * np.interp(np.log10(kap), np.log10(self.kin), self.template.power_dd) + sn0
        self.power = self.to_poles(pkmu)


class KaiserTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionMultipoles):

    pass


class BaseVelocileptorsPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles):

    _default_options = dict()

    def __init__(self, *args, **kwargs):
        self.pt_options = self._default_options.copy()
        for name, value in self._default_options.items():
            self.pt_options[name] = kwargs.pop(name, value)
        self.pt_options['threads'] = self.pt_options.pop('nthreads', 1)
        super(BaseVelocileptorsPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        if 'kmin' in self._default_options:
            self._default_options['kmin'] = self.k[0] * 0.8
        if 'kmax' in self._default_options:
            self._default_options['kmax'] = self.k[-1] * 1.2
        if 'nk' in self._default_options:
            self._default_options['nk'] = int(len(self.k) * 1.4 + 0.5)
        self.bias_options = {}

    def combine_bias_terms_poles(self, pars, **opts):
        tmp = np.array(self.pt.compute_redshift_space_power_multipoles(pars, self.template.f, apar=self.template.qpar, aperp=self.template.qper, **self.bias_options, **opts)[1:])
        return interpolate.interp1d(self.pt.kv, tmp, kind='cubic', axis=-1, copy=False, bounds_error=True, assume_sorted=True)(self.k)


class BaseVelocileptorsTracerPowerSpectrumMultipoles(BaseTheoryPowerSpectrumMultipoles):

    _default_options = dict()

    def __init__(self, *args, **kwargs):
        super(BaseVelocileptorsTracerPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        self.requires = {'pt': (self.__class__.__name__.replace('Tracer', ''), {'k': self.k, 'zeff': self.zeff, 'ells': self.ells, 'fiducial': self.fiducial})}
        self.bias_options = self._default_options.copy()
        for name, value in self._default_options.items():
            self.bias_options[name] = kwargs.pop(name, value)
        self.required_bias_params, self.optional_bias_params = {}, {}

    def set_params(self, params):
        return params.select(basename=list(self.required_bias_params.keys()) + list(self.optional_bias_params.keys()))

    def run(self, **params):
        pars = [params.get(name, value) for name, value in self.required_bias_params.items()]
        opts = {name: params.get(name, default) for name, default in self.optional_bias_params.items()}
        self.power = self.pt.combine_bias_terms_poles(pars, **opts, **self.bias_options)


class BaseVelocileptorsCorrelationFunctionMultipoles(BasePTCorrelationFunctionMultipoles):

    _default_options = dict()

    def __init__(self, *args, **kwargs):
        self.pt_options = self._default_options.copy()
        for name, value in self._default_options.items():
            self.pt_options[name] = kwargs.pop(name, value)
        self.pt_options['threads'] = self.pt_options.pop('nthreads')
        super(BaseVelocileptorsCorrelationFunctionMultipoles, self).__init__(*args, **kwargs)

    def combine_bias_terms_poles(self, pars, **opts):
        return np.array([self.pt.compute_xi_ell(ss, self.template.f, *pars, apar=self.template.qpar, aperp=self.template.qper, **self.bias_options, **opts) for ss in self.s]).T


class BaseVelocileptorsTracerCorrelationFunctionMultipoles(BaseTheoryCorrelationFunctionMultipoles):

    _default_options = dict()

    def __init__(self, *args, **kwargs):
        super(BaseVelocileptorsTracerCorrelationFunctionMultipoles, self).__init__(*args, **kwargs)
        self.requires = {'pt': (self.__class__.__name__.replace('Tracer', ''), {'k': self.k, 'zeff': self.zeff, 'ells': self.ells, 'fiducial': self.fiducial})}
        self.bias_options = self._default_options.copy()
        for name, value in self._default_options.items():
            self.bias_options[name] = kwargs.pop(name, value)

    def set_params(self, params):
        return params.select(basename=list(self.required_bias_params.keys()) + list(self.optional_bias_params.keys()))

    def run(self, **params):
        pars = [params.get(name, value) for name, value in self.required_bias_params.items()]
        opts = {name: params.get(name, default) for name, default in self.optional_bias_params.items()}
        self.corr = self.pt.combine_bias_terms_poles(pars, **opts, **self.bias_options)


class LPTPowerSpectrumMultipoles(BaseVelocileptorsPowerSpectrumMultipoles):

    _default_options = dict(kIR=0.2, cutoff=10, extrap_min=-5, extrap_max=3, N=2000, nthreads=1, jn=5)
    _bias_indices = dict(zip(['alpha0', 'alpha2', 'alpha4', 'alpha6', 'sn0', 'sn2', 'sn4'], range(12, 19)))
    # Slow, ~ 2 sec per iteration

    def run(self):
        from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
        self.lpt = LPT_RSD(self.kin, self.template.power_dd, **self.pt_options)
        self.lpt.make_pltable(self.template.f, kv=self.k, apar=self.template.qpar, aperp=self.template.qper, ngauss=3)
        lpttable = {0: self.lpt.p0ktable, 2: self.lpt.p2ktable, 4: self.lpt.p4ktable}
        self.lpttable = np.array([lpttable[ell] for ell in self.ells])

    def combine_bias_terms_poles(self, pars):
        #bias = [b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn0, sn2, sn4]
        # pkells = self.lpt.combine_bias_terms_pkell(bias)[1:]
        # return np.array([pkells[[0, 2, 4].index(ell)] for ell in self.ells])
        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn0, sn2, sn4 = pars
        bias_monomials = np.array([1, b1, b1**2, b2, b1 * b2, b2**2, bs, b1 * bs, b2 * bs, bs**2, b3, b1 * b3, alpha0, alpha2, alpha4, alpha6, sn0, sn2, sn4])
        return np.sum(self.lpttable * bias_monomials, axis=-1)

    def gradient_bias_terms_poles(self, name):
        return self.lpttable[..., self._bias_indices[name]]

    def __getstate__(self):
        state = {}
        for name in ['k', 'zeff', 'ells', 'lpttable']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


class LPTTracerPowerSpectrumMultipoles(BaseVelocileptorsTracerPowerSpectrumMultipoles):

    def __init__(self, *args, **kwargs):
        super(LPTTracerPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        self.required_bias_params = dict(b1=0.69, b2=-1.17, bs=-0.71, b3=0., alpha0=0., alpha2=0., alpha4=0., alpha6=0., sn0=0., sn2=0., sn4=0.)

    def run(self, **params):
        super(LPTTracerPowerSpectrumMultipoles, self).run(**params)
        for param in self.runtime_info.solved_params:
            self.runtime_info.gradient[param] = self.pt.gradient_bias_terms_poles(param.basename)


class LPTTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionMultipoles):

    pass


class EPTMomentsPowerSpectrumMultipoles(BaseVelocileptorsPowerSpectrumMultipoles):

    _default_options = dict(rbao=110, kmin=1e-2, kmax=0.5, nk=100, beyond_gauss=True,
                            one_loop=True, shear=True, third_order=True, cutoff=20, jn=5, N=4000,
                            nthreads=1, extrap_min=-5, extrap_max=3, import_wisdom=False)

    def __init__(self, *args, **kwargs):
        super(EPTMomentsPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        self.requires['template']['with_now'] = True

    def run(self):
        from velocileptors.EPT.moment_expansion_fftw import MomentExpansion
        self.pt = MomentExpansion(self.klin, self.template.power_dd, pnw=self.template.power_dd_now, **self.pt_options)


class EPTMomentsTracerPowerSpectrumMultipoles(BaseVelocileptorsTracerPowerSpectrumMultipoles):

    _default_options = dict(beyond_gauss=True, reduced=True)

    def __init__(self, *args, **kwargs):
        super(EPTMomentsTracerPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        for name in ['beyond_gauss']:
            self.requires['pt']['init'][name] = self.bias_options[name]
        if self.bias_options['beyond_gauss']:
            if self.bias_options['reduced']:
                self.required_bias_params = ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'alpha6', 'sn0', 'sn2', 'sn4']
            else:
                self.required_bias_params = ['b1', 'b2', 'bs', 'b3', 'alpha', 'alpha_v', 'alpha_s0', 'alpha_s2', 'alpha_g1',\
                                             'alpha_g3', 'alpha_k2', 'sn0', 'sv', 'sigma0', 'stoch_k0']
        else:
            if self.bias_options['reduced']:
                self.required_bias_params = ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'sn0', 'sn2']
            else:
                self.required_bias_params = ['b1', 'b2', 'bs', 'b3', 'alpha', 'alpha_v', 'alpha_s0', 'alpha_s2', 'sn0', 'sv', 'sigma0']

        default_values = {'b1': 1.69, 'b2': -1.17, 'bs': -0.71, 'b3': -0.479, 'counterterm_c3': 0.}
        self.required_bias_params = {name: default_values.get(name, 0.) for name in self.required_bias_params}
        self.optional_bias_params = {name: default_values.get(name, 0.) for name in self.optional_bias_params}


class EPTFullResummedPowerSpectrumMultipoles(BaseVelocileptorsPowerSpectrumMultipoles):

    _default_options = dict(rbao=110, kmin=1e-2, kmax=0.5, nk=100, sbao=None, beyond_gauss=True,
                            one_loop=True, shear=True, cutoff=20, jn=5, N=4000, nthreads=1,
                            extrap_min=-5, extrap_max=3, import_wisdom=False)

    def __init__(self, *args, **kwargs):
        super(EPTFullPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        self.requires['template']['with_now'] = True

    def run(self):
        from velocileptors.EPT.ept_fullresum_fftw import REPT
        self.pt = REPT(self.klin, self.template.power_dd, pnw=self.template.power_dd_now, **self.pt_options)


class EPTFullResummedTracerPowerSpectrumMultipoles(BaseVelocileptorsTracerPowerSpectrumMultipoles):

    def __init__(self, *args, **kwargs):
        super(EPTFullTracerPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        self.required_bias_params = ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'alpha6', 'sn0', 'sn2', 'sn4']
        self.optional_bias_params = ['bFoG']
        default_values = {'b1': 1.69, 'b2': -1.17, 'bs': -0.71, 'b3': -0.479}
        self.required_bias_params = {name: default_values.get(name, 0.) for name in self.required_bias_params}
        self.optional_bias_params = {name: default_values.get(name, 0.) for name in self.optional_bias_params}


class LPTMomentsPowerSpectrumMultipoles(BaseVelocileptorsPowerSpectrumMultipoles):

    _default_options = dict(kmin=5e-3, kmax=0.3, nk=50, beyond_gauss=False, one_loop=True,
                            shear=True, third_order=True, cutoff=10, jn=5, N=2000, nthreads=1,
                            extrap_min=-5, extrap_max=3, import_wisdom=False)

    def run(self):
        from velocileptors.LPT.moment_expansion_fftw import MomentExpansion
        self.pt = MomentExpansion(self.klin, self.template.power_dd, **self.pt_options)


class LPTMomentsTracerPowerSpectrumMultipoles(BaseVelocileptorsTracerPowerSpectrumMultipoles):

    _default_options = dict(beyond_gauss=False, shear=True, third_order=True, reduced=True, ngauss=4)

    def __init__(self, *args, **kwargs):
        super(LPTMomentsTracerPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        self.pt_options = {}
        for name in ['beyond_gauss', 'shear', 'third_order']:
            self.requires['pt']['init'][name] = self.bias_options[name]
            self.pt_options[name] = self.bias_options.pop(name)
        if self.bias_options['beyond_gauss']:
            if self.bias_options['reduced']:
                self.required_bias_params = ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'alpha6', 'sn0', 'sn2', 'sn4']
            else:
                self.required_bias_params = ['b1', 'b2', 'bs', 'b3', 'alpha', 'alpha_v', 'alpha_s0', 'alpha_s2', 'alpha_g1',\
                                             'alpha_g3', 'alpha_k2', 'sn0', 'sv', 'sigma0_stoch', 'sn4']
        else:
            if self.pt_options['reduced']:
                self.required_bias_params = ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'sn0', 'sn2']
            else:
                self.required_bias_params = ['b1', 'b2', 'bs', 'b3', 'alpha', 'alpha_v', 'alpha_s0', 'alpha_s2',
                                             'sn0', 'sv', 'sigma0_stoch']

        self.optional_bias_params = ['counterterm_c3']
        default_values = {'b1': 1.69, 'b2': -1.17, 'bs': -0.71, 'b3': -0.479}
        self.required_bias_params = {name: default_values.get(name, 0.) for name in self.required_bias_params}
        self.optional_bias_params = {name: default_values.get(name, 0.) for name in self.optional_bias_params}

    def set_params(self, params):
        params = super(LPTMomentsTracerPowerSpectrumMultipoles, self).set_params(params)
        if not self.pt_options['shear']:
            del params['bs']
        if not self.pt_options['third_order']:
            del params['b3']
        return params


class LPTFourierStreamingPowerSpectrumMultipoles(BaseVelocileptorsPowerSpectrumMultipoles):

    _default_options = dict(kmin=1e-3, kmax=3, nk=100, beyond_gauss=False, one_loop=True,
                            shear=True, third_order=True, cutoff=10, jn=5, N=2000, nthreads=None,
                            extrap_min=-5, extrap_max=3, import_wisdom=False)

    def run(self):
        from velocileptors.LPT.fourier_streaming_model_fftw import FourierStreamingModel
        self.pt = FourierStreamingModel(self.klin, self.template.power_dd, **self.pt_options)


class LPTFourierStreamingTracerPowerSpectrumMultipoles(BaseVelocileptorsTracerPowerSpectrumMultipoles):

    _default_options = dict(shear=True, third_order=True)

    def __init__(self, *args, **kwargs):
        super(LPTFourierStreamingTracerPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        self.pt_options = {}
        for name in ['shear', 'third_order']:
            self.requires['pt']['init'][name] = self.bias_options[name]
            self.pt_options[name] = self.bias_options.pop(name)
        # b3 if in third order, bs if shear bias
        self.required_bias_params = ['b1', 'b2', 'bs', 'b3', 'alpha', 'alpha_v', 'alpha_s0', 'alpha_s2', 'sn0', 'sv', 'sigma0_stoch']
        self.optional_bias_params = ['counterterm_c3']
        default_values = {'b1': 1.69, 'b2': -1.17, 'bs': -0.71, 'b3': -0.479}
        self.required_bias_params = {name: default_values.get(name, 0.) for name in self.required_bias_params}
        self.optional_bias_params = {name: default_values.get(name, 0.) for name in self.optional_bias_params}

    def set_params(self, params):
        params = super(LPTFourierStreamingTracerPowerSpectrumMultipoles, self).set_params(params)
        if not self.pt_options['shear']:
            del params['bs']
        if not self.pt_options['third_order']:
            del params['b3']
        return params


class LPTGaussianStreamingCorrelationFunctionMultipoles(BaseVelocileptorsCorrelationFunctionMultipoles):

    _default_options = dict(kmin=1e-3, kmax=3, nk=200, jn=10, cutoff=20, beyond_gauss=False, one_loop=True,
                            shear=True, N=2000, nthreads=None, extrap_min=-5, extrap_max=3, import_wisdom=False)

    def run(self):
        from velocileptors.LPT.gaussian_streaming_model_fftw import GaussianStreamingModel
        self.pt = GaussianStreamingModel(self.klin, self.template.power_dd, **self.pt_options)


class LPTGaussianStreamingTracerCorrelationFunctionMultipoles(BaseVelocileptorsTracerCorrelationFunctionMultipoles):

    _default_options = dict(shear=True, third_order=True, rwidth=100, Nint=10000, ngauss=4, update_cumulants=False)

    def __init__(self, *args, **kwargs):
        super(LPTGaussianStreamingTracerCorrelationFunctionMultipoles, self).__init__(*args, **kwargs)
        self.pt_options = {}
        for name in ['shear', 'third_order']:
            self.requires['pt']['init'][name] = self.bias_options[name]
            self.pt_options[name] = self.bias_options.pop(name)
        # alpha_s0 and alpha_s2 to be zeros
        self.required_bias_params = ['b1', 'b2', 'bs', 'b3', 'alpha', 'alpha_v', 'alpha_s0', 'alpha_s2', 's2FoG']
        self.optional_bias_params = dict()
        default_values = {'b1': 1.69, 'b2': -1.17, 'bs': -0.71, 'b3': -0.479}
        self.required_bias_params = {name: default_values.get(name, 0.) for name in self.required_bias_params}
        self.optional_bias_params = {name: default_values.get(name, 0.) for name in self.optional_bias_params}

    def set_params(self, params):
        params = super(LPTGaussianStreamingTracerCorrelationFunctionMultipoles, self).set_params(params)
        if not self.pt_options['shear']:
            del params['bs']
        if not self.pt_options['third_order']:
            del params['b3']
        return params

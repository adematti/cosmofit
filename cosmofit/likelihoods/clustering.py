import numpy as np
from scipy import constants

from .base import BaseGaussianLikelihood, BaseCalculator
from cosmofit.samples import Chain, load_samples
from cosmofit.parameter import ParameterCollection
from cosmofit.theories.base import EffectAP
from cosmofit.theories.power_template import BAOExtractor, ShapeFitPowerSpectrumExtractor, WiggleSplitPowerSpectrumExtractor, BandVelocityPowerSpectrumExtractor
from cosmofit import utils


class BaseModel(BaseCalculator):

    def __init__(self, quantities, **kwargs):
        self.quantities = list(quantities)
        self.__dict__.update(kwargs)
        self.requires = {'cosmo': ('BasePrimordialCosmology', {})}

    def run(self):
        quantities, theory = [], []
        for quantity in self.quantities:
            value = getattr(self.cosmo, quantity, None)
            if value is not None:
                quantities.append(quantity)
        self.quantities = quantities
        self.theory = np.array(theory, dtype='f8')

    def __getstate__(self):
        state = {}
        for name in ['quantities', 'theory']:
            state[name] = getattr(self, name)
        return state


class BaseParameterizationLikelihood(BaseGaussianLikelihood):

    _parambasenames = None

    def __init__(self, chains=None, select=None, burnin=None):
        self.chain = None
        if self.mpicomm.rank == 0:
            self.chain = Chain.concatenate(load_samples(source='chain', fn=chains, burnin=burnin))
        self.params = self.mpicomm.bcast(self.chain.params() if self.mpicomm.rank == 0 else None, root=0)
        if select is not None:
            self.params = self.params.select(**select)
        if self._parambasenames is not None:
            self.params = self.params.select(basename=self._parambasenames)
            for param in list(self.params):
                if param.fixed and not param.derived:
                    # if self.mpicomm.rank == 0:
                    #     self.log_info('Parameter {} is found to be fixed, ignoring.'.format(param))
                    del self.params[param]
            self.params.sort([self._parambasenames.index(param.basename) for param in self.params if param.basename in self._parambasenames])
        self.requires = {'theory': {'class': self.__class__.__name__.replace('ParameterizationLikelihood', 'Model'),
                                    'init': {'quantities': self.params.basenames()}}}

    def _prepare(self):
        params = {param.basename: param for param in self.params}
        params = [params[quantity] for quantity in self.theory.quantities]
        if self.mpicomm.rank == 0:
            self.log_info('Fitting input samples {}.'.format(params))
        mean = self.mpicomm.bcast(np.concatenate([self.chain.mean(param).ravel() for param in params]) if self.mpicomm.rank == 0 else None, root=0)
        covariance = self.mpicomm.bcast(self.chain.cov(params) if self.mpicomm.rank == 0 else None, root=0)
        super(BaseParameterizationLikelihood, self).__init__(covariance=covariance, data=mean)
        del self.chain

    def _set_meta(self, **kwargs):
        for name, value in kwargs.items():
            if value is None:
                param = self.chain.params(basename=[name])
                if not param:
                    raise ValueError('{} must be provided either as arguments or input samples'.format(name))
                value = self.chain.mean(param[0])
            elif not isinstance(value, str):
                value = np.array(value, dtype='f8')
            setattr(self, name, value)
            self.requires['theory']['init'][name] = value

    def run(self):
        if not hasattr(self, 'precision'):
            self._prepare()
        super(BaseParameterizationLikelihood, self).run()

    @property
    def flatmodel(self):
        return self.theory.theory


class FullModel(BaseModel):

    pass

class FullParameterizationLikelihood(BaseParameterizationLikelihood):

    pass


class BAOModel(BaseModel):

    def __init__(self, *args, **kwargs):
        super(BAOModel, self).__init__(*args, **kwargs)
        BAOExtractor.__init__(self, zeff=self.zeff)

    def run(self):
        BAOExtractor.run(self)
        self.theory = np.array([getattr(self, quantity) for quantity in self.quantities], dtype='f8')


class BAOParameterizationLikelihood(BaseParameterizationLikelihood):

    @property
    def _parambasenames(self):
        params = self.params.basenames()
        options = [['DM_over_rd', 'DH_over_rd'], ['DV_over_rd', 'DH_over_DM'], ['DV_over_rd'], ['DH_over_DM']]
        for ps in options:
            if all(p in params for p in ps):
                return ps
        raise ValueError('No BAO measurements found (searching for {})'.format(options))

    def __init__(self, *args, zeff=None, **kwargs):
        super(BAOParameterizationLikelihood, self).__init__(*args, **kwargs)
        self._set_meta(zeff=zeff)


class ShapeFitModel(BAOModel):

    def __init__(self, *args, **kwargs):
        super(ShapeFitModel, self).__init__(*args, **kwargs)
        ShapeFitPowerSpectrumExtractor.__init__(self, zeff=self.zeff)

    def run(self):
        BAOExtractor.run(self)
        self.n_varied = 'n' in self.quantities
        self.kp = self.kp_rs / self.cosmo.rs_drag
        ShapeFitPowerSpectrumExtractor.run(self)
        self.theory = []
        for quantity in self.quantities:
            if quantity == 'f_sqrt_Ap':
                fo = self.cosmo.get_fourier()
                f = fo.sigma8_z(z=self.zeff, of='theta_cb') / fo.sigma8_z(z=self.zeff, of='delta_cb')
                self.theory.append(self.Ap**0.5 * f)
            else:
                self.theory.append(getattr(self, quantity))
        self.theory = np.array(self.theory, dtype='f8')


class ShapeFitParameterizationLikelihood(BAOParameterizationLikelihood):

    @property
    def _parambasenames(self):
        nm = ['n', 'm']
        for param in self.params.select(basename=nm):
            if np.allclose(self.chain[param], np.mean(self.chain[param]), equal_nan=True):
                try:
                    del nm[nm.index(param.basename)]
                except IndexError:
                    pass
        return super(ShapeFitParameterizationLikelihood, self)._parambasenames + nm + ['f_sqrt_Ap']

    def __init__(self, *args, kp_rs=None, **kwargs):
        super(ShapeFitParameterizationLikelihood, self).__init__(*args, **kwargs)
        self._set_meta(kp_rs=kp_rs)


class WiggleSplitModel(BaseModel):

    def __init__(self, *args, **kwargs):
        super(WiggleSplitModel, self).__init__(*args, **kwargs)
        self.kp_fid = self.kp
        WiggleSplitPowerSpectrumExtractor.__init__(self, zeff=self.zeff, kp=self.kp_fid)
        requires = self.requires
        EffectAP.__init__(self, zeff=self.zeff, fiducial=self.fiducial, mode='distances', eta=self.eta)
        self.requires = {**requires, **self.requires}

    def run(self):
        EffectAP.run(self)
        self.theory = []
        for quantity in self.quantities:
            if quantity == 'fsigmar':
                fo = self.cosmo.get_fourier()
                r = self.r * self.qiso
                fsigmar = fo.sigma_rz(r, z=self.zeff, of='theta_cb')
                self.theory.append(fsigmar)
            elif quantity == 'm':
                self.kp = self.kp_fid / self.qiso
                WiggleSplitPowerSpectrumExtractor.run(self)
                self.theory.append(self.m)
            elif quantity == 'qbao':
                self.theory.append(self.qiso * self.fiducial.rs_drag / self.cosmo.rs_drag)
            else:  # qap
                self.theory.append(getattr(self, 'qap'))
        self.theory = np.array(self.theory, dtype='f8')


class WiggleSplitParameterizationLikelihood(BaseParameterizationLikelihood):

    _parambasenames = ('fsigmar', 'm', 'qbao', 'qap')

    def __init__(self, *args, r=None, zeff=None, kp=None, eta=1./3., fiducial=None, **kwargs):
        super(WiggleSplitParameterizationLikelihood, self).__init__(*args, **kwargs)
        self._set_meta(r=r, zeff=zeff, kp=kp, eta=eta, fiducial=fiducial)


class BandVelocityPowerSpectrumModel(BaseModel):

    def __init__(self, *args, **kwargs):
        super(BandVelocityPowerSpectrumModel, self).__init__(*args, **kwargs)
        self.kptt_fid = self.kptt
        BandVelocityPowerSpectrumExtractor.__init__(self, zeff=self.zeff, kptt=self.kptt_fid)
        EffectAP.__init__(self, zeff=self.zeff, fiducial=self.fiducial, mode='distances', eta=self.eta)

    def run(self):
        EffectAP.run(self)
        self.kptt = self.kptt_fid / self.qiso
        BandVelocityPowerSpectrumExtractor.run(self)
        self.theory = []
        if 'f' in self.quantities:
            fo = self.cosmo.get_fourier()
            r = self.r * self.qiso
            f = fo.sigma_rz(r, self.zeff, of='theta_cb') / fo.sigma_rz(r, self.zeff, of='delta_cb')
            self.theory.append(f)
        if 'qap' in self.quantities:
            self.theory.append(self.qap)
        self.theory = np.concatenate([self.ptt / self.qiso**3, self.theory], axis=0)


class BandVelocityPowerSpectrumParameterizationLikelihood(BaseParameterizationLikelihood):

    _parambasenames = ('ptt', 'f', 'qap')

    def __init__(self, *args, zeff=None, kptt=None, eta=1./3., r=8., fiducial=None, **kwargs):
        super(BandVelocityPowerSpectrumParameterizationLikelihood, self).__init__(*args, **kwargs)
        self._set_meta(zeff=zeff, kptt=kptt, eta=eta, r=r, fiducial=fiducial)

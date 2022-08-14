import glob

import numpy as np
from scipy import constants

from .base import BaseGaussianLikelihood, BaseCalculator
from cosmofit.samples import Chain
from cosmofit.parameter import ParameterCollection
from cosmofit import utils


def load_chain(chains, burnin=None):
    if not utils.is_sequence(chains):
        chains = [chains]
    if isinstance(chains[0], str):
        chains = [Chain.load(fn) for ff in chains for fn in glob.glob(ff)]
    if burnin is not None:
        chains = [chain.remove_burnin(burnin) for chain in chains]
    return Chain.concatenate(chains)


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
            self.chain = load_chain(chains, burnin=burnin)
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
            else:
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
        from cosmofit.theories.power_template import BAOExtractor
        self.requires = {'bao': (BAOExtractor, {'zeff': self.zeff})}

    def run(self):
        self.theory = np.array([getattr(self.bao, quantity) for param in self.quantities], dtype='f8')


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
        from cosmofit.theories.power_template import ShapeFitPowerSpectrumExtractor
        self.requires['shapefit'] = (ShapeFitPowerSpectrumExtractor, {'zeff': self.zeff, 'kpivot': self.kpivot, 'of': 'theta_cb'})

    def run(self):
        self.shapefit.n_varied = 'n' in self.quantities
        self.theory = [getattr(self.shapefit, quantity) for quantity in ['n', 'm'] if quantity in self.quantities]
        if 'f_sqrt_A_p' in self.quantities:
            self.theory.append(self.shapefit.A_p**0.5)  # norm of velocity divergence power spectrum
        self.theory += [getattr(self.bao, quantity) for quantity in self.quantities[len(self.theory):]]
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
        return nm + ['f_sqrt_A_p'] + super(ShapeFitParameterizationLikelihood, self)._parambasenames

    def __init__(self, *args, kpivot=0.03, **kwargs):
        super(ShapeFitParameterizationLikelihood, self).__init__(*args, **kwargs)
        self._set_meta(kpivot=kpivot)


class WiggleSplitModel(BaseModel):

    def __init__(self, *args, **kwargs):
        super(WiggleSplitModel, self).__init__(*args, **kwargs)
        from cosmofit.theories.base import EffectAP
        self.requires['effectap'] = (EffectAP, {'zeff': self.zeff, 'fiducial': self.fiducial, 'mode': 'distances'})

    def run(self):
        qiso = self.effectap.qiso
        self.theory = []
        if 'fsigmar' in self.quantities:
            fo = self.cosmo.get_fourier()
            r = self.r * qiso
            fsigmar = fo.sigma_rz(r, self.zeff, of='theta_cb')
            self.theory.append(fsigmar)
        if 'qbao' in self.quantities:
            self.theory.append(qiso * self.cosmo.rs_drag / self.effectap.fiducial.rs_drag)
        if 'qap' in self.quantities:
            self.theory.append(self.effectap.qap)
        self.theory = np.array(self.theory, dtype='f8')


class WiggleSplitPowerSpectrumParameterizationLikelihood(BaseParameterizationLikelihood):

    _parambasenames = ('fsigmar', 'qbao', 'qap')

    def __init__(self, *args, r=None, zeff=None, fiducial=None, **kwargs):
        super(WiggleSplitPowerSpectrumParameterizationLikelihood, self).__init__(*args, **kwargs)
        self._set_meta(r=r, zeff=zeff, fiducial=fiducial)


class BandVelocityPowerSpectrumModel(BaseModel):

    def __init__(self, *args, **kwargs):
        super(BandVelocityPowerSpectrumModel, self).__init__(*args, **kwargs)
        from cosmofit.theories.base import EffectAP
        from cosmofit.theories.power_template import BandVelocityPowerSpectrumExtractor
        self.requires['bandpower'] = (BandVelocityPowerSpectrumExtractor, {'zeff': self.zeff, 'kptt': self.kptt})
        self.requires['effectap'] = (EffectAP, {'zeff': self.zeff, 'fiducial': self.fiducial, 'mode': 'distances'})

    def run(self):
        qiso = self.effectap.qiso
        # Anything that will need a new run effectap will also need a new run of bandpower, so it is safe
        self.runtime_info.requires['bandpower'].kptt = self.kptt / qiso
        self.theory = []
        if 'f' in self.quantities:
            fo = self.cosmo.get_fourier()
            r = self.r * qiso
            f = fo.sigma_rz(r, self.zeff, of='theta_cb') / fo.sigma_rz(r, self.zeff, of='delta_cb')
            self.theory.append(f)
        if 'qap' in self.quantities:
            self.theory.append(self.effectap.qap)
        self.theory = np.concatenate([self.bandpower.ptt / qiso**3, self.theory], axis=0)


class BandVelocityPowerSpectrumParameterizationLikelihood(BaseParameterizationLikelihood):

    _parambasenames = ('ptt', 'f', 'qap')

    def __init__(self, *args, kptt=None, zeff=None, fiducial=None, **kwargs):
        super(BandVelocityPowerSpectrumParameterizationLikelihood, self).__init__(*args, **kwargs)
        self._set_meta(kptt=kptt, zeff=zeff, fiducial=fiducial)

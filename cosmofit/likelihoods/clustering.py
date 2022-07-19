import glob

import numpy as np
from scipy import constants

from .base import BaseGaussianLikelihood
from cosmofit.samples import Chain
from cosmofit.parameter import ParameterCollection
from cosmofit import utils


def load_chain(chains, burnin=None):
    if isinstance(chains, str):
        chains = [chains]
    if utils.is_sequence(chains) and isinstance(chains[0], str):
        chains = [Chain.load(fn) for ff in chains for fn in glob.glob(ff)]
    if burnin is not None:
        chains = [chain.remove_burnin(burnin) for chain in chains]
    return Chain.concatenate(chains)


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
        self.requires = {'cosmo': ('BasePrimordialCosmology', {})}

    def _prepare(self):
        mean = self.mpicomm.bcast(np.array([self.chain.mean(param) for param in self.params]) if self.mpicomm.rank == 0 else None, root=0)
        covariance = self.mpicomm.bcast(self.chain.cov(self.params) if self.mpicomm.rank == 0 else None, root=0)
        super(BaseParameterizationLikelihood, self).__init__(covariance=covariance, data=mean)
        del self.chain

    def _set_meta(self, **kwargs):
        for name, value in kwargs.items():
            if value is None:
                param = self.chain.params(basename=[name])
                if not param:
                    raise ValueError('{} must be provided either as arguments or input samples'.format(name))
                value = self.chain.mean(param[0])
            value = float(value)
            setattr(self, name, value)

    def run(self):
        if not hasattr(self, 'precision'):
            self._prepare()
        super(BaseParameterizationLikelihood, self).run()


class FullParameterizationLikelihood(BaseParameterizationLikelihood):

    def _prepare(self):
        self.params = ParameterCollection()
        for param in params:
            value = getattr(self.cosmo, param.basename, None)
            if value is not None:
                self.params.set(param)
        super(FullParameterizationLikelihood, self)._prepare()

    def flatmodel(self):
        return np.array([getattr(self.cosmo, param.basename) for param in self.params], dtype='f8')


class BAOParameterizationLikelihood(BaseParameterizationLikelihood):

    _parambasenames = ('DH_over_rd', 'DM_over_rd')

    def __init__(self, *args, zeff=None, **kwargs):
        super(BAOParameterizationLikelihood, self).__init__(*args, **kwargs)
        self._set_meta(zeff=zeff)
        from cosmofit.theories.power_template import BAOExtractor
        self.requires = {'bao': (BAOExtractor, {'zeff': self.zeff})}

    def flatmodel(self):
        return np.array([getattr(self.bao, param.basename) for param in self.params], dtype='f8')


class ShapeFitParameterizationLikelihood(BAOParameterizationLikelihood):

    _parambasenames = ('n', 'm', 'f_sqrt_A_p') + BAOParameterizationLikelihood._parambasenames

    def __init__(self, *args, kpivot=0.03, **kwargs):
        super(ShapeFitParameterizationLikelihood, self).__init__(*args, **kwargs)
        self._set_meta(kpivot=kpivot)
        from cosmofit.theories.power_template import ShapeFitPowerSpectrumExtractor
        self.requires['shapefit'] = (ShapeFitPowerSpectrumExtractor, {'zeff': self.zeff, 'kpivot': self.kpivot, 'of': 'theta_cb', 'n_varied': self.params[0].varied})

    def flatmodel(self):
        values = [getattr(self.shapefit, name) for name in ['n', 'm', 'A_p']]
        values[-1] **= 0.5
        return np.concatenate([values, super(ShapeFitPowerSpectrumParameterization, self).flatmodel], axis=0)


class BandVelocityPowerSpectrumParameterizationLikelihood(BaseParameterizationLikelihood):

    _parambasenames = ('ptt', 'f', 'qap')

    def __init__(self, *args, kptt=None, fiducial=None, **kwargs):
        super(ShapeFitParameterizationLikelihood, self).__init__(*args, **kwargs)
        self._set_meta(kptt=kptt)
        from cosmofit.theories.power_template import BandVelocityPowerSpectrumExtractor
        self.requires['bandpower'] = (BandVelocityPowerSpectrumExtractor, {'zeff': self.zeff, 'kptt': self.kptt})
        self.requires['effectap'] = (EffectAP, {'zeff': self.zeff, 'fiducial': fiducial, 'mode': 'distances'})

    def flatmodel(self):
        qiso = self.effectap.qiso
        self.bandpower.kptt = self.kptt / qiso
        ptt = self.bandpower.ptt / qiso**3
        fo = self.cosmo.get_fourier()
        r = 8. * qiso
        f = fo.sigma_rz(r, self.zeff, of='theta_cb') / fo.sigma_rz(r, self.zeff, of='delta_cb')
        return np.concatenate([ptt, [f, self.effectap.qap]], axis=0)

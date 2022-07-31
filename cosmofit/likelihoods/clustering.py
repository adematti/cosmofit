import glob

import numpy as np
from scipy import constants

from .base import BaseGaussianLikelihood
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


class BaseParameterizationLikelihood(BaseGaussianLikelihood):

    _parambasenames = None

    def __init__(self, chains=None, select=None, burnin=None):
        self.chain = None
        if self.mpicomm.rank == 0:
            self.chain = load_chain(chains, burnin=burnin)
        self.data_params = self.mpicomm.bcast(self.chain.params() if self.mpicomm.rank == 0 else None, root=0)
        if select is not None:
            self.data_params = self.data_params.select(**select)
        if self._parambasenames is not None:
            self.data_params = self.data_params.select(basename=self._parambasenames)
            for param in list(self.data_params):
                if param.fixed and not param.derived:
                    # if self.mpicomm.rank == 0:
                    #     self.log_info('Parameter {} is found to be fixed, ignoring.'.format(param))
                    del self.data_params[param]
            self.data_params.sort([self._parambasenames.index(param.basename) for param in self.data_params if param.basename in self._parambasenames])
        self.requires = {'cosmo': ('BasePrimordialCosmology', {})}

    def _prepare(self):
        if self.mpicomm.rank == 0:
            self.log_info('Fitting input samples {}.'.format(list(self.data_params)))
        mean = self.mpicomm.bcast(np.concatenate([self.chain.mean(param).ravel() for param in self.data_params]) if self.mpicomm.rank == 0 else None, root=0)
        covariance = self.mpicomm.bcast(self.chain.cov(self.data_params) if self.mpicomm.rank == 0 else None, root=0)
        super(BaseParameterizationLikelihood, self).__init__(covariance=covariance, data=mean)
        del self.chain
        self.base_params = {param.basename: param for param in self.data_params}

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

    def run(self):
        if not hasattr(self, 'precision'):
            self._prepare()
        super(BaseParameterizationLikelihood, self).run()


class FullParameterizationLikelihood(BaseParameterizationLikelihood):

    def _prepare(self):
        params = ParameterCollection()
        for param in self.data_params:
            value = getattr(self.cosmo, param.basename, None)
            if value is not None:
                params.set(param)
        self.data_params = params
        super(FullParameterizationLikelihood, self)._prepare()

    def flatmodel(self):
        return np.array([getattr(self.cosmo, param.basename) for param in self.data_params], dtype='f8')


class BAOParameterizationLikelihood(BaseParameterizationLikelihood):

    @property
    def _parambasenames(self):
        params = self.data_params.basenames()
        options = [('DM_over_rd', 'DH_over_rd'), ('DV_over_rd', 'DH_over_DM'), ('DV_over_rd',), ('DH_over_DM',)]
        for ps in options:
            if all(p in params for p in ps):
                return ps
        raise ValueError('No BAO measurements found (searching for {})'.format(options))

    def __init__(self, *args, zeff=None, **kwargs):
        super(BAOParameterizationLikelihood, self).__init__(*args, **kwargs)
        self._set_meta(zeff=zeff)
        from cosmofit.theories.power_template import BAOExtractor
        self.requires = {'bao': (BAOExtractor, {'zeff': self.zeff})}

    def flatmodel(self):
        return np.array([getattr(self.bao, param.basename) for param in self.data_params], dtype='f8')


class ShapeFitParameterizationLikelihood(BAOParameterizationLikelihood):

    @property
    def _parambasenames(self):
        return ('n', 'm', 'f_sqrt_A_p') + super(ShapeFitParameterizationLikelihood, self)._parambasenames

    def _prepare(self):
        params = ParameterCollection()
        for param in self.data_params:
            if param.basename in ['n', 'm'] and np.allclose(self.chain[param], np.mean(self.chain[param])):
                continue
            params.set(param)
        self.data_params = params
        super(ShapeFitParameterizationLikelihood, self)._prepare()
        self.shapefit.n_varied = 'n' in self.base_params
        self.shapefit.run()

    def __init__(self, *args, kpivot=0.03, **kwargs):
        super(ShapeFitParameterizationLikelihood, self).__init__(*args, **kwargs)
        self._set_meta(kpivot=kpivot)
        from cosmofit.theories.power_template import ShapeFitPowerSpectrumExtractor
        self.requires['shapefit'] = (ShapeFitPowerSpectrumExtractor, {'zeff': self.zeff, 'kpivot': self.kpivot, 'of': 'theta_cb'})

    def flatmodel(self):
        values = [getattr(self.shapefit, name) for name in ['n', 'm'] if name in self.base_params]
        if 'f_sqrt_A_p' in self.base_params:
            values.append(self.shapefit.A_p**0.5)  # norm of velocity power spectrum
        values += [getattr(self.bao, param.basename) for param in self.data_params[len(values):]]
        return np.array(values, dtype='f8')


class WiggleSplitPowerSpectrumParameterizationLikelihood(BaseParameterizationLikelihood):

    _parambasenames = ('fsigmar', 'qbao', 'qap')

    def __init__(self, *args, r=None, zeff=None, fiducial=None, **kwargs):
        super(WiggleSplitPowerSpectrumParameterizationLikelihood, self).__init__(*args, **kwargs)
        self._set_meta(r=r, zeff=zeff)
        from cosmofit.theories.base import EffectAP
        self.requires['effectap'] = (EffectAP, {'zeff': self.zeff, 'fiducial': fiducial, 'mode': 'distances'})

    def flatmodel(self):
        qiso = self.effectap.qiso
        fq = []
        if 'fsigmar' in self.base_params:
            fo = self.cosmo.get_fourier()
            r = 8. * qiso
            fsigmar = fo.sigma_rz(r, self.zeff, of='theta_cb')
            fq.append(fsigmar)
        if 'qbao' in self.base_params:
            fq.append(qiso * self.cosmo.rs_drag / self.effectap.fiducial.rs_drag)
        if 'qap' in self.base_params:
            fq.append(self.effectap.qap)
        return np.array(fq, dtype='f8')


class BandVelocityPowerSpectrumParameterizationLikelihood(BaseParameterizationLikelihood):

    _parambasenames = ('ptt', 'f', 'qap')

    def __init__(self, *args, kptt=None, zeff=None, fiducial=None, **kwargs):
        super(BandVelocityPowerSpectrumParameterizationLikelihood, self).__init__(*args, **kwargs)
        self._set_meta(kptt=kptt, zeff=zeff)
        from cosmofit.theories.base import EffectAP
        from cosmofit.theories.power_template import BandVelocityPowerSpectrumExtractor
        self.requires['bandpower'] = (BandVelocityPowerSpectrumExtractor, {'zeff': self.zeff, 'kptt': self.kptt})
        self.requires['effectap'] = (EffectAP, {'zeff': self.zeff, 'fiducial': fiducial, 'mode': 'distances'})

    def flatmodel(self):
        qiso = self.effectap.qiso
        self.bandpower.kptt = self.kptt / qiso
        ptt = self.bandpower.ptt / qiso**3
        fq = []
        if 'f' in self.base_params:
            fo = self.cosmo.get_fourier()
            r = 8. * qiso
            f = fo.sigma_rz(r, self.zeff, of='theta_cb') / fo.sigma_rz(r, self.zeff, of='delta_cb')
            fq.append(f)
        if 'qap' in self.base_params:
            fq.append(self.effectap.qap)
        return np.concatenate([ptt, fq], axis=0)

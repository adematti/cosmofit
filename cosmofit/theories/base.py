import numpy as np
from scipy import special

from cosmofit.base import BaseCalculator
from . import utils


class BaseTheoryPowerSpectrumMultipoles(BaseCalculator):

    def __init__(self, k=None, zeff=1., ells=(0, 2, 4), fiducial=None):
        if k is None: k = np.linspace(0.01, 0.2, 101)
        self.k = np.array(k, dtype='f8')
        self.zeff = float(zeff)
        self.ells = tuple(ells)
        self.fiducial = fiducial

    def __getstate__(self):
        state = {}
        for name in ['k', 'zeff', 'ells', 'power', 'growth_rate']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


class TrapzTheoryPowerSpectrumMultipoles(BaseTheoryPowerSpectrumMultipoles):

    def __init__(self, *args, mu=200, **kwargs):
        super(TrapzTheoryPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        self.set_k_mu(k=self.k, mu=mu, ells=self.ells)

    def set_k_mu(self, k, mu=200, ells=(0, 2, 4)):
        self.k = np.asarray(k, dtype='f8')
        if np.ndim(mu) == 0:
            self.mu = np.linspace(0., 1., mu)
        else:
            self.mu = np.asarray(mu)
        muw = utils.weights_trapz(self.mu)
        self.muweights = np.array([muw * (2 * ell + 1) * special.legendre(ell)(self.mu) for ell in ells]) / (self.mu[-1] - self.mu[0])

    def to_poles(self, pkmu):
        return np.sum(pkmu * self.muweights[:, None, :], axis=-1)


def get_cosmo(cosmo):
    import cosmoprimo
    if isinstance(cosmo, str):
        cosmo = (cosmo, {})
    if isinstance(cosmo, tuple):
        return getattr(cosmoprimo.fiducial, cosmo[0])(**cosmo[1])
    return cosmoprimo.Cosmology(**cosmo)


class EffectAP(BaseCalculator):

    def __init__(self, zeff=1., fiducial=None, mode=None):
        self.zeff = float(zeff)
        if fiducial is None:
            raise ValueError('Provide fiducial cosmology')
        fiducial = get_cosmo(fiducial)
        self.efunc_fid = fiducial.efunc(self.zeff)
        self.comoving_angular_distance_fid = fiducial.comoving_angular_distance(self.zeff)

        self.mode = mode
        if self.mode is None:
            self.mode = 'distances'
            if 'qiso' in self.params:
                self.mode = 'qiso'
            elif 'qpar' in self.params and 'qper' in self.params:
                self.mode = 'qparqper'

        if self.mode == 'qiso':
            self.params = self.params.select(name=['qiso'])
        elif self.mode == 'qparqper':
            self.params = self.params.select(name=['qpar', 'qper'])
        elif self.mode == 'distances':
            self.params = self.params.clear()
        else:
            raise ValueError('mode must be one of ["qiso", "qparqper", "distances"]')

        self.requires = {'cosmo': ('BasePrimordialCosmology', {})}

    def run(self, **params):
        if self.mode == 'distances':
            qpar, qper = self.efunc_fid / self.cosmo.efunc(self.zeff), self.cosmo.comoving_angular_distance(self.zeff) / self.comoving_angular_distance_fid
        elif self.mode == 'qiso':
            qpar = qper = params['qiso']
        else:
            qpar, qper = params['qpar'], params['qper']
        self.qpar, self.qper = qpar, qper

    def ap_k_mu(self, k, mu):
        jac = 1. / (self.qpar * self.qper ** 2)
        F = self.qpar / self.qper
        factor_ap = np.sqrt(1 + mu**2 * (1. / F**2 - 1))
        # Beutler 2016 (arXiv: 1607.03150v1) eq 44
        kap = k[..., None] / self.qper * factor_ap
        # Beutler 2016 (arXiv: 1607.03150v1) eq 45
        muap = mu / F / factor_ap
        return jac, kap, muap


class WindowedPowerSpectrumMultipoles(BaseCalculator):

    def __init__(self, kout=None, ellsout=(0, 2, 4), zeff=None, fiducial=None, wmatrix=None):
        if kout is None: kout = np.linspace(0.01, 0.2, 20)
        if np.ndim(kout[0]) == 0:
            kout = [kout] * len(ellsout)
        self.kout = [np.array(kk, dtype='f8') for kk in kout]
        self.zeff = float(zeff)
        self.ellsout = tuple(ellsout)
        self.wmatrix = wmatrix
        if wmatrix is None:
            self.ellsin = tuple(self.ellsout)
            self.kin = np.unique(np.concatenate(self.kout, axis=0))
            if all(np.allclose(kk, self.kin) for kk in self.kout):
                self.kmask = None
            else:
                self.kmask = [np.searchsorted(self.kin, kk, side='left') for kk in self.kout]
                assert all(kmask.min() >= 0 and kmask.max() < kk.size for kk, kmask in zip(self.kout, self.kmask))
                self.kmask = np.concatenate([np.searchsorted(self.kin, kk, side='left') for kk in self.kout], axis=0)
        else:
            if isinstance(wmatrix, str):
                from pypower import BaseMatrix
                wmatrix = BaseMatrix.load(wmatrix)
            wmatrix.select_proj(projsout=[(ellout, None) for ellout in self.ellsout])
            self.ellsin = []
            for proj in wmatrix.projsin:
                assert proj.wa_order in (None, 0)
                self.ellsin.append(proj.ell)
            self.kin = wmatrix.xin[0]
            assert all(np.allclose(xin, self.kin) for xin in wmatrix.xin)
            # TODO: implement best match BaseMatrix method
            for iout, (projout, kk) in enumerate(zip(wmatrix.projsout, self.kout)):
                nmk = np.sum((wmatrix.xout[iout] >= 2 * kk[0] - kk[1]) & (wmatrix.xout[iout] <= 2 * kk[-1] - kk[-2]))
                factorout = nmk // kk.size
                wmatrix.slice_x(sliceout=slice(0, wmatrix.xout[iout].size // factorout * factorout), projsout=projout)
                wmatrix.rebin_x(factorout=factorout, projsout=projout)
                istart = np.nanargmin(np.abs(wmatrix.xout[iout] - kk[0]))
                wmatrix.slice_x(sliceout=slice(istart, istart + kk.size), projsout=projout)
                if not np.allclose(wmatrix.xout[iout], kk):
                    raise ValueError('k-coordinates {} for ell = {:d} could not be found in input matrix (rebinning = {:d})'.format(kk, projout.ell, factorout))
            self.wmatrix = wmatrix.value
            if zeff is None:
                zeff = wmatrix.attrs.get('zeff', None)
            if fiducial is None:
                fiducial = wmatrix.attrs.get('fiducial', None)
        self.zeff = zeff
        self.fiducial = fiducial
        self.requires = {'theory': ('BaseTheoryPowerSpectrumMultipoles', {'k': self.kin, 'ells': self.ellsin, 'zeff': self.zeff, 'fiducial': self.fiducial})}

    def run(self):
        theory = np.ravel(self.theory.power)
        if self.wmatrix is not None:
            self.flatpower = np.dot(theory, self.wmatrix)
        elif self.kmask is not None:
            self.flatpower = theory[self.kmask]
        else:
            self.flatpower = theory

    @property
    def power(self):
        toret = []
        nout = 0
        for k in self.kout:
            sl = slice(nout, nout + len(k))
            toret.append(self.flatpower[sl])
            nout = sl.stop
        return toret

    def __getstate__(self):
        state = {}
        for name in ['kin', 'kout', 'zeff', 'ells', 'fiducial', 'wmatrix', 'kmask', 'flatpower']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

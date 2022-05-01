import numpy as np
from scipy import special

import cosmoprimo

from cosmofit.base import BaseCalculator
from . import utils


class BaseTheoryPowerSpectrumMultipoles(BaseCalculator):

    def __init__(self, k, zeff=1., ells=(0, 2, 4)):
        self.k = np.asarray(k, dtype='f8')
        self.zeff = float(zeff)
        self.ells = tuple(ells)

    def __getstate__(self):
        state = {}
        for name in ['k', 'zeff', 'ells', 'power']:
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
        muw = utils.weights_trapz(mu)
        self.muweights = np.array([muw * (2 * ell + 1) * special.legendre(ell)(self.mu) for ell in ells]) / (self.mu[-1] - self.mu[0])

    def to_poles(self, pkmu):
        return np.sum(pkmu * self.muweights[:, None, :], axis=-1)


def get_cosmo(cosmo):
    if isinstance(cosmo, str):
        cosmo = (cosmo, {})
    if isinstance(cosmo, tuple):
        return getattr(cosmoprimo.fiducial, cosmo[0])(**cosmo[1])
    return cosmoprimo.Cosmology(**cosmo)


class EffectAP(BaseCalculator):

    def __init__(self, zeff=1., fiducial='DESI'):
        self.zeff = float(zeff)
        fiducial = get_cosmo(fiducial)
        self.efunc_fid = fiducial.efunc(self.zeff)
        self.comoving_angular_distance_fid = fiducial.comoving_angular_distance(self.zeff)

        self.mode = 'distances'
        if 'aiso' in self.params:
            self.mode = 'aiso'
        elif 'apar' in self.params and 'aper' in self.params:
            self.mode = 'aparaper'
        self.requires = {'cosmoprimo': 'BasePrimordialCosmology'}

    def run(self, **params):
        if self.mode == 'distances':
            qpar, qper = self.efunc_fid / self.cosmoprimo.efunc(self.zeff), self.cosmoprimo.comoving_angular_distance(self.zeff) / self.comoving_angular_distance_fid
        elif self.mode == 'aiso':
            qpar = qper = params['aiso']
        else:
            qpar, qper = params['apar'], params['aper']
        self.qpar, self.qper = qpar, qper

    def ap_k_mu(self, k, mu):
        jac = 1. / (self.qpar * self.qper ** 2)
        F = self.qpar / self.qper
        factor_ap = np.sqrt(1 + mu**2 * (1. / F**2 - 1))
        # Beutler 2016 (arXiv: 1607.03150v1) eq 44
        kap = k / self.qper * factor_ap
        # Beutler 2016 (arXiv: 1607.03150v1) eq 45
        muap = mu / F / factor_ap
        return jac, kap, muap


class WindowedPowerSpectrumMultipoles(BaseCalculator):

    def __init__(self, kout, zeff=1., ellsout=(0, 2, 4), wmatrix=None):
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
        else:
            if isinstance(wmatrix, str):
                from pypower import BaseMatrix
                wmatrix = BaseMatrix.load(wmatrix)
            wmatrix.select_proj(projsout=self.ellsout)
            self.ellsin = []
            for proj in wmatrix.projsin:
                assert proj.wa_order in (None, 0)
                self.ellsin.append(proj.ell)
            self.kin = wmatrix.xin[0]
            assert all(np.allclose(xin, self.kin) for xin in wmatrix.xin)
            # TODO: implement best match BaseMatrix method
            for iout, (ellout, kk) in enumerate(zip(self.ellsout, self.kout)):
                mk = wmatrix.xout[iout]
                diff = np.abs(np.diff(mk)).min() / 2.
                nmk = np.sum((mk >= kk[0] - diff) & (mk <= kk[-1] + diff))
                factorout = nmk // kk.size
                wmatrix.rebin_x(factorout=factorout, projsout=ellout)
                istart = np.argmin(np.abs(wmatrix.xout[iout], kk[0]))
                wmatrix.slice_x(sliceout=slice(istart, istart + kk.size), projsout=ellout)
                if not np.allclose(wmatrix.xout[iout], kk):
                    raise ValueError('k-coordinates {} for ell = {:d} could not be found in input matrix (rebinning = {:d})'.format(kk, ellout, factorout))
            self.wmatrix = wmatrix.value

    def requires(self):
        return {'theory': ('BaseTheoryPowerSpectrumMultipoles', {'k': self.kin, 'zeff': self.zeff, 'ells': self.ellsin})}

    def run(self):
        if self.wmatrix is not None:
            self.power = np.dot(self.theory, self.wmatrix)
        elif self.kmask is not None:
            self.power = self.theory[self.kmask]
        else:
            self.power = self.theory

    def unpacked(self):
        toret = []
        nout = 0
        for k in self.kout:
            sl = slice(nout, nout + len(k))
            toret.append(self.power[sl])
            nout = sl.stop
        return toret

    def __getstate__(self):
        state = {}
        for name in ['kin', 'kout', 'zeff', 'ells', 'wmatrix', 'kmask', 'power']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

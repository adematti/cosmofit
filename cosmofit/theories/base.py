import numpy as np
from scipy import special

from cosmofit.base import BaseCalculator
from cosmofit import plotting
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
        for name in ['k', 'zeff', 'ells', 'power', 'fiducial']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


class BaseTheoryCorrelationFunctionMultipoles(BaseCalculator):

    def __init__(self, s=None, zeff=1., ells=(0, 2, 4), fiducial=None):
        if s is None: s = np.linspace(20., 200, 101)
        self.s = np.array(s, dtype='f8')
        self.zeff = float(zeff)
        self.ells = tuple(ells)
        self.fiducial = fiducial

    def __getstate__(self):
        state = {}
        for name in ['s', 'zeff', 'ells', 'corr', 'fiducial']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


class BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles(BaseTheoryCorrelationFunctionMultipoles):

    def __init__(self, s=None, ells=(0, 2, 4), **kwargs):
        super(BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles, self).__init__(s=s, ells=ells, **kwargs)
        self.k = np.logspace(min(-3, - np.log10(self.s[-1]) - 0.1), max(2, - np.log10(self.s[0]) + 0.1), 2000)
        from cosmoprimo import PowerToCorrelation
        self.fftlog = PowerToCorrelation(self.k, ell=self.ells, q=0, lowring=False)
        self.kin = np.geomspace(self.k[0], 1., 300)
        #self.kin = np.linspace(self.k[0], 0.5, 200)
        mask = self.k > self.kin[-1]
        self.lowk = self.k[~mask]
        self.pad_highk = np.exp(-(self.k[mask] - self.kin[-1])**2 / (2. * (0.5)**2))
        self.requires = {'power': {'init': {'k': self.kin, 'ells': self.ells, **kwargs}}}

    def set_params(self, params):
        self.requires['power']['params'] = params.copy()
        return params.clear()

    def run(self):
        power = [np.interp(np.log10(self.lowk), np.log10(self.kin), p) for p in self.power.power]
        power = [np.concatenate([p, p[-1] * self.pad_highk], axis=-1) for p in power]
        s, corr = self.fftlog(power)
        self.corr = np.array([np.interp(self.s, ss, cc) for ss, cc in zip(s, corr)])

    def plot(self, fn=None, kw_save=None):
        # Comparison to brute-force (non-fftlog) computation
        # Convergence towards brute-force when decreasing amping sigma
        # Difference between fftlog and brute-force: ~ effect of truncation / damping
        corr = []
        weights = utils.weights_trapz(np.log(self.kin))
        for ill, ell in enumerate(self.ells):
            # Integration in log, adding a k
            tmp = np.sum(self.kin**3 * self.power.power[ill] * weights * special.spherical_jn(ell, self.s[:, None] * self.kin), axis=-1)
            corr.append((-1) ** (ell // 2) / (2. * np.pi**2) * tmp)
        from matplotlib import pyplot as plt
        height_ratios = [max(len(self.ells), 3)] + [1] * len(self.ells)
        figsize = (6, 1.5 * sum(height_ratios))
        fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
        fig.subplots_adjust(hspace=0)
        lax[0].plot([], [], linestyle='-', color='k', label='fftlog')
        lax[0].plot([], [], linestyle='--', color='k', label='brute-force')
        for ill, ell in enumerate(self.ells):
            color = 'C{:d}'.format(ill)
            lax[0].plot(self.s, self.s**2 * self.corr[ill], color=color, linestyle='-', label=r'$\ell = {:d}$'.format(ell))
            lax[0].plot(self.s, self.s**2 * corr[ill], linestyle='--', color=color)
        for ill, ell in enumerate(self.ells):
            lax[ill + 1].plot(self.s, self.s**2 * (self.corr[ill] - corr[ill]), color='C{:d}'.format(ill))
            lax[ill + 1].set_ylabel(r'$\Delta s^{{2}}\xi_{{{0:d}}}$ [$(\mathrm{{Mpc}}/h)^{{2}}$]'.format(ell))
        for ax in lax: ax.grid(True)
        lax[0].legend()
        lax[0].set_ylabel(r'$s^{2} \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        lax[-1].set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
        if fn is not None:
            plotting.savefig(fn, fig=fig, **(kw_save or {}))
        return lax


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

    def __init__(self, zeff=1., fiducial=None, mode=None, eta=1./3.):
        self.zeff = float(zeff)
        if fiducial is None:
            raise ValueError('Provide fiducial cosmology')
        self.fiducial = get_cosmo(fiducial)
        self.efunc_fid = self.fiducial.efunc(self.zeff)
        self.comoving_angular_distance_fid = self.fiducial.comoving_angular_distance(self.zeff)
        self.mode = mode
        self.eta = float(eta)
        from .primordial_cosmology import BasePrimordialCosmology
        self.requires = {'cosmo': (BasePrimordialCosmology, {})}

    @classmethod
    def guess_mode(cls, params, default='distances'):
        mode = default
        if 'qiso' in params: mode = 'qiso'
        if 'qap' in params: mode = 'qap'
        if 'qiso' in params and 'qap' in params: mode = 'qisoqap'
        if 'qpar' in params and 'qper' in params: mode = 'qparqper'
        return mode

    def set_params(self, params):
        if self.mode is None:
            self.mode = self.guess_mode(params, default='distances')
        if self.mode == 'qiso':
            params = params.select(basename=['qiso'])
        elif self.mode == 'qap':
            params = params.select(basename=['qap'])
        elif self.mode == 'qisoqap':
            params = params.select(basename=['qiso', 'qap'])
        elif self.mode == 'qparqper':
            params = params.select(basename=['qpar', 'qper'])
        elif self.mode == 'distances':
            params = params.clear()
        else:
            raise ValueError('mode must be one of ["qiso", "qap", "qisoqap", "qparqper", "distances"]')
        return params

    def run(self, **params):
        if self.mode == 'distances':
            qpar, qper = self.efunc_fid / self.cosmo.efunc(self.zeff), self.cosmo.comoving_angular_distance(self.zeff) / self.comoving_angular_distance_fid
        elif self.mode == 'qiso':
            qpar = qper = params['qiso']
        elif self.mode == 'qap':
            qap = params['qap']  # qpar / qper
            qpar, qper = qap**(1 - self.eta), qap**(-self.eta)
        elif self.mode == 'qisoqap':
            qiso, qap = params['qiso'], params['qap']  # qpar / qper
            qpar, qper = qiso * qap**(1 - self.eta), qiso * qap**(-self.eta)
        else:
            qpar, qper = params['qpar'], params['qper']
        self.qpar, self.qper = qpar, qper
        self.qap = self.qpar / self.qper
        self.qiso = self.qpar**self.eta * self.qper**(1. - self.eta)

    def ap_k_mu(self, k, mu):
        jac = 1. / (self.qpar * self.qper**2)
        factorap = np.sqrt(1 + mu**2 * (1. / self.qap**2 - 1))
        # Beutler 2016 (arXiv: 1607.03150v1) eq 44
        kap = k[..., None] / self.qper * factorap
        # Beutler 2016 (arXiv: 1607.03150v1) eq 45
        muap = mu / self.qap / factorap
        return jac, kap, muap


class WindowedPowerSpectrumMultipoles(BaseCalculator):

    def __init__(self, k=None, ells=(0, 2, 4), ellsin=None, wmatrix=None, shotnoise=0., theory=None):
        if k is None: k = np.linspace(0.01, 0.2, 20)
        if np.ndim(k[0]) == 0:
            k = [k] * len(ells)
        self.k = [np.array(kk, dtype='f8') for kk in k]
        self.ells = tuple(ells)

        if theory is None: theory = {'init': {}}
        theory = dict(theory)
        theory.setdefault('class', 'BaseTheoryPowerSpectrumMultipoles')

        #wmatrix = None
        self.wmatrix = wmatrix
        if wmatrix is None:
            self.ellsin = tuple(self.ells)
            self.kin = np.unique(np.concatenate(self.k, axis=0))
            if all(np.allclose(kk, self.kin) for kk in self.k):
                self.kmask = None
            else:
                self.kmask = [np.searchsorted(self.kin, kk, side='left') for kk in self.k]
                assert all(kmask.min() >= 0 and kmask.max() < kk.size for kk, kmask in zip(self.k, self.kmask))
                self.kmask = np.concatenate(self.kmask, axis=0)
        else:
            if isinstance(wmatrix, str):
                from pypower import MeshFFTWindow, BaseMatrix
                fn = wmatrix
                wmatrix = MeshFFTWindow.load(fn)
                if hasattr(wmatrix, 'poles'):
                    wmatrix = wmatrix.poles
                else:
                    wmatrix = BaseMatrix.load(fn)
            self.ellsin = []
            for proj in wmatrix.projsin:
                assert proj.wa_order in (None, 0)
                self.ellsin.append(proj.ell)
            if ellsin is not None:
                self.ellsin = tuple(ellsin)
            projsin = [proj for proj in wmatrix.projsin if proj.ell in self.ellsin]
            self.ellsin = [proj.ell for proj in projsin]
            wmatrix.select_proj(projsout=[(ell, None) for ell in self.ells], projsin=projsin)
            self.kin = wmatrix.xin[0]
            assert all(np.allclose(xin, self.kin) for xin in wmatrix.xin)
            # TODO: implement best match BaseMatrix method
            for iout, (projout, kk) in enumerate(zip(wmatrix.projsout, self.k)):
                nmk = np.sum((wmatrix.xout[iout] >= 2 * kk[0] - kk[1]) & (wmatrix.xout[iout] <= 2 * kk[-1] - kk[-2]))
                factorout = nmk // kk.size
                wmatrix.slice_x(sliceout=slice(0, wmatrix.xout[iout].size // factorout * factorout), projsout=projout)
                wmatrix.rebin_x(factorout=factorout, projsout=projout)
                istart = np.nanargmin(np.abs(wmatrix.xout[iout] - kk[0]))
                wmatrix.slice_x(sliceout=slice(istart, istart + kk.size), projsout=projout)
                if not np.allclose(wmatrix.xout[iout], kk):
                    raise ValueError('k-coordinates {} for ell = {:d} could not be found in input matrix (rebinning = {:d})'.format(kk, projout.ell, factorout))
            self.wmatrix = wmatrix.value
            for name in ['zeff', 'fiducial']:
                if name in wmatrix.attrs and theory['init'].get(name, None) is None:
                    theory['init'].setdefault(name, wmatrix.attrs[name])
        shotnoise = float(shotnoise)
        self.shotnoise = np.array([shotnoise * (ell == 0) for ell in self.ellsin])
        self.flatshotnoise = np.concatenate([np.full_like(k, shotnoise * (ell == 0), dtype='f8') for ell, k in zip(self.ells, self.k)])

        theory['init'].update({'k': self.kin, 'ells': self.ellsin})
        self.requires = {'theory': theory}

    def run(self):
        theory = np.ravel(self.theory.power + self.shotnoise[:, None])
        if self.wmatrix is not None:
            self.flatpower = np.dot(theory, self.wmatrix)
        elif self.kmask is not None:
            self.flatpower = theory[self.kmask]
        else:
            self.flatpower = theory
        self.flatpower -= self.flatshotnoise

    @property
    def power(self):
        toret = []
        nout = 0
        for kk in self.k:
            sl = slice(nout, nout + len(kk))
            toret.append(self.flatpower[sl])
            nout = sl.stop
        return toret

    def __getstate__(self):
        state = {}
        for name in ['kin', 'k', 'zeff', 'ells', 'fiducial', 'wmatrix', 'kmask', 'flatpower', 'shotnoise', 'flatshotnoise']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def plot(self, fn=None, kw_save=None):
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([], [], linestyle='--', color='k', label='theory')
        ax.plot([], [], linestyle='-', color='k', label='window')
        for ill, ell in enumerate(self.ells):
            color = 'C{:d}'.format(ill)
            k = self.k[ill]
            maskin = (self.kin >= k[0]) & (self.kin <= k[-1])
            ax.plot(self.kin[maskin], self.kin[maskin] * self.theory.power[ill][maskin], color=color, linestyle='--', label=None)
            ax.plot(k, k * self.power[ill], color=color, linestyle='-', label=r'$\ell = {:d}$'.format(ell))
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        if fn is not None:
            plotting.savefig(fn, fig=fig, **(kw_save or {}))
        return ax


class WindowedCorrelationFunctionMultipoles(BaseCalculator):

    def __init__(self, s=None, ells=(0, 2, 4), theory=None):
        if s is None: s = np.linspace(20., 120., 101)
        if np.ndim(s[0]) == 0:
            s = [s] * len(ells)
        self.s = [np.array(ss, dtype='f8') for ss in s]
        self.ells = tuple(ells)

        if theory is None: theory = {}
        theory = dict(theory)
        theory.setdefault('class', 'BaseTheoryCorrelationFunctionMultipoles')

        # No matrix for the moment
        self.ellsin = tuple(self.ells)
        self.sin = np.unique(np.concatenate(self.s, axis=0))
        if all(np.allclose(ss, self.sin) for ss in self.s):
            self.smask = None
        else:
            self.smask = [np.searchsorted(self.sin, ss, side='left') for ss in self.k]
            assert all(smask.min() >= 0 and smask.max() < ss.size for ss, smask in zip(self.s, self.smask))
            self.smask = np.concatenate(self.smask, axis=0)

        theory['init'].update({'s': self.sin, 'ells': self.ellsin})
        self.requires = {'theory': theory}

    def run(self):
        theory = np.ravel(self.theory.corr)
        if self.smask is not None:
            self.flatcorr = theory[self.smask]
        else:
            self.flatcorr = theory

    @property
    def corr(self):
        toret = []
        nout = 0
        for ss in self.s:
            sl = slice(nout, nout + len(ss))
            toret.append(self.flatcorr[sl])
            nout = sl.stop
        return toret

    def __getstate__(self):
        state = {}
        for name in ['sin', 's', 'ells', 'smask', 'flatcorr']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

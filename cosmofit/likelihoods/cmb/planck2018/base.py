import os

import numpy as np
from scipy import constants

from cosmofit.likelihoods.base import BaseCalculator
from cosmofit.theories.primordial_cosmology import BasePrimordialCosmology


class ClTheory(BaseCalculator):

    def __init__(self, cls, lensing=None, non_linear=None, unit=None):
        self.requested_cls = dict(cls)
        self.ell_max_lensed_cls, self.ell_max_lens_potential_cls = 0, 0
        for cl, ellmax in self.requested_cls.items():
            if cl in ['tt', 'ee', 'bb', 'te']: self.ell_max_lensed_cls = max(self.ell_max_lensed_cls, ellmax)
            elif cl in ['pp', 'tp', 'ep']: self.ell_max_lens_potential_cls = max(self.ell_max_lens_potential_cls, ellmax)
            elif cl in ['tb', 'eb']: pass
            else: raise ValueError('Unknown Cl {}'.format(cl))
        if lensing is None:
            lensing = bool(self.ell_max_lens_potential_cls)
        ellmax = max(self.ell_max_lensed_cls, self.ell_max_lens_potential_cls)
        if non_linear is None:
            if bool(self.ell_max_lens_potential_cls) or max(ellmax if 'b' in cl.lower() else 0 for cl, ellmax in self.requested_cls.items()) > 50:
                non_linear = 'mead'
            else:
                non_linear = ''
        self.unit = unit
        allowed_units = [None, 'muK']
        if self.unit not in allowed_units:
            raise ValueError('Input unit must be one of {}, found {}'.format(allowed_units, self.unit))
        self.requires = {'cosmo': {'class': BasePrimordialCosmology,
                                   'init': {'params': {'lensing': lensing, 'ellmax_cl': ellmax, 'non_linear': non_linear}}}}

    def run(self):
        self.cls = {}
        T0_cmb = self.cosmo.T0_cmb
        hr = self.cosmo.get_harmonic()
        if self.ell_max_lensed_cls:
            lensed_cl = hr.lensed_cl(ellmax=self.ell_max_lensed_cls)
        if self.ell_max_lens_potential_cls:
            lens_potential_cl = hr.lens_potential_cl(ellmax=self.ell_max_lens_potential_cls)
        for cl, ellmax in self.requested_cls.items():
            if cl in ['tb', 'eb']:
                tmp = np.zeros(ellmax + 1, dtype='f8')
            if 'p' in cl:
                tmp = lens_potential_cl[cl][:ellmax + 1]
            else:
                tmp = lensed_cl[cl][:ellmax + 1]
            if self.unit == 'muK':
                npotential = cl.count('p')
                unit = (T0_cmb * 1e6)**(2 - npotential)
                tmp = tmp * unit
            self.cls[cl] = tmp

    def __getstate__(self):
        state = {}
        for name in ['requested_cls', 'unit']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return {**state, **self.cls}

    def __setstate__(self, state):
        state = state.copy()
        self.unit = state.pop('unit')
        self.cls = state


class PlanckClikLikelihood(BaseCalculator):

    def __init__(self, clik_fn, clik_dir=None, product_id=None):

        import clik
        if clik_dir is not None:
            clik_fn = os.path.join(clik_dir, clik_fn)
        self.lensing = clik.try_lensing(clik_fn)
        try:
            if self.lensing:
                self.clik = clik.clik_lensing(clik_fn)
            else:
                self.clik = clik.clik(clik_fn)
        except clik.lkl.CError as exc:
            if not os.path.exists(clik_fn):
                raise IOError('The path to the .clik file for the likelihood {} was not found at {}'.format(self.__class__.__name__, clik_fn))
            else:
                raise exc
        self.ells_max = self.clik.get_lmax()
        self.nuisance_params = list(self.clik.extra_parameter_names)

        requested_cls = ['tt', 'ee', 'bb', 'te', 'tb', 'eb']
        if self.lensing:
            has_cls = [ellmax != -1 for ellmax in self.ells_max]
            requested_cls = ['pp'] + requested_cls
        else:
            has_cls = self.clik.get_has_cl()
        self.requested_cls, sizes, cls = [], [], {}
        for cl, ellmax, has_cl in zip(requested_cls, self.ells_max, has_cls):
            if int(has_cl):
                self.requested_cls.append(cl)
                sizes.append(ellmax + 1)
                if cl not in ['tb', 'eb']: cls[cl] = ellmax

        # Placeholder for vector passed to clik
        self.cumsizes = np.insert(np.cumsum(sizes), 0, 0)
        self.vector = np.zeros(self.cumsizes[-1] + len(self.nuisance_params))
        self.requires = {'theory': {'class': ClTheory, 'init': {'cls': cls, 'lensing': True, 'unit': 'muK'}}}

    def set_params(self, params):
        basenames = [param.basename for param in params if not param.get('drop', False)]
        if set(basenames) != set(self.nuisance_params):
            raise ValueError('Expected nuisance parameters {}, received {}'.format(self.nuisance_params, basenames))
        return params

    def run(self, **params):
        self.loglikelihood = -np.inf
        for cl, start, stop in zip(self.requested_cls, self.cumsizes[:-1], self.cumsizes[1:]):
            if cl in ['tb', 'eb']: continue
            tmp = self.theory.cls[cl]
            # Check for nan's: may produce a segfault in clik
            if np.isnan(tmp).any():
                return
            self.vector[start:stop] = tmp

        # Fill with likelihood parameters
        self.vector[self.cumsizes[-1]:] = [params[p] for p in self.nuisance_params]
        self.loglikelihood = self.clik(self.vector)[0]
        # "zero" of clik, and sometimes nan's returned
        if np.allclose(self.loglikelihood, -1e30) or np.isnan(self.loglikelihood):
            self.loglikelihood = -np.inf

    def __del__(self):
        del self.clik


class Planck2018ClikLikelihood(PlanckClikLikelihood):

    pass

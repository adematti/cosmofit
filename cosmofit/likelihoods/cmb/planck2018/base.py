import os

import numpy as np
from scipy import constants

from cosmofit.likelihoods.base import BaseCalculator
from cosmofit.theories.primordial_cosmology import BasePrimordialCosmology


class PlanckClickLikelihood(BaseCalculator):

    def __init__(self, click_fn, click_dir=None, product_id=None):

        import click
        if click_dir is not None:
            click_fn = os.path.join(click_dir, click_fn)
        self.lensing = clik.try_lensing(clik_fn)
        try:
            if self.lensing:
                self.clik = clik.clik_lensing(click_fn)
            else:
                self.clik = clik.clik(click_fn)
        except clik.lkl.CError as exc:
            if not os.path.exists(click_fn):
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
        self.requested_cls, sizes = [], []
        self.ell_max_lensed_cls, self.ell_max_lens_potential_cls = 0, 0
        for cl, ellmax, has_cl in zip(requested_cls, self.ells_max, has_cls):
            if has_cl:
                self.requested_cls.append(cl)
                self.ells_max_cls.append(ellmax)
                sizes.append(ellmax + 1)
                if cl in ['tt', 'ee', 'bb', 'te']: self.ell_max_lensed_cls = max(self.ell_max_lensed_cls, ellmax)
                if cl in ['tb', 'eb']: self.ell_max_lens_potential_cls = max(self.ell_max_lens_potential_cls, ellmax)

        # Placeholder for vector passed to clik
        self.cumsizes = np.insert(np.cumsum(sizes), 0, 0)
        self.vector = np.zeros(self.cumsizes[-1] + len(self.nuisance_params))

        self.requires = {'cosmo': {'class': BasePrimordialCosmology, 'init': {'params': {'lensing': self.lensing}, 'extra_params': {'l_max_scalars': max(self.ells_max)}}}}

    def set_params(self, params):
        basenames = params.basenames()
        if set(basenames) != set(self.nuisance_params):
            raise ValueError('Expected nuisance parameters {}, received {}'.format(self.nuisance_params, basenames))
        return params

    def run(self, **params):
        hr = self.cosmo.get_harmonic()
        if self.ell_max_lensed_cls:
            lensed_cl = hr.lensed_cl(ellmax=self.ell_max_lensed_cls)
        if self.ell_max_lens_potential_cls:
            lens_potential_cl = hr.lens_potential_cl(ellmax=self.ell_max_lens_potential_cls)
        self.loglikelihood = -np.inf
        for cl, start, stop in zip(self.requested_cls, self.cumsizes[:-1], self.cumsizes[1:]):
            if cl in ['tb', 'eb']: continue
            if 'p' in cl:
                tmp = lens_potential_cl[cl][:stop - start]
            else:
                tmp = lensed_cl[cl][:stop - start]
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

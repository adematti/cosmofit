import os
from collections import UserDict
from configparser import ConfigParser

import numpy as np

from cosmofit.likelihoods.base import BaseCalculator
from cosmofit.theories.primordial_cosmology import BasePrimordialCosmology


class SNLikelihood(BaseCalculator):

    def __init__(self, config_fn, data_dir=None):
        if data_dir is None:
            from cosmofit.install import InstallerConfig
            data_dir = InstallerConfig()[self.__class__.__name__]['data_dir']
        self.config = self.read_config(os.path.join(data_dir, config_fn))
        self.covariance = self.read_covariance(os.path.join(data_dir, self.config['mag_covmat_file']))
        self.light_curve_params = self.read_light_curve_params(os.path.join(data_dir, self.config['data_file']))
        self.requires = {'cosmo': {'class': BasePrimordialCosmology}}

    def read_config(self, fn):

        class Parser(UserDict):

            def __init__(self, fn):
                data = {}
                with open(fn, 'r') as file:
                    for line in file.readlines():
                        kv = [v.strip() for v in line.split('=')]
                        if len(kv) == 1:
                            data[kv] = ''
                        elif len(kv) == 2:
                            data[kv[0]] = kv[1]
                        else:
                            raise ValueError('Could not read {} of {}'.format(line, fn))
                self.data = data

            def int(self, key):
                return int(self[key])

            def float(self, key):
                return float(self[key])

            def bool(self, key):
                toret = self[key].lower()
                if toret in ['true', 't']:
                    return True
                if toret in ['false', 'f']:
                    return False
                raise ValueError('Cannot interpret {} as bool'.format(key))

        return Parser(fn)

    def read_covariance(self, fn):
        with open(fn, 'r') as file:
            size = int(file.readline())
        return np.loadtxt(fn, skiprows=1).reshape(size, size)

    def read_light_curve_params(self, fn):
        sep = ' '
        with open(fn, 'r') as file:
            for line in file.readlines():
                if line.startswith('#'):
                    names = [name.strip() for name in line[1:].split(sep)]
                    values = {name: [] for name in names}
                elif line:
                    for name, value in zip(names, line.split(sep)):
                        values[name].append(value if name == 'name' else float(value))
        return {name: np.array(value) for name, value in values.items()}

"""Definition of :class:`Chain`, to hold products of likelihood sampling."""

import os
import re
import glob

import numpy as np

from cosmofit.parameter import ParameterCollection, Parameter, ParameterArray

from .profile import ParameterValues
from . import utils


class Chain(ParameterValues):

    """Class that holds samples drawn from likelihood."""

    _type = ParameterArray
    _attrs = ParameterValues._attrs + ['_logposterior', '_aweight', '_fweight', '_weight']

    def __init__(self, data=None, params=None, logposterior='logposterior', aweight='aweight', fweight='fweight', weight='weight', **kwargs):
        self._logposterior = logposterior
        self._aweight = aweight
        self._fweight = fweight
        self._weight = weight
        super(Chain, self).__init__(data=data, params=params, **kwargs)

    @property
    def _metrics(self):
        return [self._logposterior, self._aweight, self._fweight, self._weight]

    @property
    def aweight(self):
        if self._aweight not in self:
            self[self._aweight] = np.ones(self.shape, dtype='f8')
        return self[self._aweight]

    @property
    def fweight(self):
        if self._fweight not in self:
            self[self._fweight] = np.ones(self.shape, dtype='f8')
        return self[self._fweight]

    @property
    def logposterior(self):
        if self._logposterior not in self:
            self[self._logposterior] = np.zeros(self.shape, dtype='f8')
        return self[self._logposterior]

    @aweight.setter
    def aweight(self, item):
        self[self._aweight] = item

    @fweight.setter
    def aweight(self, item):
        self[self._aweight] = item

    @logposterior.setter
    def logposterior(self, item):
        self[self._aweight] = item

    @property
    def weight(self):
        return ParameterArray(self.aweight * self.fweight, Parameter(self._weight, latex=utils.metrics_to_latex(self._weight)))

    def ravel(self):
        new = self.copy()
        for param in self.names():
            new[param] = self[param].ravel()
        return new

    def remove_burnin(self, burnin=0):
        """
        Return new samples with burn-in removed.

        Parameters
        ----------
        burnin : float, int
            If burnin between 0 and 1, remove that fraction of samples.
            Else, remove burnin first points.

        Returns
        -------
        samples : Chain
        """
        if 0 < burnin < 1:
            burnin = burnin * len(self)
        burnin = int(burnin + 0.5)
        return self[burnin:]

    def get(self, name, *args, **kwargs):
        """
        Return parameter of name ``name`` in collection.

        Parameters
        ----------
        name : Parameter, string
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.

        Returns
        -------
        param : Parameter
        """
        has_default = False
        if args:
            if len(args) > 1:
                raise SyntaxError('Too many arguments!')
            has_default = True
            default = args[0]
        if kwargs:
            if len(kwargs) > 1:
                raise SyntaxError('Too many arguments!')
            has_default = True
            default = kwargs['default']
        try:
            return self.data[self.index(name)]
        except IndexError:
            if name == self._weight:
                return self.weight
            if has_default:
                return default
            raise KeyError('Column {} does not exist'.format(name))

    def __repr__(self):
        """Return string representation, including shape and columns."""
        return 'Chain(shape={}, params={})'.format(self.shape, self.params())

    @classmethod
    def read_cosmomc(cls, base_filename, ichains=None):
        """
        Load samples in *CosmoMC* format, i.e.:

        - '_{ichain}.txt' files for sample values
        - '.paramnames' files for parameter names / latex
        - '.ranges' for parameter ranges

        Parameters
        ----------
        base_filename : string
            Base *CosmoMC* file name. Will be prepended by '_{ichain}.txt' for sample values,
            '.paramnames' for parameter names and '.ranges' for parameter ranges.

        ichains : int, tuple, list, default=None
            Chain numbers to load. Defaults to all chains matching pattern '{base_filename}*.txt'

        Returns
        -------
        samples : Chain
        """
        self = cls()

        params_filename = '{}.paramnames'.format(base_filename)
        self.log_info('Loading params file: {}.'.format(params_filename))
        params = ParameterCollection()
        with open(params_filename) as file:
            for line in file:
                name, latex = line.split()
                name = name.strip()
                if name.endswith('*'): name = name[:-1]
                latex = latex.strip().replace('\n', '')
                params.set(Parameter(basename=name.strip(), latex=latex, fixed=False))

            ranges_filename = '{}.ranges'.format(base_filename)
            if os.path.exists(ranges_filename):
                self.log_info('Loading parameter ranges from {}.'.format(ranges_filename))
                with open(ranges_filename) as file:
                    for line in file:
                        name, low, high = line.split()
                        latex = latex.replace('\n', '')
                        limits = []
                        for lh, li in zip([low, high], [-np.inf, np.inf]):
                            lh = lh.strip()
                            if lh == 'N': lh = li
                            else: lh = float(lh)
                            limits.append(lh)
                        params[name.strip()].prior.set_limits(limits=limits)
            else:
                self.log_info('Parameter ranges file {} does not exist.'.format(ranges_filename))

            chain_filename = '{}{{}}.txt'.format(base_filename)
            chain_filenames = []
            if ichains is not None:
                if np.ndim(ichains) == 0:
                    ichains = [ichains]
                for ichain in ichains:
                    chain_filenames.append(chain_filename.format('_{:d}'.format(ichain)))
            else:
                chain_filenames = glob.glob(chain_filename.format('*'))

            samples = []
            for chain_filename in chain_filenames:
                self.log_info('Loading chain file: {}.'.format(chain_filename))
                samples.append(np.loadtxt(chain_filename, unpack=True))

            samples = np.concatenate(samples, axis=-1)
            self.aweight = samples[0]
            self.logposterior = -samples[1]
            for param, values in zip(params, samples[2:]):
                self.set(ParameterArray(values, param))

        return self

    def write_cosmomc(self, base_filename, params=None, ichain=None, fmt='%.18e', delimiter=' ', **kwargs):
        """
        Save samples to disk in *CosmoMC* format.

        Parameters
        ----------
        base_filename : string
            Base *CosmoMC* file name. Will be prepended by '_{ichain}.txt' for sample values,
            '.paramnames' for parameter names and '.ranges' for parameter ranges.

        columns : list, ParameterCollection, default=None
            Parameters to save samples of. Defaults to all parameters (weight and logposterior treated separatey).

        ichain : int, default=None
            Chain number to append to file name, i.e. sample values will be saved as '{base_filename}_{ichain}.txt'.
            If ``None``, does not append any number, sample values will be saved as '{base_filename}.txt'.

        kwargs : dict
            Arguments for :func:`numpy.savetxt`.
        """
        if params is None: params = self.names()
        columns = list([str(param) for param in params])
        metrics_columns = [self._weight, self._logposterior]
        for column in self._metrics:
            if column in columns: del columns[columns.index(column)]
        data = self.to_array(params=metrics_columns + columns, struct=False).reshape(-1, self.size)
        data[1] *= -1
        data = data.T
        utils.mkdir(os.path.dirname(base_filename))
        chain_filename = '{}.txt'.format(base_filename) if ichain is None else '{}_{:d}.txt'.format(base_filename, ichain)
        self.log_info('Saving chain to {}.'.format(chain_filename))
        np.savetxt(chain_filename, data, header='', fmt=fmt, delimiter=delimiter, **kwargs)

        output = ''
        params = self.params(name=columns)
        for param in params:
            tmp = '{}* {}\n' if getattr(param, 'derived', getattr(param, 'fixed')) else '{} {}\n'
            output += tmp.format(param.name, param.latex())
        params_filename = '{}.paramnames'.format(base_filename)
        self.log_info('Saving parameter names to {}.'.format(params_filename))
        with open(params_filename, 'w') as file:
            file.write(output)

        output = ''
        for param in params:
            limits = param.prior.limits
            limits = tuple('N' if limit is None or np.abs(limit) == np.inf else limit for limit in limits)
            output += '{} {} {}\n'.format(param.name, limits[0], limits[1])
        ranges_filename = '{}.ranges'.format(base_filename)
        self.log_info('Saving parameter ranges to {}.'.format(ranges_filename))
        with open(ranges_filename, 'w') as file:
            file.write(output)

    def to_getdist(self, params=None, label=None):
        """
        Return *GetDist* hook to samples.

        Parameters
        ----------
        columns : list, ParameterCollection, default=None
            Parameters to share to *GetDist*. Defaults to all parameters (weight and logposterior treated separatey).

        Returns
        -------
        samples : getdist.MCSamples
        """
        from getdist import MCSamples
        toret = None
        if params is None: params = self.params()
        labels = [param.latex() for param in params]
        samples = self.to_array(params=params, struct=False).reshape(-1, self.size)
        names = [str(param) for param in params]
        toret = MCSamples(samples=samples.T, weights=self.weight.ravel(), loglikes=-self.logposterior.ravel(), names=names, labels=labels, label=label)
        return toret

    def var(self, param, ddof=1):
        """
        Estimate weighted param variance.

        Parameters
        ----------
        columns : list, ParameterCollection, default=None
            Parameters to compute variance for.

        ddof : int, default=1
            Number of degrees of freedom.

        Returns
        -------
        var : scalar, array
            If single parameter provided as ``columns``, returns variance for that parameter (scalar).
            Else returns variance array.
        """
        return np.cov(self[param].ravel(), fweights=self.fweight.ravel(), aweights=self.aweight.ravel(), ddof=ddof)

    def std(self, param, ddof=1):
        return self.var(param, ddof=ddof) ** 0.5

    def mean(self, param):
        """Return weighted mean."""
        return np.average(self[param].ravel(), weights=self.weight.ravel())

    def argmax(self, param):
        """Return parameter value for maximum of ``cost.``"""
        return self[param].ravel()[np.argmax(self.logposterior.ravel())]

    def median(self, param):
        """Return weighted quantiles."""
        return utils.weighted_quantile(self[param].ravel(), q=0.5, weights=self.weight.ravel())

    def quantile(self, param, q=(0.1587, 0.8413)):
        """Return weighted quantiles."""
        return utils.weighted_quantile(self[param].ravel(), q=q, weights=self.weight.ravel())

    def interval(self, param, **kwargs):
        """
        Return n-sigmas confidence interval(s).

        Parameters
        ----------
        columns : list, ParameterCollection, default=None
            Parameters to compute confidence interval for.

        nsigmas : int
            Return interval for this number of sigmas.

        Returns
        -------
        interval : array
        """
        return utils.interval(self[param].ravel(), self.weight.ravel(), **kwargs)

    def cov(self, params=None, ddof=1):
        """
        Estimate weighted param covariance.

        Parameters
        ----------
        columns : list, ParameterCollection, default=None
            Parameters to compute covariance for.
            Defaults to all varied parameters.

        ddof : int, default=1
            Number of degrees of freedom.

        Returns
        -------
        cov : scalar, array
            If single parameter provided as ``columns``, returns variance for that parameter (scalar).
            Else returns covariance (2D array).
        """
        if params is None:
            params = self.params()
        if not utils.is_sequence(params):
            params = [params]
        return np.atleast_2d(np.cov([self[param].ravel() for param in params], fweights=self.fweight.ravel(), aweights=self.aweight.ravel(), ddof=ddof))

    def invcov(self, params=None, ddof=1):
        """
        Estimate weighted parameter inverse covariance.

        Parameters
        ----------
        columns : list, ParameterCollection, default=None
            Parameters to compute inverse covariance for.
            Defaults to all varied parameters.

        ddof : int, default=1
            Number of degrees of freedom.

        Returns
        -------
        cov : scalar, array
            If single parameter provided as ``columns``, returns inverse variance for that parameter (scalar).
            Else returns inverse covariance (2D array).
        """
        return utils.inv(self.cov(params, ddof=ddof))

    def corrcoef(self, params=None, **kwargs):
        """
        Estimate weighted parameter correlation matrix.
        See :meth:`cov`.
        """
        return utils.cov_to_corrcoef(self.cov(params, **kwargs))

    def to_stats(self, params=None, quantities=None, sigfigs=2, tablefmt='latex_raw', filename=None):
        """
        Export samples summary quantities.

        Parameters
        ----------
        columns : list, default=None
            Parameters to export quantities for.
            Defaults to all parameters.

        quantities : list, default=None
            Quantities to export. Defaults to ``['argmax','mean','median','std','quantile:1sigma','interval:1sigma']``.

        sigfigs : int, default=2
            Number of significant digits.
            See :func:`utils.round_measurement`.

        tablefmt : string, default='latex_raw'
            Format for summary table.
            See :func:`tabulate.tabulate`.

        filename : string default=None
            If not ``None``, file name where to save summary table.

        Returns
        -------
        tab : string
            Summary table.
        """
        import tabulate
        if params is None: params = self.params(varied=True)
        data = []
        if quantities is None: quantities = ['argmax', 'mean', 'median', 'std', 'quantile:1sigma', 'interval:1sigma']
        is_latex = 'latex_raw' in tablefmt

        def round_errors(low, up):
            low, up = utils.round_measurement(0.0, low, up, sigfigs=sigfigs)[1:]
            if is_latex: return '${{}}_{{{}}}^{{+{}}}$'.format(low, up)
            return '{}/+{}'.format(low, up)

        for iparam, param in enumerate(params):
            row = []
            if is_latex: row.append(param.latex(inline=True))
            else: row.append(str(param.name))
            ref_center = self.mean(param)
            ref_error = self.var(param) ** 0.5
            for quantity in quantities:
                if quantity in ['argmax', 'mean', 'median', 'std']:
                    value = getattr(self, quantity)(param)
                    value = utils.round_measurement(value, ref_error, sigfigs=sigfigs)[0]
                    if is_latex: value = '${}$'.format(value)
                    row.append(value)
                elif quantity.startswith('quantile'):
                    nsigmas = int(re.match('quantile:(.*)sigma', quantity).group(1))
                    low, up = self.quantile(param, q=utils.nsigmas_to_quantiles_1d_sym(nsigmas))
                    row.append(round_errors(low - ref_center, up - ref_center))
                elif quantity.startswith('interval'):
                    nsigmas = int(re.match('interval:(.*)sigma', quantity).group(1))
                    low, up = self.interval(param, nsigmas=nsigmas)
                    row.append(round_errors(low - ref_center, up - ref_center))
                else:
                    raise RuntimeError('Unknown quantity {}.'.format(quantity))
            data.append(row)
        tab = tabulate.tabulate(data, headers=quantities, tablefmt=tablefmt)
        if filename is not None:
            utils.mkdir(os.path.dirname(filename))
            self.log_info('Saving to {}.'.format(filename))
            with open(filename, 'w') as file:
                file.write(tab)
        return tab

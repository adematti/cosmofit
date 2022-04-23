"""Definition of :class:`Chain`, to hold products of likelihood sampling."""

import os
import re
import glob

import numpy as np

from cosmofit import utils
from cosmofit.parameter import ParameterCollection, Parameter, ParameterArray

from .profile import ParameterValues
from .utils import nsigmas_to_quantiles_1d_sym, metrics_to_latex


class Chain(ParameterValues):

    """Class that holds samples drawn from likelihood."""

    _type = ParameterArray
    _attrs = []

    def __init__(self, data=None, parameters=None, logposterior='logposterior', aweight='aweight', fweight='fweight', weight='weight'):
        super(Chain, self).__init__(data=data, parameters=parameters)
        self._logposterior = logposterior
        self._aweight = aweight
        self._fweight = fweight
        self._weight = weight

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
        return ParameterArray(self.aweight * self.fweight, Parameter(self._weight, latex=metrics_to_latex(self._weight)))

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
        return 'Chain(shape={:d}, parameters={})'.format(self.shape, self.parameters())

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

        parameters_filename = '{}.paramnames'.format(base_filename)
        self.log_info('Loading parameters file: {}.'.format(parameters_filename))
        parameters = ParameterCollection()
        with open(parameters_filename) as file:
            for line in file:
                name, latex = line.split()
                name = name.strip()
                if name.endswith('*'): name = name[:-1]
                latex = latex.strip().replace('\n', '')
                parameters.set(Parameter(name=name.strip(), latex=latex, fixed=False))

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
                        parameters[name.strip()].prior.set_limits(limits=limits)
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
            for param, values in zip(parameters, samples[2:]):
                self.set(ParameterArray(values, param))

        return self

    def write_cosmomc(self, base_filename, parameters=None, ichain=None, fmt='%.18e', delimiter=' ', **kwargs):
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
        if parameters is None: parameters = self.names()
        columns = list([str(param) for param in parameters])
        metrics_columns = [self._weight, self._logposterior]
        for column in metrics_columns:
            if column in columns: del columns[columns.index(column)]
        data = self.to_array(columns=metrics_columns + columns, struct=False).reshape(-1, self.size)
        data[1] *= -1
        data = data.T
        utils.mkdir(os.path.dirname(base_filename))
        chain_filename = '{}.txt'.format(base_filename) if ichain is None else '{}_{:d}.txt'.format(base_filename, ichain)
        self.log_info('Saving chain to {}.'.format(chain_filename))
        np.savetxt(chain_filename, data, header='', fmt=fmt, delimiter=delimiter, **kwargs)

        output = ''
        parameters = self.parameters()
        parameters = [parameters[column] for column in columns]
        for param in parameters:
            tmp = '{}* {}\n' if getattr(param, 'derived', getattr(param, 'fixed')) else '{} {}\n'
            output += tmp.format(param.name, param.latex if param.latex is not None else param.name)
        parameters_filename = '{}.paramnames'.format(base_filename)
        self.log_info('Saving parameter names to {}.'.format(parameters_filename))
        with open(parameters_filename, 'w') as file:
            file.write(output)

        output = ''
        for param in parameters:
            limits = param.prior.limits
            limits = tuple('N' if limit is None or np.abs(limit) == np.inf else limit for limit in limits)
            output += '{} {} {}\n'.format(param.name, limits[0], limits[1])
        ranges_filename = '{}.ranges'.format(base_filename)
        self.log_info('Saving parameter ranges to {}.'.format(ranges_filename))
        with open(ranges_filename, 'w') as file:
            file.write(output)

    def to_getdist(self, parameters=None):
        """
        Return *GetDist* hook to samples.

        Parameters
        ----------
        columns : list, ParameterCollection, default=None
            Parameters to share to *GetDist*. Defaults to all parameters (weight and logposterior treated separatey).

        Returns
        -------
        samples : getdist.MCChain
        """
        from getdist import MCChain
        toret = None
        if parameters is None: parameters = self.parameters()
        labels = [param.latex for param in parameters]
        samples = self.to_array(parameters=parameters, struct=False).reshape(-1, self.size)
        names = [str(param) for param in parameters]
        toret = MCChain(samples=samples.T, weights=self.weight, loglikes=-self.logposterior, names=names, labels=labels)
        return toret

    def var(self, parameter, ddof=1):
        """
        Estimate weighted parameter variance.

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
        return np.cov(self[parameter].ravel(), fweights=self.fweight.ravel(), aweights=self.aweight.ravel(), ddof=ddof)

    def mean(self, parameter):
        """Return weighted mean."""
        return np.average(self[parameter].ravel(), weights=self.weight.ravel())

    def argmax(self, parameter):
        """Return parameter value for maximum of ``cost.``"""
        return self[parameter].ravel()[np.argmax(self.logposterior.ravel())]

    def quantile(self, parameter, q=(0.1587, 0.8413)):
        """Return weighted quantiles."""
        return utils.weighted_quantile(self[parameter].ravel(), q=q, weights=self.weight.ravel())

    def interval(self, parameter, **kwargs):
        """
        Return n-sigmas confidence interval(s).

        Parameters
        ----------
        columns : list, ParameterCollection, default=None
            Parameters to compute confidence interval for.

        nsigmas : int
            Return interval for this number of sigmas.

        bins : int, default=100
            Number of bins i.e. mesh nodes.
            See :meth:`Mesh.from_samples`.

        method : string
            Method to interpolate (weighted) samples on mesh.
            See :meth:`Mesh.from_samples`.

        bw_method : string, default='scott'
            If ``method`` is ``'gaussian_kde'``, method to determine KDE bandwidth, see :class:`scipy.stats.gaussian_kde`.

        Returns
        -------
        interval : array
        """
        return utils.interval(self[parameter].ravel(), self.weight.ravel(), **kwargs)

    def cov(self, parameters=None, ddof=1):
        """
        Estimate weighted parameter covariance.

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
        return np.cov([self[param].ravel() for param in parameters], fweights=self.fweight.ravel(), aweights=self.aweight.ravel(), ddof=ddof)

    def invcov(self, parameters=None, ddof=1):
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
        return utils.inv(self.cov(parameters, ddof=ddof))

    def corrcoef(self, parameters=None, **kwargs):
        """
        Estimate weighted parameter correlation matrix.
        See :meth:`cov`.
        """
        return utils.cov_to_corrcoef(self.cov(parameters, **kwargs))

    def to_stats(self, parameters=None, quantities=None, sigfigs=2, tablefmt='latex_raw', filename=None):
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
        # if columns is None: columns = self.columns(exclude='metrics.*')
        if parameters is None: parameters = self.parameters(varied=True)
        data = []
        if quantities is None: quantities = ['argmax', 'mean', 'median', 'std', 'quantile:1sigma', 'interval:1sigma']
        is_latex = 'latex_raw' in tablefmt

        def round_errors(low, up):
            low, up = utils.round_measurement(0.0, low, up, sigfigs=sigfigs)[1:]
            if is_latex: return '${{}}_{{{}}}^{{+{}}}$'.format(low, up)
            return '{}/+{}'.format(low, up)

        for iparam, param in enumerate(parameters):
            row = []
            if is_latex: row.append(param.get_label())
            else: row.append(str(param.name))
            ref_center = self.mean(param)
            ref_error = self.std(param)
            for quantity in quantities:
                if quantity in ['argmax', 'mean', 'median', 'std']:
                    value = getattr(self, quantity)(param)
                    value = utils.round_measurement(value, ref_error, sigfigs=sigfigs)[0]
                    if is_latex: value = '${}$'.format(value)
                    row.append(value)
                elif quantity.startswith('quantile'):
                    nsigmas = int(re.match('quantile:(.*)sigma', quantity).group(1))
                    low, up = self.quantile(param, q=nsigmas_to_quantiles_1d_sym(nsigmas))
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

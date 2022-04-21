"""Definition of :class:`Profiles`, to hold products of likelihood profiling."""

import os

import numpy as np

from cosmofit import utils
from cosmofit.utils import BaseClass, is_sequence
from cosmofit.parameter import Parameter, ParameterArray, ParameterCollection
from .utils import metrics_to_latex


class ParameterValues(ParameterCollection):

    """Class that holds samples drawn from likelihood."""

    _type = ParameterArray
    _attrs = []
    _metrics = []

    @staticmethod
    def _get_name(param):
        return getattr(param, 'name', str(param))

    @staticmethod
    def _get_param(item):
        return item.parameter

    @property
    def shape(self):
        if len(self.data):
            return self.data[0].shape
        return ()

    @property
    def size(self):
        """Equivalent for :meth:`__len__`."""
        return np.prod(self.shape, dtype='intp')

    def parameters(self, include_metrics=False, **kwargs):
        parameters = super(ParameterValues, self).parameters(**kwargs)
        if not include_metrics:
            parameters = [param for param in parameters if str(param) not in self._metrics]
        return parameters

    @classmethod
    def concatenate(cls, *others):
        """
        Concatenate input collections.
        Unique items only are kept.
        """
        if not others: return cls()
        if len(others) == 1 and is_sequence(others[0]):
            others = others[0]
        new = cls(others[0])
        new_names = new.names()
        for other in others:
            other_names = other.names()
            if new_names and other_names and set(other_names) != set(new_names):
                raise ValueError('Cannot concatenate values as parameters do not match: {} != {}.'.format(other_names, new_names))
        for name in new_names:
            new[name] = np.concatenate([other[name] for other in others], axis=0)
        return new

    def __setitem__(self, name, item):
        """
        Update parameter in collection (a parameter with same name must already exist).
        See :meth:`set` to set a new parameter.

        Parameters
        ----------
        name : Parameter, string, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
            If integer, index in collection.

        item : Parameter
            Parameter.
        """
        if not isinstance(item, self._type):
            param = Parameter(name, latex=metrics_to_latex(name) if name in self._metrics else None)
            if param in self:
                param = self[param].parameter.clone(param)
            item = ParameterArray(item, param)
        try:
            self.data[name] = item
        except TypeError:
            name = self._get_name(item)
            if self._get_name(item) != name:
                raise KeyError('Parameter {} must be indexed by name (incorrect {})'.format(self._get_name(item), name))
            self.data[self._index_name(name)] = item

    def __getitem__(self, name):
        """
        Get samples parameter ``name`` if :class:`Parameter` or string,
        else return copy with local slice of samples.
        """
        if isinstance(name, (Parameter, str)):
            return self.get(name)
        new = self.copy()
        new.data = [column[name] for column in self.data]
        return new

    def __repr__(self):
        """Return string representation, including shape and columns."""
        return 'ParameterValues(shape={:d}, parameters={})'.format(self.shape, self.parameters())

    def to_array(self, parameters=None, struct=True):
        """
        Return samples as numpy array.

        Parameters
        ----------
        columns : list, default=None
            Columns to use. Defaults to all columns.

        struct : bool, default=True
            Whether to return structured array, with columns accessible through e.g. ``array['Position']``.
            If ``False``, numpy will attempt to cast types of different columns.

        Returns
        -------
        array : array
        """
        if parameters is None:
            parameters = self.parameters()
        names = [self._get_name(name) for name in parameters]
        if struct:
            toret = np.empty(len(self), dtype=[(name, self[name].dtype, self.shape[1:]) for name in names])
            for name in names: toret[name] = self[name]
            return toret
        return np.array([self[name] for name in names])


class ParameterBestFit(ParameterValues):

    def __init__(self, data=None, logposterior='logposterior'):
        super(ParameterBestFit, self).__init__(data=data)
        self._logposterior = logposterior

    @property
    def _metrics(self):
        return [self._logposterior]

    @property
    def logposterior(self):
        if self._logposterior not in self:
            self[self._logposterior] = np.zeros(self.shape, dtype='f8')
        return self[self._logposterior]


class ParameterCovariance(BaseClass):

    """Class that represents a parameter covariance."""

    def __init__(self, covariance, parameters):
        """
        Initialize :class:`ParameterCovariance`.

        Parameters
        ----------
        covariance : array
            2D array representing covariance.

        parameters : list, ParameterCollection
            Parameters corresponding to input ``covariance``.
        """
        self._cov = np.asarray(covariance)
        self.parameters = ParameterCollection(parameters)

    def cov(self, parameters=None):
        """Return covariance matrix for input parameters ``parameters``."""
        if parameters is None:
            parameters = self.parameters
        idx = np.array([self.parameters.index(param) for param in parameters])
        toret = self._cov[np.ix_(idx, idx)]
        return toret

    def invcov(self, parameters=None):
        """Return inverse covariance matrix for input parameters ``parameters``."""
        return utils.inv(self.cov(parameters))

    def corrcoef(self, parameters=None):
        """Return correlation matrix for input parameters ``parameters``."""
        return utils.cov_to_corrcoef(self.cov(parmeters=parameters))

    def __getstate__(self):
        """Return this class state dictionary."""
        return {'cov': self._cov, 'parameters': self.parameters.__getstate__()}

    def __setstate__(self, state):
        """Set this class state dictionary."""
        self._cov = state['cov']
        self.parameters = ParameterCollection.from_state(state['parameters'])

    def __repr__(self):
        """Return string representation of parameter covariance, including parameters."""
        return '{}({})'.format(self.__class__.__name__, self.parameters)


class ProfilingResult(BaseClass):
    r"""
    Class holding results of likelihood profiling.

    Attributes
    ----------
    init : ParamDict
        Initial parameter values.

    bestfit : ParamDict
        Best fit parameters.

    parabolic_errors : ParamDict
        Parameter parabolic errors.

    deltachi2_errors : ParamDict
        Lower and upper errors corresponding to :math:`\Delta \chi^{2} = 1`.

    covariance : ParameterCovariance
        Parameter covariance at best fit.
    """
    _attrs = {'init': ParameterValues, 'bestfit': ParameterBestFit, 'parabolic_errors': ParameterBestFit, 'deltachi2_errors': ParameterBestFit, 'covariance': ParameterCovariance}

    def __init__(self, attrs=None, **kwargs):
        """
        Initialize :class:`Profiles`.

        Parameters
        ----------
        parameters : list, ParameterCollection
            Parameters used in likelihood profiling.

        attrs : dict, default=None
            Other attributes.
        """
        self.attrs = attrs or {}
        self.set(**kwargs)

    def parameters(self, **kwargs):
        return self.init.parameters(**kwargs)

    def set(self, parameters=None, **kwargs):
        for name, cls in self._attrs.items():
            if name in kwargs:
                item = kwargs[name]
                if name == 'covariance':
                    if not isinstance(item, cls):
                        item = cls(item, parameters=self.parameters() if parameters is None else parameters)
                else:
                    item = cls(item)
                setattr(self, name, item)

    def get(self, name):
        """Access attribute by name."""
        return getattr(self, name)

    def has(self, name):
        """Has this attribute?"""
        return hasattr(self, name)

    @classmethod
    def concatenate(cls, *others):
        """
        Concatenate profiles together.

        Parameters
        ----------
        others : list
            List of :class:`Profiles` instances.

        Returns
        -------
        new : Profiles

        Warning
        -------
        :attr:`attrs` of returned profiles contains, for each key, the last value found in ``others`` :attr:`attrs` dictionaries.
        """
        if not others: return cls()
        if len(others) == 1 and is_sequence(others[0]):
            others = others[0]
        new = others[0].copy()
        attrs = [name for name in new._attrs if new.has(name) and name != 'covariance']
        for other in others:
            if [name for name in other._attrs if other.has(name) and name != 'covariance'] != attrs:
                raise ValueError('Cannot concatenate two profiles if both do not have same attributes.')
        for name in attrs:
            setattr(new, name, new._attrs[name].concatenate([other.get(name) for other in others]))
        return new

    def extend(self, other):
        """Extend profiles with ``other``."""
        new = self.concatenate(self, other)
        self.__dict__.update(new.__dict__)

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        state['attrs'] = self.attrs
        for name in self._attrs:
            if hasattr(self, name):
                state[name] = getattr(self, name).__getstate__()
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        self.attrs = state.get('attrs', {})
        for name, cls in self._attrs.items():
            if name in state:
                setattr(self, name, cls.from_state(state[name]))

    def to_stats(self, parameters=None, quantities=None, sigfigs=2, tablefmt='latex_raw', filename=None):
        """
        Export profiling quantities.

        Parameters
        ----------
        parameters : list, ParameterCollection
            Parameters to export quantities for.
            Defaults to all parameters.

        quantities : list, default=None
            Quantities to export. Defaults to ``['bestfit','parabolic_errors','deltachi2_errors']``.

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
        if parameters is None: parameters = self.parameters
        data = []
        if quantities is None: quantities = [quantity for quantity in ['bestfit', 'parabolic_errors', 'deltachi2_errors'] if self.has(quantity)]
        is_latex = 'latex_raw' in tablefmt
        argmax = self.besfit.logposterior.argmax()

        def round_errors(low, up):
            low, up = utils.round_measurement(0.0, low, up, sigfigs=sigfigs)[1:]
            if is_latex: return '${{}}_{{{}}}^{{+{}}}$'.format(low, up)
            return '{}/+{}'.format(low, up)

        for iparam, param in enumerate(parameters):
            row = []
            if is_latex: row.append(param.get_label())
            else: row.append(str(param.name))
            row.append(str(param.varied))
            ref_error = self.parabolic_errors[param][argmax]
            for quantity in quantities:
                if quantity in ['bestfit', 'parabolic_errors']:
                    value = self.get(quantity)[param][argmax]
                    value = utils.round_measurement(value, ref_error, sigfigs=sigfigs)[0]
                    if is_latex: value = '${}$'.format(value)
                    row.append(value)
                elif quantity == 'deltachi2_errors':
                    low, up = self.get(quantity)[param][argmax]
                    row.append(round_errors(low, up))
                else:
                    raise ValueError('Unknown quantity {}.'.format(quantity))
            data.append(row)
        headers = []
        chi2min = '{:.2f}'.format(-2. * self.bestfit.logposterior[argmax])
        headers.append((r'$\chi^{{2}} = {}$' if is_latex else 'chi2 = {}').format(chi2min))
        headers.append('varied')
        headers += [quantity.replace('_', ' ') for quantity in quantities]
        tab = tabulate.tabulate(data, headers=headers, tablefmt=tablefmt)
        if filename is not None:
            utils.mkdir(os.path.dirname(filename))
            self.log_info('Saving to {}.'.format(filename))
            with open(filename, 'w') as file:
                file.write(tab)
            return tab

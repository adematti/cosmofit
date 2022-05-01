import os
import re
import sys
import importlib
import inspect
from collections import UserDict

import numpy as np
import yaml
from mpytools import CurrentMPIComm

from .utils import BaseClass
from .parameter import ParameterCollection


namespace_delimiter = '.'


class FileSystem(BaseClass):

    def __init__(self, output='./'):
        self.output_dir = os.path.dirname(output)
        self.namespace = os.path.basename(output)

    def filename(self, fn, i=None):
        if i is not None:
            fni = fn.format(i)
            if fni == fn:
                base, ext = os.path.splitext(fn)
                fn = '{}_{}.{}'.format(base, i, ext)
            else:
                fn = fni
        return os.path.join(self.output_dir, self.namespace + fn)


class YamlLoader(yaml.SafeLoader):
    """
    *yaml* loader that correctly parses numbers.
    Taken from https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number.
    """


YamlLoader.add_implicit_resolver(u'tag:yaml.org,2002:float',
                                 re.compile(u'''^(?:
                                 [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                                 |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                                 |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                                 |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                                 |[-+]?\\.(?:inf|Inf|INF)
                                 |\\.(?:nan|NaN|NAN))$''', re.X),
                                 list(u'-+0123456789.'))

YamlLoader.add_implicit_resolver('!none', re.compile('None$'), first='None')


def none_constructor(loader, node):
    return None


YamlLoader.add_constructor('!none', none_constructor)


def yaml_parser(string, index=None):
    """Parse string in *yaml* format."""
    # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    alls = list(yaml.load_all(string, Loader=YamlLoader))
    if index is not None:
        if isinstance(index, dict):
            for config in alls:
                if all([config.get(name) == value for name, value in index.items()]):
                    break
        else:
            config = alls[index]
    else:
        config = yaml.load(string, Loader=YamlLoader)
    data = dict(config)
    return data


class Decoder(UserDict):
    """
    Class that decodes configuration dictionary, taking care of template forms.

    Attributes
    ----------
    data : dict
        Decoded configuration dictionary.

    raw : dict
        Raw (without decoding of template forms) configuration dictionary.

    filename : string
        Path to corresponding configuration file.

    parser : callable
        *yaml* parser.
    """
    def __init__(self, data=None, string=None, parser=None, decode=True, **kwargs):
        """
        Initialize :class:`Decoder`.

        Parameters
        ----------
        data : dict, string, default=None
            Dictionary or path to a configuration *yaml* file to decode.

        string : string
            If not ``None``, *yaml* format string to decode.
            Added on top of ``data``.

        parser : callable, default=yaml_parser
            Function that parses *yaml* string into a dictionary.
            Used when ``data`` is string, or ``string`` is not ``None``.

        decode : bool, default=True
            Whether to decode configuration dictionary, i.e. solve template forms.

        kwargs : dict
            Arguments for :func:`parser`.
        """
        self.parser = parser
        if parser is None:
            self.parser = yaml_parser

        data_ = {}

        self.base_dir = '.'
        if isinstance(data, str):
            if string is None: string = ''
            # if base_dir is None: self.base_dir = os.path.dirname(data)
            string += self.read_file(data)
        elif data is not None:
            data_ = dict(data)

        if string is not None:
            data_.update(self.parser(string, **kwargs))

        self.data = data_
        if decode: self.decode()

    def read_file(self, filename):
        """Read file at path ``filename``."""
        with open(filename, 'r') as file:
            toret = file.read()
        return toret

    def decode(self):

        eval_re_pattern = re.compile("e'(.*?)'$")

        def decode_eval(word):
            m = re.match(eval_re_pattern, word)
            if m:
                words = m.group(1)
                return eval(words, {'np': np}, {})
            return word

        def callback(di):
            for key, value in (di.items() if isinstance(di, dict) else enumerate(di)):
                if isinstance(value, (dict, list)):
                    callback(di)
                elif isinstance(value, str):
                    di[key] = decode_eval(value)

        callback(self.data)

    def update_from_namespace(self, string, value, inherit_type=True):
        namespaces = string.split(namespace_delimiter)
        namespaces, name = namespaces[:-1], namespaces[-1]
        d = self.data
        for namespace in namespaces:
            d = d[namespace]
        if inherit_type and name in d:
            d[name] = type(d[name])(value)
        else:
            d[name] = value


class RegisteredCalculator(type(BaseClass)):

    _registry = set()

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry.add(cls)
        return cls

    @property
    def config_fn(cls):
        fn = inspect.getfile(cls)
        return os.path.splitext(fn)[0] + '.yaml'


class BaseCalculator(BaseClass, metaclass=RegisteredCalculator):

    is_vectorized = False
    parameters = ParameterCollection()

    def requires(self):
        return {}

    def __getattr__(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        if name == 'runtime_info':
            self.runtime_info = RuntimeInfo(self)
            return self.runtime_info
        if name in self.runtime_info.requires:
            toret = self.runtime_info.requires[name]
            if toret.runtime_info.torun:
                toret.run(**toret.runtime_info.params)
            return toret
        raise AttributeError('Attribute {} does not exist'.format(name))

    def __repr__(self):
        return '{}(namespace={}, basename={})'.format(self.__class__.__name__, self.info.namespace, self.info.basename)

    @property
    def mpicomm(self):
        mpicomm = getattr(self, '_mpicomm', None)
        if mpicomm is None: mpicomm = CurrentMPIComm.get()
        self._mpicomm = mpicomm
        return self._mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self._mpicomm = mpicomm


class PipelineError(Exception):

    pass


def import_cls(clsname, pythonpath=None, registry=BaseCalculator._registry):

    if isinstance(clsname, type):
        return clsname
    tmp = clsname.rsplit('.', 1)
    if len(tmp) == 1:
        allcls = []
        for cls in registry:
            if cls.__name__ == tmp: allcls.append(cls)
        if len(allcls) == 1:
            return allcls[0]
        if len(allcls) > 1:
            raise PipelineError('Multiple calculator classes are named {}'.fomat(tmp))
        raise PipelineError('No calculator class {} found'.fomat(tmp))
    modname, clsname = tmp
    if pythonpath is not None:
        sys.path.insert(0, pythonpath)
    module = importlib.import_module(modname)
    return getattr(module, clsname)


def format_clsdict(clsdict, **kwargs):
    # cls, init kwargs
    if isinstance(clsdict, str):
        clsdict = {'class': clsdict}
    if isinstance(clsdict, (list, tuple)):
        if len(clsdict) == 1:
            clsdict = (clsdict[0], {})
        if len(clsdict) == 2:
            clsdict = {**clsdict[1], **{'class': clsdict[0]}}
        else:
            raise PipelineError('Provide (ClassName, {}) or {class: ClassName, ...}')
    else:
        clsdict = dict(clsdict)
    clsdict['class'] = import_cls(clsdict.pop('class'), pythonpath=clsdict.pop('pythonpath', None), **kwargs)
    config_fn = clsdict.pop('config_fn', clsdict['class'].config_fn)
    clsdict = {**Decoder(config_fn), **clsdict}
    load_fn = clsdict.pop('load_fn', None)
    save_fn = clsdict.pop('save_fn', None)
    if not isinstance(load_fn, str):
        if load_fn and isinstance(save_fn, str):
            load_fn = save_fn
        else:
            load_fn = None
    if os.path.isfile(load_fn):
        clsdict = {'class': clsdict['class']}
    clsdict['load_fn'], clsdict['save_fn'] = load_fn, save_fn
    return clsdict


def init_from_clsdict(clsdict, namespace=None, params=None, **kwargs):
    # cls, init kwargs
    clsdict = format_clsdict(clsdict)
    cls = clsdict.pop('class')
    new = cls.__new__(cls)
    new.params = ParameterCollection(getattr(new, 'params', None), namespace=namespace)
    new.params.update(ParameterCollection(clsdict.pop('params', None), namespace=namespace))
    if params is not None:
        for param in params:
            if param in new.params:
                new.params[param].update(param)
    info = RuntimeInfo(new, namespace=namespace, initargs=clsdict, **kwargs)
    load_fn = clsdict['load_fn']
    save_fn = clsdict['save_fn']
    if load_fn:
        new = cls.load(save_fn)
    else:
        new.__init__(**clsdict)
    new.info = info
    if save_fn is not None:
        new.save(save_fn)
    return new


class RuntimeInfo(BaseClass):

    def __init__(self, calculator, namespace=None, requires=None, required_by=None, initargs=None, basename=None):
        self.initargs = dict(initargs or {})
        self.basename = basename
        self.namespace = namespace
        self.required_by = set(required_by or [])
        self.requires = requires
        if requires is None:
            self.requires = {}
            for name, clsdict in calculator.requires:
                self.requires.update(init_from_clsdict(clsdict, namespace=namespace))
        self.params = {param.basename: param.value for param in calculator.params}

    @property
    def torun(self):
        return getattr(self, '_torun', True)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._torun = True
        self._params = params

    @property
    def name(self):
        if self.namespace:
            return namespace_delimiter.join(self.namespace, self.basename)
        return self.basename


class BasePipeline(BaseClass):

    @CurrentMPIComm.enable
    def __init__(self, calculators, params=None, mpicomm=None):
        self.params = ParameterCollection(params)
        if isinstance(calculators, str):
            calculators = Decoder(calculators)
        else:
            import copy
            calculators = copy.deepcopy(calculators)

        instantiated = []

        def callback_namespace(calculators_in_basenamespace, todo, basenamespace=None):

            for newnamespace, calculators_in_namespace in calculators_in_basenamespace.items():
                if namespace_delimiter in newnamespace:
                    raise PipelineError('Do not use {} in namespace'.format(namespace_delimiter))
                if basenamespace is None:
                    namespace = newnamespace
                else:
                    namespace = namespace_delimiter.join([basenamespace, newnamespace])
                if 'class' not in calculators_in_namespace:  # check for another name space
                    for key in calculators_in_namespace: break
                    if key != 'class':
                        callback_namespace(key, calculators_in_namespace[key], basenamespace=namespace)

                todo(namespace, calculators_in_namespace)

        def search_parent_namespace(calculators, namespace):
            # first in this namespace, then go back
            splitnamespace = namespace.split(namespace_delimiter)

            def todo(tmpnamespace, calculators_in_namespace):
                tmpsplitnamespace = tmpnamespace.split(namespace_delimiter)
                if tmpsplitnamespace == splitnamespace[:len(tmpsplitnamespace)]:
                    for basename, clsdict in calculators_in_namespace:
                        yield tmpnamespace, basename, clsdict

            callback_namespace(calculators, todo, basenamespace=None)

        def todo(namespace, calculators_in_namespace):
            for name, clsdict in calculators_in_namespace.items():
                calculators_in_namespace[name] = format_clsdict(clsdict)

        callback_namespace(calculators, todo, basenamespace=None)

        def is_equal(obj1, obj2):
            if type(obj2) is type(obj1):
                if isinstance(obj1, dict):
                    if obj2.keys() == obj1.keys():
                        return all(is_equal(obj1[name], obj2[name]) for name in obj1)
                elif isinstance(obj1, (tuple, list)):
                    if len(obj2) == len(obj1):
                        return all(is_equal(o1, o2) for o1, o2 in zip(obj1, obj2))
                elif isinstance(obj1, np.ndarray):
                    return np.all(obj2 == obj1)
                else:
                    return obj2 == obj1
            return False

        def callback_instantiate(namespace, basename, clsdict, required_by=None):
            new = init_from_clsdict(clsdict, namespace=namespace, params=self.params, requires={}, required_by=required_by, basename=basename)
            instantiated.append(new)
            for requirementbasename, clsdict in new.requires:
                # Search for parameter
                clsdict = format_clsdict(clsdict)
                requirementnamespace = namespace
                match_first, match_name = None, None
                for tmpnamespace, tmpbasename, tmpclsdict in search_parent_namespace(calculators, namespace):
                    if issubclass(tmpclsdict['class'], clsdict['class']):
                        tmp = (tmpnamespace, tmpbasename, {**clsdict, **tmpclsdict})
                        if match_first is None:
                            match_first = tmp
                        if tmpbasename == requirementbasename:
                            match_name = tmp
                            break
                if match_name:
                    requirementnamespace, requirementbasename, clsdict = match_name
                elif match_first:
                    requirementnamespace, requirementbasename, clsdict = match_first
                for inst in instantiated:
                    already_instantiated = is_equal((inst.__class__, inst.info.namespace, inst.info.basename, inst.info.clsdict),
                                                    (clsdict['class'], requirementnamespace, requirementbasename, clsdict))
                    if already_instantiated:
                        break
                if already_instantiated:
                    requirement = inst
                    requirement.info.required_by.add(new)
                else:
                    requirement = callback_instantiate(requirementnamespace, requirementbasename, clsdict, required_by={new})
                new.info.requires[requirementbasename] = requirement

        def todo(namespace, calculators_in_namespace):
            for basename, clsdict in calculators_in_namespace.items():
                callback_instantiate(namespace, basename, clsdict)

        callback_namespace(calculators, todo, basenamespace=None)

        self.end_calculators = []
        self.calculators = instantiated
        for calculator in self.calculators:
            self.params.update(calculator.params)
            if not calculator.info.required_by:
                self.end_calculators.append(calculator)

        # Checks
        for param in self.params:
            if not any(param in calculator.params for calculator in self.calculators):
                raise PipelineError('Parameter {} is not used by any calculator')

        def callback(calculator, required_by):
            for calc in required_by:
                if calc is calculator:
                    raise PipelineError('Circular dependency for calculator {}'.format(calc))
                callback(calculator, calc.info.required_by)

        for calculator in self.calculator:
            callback(calculator, calculator.info.required_by)

        self.mpicomm = mpicomm

        # Init run, e.g. for fixed parameters
        for calculator in self.end_calculators:
            calculator.run(**calculator.info.params)

    def run(self, **params):  # params with namespace
        for calculator in self.calculators:
            for param in calculator.params:
                value = params.get(param.name, None)
                if value is not None and value != calculator.info.params[param.basename]:
                    calculator.info.params[param.basename] = value
                    calculator.info.torun = True
        for calculator in self.end_calculators:
            calculator.run(**calculator.info.params)

    @property
    def mpicomm(self):
        mpicomm = getattr(self, '_mpicomm', None)
        if mpicomm is None: mpicomm = CurrentMPIComm.get()
        self._mpicomm = mpicomm
        return self._mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self._mpicomm = mpicomm
        for calculator in self.calculators:
            calculator.mpicomm = mpicomm


class LikelihoodPipeline(BasePipeline):

    def __init__(self, *args, **kwargs):
        super(LikelihoodPipeline, self).__init__(*args, **kwargs)
        # Check end_calculators are likelihoods
        for calculator in self.calculators:
            likelihood_name = 'loglikelihood'
            if not hasattr(calculator, 'loglikelihood'):
                raise PipelineError('End calculator {} has no attribute {}'.format(calculator, likelihood_name))
            loglikelihood = getattr(calculator, likelihood_name)
            if not np.ndim(loglikelihood) == 0:
                raise PipelineError('End calculator {} attribute {} must be scalar'.format(calculator, likelihood_name))

    def run(self, **params):
        super(self, LikelihoodPipeline).run(**params)
        for calculator in self.end_calculators:
            self.loglikelihoods[calculator.info.name] = calculator.loglikelihood
        self.loglikelihood = sum(loglike for loglike in self.loglikelihoods)

    def logprior(self, **params):
        logprior = 0.
        for name, value in params.items():
            logprior += self.params[name].prior(value)
        return logprior

import os
import re
from collections import UserDict

import numpy as np
import yaml

from .utils import BaseClass


class FileSystem(BaseClass):

    def __init__(self, name=None):
        if isinstance(name, self.__class__):
            self.__dict__.update(name.__dict__)
            return
        if name is None:
            name = './'
        self.base_dir = os.path.dirname(name)
        self.namespace = os.path.basename(name)

    def filename(self, fn, i=None):
        if i is not None:
            fni = fn.format(i)
            if fni == fn:
                base, ext = os.path.splitext(fn)
                fn = '{}_{}{}'.format(base, i, ext)
            else:
                fn = fni
        return os.path.join(self.base_dir, self.namespace + fn)

    def __call__(self, *args, **kwargs):
        return self.filename(*args, **kwargs)


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
            match = False
            for config in alls:
                match = all([config.get(name) == value for name, value in index.items()])
                if match: break
            if not match:
                raise IndexError('No match found for index {}'.format(index))
        else:
            config = alls[index]
    else:
        config = yaml.load(string, Loader=YamlLoader)
    data = dict(config)
    return data


class MetaClass(type(BaseClass), type(UserDict)):

    pass


class BaseConfig(BaseClass, UserDict, metaclass=MetaClass):
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
    _attrs = ['data']

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
        if isinstance(data, self.__class__):
            self.__dict__.update(data.copy().__dict__)
            return

        self.parser = parser
        if parser is None:
            self.parser = yaml_parser

        datad = {}

        self.base_dir = '.'
        if isinstance(data, str):
            if string is None: string = ''
            # if base_dir is None: self.base_dir = os.path.dirname(data)
            string += self.read_file(data)
        elif data is not None:
            datad = dict(data)

        if string is not None:
            datad.update(self.parser(string, **kwargs))

        self.data = datad
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
                    callback(value)
                elif isinstance(value, str):
                    di[key] = decode_eval(value)

        callback(self.data)

    def update_from_namespace(self, string, value, inherit_type=True, delimiter=None):
        if delimiter is None:
            from .base import namespace_delimiter as delimiter
        namespaces = string.split(delimiter)
        namespaces, name = namespaces[:-1], namespaces[-1]
        d = self.data
        for namespace in namespaces:
            d = d[namespace]
        if inherit_type and name in d:
            d[name] = type(d[name])(value)
        else:
            d[name] = value

    def __copy__(self):
        import copy
        new = super(BaseConfig, self).__copy__()
        for name in self._attrs:
            if hasattr(self, name):
                setattr(self, name, copy.copy(getattr(self, name)))
        return new

    def deepcopy(self):
        import copy
        return copy.deepcopy(self)

    def clone(self, *args, **kwargs):
        new = self.copy()
        new.update(*args, **kwargs)
        return new

    def __eq__(self, other):

        def callback(obj1, obj2):
            if type(obj2) is type(obj1):
                if isinstance(obj1, dict):
                    if obj2.keys() == obj1.keys():
                        return all(callback(obj1[name], obj2[name]) for name in obj1)
                elif isinstance(obj1, (tuple, list)):
                    if len(obj2) == len(obj1):
                        return all(callback(o1, o2) for o1, o2 in zip(obj1, obj2))
                elif isinstance(obj1, np.ndarray):
                    return np.all(obj2 == obj1)
                else:
                    return obj2 == obj1
            return False

        return type(other) == type(self) and callback(self.data, other.data)


class FileSystemConfig(BaseConfig):

    def init(self):
        return FileSystem(self.get('input', None)), FileSystem(self.get('output', None))

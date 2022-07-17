import os
import re
from collections import UserDict

import numpy as np
import yaml

from .utils import BaseClass


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


class ConfigError(Exception):

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
    _attrs = []

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

    @staticmethod
    def read_file(filename):
        """Read file at path ``filename``."""
        with open(filename, 'r') as file:
            toret = file.read()
        return toret

    def decode(self):

        eval_re_pattern = re.compile("e'(.*?)'$")
        format_re_pattern = re.compile("f'(.*?)'$")

        def decode_eval(word):
            m = re.match(eval_re_pattern, word)
            if m:
                word = m.group(1)
                placeholders = re.finditer(r'\{.*?\}', word)
                word_letters = re.sub(r'[^a-zA-Z]', '', word)
                di = {}
                for placeholder in placeholders:
                    placeholder = placeholder.group()
                    key = placeholder[1:-1]
                    freplace = replace = self.search(key)
                    if isinstance(replace, str):
                        freplace = decode_eval(replace)
                        if freplace is None: freplace = replace
                    key = '__variable_of_{}_{:d}__'.format(word_letters, len(di) + 1)
                    assert key not in word
                    di[key] = freplace
                    word = word.replace(placeholder, key)
                return eval(word, {'np': np}, di)
            return None

        def decode_format(word):
            m = re.match(format_re_pattern, word)
            if m:
                word = m.group(1)
                placeholders = re.finditer(r'\{.*?\}', word)
                for placeholder in placeholders:
                    placeholder = placeholder.group()
                    if placeholder.startswith('{{'):
                        word = word.replace(placeholder, placeholder[1:-1])
                    else:
                        keyfmt = placeholder[1:-1].split(':', 2)
                        if len(keyfmt) == 2: key, fmt = keyfmt[0], ':' + keyfmt[1]
                        else: key, fmt = keyfmt[0], ''
                        freplace = replace = self.search(key)
                        if isinstance(replace, str):
                            freplace = decode_format(replace)
                            if freplace is None: freplace = replace
                        word = word.replace(placeholder, ('{' + fmt + '}').format(freplace))
                return word
            return None

        def callback(di, decode):
            for key, value in (di.items() if isinstance(di, dict) else enumerate(di)):
                if isinstance(value, (dict, list)):
                    callback(value, decode)
                elif isinstance(value, str):
                    tmp = decode(value)
                    if tmp is not None:
                        di[key] = tmp

        callback(self.data, decode_eval)
        callback(self.data, decode_format)

    def search(self, namespaces, delimiter=None):
        if isinstance(namespaces, str):
            if delimiter is None:
                from .base import namespace_delimiter as delimiter
            namespaces = namespaces.split(delimiter)
        d = self
        for namespace in namespaces:
            d = d[namespace]
        return d

    def update_from_namespace(self, string, value, inherit_type=True, delimiter=None):
        if delimiter is None:
            from .base import namespace_delimiter as delimiter
        namespaces = string.split(delimiter)
        namespaces, basename = namespaces[:-1], namespaces[-1]
        d = self.search(namespaces)
        if inherit_type and basename in d:
            d[basename] = type(d[basename])(value)
        else:
            d[basename] = value

    def __copy__(self):
        import copy
        new = super(BaseConfig, self).__copy__()
        new.data = self.data.copy()
        for name in self._attrs:
            if hasattr(self, name):
                setattr(new, name, copy.copy(getattr(self, name)))
        return new

    def deepcopy(self):
        import copy
        return copy.deepcopy(self)

    def update(self, *args, **kwargs):
        super(BaseConfig, self).update(*args, **kwargs)
        if len(args) == 1 and isinstance(args[0], self.__class__):
            self.__dict__.update({name: value for name, value in args[0].__dict__.items() if name != 'data'})

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

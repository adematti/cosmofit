import os
import sys
import datetime
import argparse

from mpytools import CurrentMPIComm

from ._version import __version__
from .io import BaseConfig, ConfigError
from .base import BasePipeline, LikelihoodPipeline, DoConfig, namespace_delimiter
from .samples import SourceConfig, SummaryConfig
from .utils import setup_logging
from cosmofit.profilers import ProfilerConfig
from cosmofit.samplers import SamplerConfig
from cosmofit.emulators import EmulatorConfig


def ascii_art(section):

    ascii_art = r"""
                                 __ _ _
                                / _(_) |
   ___ ___  ___ _ __ ___   ___ | |_ _| |_
  / __/ _ \/ __| '_ ` _ \ / _ \|  _| | __|
 | (_| (_) \__ \ | | | | | (_) | | | | |_
  \___\___/|___/_| |_| |_|\___/|_| |_|\__|
                                          """ + """\n"""\
    + """{}\n""".format(section)\
    + """version: {}               date: {}\n""".format(__version__, datetime.date.today())
    return ascii_art


def parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config_fn', type=str, help='Name of configuration file')
    parser.add_argument('--verbose', '-v', type=str, choices=['warning', 'info', 'debug'], default='info', help='Verbosity level')
    parser.add_argument('--update', type=str, nargs='*', default=[], help='List of namespace1.....key=value to update config file')
    parser.add_argument('remainder', nargs=argparse.REMAINDER)
    return parser


def update_config(config_fn, section=None, update=None, remainder=None):
    config = BaseConfig(config_fn, decode=False)
    if section is not None:
        config.setdefault(section, {})
    for string in (update or []):
        keyvalue = string.split('=')
        if len(keyvalue) != 2:
            raise ConfigError('Provide updates as namespace1....name.key=value format')
        config.update_from_namespace(keyvalue[0], keyvalue[1], inherit_type=True)
    for string in (remainder or []):
        keyvalue = string.split('=')
        if len(keyvalue) != 2:
            raise ConfigError('Provide updates as key=value format')
        config.update_from_namespace(namespace_delimiter.join([section, keyvalue[0]]), keyvalue[1], inherit_type=True)
    config.decode()
    return config


@CurrentMPIComm.enable
def read_args(args=None, mpicomm=None, parser=parser(), section='sample'):
    args = parser.parse_args(args=args)
    if mpicomm.rank == 0:
        print(ascii_art(section))
    setup_logging(args.verbose)
    config = update_config(args.config_fn, section=section, update=args.update, remainder=args.remainder)
    return config, args


@CurrentMPIComm.enable
def sample_from_args(args=None, mpicomm=None):
    config, args = read_args(args=args, mpicomm=mpicomm, section='sample')
    return sample_from_config(config, mpicomm=mpicomm)


@CurrentMPIComm.enable
def sample_from_config(config, mpicomm=None):
    config = BaseConfig(config)
    if 'sample' not in config:
        raise ConfigError('Provide "sample"')
    config_sampler = SamplerConfig(config['sample'])

    if 'pipeline' not in config:
        raise ConfigError('Provide pipeline')

    cls = LikelihoodPipeline if config_sampler.is_posterior_sampler else BasePipeline
    pipeline = cls(config['pipeline'], params=config.get('params', None), mpicomm=mpicomm)

    return config_sampler.run(pipeline)


@CurrentMPIComm.enable
def profile_from_args(args=None, mpicomm=None):
    config, args = read_args(args=args, mpicomm=mpicomm, section='profile')
    return profile_from_config(config, mpicomm=mpicomm)


@CurrentMPIComm.enable
def profile_from_config(config, mpicomm=None):
    config = BaseConfig(config)
    if 'profile' not in config:
        raise ConfigError('Provide "profile"')
    config_profiler = ProfilerConfig(config['profile'])

    if 'pipeline' not in config:
        raise ConfigError('Provide pipeline')
    likelihood = LikelihoodPipeline(config['pipeline'], params=config.get('params', None), mpicomm=mpicomm)

    return config_profiler.run(likelihood)


@CurrentMPIComm.enable
def do_from_args(args=None, mpicomm=None):
    config, args = read_args(args=args, mpicomm=mpicomm, section='do')
    return do_from_config(config, mpicomm=mpicomm)


@CurrentMPIComm.enable
def do_from_config(config, mpicomm=None):
    config = BaseConfig(config)
    if 'do' not in config:
        raise ConfigError('Provide "do"')
    config_do = DoConfig(config['do'])

    if 'pipeline' not in config:
        raise ConfigError('Provide pipeline')
    pipeline = BasePipeline(config['pipeline'], params=config.get('params', None), mpicomm=mpicomm)

    config_do.run(pipeline)


@CurrentMPIComm.enable
def summarize_from_args(args=None, mpicomm=None):
    config, args = read_args(args=args, mpicomm=mpicomm, section='summarize')
    return summarize_from_config(config, mpicomm=mpicomm)


@CurrentMPIComm.enable
def summarize_from_config(config, mpicomm=None):
    config = BaseConfig(config)
    if 'summarize' not in config:
        raise ConfigError('Provide "summarize"')
    config_summary = SummaryConfig(config['summarize'])
    return config_summary.run()


@CurrentMPIComm.enable
def emulate_from_args(args=None, mpicomm=None):
    config, args = read_args(args=args, mpicomm=mpicomm, section='emulate')
    return emulate_from_config(config, mpicomm=mpicomm)


@CurrentMPIComm.enable
def emulate_from_config(config, mpicomm=None):
    config = BaseConfig(config)
    if 'emulate' not in config:
        raise ConfigError('Provide "emulate"')
    config_emulator = EmulatorConfig(config['emulate'])

    if 'pipeline' not in config:
        raise ConfigError('Provide pipeline')
    pipeline = BasePipeline(config['pipeline'], params=config.get('params', None), mpicomm=mpicomm)

    return config_emulator.run(pipeline)


@CurrentMPIComm.enable
def install_from_args(args=None, mpicomm=None):
    from .install import InstallerConfig
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config_fn_or_classes', type=str, nargs='*', help='Name of configuration file(s) or classes to install')
    parser.add_argument('--verbose', '-v', type=str, choices=['warning', 'info', 'debug'], default='info', help='Verbosity level')
    parser.add_argument('--update', type=str, nargs='*', default=[], help='List of namespace1.....key=value to update config file')
    parser.add_argument('--exclude', type=str, nargs='*', default=[], help='Name of classes to ignore')
    parser = InstallerConfig.parser(parser)
    args = parser.parse_args(args=args)
    section = 'install'
    if mpicomm.rank == 0:
        print(ascii_art(section))
    setup_logging(args.verbose)
    config_installer = InstallerConfig.from_args(args)
    config_installer.update(exclude=args.exclude)
    for config_fn_or_class in args.config_fn_or_classes:
        if os.path.isfile(config_fn_or_class):
            config = update_config(config_fn_or_class, section=section, update=args.update)
            config[section] = InstallerConfig(config[section]).clone(config_installer)
            install_from_config(config, mpicomm=mpicomm)
        else:
            import_class(config_fn_or_class, install=config_installer)


@CurrentMPIComm.enable
def install_from_config(config, sections=('profile', 'sample', 'emulate'), mpicomm=None):
    from .base import PipelineConfig
    config = BaseConfig(config)

    if 'install' not in config:
        raise ConfigError('Provide "install"')

    from .install import InstallerConfig
    config_installer = InstallerConfig(config['install'])

    if 'pipeline' not in config:
        raise ConfigError('Provide pipeline')

    pipeline = PipelineConfig(config['pipeline'], params=config.get('params', None), mpicomm=mpicomm, install=config_installer).init()
    for section in sections:
        if section in config:
            do = {'profile': ProfilerConfig, 'sample': SamplerConfig, 'emulate': EmulatorConfig}[section](config[section], install=config_installer)
            if section == 'emulate': do.run(BasePipeline(pipeline))

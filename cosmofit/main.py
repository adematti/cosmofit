import sys
import datetime
import argparse

from mpytools import CurrentMPIComm

from ._version import __version__
from .io import BaseConfig, ConfigError
from .base import BasePipeline, LikelihoodPipeline, DoConfig
from .samples import SourceConfig, SummaryConfig
from .utils import setup_logging


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


@CurrentMPIComm.enable
def read_args(args=None, mpicomm=None, parser=None, section='sample'):
    if parser is None:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config_fn', type=str, help='Name of configuration file')
    parser.add_argument('--verbose', '-v', type=str, choices=['warning', 'info', 'debug'], default='info', help='Verbosity level')
    parser.add_argument('--update', nargs='*', type=str, default=[], help='List of namespace1.....key=value to update config file')
    args = parser.parse_args(args=args)
    if mpicomm.rank == 0:
        print(ascii_art(section))
    setup_logging(args.verbose)
    config = BaseConfig(args.config_fn, decode=False)
    for string in args.update:
        keyvalue = string.split('=')
        if len(keyvalue) != 2:
            raise ValueError('Provide updates as namespace1....name.key=value format')
        config.update_from_namespace(keyvalue[0], keyvalue[1], inherit_type=True)
    config.decode()
    config['output'] = config.get('output', './')
    return config, args.config_fn


@CurrentMPIComm.enable
def sample_from_args(args=None, mpicomm=None):
    config, config_fn = read_args(args=args, mpicomm=mpicomm, section='sample')
    return sample_from_config(config, mpicomm=mpicomm)


@CurrentMPIComm.enable
def sample_from_config(config, mpicomm=None):
    from cosmofit.samplers import SamplerConfig
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
    config, config_fn = read_args(args=args, mpicomm=mpicomm, section='profile')
    return profile_from_config(config, mpicomm=mpicomm)


@CurrentMPIComm.enable
def profile_from_config(config, mpicomm=None):
    from cosmofit.profilers import ProfilerConfig
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
    config, config_fn = read_args(args=args, mpicomm=mpicomm, section='do')
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

    params = SourceConfig(config_do['source']).choice(params=pipeline.params)
    pipeline.run(**params)
    config_do.run(pipeline)


@CurrentMPIComm.enable
def summarize_from_args(args=None, mpicomm=None):
    config, config_fn = read_args(args=args, mpicomm=mpicomm, section='summarize')
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
    config, config_fn = read_args(args=args, mpicomm=mpicomm, section='emulate')
    return emulate_from_config(config, mpicomm=mpicomm)


@CurrentMPIComm.enable
def emulate_from_config(config, mpicomm=None):
    from cosmofit.emulators import EmulatorConfig
    config = BaseConfig(config)
    if 'emulate' not in config:
        raise ConfigError('Provide "emulate"')
    config_emulator = EmulatorConfig(config['emulate'])

    if 'pipeline' not in config:
        raise ConfigError('Provide pipeline')
    pipeline = BasePipeline(config['pipeline'], params=config.get('params', None), mpicomm=mpicomm)

    params = SourceConfig(config_emulator['source']).choice(params=pipeline.params)
    pipeline.run(**params)
    return config_emulator.run(pipeline)

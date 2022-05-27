import sys
import datetime
import argparse

from mpytools import CurrentMPIComm

from ._version import __version__
from .base import BaseConfig, PipelineError, BasePipeline, LikelihoodPipeline, RunnerConfig
from .io import FileSystemConfig
from .utils import setup_logging


def ascii_art(section):

    ascii_art = r"""
                                 __ _ _
                                / _(_) |
   ___ ___  ___ _ __ ___   ___ | |_ _| |_
  / __/ _ \/ __| '_ ` _ \ / _ \|  _| | __|
 | (_| (_) \__ \ | | | | | (_) | | | | |_
  \___\___/|___/_| |_| |_|\___/|_| |_|\__|
                                          """ + "\n"
    + "{}\n".format(section)
    + """version: {}               date: {}\n""".format(__version__, datetime.date.today())
    return ascii_art


@CurrentMPIComm.enable
def read_args(args=None, mpicomm=None, parser=None, section='sample'):
    if parser is None:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config_fn', action='store', type=str, help='Name of configuration file')
    parser.add_argument('--verbose', '-v', action='verbosity', type=str, choices=['warning', 'info', 'debug'], default='info', help='Verbosity level')
    parser.add_argument('--update', nargs='*', type=str, help='List of namespace1....name.key=value to update config file')
    args = parser.parse_args(args=args)
    if mpicomm.rank == 0:
        print(ascii_art(section))
    setup_logging(args.verbose)
    config = BaseConfig(args.config_fn)
    for string in args.update:
        keyvalue = string.split('=')
        if len(keyvalue) != 2:
            raise ValueError('Provide updates as namespace1....name.key=value format')
        config.update_from_namespace(keyvalue[0], keyvalue[1], inherit_type=True)
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
        raise PipelineError('Provide "sample"')
    config_sampler = SamplerConfig(config['sample'])

    if 'pipeline' not in config:
        raise PipelineError('Provide pipeline')
    likelihood = LikelihoodPipeline(config['pipeline'], params=config.get('params', None))

    sampler = config_sampler.init(likelihood, mpicomm=mpicomm)
    check = config_sampler.get('check', {})
    run_check = bool(check)
    if isinstance(check, bool): check = {}
    min_iterations = config_sampler['run'].get('min_iterations', 0)
    max_iterations = config_sampler['run'].get('max_iterations', int(1e5) if run_check else sys.maxsize)
    check_every = config_sampler['run'].get('check_every', 200)

    run_kwargs = {}
    for name in ['thin_by']:
        if name in config_sampler['run']: run_kwargs = config_sampler['run'].get(name)

    filesystem = config.get('filesystem', None)
    chains_fn = config_sampler.get('save_fn', None)
    if filesystem is not None and chains_fn is None:
        chains_fn = 'chain'

    if isinstance(filesystem, str):
        filesystem = {'output': filesystem}
    filesystem = FileSystemConfig(filesystem).init()[1]

    if chains_fn is not None:
        if isinstance(chains_fn, str):
            chains_fn = [filesystem(chains_fn, i=i) for i in range(sampler.nchains)]
        else:
            if len(chains_fn) != sampler.nchains:
                raise PipelineError('Provide {:d} chain file names'.format(sampler.nchains))
            chains_fn = [filesystem(chain_fn) for chain_fn in chains_fn]

    count_iterations = 0
    is_converged = False
    while not is_converged:
        niter = min(max_iterations - count_iterations, check_every)
        count_iterations += niter
        sampler.run(niterations=niter, **run_kwargs)
        if chains_fn is not None:
            for ichain in range(sampler.nchains):
                sampler.chains[ichain].save(chains_fn[ichain])
        is_converged = sampler.check(**check) if run_check else False
        if count_iterations < min_iterations:
            is_converged = False
        if count_iterations >= max_iterations:
            is_converged = True
    return sampler


@CurrentMPIComm.enable
def profile_from_args(args=None, mpicomm=None):
    config, config_fn = read_args(args=args, mpicomm=mpicomm, section='profile')
    return sample_from_config(config, mpicomm=mpicomm)


@CurrentMPIComm.enable
def profile_from_config(config, mpicomm=None):
    from cosmofit.profilers import ProfilerConfig
    config = BaseConfig(config)
    if 'profile' not in config:
        raise PipelineError('Provide "profile"')
    config_profiler = ProfilerConfig(config['profile'])

    if 'pipeline' not in config:
        raise PipelineError('Provide pipeline')
    likelihood = LikelihoodPipeline(config['pipeline'], params=config.get('params', None))

    profiler = config_profiler.init(likelihood, mpicomm=mpicomm)

    filesystem = config.get('filesystem', None)
    profiles_fn = config_profiler.get('save_fn', None)
    if filesystem is not None and profiles_fn is None:
        profiles_fn = 'profiles'

    if isinstance(filesystem, str):
        filesystem = {'output': filesystem}
    filesystem = FileSystemConfig(filesystem).init()[1]

    if profiles_fn is not None:
        profiles_fn = filesystem(profiles_fn)

    profiler.run(**config_profiler['run'])
    if profiles_fn is not None:
        profiler.profiles.save(profiles_fn)
    return profiler


@CurrentMPIComm.enable
def run_from_args(args=None, mpicomm=None):
    config, config_fn = read_args(args=args, mpicomm=mpicomm, section='run')
    return run_from_config(config, mpicomm=mpicomm)


@CurrentMPIComm.enable
def run_from_config(config, mpicomm=None):
    config = BaseConfig(config)
    if 'run' not in config:
        raise PipelineError('Provide "run"')
    config_runner = RunnerConfig(config['run'])

    filesystem = config.get('filesystem', None)
    if isinstance(filesystem, str):
        filesystem = {'input': filesystem, 'output': filesystem}
    filesystem_input, filesystem_output = FileSystemConfig(filesystem).init()

    if 'pipeline' not in config:
        raise PipelineError('Provide pipeline')
    pipeline = BasePipeline(config['pipeline'], params=config.get('params', None))
    params = config_runner.params(params=pipeline.params, filesystem=filesystem_input)
    pipeline.run(**params)
    config_runner.run(pipeline, filesystem=filesystem_output)

import sys
import datetime
import argparse

from mpytools import CurrentMPIComm

from ._version import __version__
from .base import Decoder, PipelineError, LikelihoodPipeline, format_clsdict, FileSystem
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
def read_args(args=None, mpicomm=None, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config_fn', action='store', type=str, help='Name of configuration file')
    parser.add_argument('--verbose', '-v', action='verbosity', type=str, choices=['warning', 'info', 'debug'], default='info', help='Verbosity level')
    parser.add_argument('--update', nargs='*', type=str, help='List of namespace1....name.key=value to update config file')
    args = parser.parse_args(args=args)
    if mpicomm.rank == 0:
        print(ascii_art('sample'))
    setup_logging(args.verbose)
    config = Decoder(args.config_fn)
    for string in args.update:
        keyvalue = string.split('=')
        if len(keyvalue) != 2:
            raise ValueError('Provide updates as namespace1....name.key=value format')
        config.update_from_namespace(keyvalue[0], keyvalue[1], inherit_type=True)
    config['output'] = config.get('output', './')
    return config, args.config_fn


@CurrentMPIComm.enable
def sample_from_args(args=None, mpicomm=None):
    config, config_fn = read_args(args=args, mpicomm=mpicomm)
    return sample_from_config(config, mpicomm=mpicomm)


@CurrentMPIComm.enable
def sample_from_config(config, mpicomm=None):
    from cosmofit.samplers import BaseSampler
    config = Decoder(config)
    if 'sampler' not in config:
        raise PipelineError('Provide sampler')
    cls, clsdict = format_clsdict(config['sampler'], registry=BaseSampler._registry)

    if 'pipeline' not in config:
        raise PipelineError('Provide pipeline')
    likelihood = LikelihoodPipeline(config['pipeline'], params=config.get('params', None))

    diagnostics = clsdict.pop('diagnostics', None)
    min_iterations = clsdict.pop('min_iterations', 0)
    max_iterations = clsdict.pop('max_iterations', int(1e5) if diagnostics is None else sys.maxint)
    check_every = clsdict.pop('check_every', 200)

    run_kwargs = {}
    for name in ['thin_by']:
        if name in clsdict: run_kwargs = clsdict.pop(name)

    filesystem = None
    output = config.get('output', None)
    if output is not None:
        filesystem = FileSystem(output)

    chains_fn = clsdict.pop('chains_fn', None)
    if filesystem is not None and chains_fn is None:
        chains_fn = 'chain'

    sampler = cls(likelihood, **clsdict, mpicomm=mpicomm)
    if chains_fn is not None:
        if filesystem is None:
            filesystem = FileSystem('./')
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
            for ichain in sampler.nchains:
                sampler.chains[ichain].save(chains_fn[ichain])
        is_converged = sampler.diagnose(**diagnostics)
        if count_iterations < min_iterations:
            is_converged = False
        if count_iterations > max_iterations:
            is_converged = True
    return sampler

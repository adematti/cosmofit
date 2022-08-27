"""Classes and functions dedicated to handling samples drawn from likelihood."""

import glob

from .chain import Chain
from .profile import ParameterValues, Profiles
from . import diagnostics

__all__ = ['Chain', 'Profiles', 'diagnostics']


from cosmofit.parameter import ParameterCollection
from cosmofit.io import BaseConfig, ConfigError
from cosmofit import utils


def load_samples(source='profiles', fn=None, choice=None, burnin=None):
    if not utils.is_sequence(fn): fn = [fn]
    fns = []
    for ff in fn: fns += glob.glob(ff)
    if source == 'profiles':
        profiles = [fn if isinstance(fn, Profiles) else Profiles.load(fn) for fn in fns]
        if choice == 'argmax':
            profiles = Profiles.concatenate(profiles)
            argmax = profiles.bestfit.logposterior.argmax()
            return {str(param): profiles.bestfit[param][argmax] for param in profiles.bestfit.params()}
        return profiles
    if source == 'chain':
        chains = [fn if isinstance(fn, Chain) else Chain.load(fn) for fn in fns]
        if burnin is not None:
            chains = [chain.remove_burnin(burnin) for chain in chains]
        if choice == 'argmax':
            chain = Chain.concatenate(chains)
            argmax = chain.logposterior.argmax()
            return {str(param): chain[param].flat[argmax] for param in chain.params()}
        return chains
    raise ConfigError('source must be one of ["profiles", "chain"]')


class SourceConfig(BaseConfig):

    def choice(self, params=None):
        params = {param.name: param.value for param in ParameterCollection(params)}
        for source, value in self.items():
            if isinstance(value, str):
                value = {'fn': value}
            if source == 'params':
                params.update({param.name: param.value for param in ParameterCollection(value) if param.name in params})
            else:
                params.update({param: value for param, value in load_samples(source=source, **{'choice': 'argmax', **value}).items() if param in params})
        return params


class SummaryConfig(BaseConfig):

    _allowed_sources = ['profiles', 'chain']

    def __init__(self, *args, **kwargs):
        super(SummaryConfig, self).__init__(*args, **kwargs)
        from cosmofit.io import ConfigError
        sources = []
        for source in self._allowed_sources:
            if self.get(source, None) is not None:
                sources.append(source)
        if len(sources) == 0 or len(sources) > 1:
            raise ConfigError('Provide one source (one of {}), found {}'.format(self._allowed_sources, sources))
        value = self[sources[0]]
        if isinstance(value, dict):
            self.source = load_samples(source=sources[0], **value)
        else:
            self.source = load_samples(source=sources[0], fn=value)

    def run(self):
        from . import plotting
        for section, options in self.items():
            if section == 'stats':
                if not utils.is_sequence(options):
                    options = [options]
                for source, option in zip(self.source, options):
                    if isinstance(option, str):
                        option = {'fn': option}
                    source.to_stats(**option)
            elif section not in self._allowed_sources:
                if section == 'plot_triangle':
                    sources = [Chain.concatenate(self.source)]
                elif section == 'plot_profile_comparison':
                    nsources = len(self.source)
                    if nsources % 2:
                        raise ConfigError('Number of profiles must be even (profiles + profiles_ref) for {}'.format(section))
                    sources = [self.source[:nsources // 2], self.source[nsources // 2:]]
                else:
                    sources = [self.source]
                func = getattr(plotting, section)
                if isinstance(options, str):
                    options = {'fn': options}
                func(*sources, **options)

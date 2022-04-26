from mpytools import CurrentMPIComm

from .utils import BaseClass
from .parameter import ParameterCollection


class BaseCalculator(BaseClass):

    is_vectorized = False

    def get_output_names(self):
        return []

    def get_input_names(self, outputs=None):
        if outputs is None:
            outputs = self.get_output_names()
        return outputs

    def get_output(self):
        output = {}
        for name in self.output_names:
            output[name] = self.get_input(name)
        return output


class BaseLikelihood(BaseCalculator):

    is_vectorized = False

    def __init__(self, *args, **kwargs):
        super(BaseLikelihood, self).__init__(*args, **kwargs)
        self.name = 'base'

    def get_output_names(self):
        return ['loglikelihood_{}'.format(self.name)]

    def get_input_names(self, outputs=None):
        return self.parameters

    def get_output(self):
        raise NotImplementedError


class PipelineError(Exception):

    pass


class likelihoodPipeline(BaseClass):

    @CurrentMPIComm.enable
    def __init__(self, *calculators, parameters=None, mpicomm=None):
        self.mpicomm = mpicomm
        parameters = ParameterCollection(parameters)
        calculators = list(calculators)
        self.likelihoods, self.theories = [], []
        for calculator in calculators:
            if isinstance(calculator, BaseLikelihood):
                self.likelihoods.append(calculator)
            else:
                self.theories.append(calculator)
        dependencies = {}
        for calculator in self.likelihoods:
            for name in calculator.get_input_names():
                dependencies[name] = dependencies.get(name, []) + [calculator]

        def get_base_inputs(dependencies, theories):
            dependencies = dependencies.copy()
            keep_iterating = True
            while keep_iterating:
                keep_iterating = False
                for name in list(dependencies.keys()):
                    providers = []
                    for theory in theories:
                        if name in theory.get_output_names():
                            providers.append(theory)
                    if len(providers) > 1:
                        raise PipelineError('{} required by {} is provided by several calculators: {}'.format(name, dependencies[name], providers))
                    elif len(providers) == 1:
                        users = dependencies.pop(name)
                        if theory in users:
                            raise PipelineError('Circular dependency for {}, input/output by {}'.format(name, users))
                        for name in theory.get_input_names():
                            dependencies[name] = dependencies.get(name, []) + [theory] + users
                        keep_iterating = True

        def get_parameters(dependencies):
            toret, remaining = [], {}
            for name in dependencies:
                parameters, providers = [], []
                for calculator in dependencies[name]:
                    if name in calculator.parameters and calculator.parameters[name] not in parameters:
                        parameters.append(calculator.parameters[name])
                        providers.append(calculator)
                if len(providers) > 1:
                    raise PipelineError('Parameter {} required by {} is required by several calculators with different specs: {}'.format(name, dependencies[name], providers))
                elif len(providers) == 1:
                    toret.append(parameters[0])
                else:
                    remaining[name] = dependencies[name]
            return toret

        dependencies = get_base_inputs(dependencies, self.theories)
        for theory in self.theories:
            if theory not in concatenate_list(dependencies.values()):
                raise PipelineError('Theory {} is useless, please remove it from the list'.format(theory))
        self.parameters, self.dependencies = get_parameters(dependencies)
        if dependencies:
            base_theories = get_base_theories()
            dependencies = get_base_inputs(dependencies, self.theories + base_theories)
            for theory in base_theories:
                if theory in concatenate_list(dependencies.values()):
                    self.theories = [theory] + self.theories
            self.parameters, self.dependencies = get_parameters(dependencies)
            msg = '\n'.join(['- {} required by {} is neither a parameter nor is provided by any known calculator'.format(name, dependencies[name]) for name in dependencies])
            if msg:
                raise PipelineError('\n{}\n'.format(msg))

        def loglikelihood(self, values):
            pass

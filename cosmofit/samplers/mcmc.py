import itertools

import numpy as np
from numpy import linalg
from numpy.linalg import LinAlgError
from scipy.stats import special_ortho_group

from cosmofit.samples import Chain, load_source
from .base import BasePosteriorSampler


class State(object):

    _attrs = ['coords', 'log_prob', 'weight']

    def __init__(self, *args, **kwargs):
        attrs = {k: v for k, v in zip(self._attrs, args)}
        attrs.update(kwargs)
        self.__dict__.update(attrs)


class MHSampler(object):

    """We follow emcee interface."""

    def __init__(self, ndim, log_prob_fn, propose, nsteps_drag=None, max_tries=1000, rng=None):
        self.ndim = ndim
        self.log_prob_fn = log_prob_fn
        self.propose = propose
        self.nsteps_drag = nsteps_drag
        if nsteps_drag is not None:
            self.nsteps_drag = int(nsteps_drag)
            if self.nsteps_drag < 2:
                raise ValueError('With dragging nsteps_drag must be >= 2')
            if len(self.propose) != 2:
                raise ValueError('With dragging give list of two propose methods, slow and fast')
        self.max_tries = int(max_tries)
        self.rng = rng or np.random.RandomState()
        self.states = []

    def sample(self, start, iterations=1, thin_by=1):
        self.state = State(start, self.log_prob_fn(start), 1.)
        for i in range(iterations):
            accept = False
            for itry in range(self.max_tries):
                coords = self.state.coords.copy()
                if self.nsteps_drag:  # dragging
                    current_coords_start = coords
                    sum_log_prob_start = current_log_prob_start = self.state.log_prob
                    current_coords_end = current_coords_start + self.propose[0]()  # slow
                    sum_log_prob_end = current_log_prob_end = self.log_prob_fn(trial_end)
                    if current_log_prob_end == -np.inf:
                        self.state.weight += 1
                        continue
                    n_average = 1 + self.nsteps_drag
                    for i in range(1, n_average):
                        proposal_coords_start = current_coords_start + self.propose[1]()  # fast
                        proposal_log_prob_start = self.log_prob_fn(proposal_coords_start)
                        if proposal_log_prob_start != -np.inf:
                            proposal_coords_end = current_coords_end + self.propose[1]()  # fast
                            proposal_log_prob_end = self.log_prob_fn(proposal_coords_end)
                            if log_prob_end != -np.inf:
                                # Create the interpolated probability and perform a Metropolis test
                                frac = i_step / n_average
                                proposal_log_prob_interp = (1 - frac) * proposal_log_prob_start + frac * proposal_log_prob_end
                                current_log_prob_interp = (1 - frac) * current_log_prob_start + frac * current_log_prob_end
                                if self._mh_accept(proposal_log_prob_interp, current_log_prob_interp):
                                    # The dragging step was accepted, do the drag
                                    current_coords_start = proposal_coords_start
                                    current_log_prob_start = proposal_log_prob_start
                                    current_coords_end = proposal_coords_end
                                    current_log_prob_end = proposal_log_prob_end
                        sum_log_prob_start += current_log_prob_start
                        sum_log_prob_end += current_log_prob_end
                    accept = self._mh_accept(sum_log_prob_end / n_average, sum_log_prob_start / n_average)
                    coords, log_prob = current_coords_end, current_log_prob_end
                else:  # standard MH
                    coords += self.propose()
                    log_prob = self.log_prob_fn(coords)
                    accept = self._mh_accept(log_prob, self.state.log_prob)
                if accept:
                    if i % thin_by == 0:
                        self.states.append(self.state)
                    self.state = State(coords, log_prob, 1.)
                    break
                else:
                     self.state.weight += 1
            if not accept:
                raise ValueError('Could not find finite log posterior after {:d} tries'.format(self.max_tries))
            yield self.state

    def _mh_accept(self, log_prob_trial, log_prob_current):
        if log_prob_trial == -np.inf:
            return False
        if log_prob_trial > log_prob_current:
            return True
        return self.rng.standard_exponential() > (log_prob_current - log_prob_trial)

    def get_chain(self):
        return np.array([state.coords for state in self.states])

    def get_weight(self):
        return np.array([state.weight for state in self.states])

    def get_log_prob(self):
        return np.array([state.log_prob for state in self.states])

    def get_acceptance_rate(self):
        return len(states) / self.get_weight().sum()

    def reset(self):
        self.states = []


class IndexCycler(object):

    def __init__(self, ndim, rng):
        self.ndim = ndim
        self.loop_index = -1
        self.rng = rng


class CyclicIndexRandomizer(IndexCycler):

    def __init__(self, ndim, rng):
        if np.ndim(ndim) == 0:
            self.sorted_indices = list(range(ndim))
        else:
            self.sorted_indices = ndim
            ndim = len(ndim)
        super(CyclicIndexRandomizer, self).__init__(ndim, rng)
        if self.ndim <= 2:
            self.indices = list(range(ndim))

    def next(self):
        """Get the next random index, or alternate for two or less."""
        self.loop_index = (self.loop_index + 1) % self.ndim
        if self.loop_index == 0 and self.ndim > 2:
            self.indices = self.rng.permutation(self.sorted_indices)
        return self.indices[self.loop_index]


class SOSampler(IndexCycler):

    def __call__(self):
        return self.sample()

    def sample(self):
        """Propose a random n-dimension vector."""
        if self.ndim == 1:
            return np.array([self.rng.choice([-1, 1]) * self.sample_r()])
        self.loop_index = (self.loop_index + 1) % self.ndim
        if self.loop_index == 0:
            self.rotmat = special_ortho_group.rvs(self.ndim, random_state=self.rng)
        return self.rotmat[:, self.loop_index] * self.sample_r()

    def sample_r(self):
        """
        Radial proposal. A mixture of an exponential and 2D Gaussian radial
        proposal (to make wider tails and more mass near zero, so more robust to scale
        misestimation).
        """
        if self.rng.uniform() < 0.33:
            return self.rng.standard_exponential()
        return np.sqrt(self.rng.chisquare(min(self.ndim, 2)))


class BlockProposer(object):

    def __init__(self, blocks, oversample_factors=None,
                 last_slow_block_index=None, proposal_scale=2.4, rng=None):
        """
        Proposal density for fast and slow parameters, where parameters are
        grouped into blocks which are changed at the same time.

        Parameters
        ----------
        blocks : array
            Number of parameters in each block, with blocks sorted by ascending speed.

        oversample_factors : list
            List of *int* oversampling factors *per parameter*,
            i.e. a factor of n for a block of dimension d would mean n*d jumps for that
            block per full cycle, whereas a factor of 1 for all blocks (default) means
            that all *directions* are treated equally (but the proposals are still
            block-wise).

        last_slow_block_index : int
            Index of the last block considered slow.
            By default, all blocks are considered slow.

        proposal_scale : float
            Overall scale for the proposal.
        """
        self.rng = rng or np.random.RandomState()
        self.proposal_scale = float(proposal_scale)
        self.blocks = np.array(blocks, dtype='i4')
        if np.any(blocks != self.blocks):
            raise ValueError('blocks must be integer! Got {}.'.format(blocks))
        if oversample_factors is None:
            self.oversample_factors = np.ones(len(blocks), dtype='i4')
        else:
            if len(oversample_factors) != len(self.blocks):
                raise ValueError('List of oversample_factors has a different length than list of blocks: {:d} vs {:d}'.format(len(oversample_factors), len(self.blocks)))
            self.oversample_factors = np.array(oversample_factors, dtype='i4')
            if np.any(oversample_factors != self.oversample_factors):
                raise ValueError('oversample_factors must be integer! Got {}.'.format(oversample_factors))
        # Binary fast-slow split
        self.last_slow_block_index = last_slow_block_index
        if self.last_slow_block_index is None:
            self.last_slow_block_index = len(blocks) - 1
        else:
            if self.last_slow_block_index > len(blocks) - 1:
                raise ValueError('The index given for the last slow block, {:d}, is not valid: there are only {:d} blocks'.format(self.last_slow_block_index, len(self.blocks)))
        n_all = sum(self.blocks)
        n_slow = sum(self.blocks[:1 + self.last_slow_block_index])
        self.nsamples_slow = self.nsamples_fast = 0
        # Starting index of each block
        self.block_starts = np.insert(np.cumsum(self.blocks), 0, 0)
        # Prepare indices for the cycler, repeated if there is oversampling
        indices_repeated = np.concatenate([np.repeat(np.arange(b) + s, o) for b, s, o in zip(self.blocks, self.block_starts, self.oversample_factors)])
        self.param_block_indices = np.concatenate([np.full(b, ib, dtype='i4') for ib, b in enumerate(self.blocks)])
        # Creating the blocked proposers
        self.proposer = [SOSampler(b, self.rng) for b in self.blocks]
        # Parameter cyclers, cycling over the j's
        self.param_cycler = CyclicIndexRandomizer(indices_repeated, self.rng)
        # These ones are used by fast dragging only
        self.param_cycler_slow = CyclicIndexRandomizer(n_slow, self.rng)
        self.param_cycler_fast = CyclicIndexRandomizer(n_all - n_slow, self.rng)

    @property
    def ndim(self):
        return len(self.param_block_indices)

    def __call__(self, params=None):
        current_iblock = self.param_block_indices[self.param_cycler.next()]
        if current_iblock <= self.last_slow_block_index:
            self.nsamples_slow += 1
        else:
            self.nsamples_fast += 1
        return self._get_block_proposal(current_iblock, params)

    def slow(self, params=None):
        current_iblock_slow = self.param_block_indices[self.param_cycler_slow.next()]
        self.nsamples_slow += 1
        return self.get_block_proposal(current_iblock_slow, params)

    def fast(self, params=None):
        current_iblock_fast = self.param_block_indices[self.param_cycler_slow.ndim + self.param_cycler_fast.next()]
        self.nsamples_fast += 1
        return self._get_block_proposal(current_iblock_fast, params)

    def _get_block_proposal(self, iblock, params=None):
        if params is None:
            params = np.zeros(self.ndim, dtype='f8')
        else:
            params = np.array(params)
        params[self.block_starts[iblock]:] += self.transform[iblock].dot(self.proposer[iblock]() * self.proposal_scale)
        return params

    def set_covariance(self, matrix):
        """
        Take covariance of sampled parameters (matrix), and construct orthonormal
        parameters where orthonormal parameters are grouped in blocks by speed, so changes
        in the slowest block changes slow and fast parameters, but changes in the fastest
        block only changes fast parameters.
        """
        matrix = np.array(matrix)
        if matrix.shape[0] != self.ndim:
            raise ValueError('The covariance matrix does not have the correct dimension: '
                             'it is {:d}, but it should be {:d}.'.format(matrix.shape[0], self.ndim))
        if not (np.allclose(matrix.T, matrix) and np.all(np.linalg.eigvals(matrix) > 0)):
            raise linalg.LinAlgError('The given covmat is not a positive-definite, symmetric square matrix.')
        L = linalg.cholesky(matrix)
        # Store the basis as transformation matrices
        self.transform = []
        for block_start, bp in zip(self.block_starts, self.proposer):
            block_end = block_start + bp.ndim
            self.transform += [L[block_start:, block_start:block_end]]
        return True


def _format_blocks(blocks, params):
    blocks, oversample_factors = [b[1] for b in blocks], [b[0] for b in blocks]
    blocks = [[params[name] for name in block if name in params] for block in blocks]
    blocks, oversample_factors = [b for b in blocks if b], [s for s, b in zip(oversample_factors, blocks) in b]
    params_in_blocks = set(iterations.chain(*blocks))
    if params_in_blocks != set(params):
        raise ValueError('Missing sampled parameters in provided blocks: {}'.format(set(params) - params_in_blocks))
    argsort = np.argsort(oversample_factors)
    return [b[i] for i in argsort], [oversample_factors[i] for i in argsort]


class MCMCSampler(BasePosteriorSampler):

    def __init__(self, *args, blocks=None, covariance=None, proposal_scale=2.4, learn=True, drag=False, **kwargs):
        super(MCMCSampler, self).__init__(*args, **kwargs)
        if blocks is None:
            blocks, oversample_factors = self.likelihood.block_params(params=self.varied_params, nblocks=2 if drag else None)
        else:
            blocks, oversample_factors = _format_blocks(blocks, self.varied_params)
        last_slow_block_index = nsteps_drag = None
        if drag:
            if len(blocks) == 1:
                drag = False
                if self.mpicomm.rank == 0:
                    self.log_warning('Dragging disabled: not possible if there is only one block.')
            if max(oversample_factors) / min(oversample_factors) < 2:
                drag = False
                if self.mpicomm.rank == 0:
                    self.log_warning('Dragging disabled: speed ratios between blocks < 2.')
            for first_fast_block_index, speed in enumerate(oversample_factors):
                if speed != 1: break
            last_slow_block_index = first_fast_block_index - 1
            n_slow = sum(len(b) for b in blocks[:first_fast_block_index])
            n_fast = len(self.varied_params) - n_slow
            nsteps_drag = int(oversample_factors[first_fast_block_index] * n_fast / n_slow + 0.5)
            if self.mpicomm.rank == 0:
                self.log_info('Dragging:')
                self.log_info('1 step: {}'.format(blocks[:first_fast_block_index]))
                self.log_info('{:d} steps: {}'.format(nsteps_drag, blocks[first_fast_block_index:]))
        elif np.any(oversample_factors > 1):
            if self.mpicomm.rank == 0:
                self.log_info('Oversampling with factors:')
                for s, b in zip(oversample_factors, blocks):
                    self.log_info('{:d}: {}'.format(s, b))

        self.varied_params = self.varied_params.sort(itertools.chain(*blocks))
        self.proposer = BlockProposer(blocks=[len(b) for b in blocks], oversample_factors=oversample_factors, last_slow_block_index=last_slow_block_index, proposal_scale=proposal_scale, rng=self.rng)
        self.learn = bool(learn)
        self.learn_check = None
        burnin = 0.5
        if isinstance(learn, dict):
            self.learn = True
            self.learn_check = dict(learn)
            burnin = self.learn_check['burnin'] = self.learn_check.get('burnin', burnin)
        covariance = load_source(covariance, cov=True, params=self.varied_params, burnin=burnin)
        self.proposer.set_covariance(covariance)
        self.learn_diagnostics = {}
        propose = [self.proposer.slow, self.proposer.fast] if drag else self.proposer
        self.sampler = MHSampler(len(self.varied_params), self.logposterior, propose=propose, nsteps_drag=nsteps_drag, max_tries=self.max_tries, rng=self.rng)

    def _prepare(self):
        covariance = None
        if self.learn and self.mpicomm.bcast(self.chains[0] is not None, root=0):
            learn = self.learn_check is None
            burnin = 0.5
            if not learn:
                burnin = self.learn_check.get('burnin', burnin)
                learn = self.check(**self.learn_check, diagnostics=self.learn_diagnostics, quiet=True)
            if learn and self.mpicomm.rank == 0:
                chain = Chain.concatenate([chain.remove_burnin(burnin) for chain in self.chains])
                covariance = chain.cov(params=self.varied_params)
        covariance = self.mpicomm.bcast(covariance, root=0)
        if covariance is not None:
            try:
                self.proposer.set_covariance(covariance)
            except LinAlgError:
                self.log_info('New proposal covariance is ill-conditioned, skipping update.')
            else:
                self.log_info('Updating proposal covariance.')

    def _run_one(self, start, niterations=300, thin_by=1):
        if thin_by == 'auto':
            thin_by = int(sum(b * s for b, s in zip(self.proposer.blocks, self.proposer.oversample_factors)) / len(self.varied_params))
        for _ in self.sampler.sample(start=np.ravel(start), iterations=niterations, thin_by=thin_by):
            pass
        chain = self.sampler.get_chain()
        data = [chain[..., iparam] for iparam, param in enumerate(self.varied_params)] + [self.sampler.get_log_prob()]
        self.sampler.reset()
        return Chain(data=data, params=self.varied_params + ['logposterior'])

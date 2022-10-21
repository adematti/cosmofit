Samplers
========

Typically, samplers are instantiated this way:

.. code-block:: yaml

  sample:
    # Specify the sampler class
    class: cosmofit.samplers.EmceeSampler
    init:
      # Any sampler option
      chains: 4  # number of chains or paths to existing chains to resume from
      seed: None  # random seed
      ref_scale: 1.  # rescale all parameter reference distribution (from which they are initially sampled from) by this factor
      max_tries: 1000  # maximum number of calls to get finite posterior
    run:
      check_every: 200  # save samples every 200 iterations
      min_iterations: 100  # minimum number of iterations to run (useful if convergence criteria below are satisfied by chance at the beginning of the run)
      max_iterations: sys.maxsize  # maximum number of iterations to run
      check:
        # Fraction of samples to remove for convergence tests
        burnin: 0.5
        # For how many "checks" the criteria must be fulfilled for the inference to stop
        stable_over: 2
        # Gelman-Rubin criterion (on eigenvalues of the parameter covariance matrix) < 0.03
        max_eigen_gr: 0.03
        # Gelman-Rubin criterion (on parameter covariance matrix diagonal) < 0.03
        max_diag_gr: 0.03
        # Gelman-Rubin criterion on variance of nsigmas_cl_diag_gr = 1 interval limits < 0.03
        max_cl_diag_gr: 0.03
        nsigmas_cl_diag_gr: 1
        # Minimal number of iterations over integrated auto-correlation time (~ # of independent samples)
        min_iterations_over_iact: 1000
        reliable_iterations_over_iact: 50  # after how many samples is auto-correlation time estimation reliable
        # All max_* have a min_* counterpart, and vice-versa
    save: 'chain_*.npy'  # where to save chains


Note however that for nested samplers (dynesty, polychord, "check" options are not the same).
Sampling can be interrupted anytime, and resumed by providing the path to the saved chains in "chains" argument of "init".

One will typically run sampling ``nchains * nprocs_per_chain + 1`` processes, with ``nchains >= 1`` the number of chains and ``nprocs_per_chain >= 1``
the number of processes per chain --- plus 1 root process to distribute the work.

In the following we present samplers' default options (in addition to those presented above).

.. include:: samplers_configs.rst

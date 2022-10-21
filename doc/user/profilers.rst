Profilers
=========

Typically, profilers are instantiated this way:

.. code-block:: yaml

  profile:
    # Specify the profiler class
    class: cosmofit.samplers.MinuitSampler
    init:
      # Any profiler option
      profiles: # optionally, profiles to resume from (e.g. to add 2D contours to)
      seed: None  # random seed
      max_tries: 1000  # maximum number of calls to get finite posterior
      # Normalize parameter by their variance, useful if parameters are of different orders of magnitude
      transform: False
      # If transform is True, covariance to normalize parameters
      # Can be previous samples ({fn: chain.npy, burnin: 0.5}), or profiles (containing covariance matrix)
      # If variance for a given parameter is not provided, use parameter 'proposal' squared
      covariance:
      ref_scale: 1.  # rescale all parameter reference distribution (from which they are initially sampled from) by this factor
    maximize:
      # Number of optimization runs, starting from independent points
      # Defaults to the number of MPI processes - 1
      niterations: None
    save: profiles.npy  # where to save profiles

One will typically run several independent likelihood maximizations in parallel, on number of MPI processes - 1 ranks (1 if single process),
to make sure the global maximum is found.
minuit profiler also allows to compute parameter's interval (:math:`\Delta \chi^{2} = 1`), 1D profiles and 2D contours; these calculations
can be performed in parallel, for several parameters.
In the following we present profilers' default options (in addition to those presented above).

.. include:: profilers_configs.rst

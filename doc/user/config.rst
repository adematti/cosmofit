.. _user-config:


Code structure
==============
**cosmofit** is structured the following way:

* a pipeline: compute things given some parameters
* emulators: to (optionally) emulate the pipeline
* profilers: find likelihood/posterior best fit, 1D profiles and 2D contours
* samplers: sample the posterior


**cosmofit** can be run from Python directly, as examplified in `notebooks <https://github.com/adematti/cosmofit/blob/main/nb>`_.
This makes it easy to query and plot any quantity calculated by the pipeline for any given set of parameters, which is also helpful for debugging purposes.
Yet, it will usually be most convienent to work with configuration files.


Configuration file
==================
Below is an example configuration file.

.. code-block:: yaml

  # First, let us define our pipeline, i.e. a sequence of calculators
  # (named like, theory, pt, cosmo here)
  pipeline:
    # We open the namespace "QSO": all parameters will be automatically prefixed by "QSO"
    # (except if those are defined in higher-level namespace)
    QSO:
      like:  # Calculator name, which is purely arbitrary
        # We need to specify where our likelihood "like" is defined, in the format module.Class --* it can be anywhere!
        # We could also specify a pythonpath: location where to search for the said module
        class: cosmofit.likelihoods.galaxy_clustering.PowerSpectrumMultipolesLikelihood
        init: # Whatever parameters used for initialization of the likelihood class
          # Note the f'' syntax: data_dir is replaced by its value in the configuration file (scroll down!)
          data: f'{data_dir}/data.npy'  # data path
          covariance: f'{data_dir}/mock_*.npy'  # list of mocks to use for on-the-fly covariance computation
          klim:  # k-limits
            0: [0.02, 0.2]
            2: [0.02, 0.2]
            4: [0.02, 0.2]
          kstep: 0.01  # k-binning
          zeff: 0.5
          fiducial: DESI  # fiducial cosmology
        # Since loading all mocks take a couple of seconds, save the likelihood to disk...
        save: f'{output_dir}/QSO.like.npy'
        # ... it will be directly reloaded by specifying load: True
        #load: True
        # This calculator has a method "plot" (to plot theory vs. data power spectrum)
        # Give arguments there: it will be called when running 'cosmofit do config.yaml'
        plot: f'{output_dir}/power.png'
      theory:
        class: cosmofit.theories.galaxy_clustering.LPTMomentsTracerPowerSpectrumMultipoles
        # Some parameters are attached to this calculator
        # Those are already specified in the yaml file in cosmofit/theories/galaxy_clustering directory,
        # but we can update them here:
        params:
          b1:
            # Reset b1 prior as a normal distribution of mean 1. and standard deviation 3.
            prior:
              dist: norm
              loc: 1.
              scale: 3.
          # Look at the wildcard syntax, which captures all alpha0, alpha2, etc. parameters
          alpha*:
            # '.auto' is a keyword, meaning this parameter is analytically solved for
            # '.auto' is equivalent to '.best' when running likelihood profiling (parameter is adjusted to minimize chi2 at each step)
            # '.auto' is equivalent to '.marg' when sampling the posterior (parameter is marginalized over)
            derived: '.auto'
          sn*:
            derived: '.auto'
      pt:
        class: cosmofit.theories.galaxy_clustering.LPTPowerSpectrumMultipoles
        #load: f'{output_dir}/emulator.npy'
        # This calculator is computationally expensive, so we compute an emulator for it; to do so, run e.g.:
        # mpiexec -np 20 cosmofit emulate config.yaml
        # Then, uncomment the 'load' line above
        emulator:
          init:
            # Specify the derivative order for each parameter
            # By default ('*'), 1, and for q's (qpar and qper), second order
            order: {'*': 1, 'q*': 2}
          save: f'{output_dir}/emulator.npy'
      shapefit:
        # Shapefit parameterization
        class: cosmofit.theories.galaxy_clustering.ShapeFitPowerSpectrumParameterization
    cosmo:
      class: cosmofit.theories.primordial_cosmology.Cosmoprimo

  # What emulator engine to use; here, Taylor expansion
  emulate:
    class: TaylorEmulator

  # What sampler to use; here emcee
  sample:
    class: cosmofit.samplers.EmceeSampler
    init:
      #nwalkers: 40
      # How many chains to run in parallel (rest of processes for internal likelihood parallelization)
      chains: 1
      max_tries: 10
    run:
      max_iterations: 10000
      # Dump to disk every 100 step
      check_every: 100
      # Perform convergence checks
      check: True
    save: f'{output_dir}/chain_*.npy'

  # What profiler to use; here minuit
  profile:
    class: cosmofit.profilers.MinuitProfiler
    maximize:
      # How many posterior maximation to run (from different seeds)
      niterations: 10
    save: f'{output_dir}/profiles.npy'

  # Summary for chains
  summarize:
    source:
      fn: f'{output_dir}/chain_0.npy'
      burnin: 0.5
    plot_triangle: f'{output_dir}/triangle.png'
    plot_trace: f'{output_dir}/trace.png'
    plot_autocorrelation_time: f'{output_dir}/autocorrelation_time.png'
    plot_gelman_rubin:
      fn: f'{output_dir}/gelman_rubin.png'
      nsplits: 4
    plot_geweke: f'{output_dir}/geweke.png'

  # Summary for profiles
  summarize:
    source: f'{output_dir}/profiles.npy'
    stats: f'{output_dir}/stats.tex'

  # Run the pipeline at a given point (here, mean of the chains)
  do:
    source: f'{output_dir}/chain_0.npy'
    do: plotcosmofit.theories.galaxy_clustering.LPTPowerSpectrumMultipoles

  # Any variable that you can use anywhere
  data_dir: _pk
  # Note the '${}' syntax: ${HOME} will be replaced by the environment variable
  output_dir: f'${HOME}/_fs'


If you are familiar with **Cobaya**, this should not look so different, but there are some notable differences.
Before going into the details, let us just specify that:

.. code-block:: bash

  cosmofit install config.yaml  # installs all dependencies
  cosmofit emulate config.yaml  # builds the emulator for cosmofit.theories.galaxy_clustering.LPTPowerSpectrumMultipoles
  # Then, just uncomment load: True in pipeline.QSO.pt section of config.yaml
  cosmofit profile config.yaml  # runs best fits
  cosmofit summarize config.yaml  # print bestfits
  cosmofit sample config.yaml  # sample posterior
  # Re-arrange summary for chains as last one in config.yaml (last definition is kept)
  cosmofit summarize config.yaml  # some chain plots

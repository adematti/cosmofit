.. title:: cosmofit docs

**************************************
Welcome to cosmofit's documentation!
**************************************

.. toctree::
  :maxdepth: 1
  :caption: User documentation

  user/building
  user/config
  user/parameters
  user/samplers
  user/profilers
  user/emulators
  api/api

.. toctree::
  :maxdepth: 1
  :caption: Developer documentation

  developer/documentation
  developer/tests
  developer/contributing
  developer/changes

.. toctree::
  :hidden:

************
Introduction
************

**cosmofit** is a Python package for cosmological inference.
As other cosmological inference codes (e.g. `cosmosis <https://github.com/joezuntz/cosmosis>`_, `cobaya <https://github.com/CobayaSampler/cobaya>`_)
**cosmofit** includes some syntax to build a "pipeline", i.e. a suite of "calculators", which computes quantities, given a set of input parameters.
**cosmofit** embeds:

- tools to emulate this pipeline (in-place) at any step
- if the pipeline computes a likelihood: profilers, and many samplers

Just as other cosmological inference codes, **cosmofit** also includes:

- an advanced parameterization infrastructure (priors, reference distributions, derived parameters, etc.)
- speed hierarchy between various parameters, exploited in some samplers (MCMCSampler, PolychordSampler)
- primordial cosmology computations with `cosmoprimo <https://github.com/cosmodesi/cosmoprimo>`_
- likelihoods (Planck2018, SN, ...): in progress, though!
- tools to install external data/packages
- convergence diagnostics
- MPI support to run several chains in parallel

In addition:

- consistent parameterization between Boltzmann codes (through `cosmoprimo <https://github.com/cosmodesi/cosmoprimo>`_)
- transparent namespace scheme for parameters, to avoid keeping track of all possible parameter (base) names among all various calculators (and likelihoods)
- in-place emulation, to considerably speed-up inference, while allowing for easy checks of the emulation strategy at the posterior level (what we care about!)
- automatic differentiation (with jax, for some calculators) to perform analytic marginalization (next: gradient to be used in profilers and samplers)
- double parallelization level (several chains, and several processes per chain), for all samplers
- more likelihood profiling tools (1D and 2D profiles in addition to likelihood/posterior maximization)
- the possibility to save any array (of any shape) quantity to disk, no matter the sampler/profiler under use, to facilitate debugging, set up model template bases, build emulators within the relevant parameter space, etc.

Development so far has focused on galaxy clustering / compression techniques.


Quick start-up
==============

For a quick start-up, see `notebooks <https://github.com/adematti/cosmofit/blob/main/nb>`_.


Changelog
=========

* :doc:`developer/changes`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

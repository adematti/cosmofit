Emulators
=========

Typically, emulators are instantiated this way:

.. code-block:: yaml

  emulate:
    # Specify the emulator class
    class: cosmofit.emulators.TaylorEmulator
    init:
      # Any emulator option
    sample:
      # Any option specifying how to sample the pipeline to emulate
      # Or path to samples (e.g. MCMC chains or grid) to use
    fit:
      # Any option to fit the pipeline to emulate
    check:
      # Any option to check emulation is reliable
    plot:
      # Path to plot(s) of predicted vs. true emulated quantities
    # Where to save emulator
    # Then, the calculator up to which the pipeline has been emulated
    # can be loaded with load: emulator.npy
    save: emulator.npy

One will typically run pipeline sampling in parallel.
In the following we present emulators' default options (in addition to those presented above).

.. include:: emulators_configs.rst

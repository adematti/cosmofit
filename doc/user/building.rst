.. _user-building:

Building
========

Requirements
------------
Only strict requirements are:

  * numpy
  * scipy
  * tabulate
  * `mpytools <https://github.com/cosmodesi/mpytools>`_

Extra requirements are:

  * plotting: getdist, anesthetic to make nice contour plots
  * jax: for automatic differentiation (calculation of gradient)

pip
---
To install **cosmofit**, simply run::

  python -m pip install git+https://github.com/adematti/cosmofit

If you want to install extra requirements, run::

  python -m pip install git+https://github.com/adematti/cosmofit#egg=cosmofit[plotting,jax]

git
---

First:

  git clone https://github.com/adematti/cosmofit.git

To install the code::

  python setup.py install --user

Or in development mode (any change to Python code will take place immediately)::

  python setup.py develop --user


Pipeline dependencies, samplers, profilers, emulators
-----------------------------------------------------
**cosmofit** comes with an infrastructure to install packages.
One can install all dependencies given a configuration file ``config.yaml`` with::

  cosmofit install config.yaml

Run::

  cosmofit install --help

to print some help. Syntax is close to that of **pip**, e.g. dependencies can be installed locallly with::

  cosmofit install --user config.yaml

A notable difference w.r.t. to **pip** is the --install-dir option, that allows one to specify where to save dependencies
(on top of possibly already exising environment, just like the **pip** --user option)::

  cosmofit install --install-dir your/directory/ config.yaml

In future calls, if omitted the installation directory will default to the first used.
The default installation directory can be changed directly in ``$HOME/.cosmofit/config.yaml``.

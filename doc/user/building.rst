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
  * `cosmoprimo <https://github.com/cosmodesi/cosmoprimo>`_

Extra requirements are:

  * profilers: imuinuit
  * samplers: emcee, zeus-mcmc, dynesty
  * emulators: findiff, tensorflow

pip
---
To install **cosmofit**, simply run::

  python -m pip install git+https://github.com/adematti/cosmofit

If you want to install extra requirements, e.g. profilers and samplers, run::

  python -m pip install git+https://github.com/adematti/cosmofit#egg=cosmofit[profilers,samplers]

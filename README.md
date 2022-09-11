# cosmofit

**cosmofit** is a Python package for cosmological inference.

WARNING: this is ongoing work! Most developments are targeted at 2-pt galaxy clustering measurements.
Within DESI context, the goal is to set up BAO and RSD fits, test various models, emulation and compression techniques,
explore combination with external datasets, in a coherent framework.
This should be helpful for developments, before including the 'official' likelihood in popular cosmological inference codes for legacy.

Example notebooks are provided in directory nb/.

## TODOs

- reparametrization (in process)
- dimensionality reduction (in process)
- jax.jit
- use Jacobian, if available, in Taylor emulator?
- finish documentation
- proper example with cross-covariance (e.g. many redshift slices), potential changes to the infrastructure
- autodetection of Chain/Profiles types; improve Profiles type
- allow emulators to use result of previous calculators as input (not only parameters)
- implement external likelihoods (Planck, Pantheon, DES, etc.)
- more PT theory models
- real science work...

## Documentation

Documentation is hosted on Read the Docs, [cosmofit docs](https://cosmofit.readthedocs.io/).

## Requirements

Only strict requirements are:

  - numpy
  - scipy

## Installation

### pip

Simply run:
```
python -m pip install git+https://github.com/adematti/cosmofit
```

### git

First:
```
git clone https://github.com/adematti/cosmofit.git
```
To install the code:
```
python setup.py install --user
```
Or in development mode (any change to Python code will take place immediately):
```
python setup.py develop --user
```

## License

**cosmofit** is free software distributed under a BSD3 license. For details see the [LICENSE](https://github.com/adematti/cosmofit/blob/main/LICENSE).

## Acknowledgments

- Some inspiration from cobaya: https://github.com/CobayaSampler/cobaya
- BAO models: https://github.com/Samreay/Barry
- Taylor expansion emulator: https://github.com/sfschen/velocileptors_shapefit
- MLP emulator: https://github.com/sfschen/EmulateLSS

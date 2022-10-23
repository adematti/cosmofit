# cosmofit

**cosmofit** is a Python package for cosmological inference.

WARNING: this is ongoing work! Most developments are targeted at 2-pt galaxy clustering measurements.
Within DESI context, the goal is to set up BAO and RSD fits, test various models, emulation and compression techniques,
explore combination with external datasets, in a coherent framework.
This should be helpful for developments, before including the 'official' likelihood in popular cosmological inference codes for legacy.

Example notebooks are provided in directory nb/.

## TODOs

- dimensionality reduction (in progress)
- jax.jit
- use Jacobian, if available, in Taylor emulator?
- finish documentation
- proper example with cross-covariance (e.g. many redshift slices), potential changes to the infrastructure
- implement external likelihoods (Planck, Pantheon, DES, etc.)
- real science work...

## Documentation

Documentation is hosted on Read the Docs, [cosmofit docs](https://cosmofit.readthedocs.io/).

## Installation

See [cosmofit docs](https://cosmofit.readthedocs.io/en/latest/user/building.html).

## License

**cosmofit** is free software distributed under a BSD3 license. For details see the [LICENSE](https://github.com/adematti/cosmofit/blob/main/LICENSE).

## Acknowledgments

- Inspiration from cobaya: https://github.com/CobayaSampler/cobaya
- BAO models (Sam Ray and Cullan Howlett): https://github.com/Samreay/Barry
- Taylor expansion emulator (Stephen Chen): https://github.com/sfschen/velocileptors_shapefit
- MLP emulator (Stephen Chen and Joe DeRose): https://github.com/sfschen/EmulateLSS

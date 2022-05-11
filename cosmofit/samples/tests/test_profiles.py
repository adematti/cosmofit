import os
import numpy as np

from cosmofit import setup_logging
from cosmofit.samples.profile import Profiles, ParameterBestFit, ParameterValues, ParameterCovariance


def get_profiles(params):
    rng = np.random.RandomState()
    profiles = Profiles()
    profiles.set(start=ParameterValues([0. for param in params], params=params))
    profiles.set(bestfit=ParameterBestFit([rng.normal(0., 0.1) for param in params] + [rng.normal(0., 0.1)], params=params + ['logposterior']))
    profiles.set(parabolic_errors=ParameterValues([0.5 for param in params], params=params))
    profiles.set(deltachi2_errors=ParameterValues([(0.5, 0.5) for param in params], params=params, enforce={'ndmin': 2}))
    profiles.set(covariance=ParameterCovariance(np.eye(len(params)), params=params))

    return profiles


def test_misc():
    profiles_dir = '_profiles'
    params = ['params.a', 'params.b', 'params.c', 'params.d']
    profiles = Profiles.concatenate(*[get_profiles(params) for i in range(5)])
    assert profiles.bestfit.shape == profiles.bestfit['logposterior'].shape == (5,)
    fn = os.path.join(profiles_dir, 'profile.npy')
    profiles.save(fn)
    profiles2 = profiles.load(fn)
    assert profiles2 == profiles
    profiles.bcast(profiles)
    del profiles.deltachi2_errors
    profiles.bcast(profiles)


def test_stats():
    params = ['params.a', 'params.b', 'params.c', 'params.d']
    profiles = get_profiles(params)
    print(profiles.to_stats(tablefmt='latex_raw'))
    print(profiles.to_stats(tablefmt='pretty'))


if __name__ == '__main__':

    setup_logging()

    test_misc()
    test_stats()

import os
import numpy as np

from cosmofit import setup_logging
from cosmofit.samples.profile import Profiles, ParameterBestFit, ParameterValues, ParameterCovariance, ParameterContours
from cosmofit.samples import plotting


def get_profiles(params):
    rng = np.random.RandomState()
    profiles = Profiles()
    profiles.set(start=ParameterValues([0. for param in params], params=params))
    params = profiles.start.params()
    for param in params: param.fixed = False
    profiles.set(bestfit=ParameterBestFit([rng.normal(0., 0.1) for param in params] + [-0.5], params=params + ['logposterior']))
    profiles.set(error=ParameterValues([0.5 for param in params], params=params))
    profiles.set(covariance=ParameterCovariance(np.eye(len(params)), params=params))
    profiles.set(interval=ParameterValues([(-0.5, 0.5) for param in params], params=params))
    x = np.linspace(-1., 1., 101)
    profiles.set(profile=ParameterValues([[x, 1. + x**2] for param in params], params=params))
    t = np.linspace(0., 2. * np.pi, 101)
    params2 = [(param1, param2) for param1 in params for param2 in params]
    profiles.set(contour=ParameterContours([(np.cos(t), np.sin(t)) for param in params2], params=params2))
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
    del profiles.error
    profiles.bcast(profiles)


def test_stats():
    params = ['params.a', 'params.b', 'params.c', 'params.d']
    profiles = get_profiles(params)
    print(profiles.to_stats(tablefmt='latex_raw'))
    print(profiles.to_stats(tablefmt='pretty'))


def test_plot():
    profiles_dir = '_profiles'
    params = ['like.a', 'like.b', 'like.c', 'like.d']
    profiles = [get_profiles(params)] * 2
    plotting.plot_aligned_stacked(profiles, fn=os.path.join(profiles_dir, 'aligned.png'))

    profiles = [get_profiles(params)] * 2
    plotting.plot_profile(profiles, fn=os.path.join(profiles_dir, 'profile.png'))
    plotting.plot_profile_comparison(profiles[0], profiles[1], fn=os.path.join(profiles_dir, 'profile_comparison.png'))


if __name__ == '__main__':

    setup_logging()

    test_misc()
    test_stats()
    test_plot()

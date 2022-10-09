import numpy as np
from matplotlib import pyplot as plt

from cosmofit.samplers.mcmc import SOSampler


def test_mcmc():
    rng = np.random.RandomState()
    size = 10000
    sampler = SOSampler(1, rng)
    samples = np.array([sampler.sample() for i in range(size)])
    plt.hist(samples.ravel(), bins=20)
    plt.show()
    sampler = SOSampler(2, rng)
    samples = np.array([sampler.sample() for i in range(size)])
    plt.scatter(*samples.T)
    plt.show()


if __name__ == '__main__':

    test_mcmc()

import os
import numpy as np

from cosmoprimo.fiducial import DESI
from pypower import CatalogFFTPower, MeshFFTWindow, PowerSpectrumStatistics, BaseMatrix

from mockfactory import EulerianLinearMock, RandomBoxCatalog, setup_logging


nmesh = 100
boxsize = 500
boxcenter = 0
los = 'x'
pklin = DESI().get_fourier().pk_interpolator().to_1d(z=2)
f = 0.8
bias = 2.


def run_box_mock(save_fn, unitary_amplitude=False, seed=42):

    # unitary_amplitude forces amplitude to 1
    mock = EulerianLinearMock(pklin, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=unitary_amplitude)
    mock.set_real_delta_field(bias=bias)
    mock.set_rsd(f=f, los=los)

    data = RandomBoxCatalog(nbar=4e-3, boxsize=boxsize, boxcenter=boxcenter, seed=seed)
    data['Weight'] = mock.readout(data['Position'], field='delta', resampler='tsc', compensate=True) + 1.

    poles = CatalogFFTPower(data_positions1=data['Position'], data_weights1=data['Weight'], edges={'step': 0.005},
                            los=los, boxsize=boxsize, boxcenter=boxcenter, nmesh=100,
                            resampler='tsc', interlacing=2,
                            position_type='pos', mpicomm=data.mpicomm).poles
    poles.save(save_fn)


def run_box_window(save_fn, poles_fn):

    edgesin = np.linspace(0., 0.5, 100)
    window = MeshFFTWindow(edgesin=edgesin, power_ref=PowerSpectrumStatistics.load(poles_fn), periodic=True).poles
    window.save(save_fn)


def plot_power(save_fn):
    PowerSpectrumStatistics.load(save_fn).plot(show=True)


def plot_window(save_fn):
    from matplotlib import pyplot as plt
    wmatrix = BaseMatrix.load(save_fn)
    factorout = 2
    wmatrix.slice_x(sliceout=slice(0, wmatrix.xout[0].size // factorout * factorout))
    wmatrix.rebin_x(factorout=factorout)

    #4 slice(0, 28, None) slice(2, 10, None)

    kin = wmatrix.xin[0]
    kout = wmatrix.xout[0]

    pkin = pklin(kin)
    beta = f/bias
    pk = []
    pk.append(bias**2 * (1. + 2. / 3. * beta + 1. / 5. * beta**2) * pkin)
    pk.append(bias**2 * (4. / 3. * beta + 4. / 7. * beta**2) * pkin)
    pk.append(bias**2 * 8. / 35. * beta**2 * pkin)
    pkconv = wmatrix.dot(pk, unpack=True)

    ax = plt.gca()
    ax.plot([], [], linestyle='--', color='k', label='theory')
    ax.plot([], [], linestyle='-', color='k', label='window')
    for ill, proj in enumerate(wmatrix.projsout):
        ell = proj.ell
        ax.plot(kin, kin * pk[ill], color='C{:d}'.format(ill), linestyle='--', label=None)
        ax.plot(kout, kout * pkconv[ill], color='C{:d}'.format(ill), linestyle='-', label=r'$\ell = {:d}$'.format(ell))
    ax.set_xlim(0., 0.3)
    ax.legend()
    ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
    ax.set_ylabel(r'$k P(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
    plt.show()


if __name__ == '__main__':

    setup_logging()

    output_dir = 'pk'
    data_fn = os.path.join(output_dir, 'data.npy')
    window_fn = os.path.join(output_dir, 'window.npy')
    todo = ['plot']

    if 'mock' in todo:
        run_box_mock(data_fn, pklin=pklin, unitary_amplitude=True, seed=0)
        for i in range(500):
            run_box_mock(os.path.join(output_dir, 'mock_{:d}.npy'.format(i)), pklin=pklin, seed=(i + 1) * 42)

    if 'window' in todo:
        run_box_window(window_fn, data_fn)

    if 'plot' in todo:
        plot_power(data_fn)
        plot_window(window_fn)

import os

from cosmoprimo.fiducial import DESI
from pypower import CatalogFFTPower

from mockfactory import EulerianLinearMock, RandomBoxCatalog, setup_logging


def run(save_fn, pklin, unitary_amplitude=False, seed=42):
    nmesh = 100
    boxsize = 500
    boxcenter = 0
    f = 0.8
    bias = 2.
    los = 'x'

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


if __name__ == '__main__':

    setup_logging()

    output_dir = 'pk'
    pklin = DESI().get_fourier().pk_interpolator().to_1d(z=2)
    run(os.path.join(output_dir, 'data.npy'), pklin=pklin, unitary_amplitude=True, seed=0)
    for i in range(500):
        run(os.path.join(output_dir, 'mock_{:d}.npy'.format(i)), pklin=pklin, seed=(i + 1) * 42)

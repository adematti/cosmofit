import os
import numpy as np

from cosmoprimo.fiducial import DESI
from pypower import CatalogFFTPower, MeshFFTWindow, PowerSpectrumStatistics

from mockfactory import EulerianLinearMock, RandomBoxCatalog, setup_logging


nmesh = 100
boxsize = 500
boxcenter = 0
los = 'x'


def run_box_mock(save_fn, pklin, unitary_amplitude=False, seed=42):
    f = 0.8
    bias = 2.

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
    window = MeshFFTWindow(edgesin=edgesin, power_ref=PowerSpectrumStatistics.load(poles_fn), los=los, periodic=True).poles
    window.save(save_fn)


if __name__ == '__main__':

    setup_logging()

    output_dir = 'pk'
    data_fn = os.path.join(output_dir, 'data.npy')
    window_fn = os.path.join(output_dir, 'window.npy')
    todo = ['window']

    if 'mock' in todo:
        pklin = DESI().get_fourier().pk_interpolator().to_1d(z=2)
        run_box_mock(data_fn, pklin=pklin, unitary_amplitude=True, seed=0)
        for i in range(500):
            run_box_mock(os.path.join(output_dir, 'mock_{:d}.npy'.format(i)), pklin=pklin, seed=(i + 1) * 42)

    if 'window' in todo:
        run_box_window(window_fn, data_fn)

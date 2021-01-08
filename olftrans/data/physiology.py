import os
import h5py
import numpy as np
from olftrans import DATADIR
from . import utils

def get_data(dt, fpath='antenna_data.h5'):
    data = dict()
    waveforms = ['white_noise', 'staircase', 'elife15/parabola', 'elife15/ramp', 'elife15/step']
    with h5py.File(fpath, 'r') as f:
        for w in waveforms:
            _stim_t = f[f'/{w}/stimulus/x'][()]
            _stim_orig = f[f'/{w}/stimulus/y'][()]
            _psth_t = f[f'/{w}/psth/x'][()]
            _psth_orig = f[f'/{w}/psth/y'][()]
            _t, _stim, _psth = utils.process_io(_stim_t, _stim_orig, _psth_t, _psth_orig, dt=dt)
            index = np.argsort(_stim.max(1)) # sort I/O in increasing order of the stimulus' maximum amplitudes
            data[w] = dict(t=_t, input=_stim[index], output=_psth[index])
    return data

DT = 1e-5
DATA = get_data(DT, os.path.join(DATADIR, 'antenna_data.h5'))

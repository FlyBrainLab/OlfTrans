import os
import h5py
import numpy as np
from olftrans import DATADIR
from warnings import warn
from . import utils
from .. import errors as err

def get_data(dt, fpath="antenna_data.h5"):
    data = dict()
    waveforms = [
        "white_noise",
        "staircase",
        "elife15/parabola",
        "elife15/ramp",
        "elife15/step",
    ]
    with h5py.File(fpath, "r") as f:
        for w in waveforms:
            _stim_t = f[f"/{w}/stimulus/x"][()]
            _stim_orig = f[f"/{w}/stimulus/y"][()]
            _psth_t = f[f"/{w}/psth/x"][()]
            _psth_orig = f[f"/{w}/psth/y"][()]
            _t, _stim, _psth = utils.process_io(
                _stim_t, _stim_orig, _psth_t, _psth_orig, dt=dt
            )
            index = np.argsort(
                _stim.max(1)
            )  # sort I/O in increasing order of the stimulus' maximum amplitudes
            data[w] = dict(t=_t, input=_stim[index], output=_psth[index])
    return data


DT = 1e-5
DATA = None

try:
    DATA = get_data(DT, os.path.join(DATADIR, "antenna_data.h5"))
except OSError as e:
    warn(err.MissingFileWarning(
        'Antenna Physiology Data Not Found. Make sure the file olftrans/data/antenna_data.h5 '
        'is found and named as antenna_data.h5. Follow instructions at '
        'http://amacrine.ee.columbia.edu:15000/ to download and place'
        'in olftrans/data folder.'
    ))
except Exception:
    raise Exception("Loading Antenna Data encountered unknown error.") from e

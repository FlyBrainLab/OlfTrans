import os
import h5py
import numpy as np
from olftrans import ROOTDIR


def process_io(stim_t, stim, psth_t, psth, dt=5e-4):
    """Process I/O

    This function takes stimulus and PSTH, interpolates both
    at the same time resolution and make sure stimulus and
    PSTH are the same length. Padding is added with terminal values
    as needed.

    Arguments:
        stim_t: Raw Stimulus Time Series
        stim: Raw Stimulus
            - Should have the same dimensionality as `stim_t`
        psth_t: Raw PSTH Time Stamps
        psth: Raw PSTH
            - Should have the same dimensionality as `psth_t`

    Keyword Arguments:
        dt: time step at which to interpolate the stimulus and PSTH

    Returns:
        t: interpolated time series
        stim: interpolated stimulus
        psth: interpolated PSTH
    """
    stim = np.atleast_2d(stim)
    psth = np.atleast_2d(psth)
    assert stim_t.ndim == 1
    assert psth_t.ndim == 1

    dt_orig = stim_t[1] - stim_t[0]
    lower_bound = max(min(stim_t), min(psth_t))
    upper_bound = min(max(stim_t), max(psth_t))
    stim_mask = np.logical_and(
        stim_t >= lower_bound, stim_t <= upper_bound + dt_orig / 2
    )
    stim_t = stim_t[stim_mask]
    stim = stim[:, stim_mask]
    psth_mask = np.logical_and(
        psth_t >= lower_bound, psth_t <= upper_bound + dt_orig / 2
    )
    psth_t = psth_t[psth_mask]
    psth = psth[:, psth_mask]

    t = np.arange(min(stim_t), max(stim_t), dt)

    stim_intp = []
    psth_intp = []
    for n, (_stim, _psth) in enumerate(zip(stim, psth)):
        stim_intp.append(np.interp(t, stim_t, _stim))
        psth_intp.append(np.interp(t, psth_t, _psth))
    return t, np.vstack(stim_intp), np.vstack(psth_intp)

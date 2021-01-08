"""Main module."""
import math
import numpy as np
from warnings import warn
from .data import NoisyConnorStevens, OTP
from .neurodriver import model
from scipy.interpolate import interp2d
from dataclasses import dataclass

@dataclass(frozen=True)
class Estimation:
    sigma: float
    steady_state_spike_rate: float
    peak_spike_rate: float
    steady_state_current: float
    peak_current: float
    affs: float
    br: float
    dr: float


def estimate(
    amplitude, resting_spike_rate, steady_state_spike_rate, peak_spike_rate=None, decay_time=None, cache=True
):
    """Main Entry Point of the Estimation Procedure"""
    sigma = estimate_sigma(resting_spike_rate, cache=cache)
    ss_current = estimate_current(steady_state_spike_rate, sigma, cache=cache)
    affinity = estimate_affinity(amplitude, ss_current, cache=cache)
    if peak_spike_rate is not None:
        peak_current = estimate_current(peak_spike_rate, sigma, cache=cache)
    else:
        peak_current = None
    dr = estimate_dissociation(amplitude, affinity, peak_current=peak_current, decay_time=decay_time, cache=cache)
    br = affinity * dr
    return Estimation(br=br, dr=dr, sigma=sigma, affs=affinity,
                      steady_state_spike_rate=steady_state_spike_rate,
                      steady_state_current=ss_current,
                      peak_spike_rate=peak_spike_rate,
                      peak_current=peak_current)


def estimate_sigma(resting_spike_rate, cache=True):
    """Estimate Sigma Value given Resting Spike Rate"""
    return np.interp(x=resting_spike_rate, xp=NoisyConnorStevens.DATA.rest_f, fp=NoisyConnorStevens.DATA.rest_sigma)

def estimate_current(spike_rate, sigma, sigma_atol=1e-4, sigma_rtol=1e-4, cache=True):
    """Estimate Current Value given Spike Rate"""
    sigma_err = sigma - NoisyConnorStevens.DATA.sigma
    sigma_rerr = abs(sigma_err)/min([abs(sigma), abs(NoisyConnorStevens.DATA.sigma)])
    if abs(sigma_err) > sigma_atol or sigma_rerr > sigma_rtol:
        warn(f"Required Sigma value too different from cached Sigma, recomputing F-I curve...")
        currents = np.linspace(0, 150, 100)
        _, target_f = model.compute_fi(
            model.NoisyConnorStevens,
            currents,
            dt=5e-6,
            repeat=50,
            neuron_params={'sigma':sigma/math.sqrt(5e-6)},
            save=cache
        )
    else:
        currents = NoisyConnorStevens.DATA.I.copy()
        target_f = NoisyConnorStevens.DATA.f.copy()

    idx = np.argsort(currents)
    return np.interp(x=spike_rate, xp=target_f[idx], fp=currents[idx])

def estimate_affinity(amplitude, steady_state_current, cache=True):
    """Estimate Affinity Value From Steady-State Current"""
    br = OTP.DATA.br
    dr = OTP.DATA.dr
    DR,BR = np.meshgrid(dr, br)
    v = OTP.DATA.amplitude
    bvd = (BR*v/DR).ravel()
    idx = np.argsort(bvd)
    max_current = np.percentile(OTP.DATA.ss, 99) # clip current value at 99 percentile of available values
    steady_state_current = np.clip(steady_state_current, 0., max_current)

    bvd_intp = 10**np.linspace(-2, 4, 100)
    ss_intp = np.interp(bvd_intp, fp=OTP.DATA.ss.ravel()[idx], xp=bvd[idx])
    target_bvd = np.interp(x=steady_state_current, xp=ss_intp, fp=bvd_intp)
    return target_bvd/amplitude

def estimate_dissociation(
    amplitude, affinity, peak_current=None, decay_time=0.1, cache=True
):
    if peak_current is None:
        if decay_time is not None:
            return 1./decay_time
        else:
            return 10.
    else:
        intp = interp2d(OTP.DATA.br*OTP.DATA.amplitude, OTP.DATA.dr, OTP.DATA.peak, kind='linear')
        drs = np.linspace(1e-2, 1000., 500)
        errs = np.zeros_like(drs)
        for n_d, dr in enumerate(drs):
            _br = affinity*dr/amplitude
            errs[n_d] = intp(_br*amplitude, dr) - peak_current
        return drs[np.argmin(np.abs(errs))]

def estimate_resting_spike_rate(sigma, cache=True):
    """Estimate Resting Spike Rate from Sigma Value"""
    return np.interp(x=sigma, xp=NoisyConnorStevens.DATA.rest_sigma, fp=NoisyConnorStevens.DATA.rest_f)

def run(t, waveform, br, dr, sigma):
    pass
"""Main module."""
import math
import numpy as np
from warnings import warn
from .data import olfdata
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
    amplitude,
    resting_spike_rate,
    steady_state_spike_rate,
    peak_spike_rate=None,
    decay_time=None,
    cache=True,
):
    """Main Entry Point of the Estimation Procedure"""
    sigma = estimate_sigma(resting_spike_rate, cache=cache)
    ss_current = estimate_current(steady_state_spike_rate, sigma, cache=cache)
    affinity = estimate_affinity(amplitude, ss_current, cache=cache)
    if peak_spike_rate is not None:
        peak_current = estimate_current(peak_spike_rate, sigma, cache=cache)
    else:
        peak_current = None
    dr = estimate_dissociation(
        amplitude,
        affinity,
        peak_current=peak_current,
        decay_time=decay_time,
        cache=cache,
    )
    br = affinity * dr
    return Estimation(
        br=br,
        dr=dr,
        sigma=sigma,
        affs=affinity,
        steady_state_spike_rate=steady_state_spike_rate,
        steady_state_current=ss_current,
        peak_spike_rate=peak_spike_rate,
        peak_current=peak_current,
    )


def estimate_sigma(resting_spike_rate, cache=True):
    """Estimate Sigma Value given Resting Spike Rate"""
    res = olfdata.find('rest', ['ParamValue', 'Frequencies', 'metadata.dt'], lambda df: df.ParamKey == 'sigma')
    sigmas = res.iloc[-1].ParamValue*np.sqrt(res.iloc[-1]['metadata.dt'])
    rest_fs = res.iloc[-1].Frequencies
    return np.interp(resting_spike_rate, xp=rest_fs, fp=sigmas)

def estimate_current(spike_rate, sigma, sigma_atol=1e-4, sigma_rtol=1e-4, cache=True):
    """Estimate Current Value given Spike Rate"""
    res = olfdata.find("FI", ["Currents", "Frequencies", "Params", "metadata.dt"])
    feasible_indices = []
    for n, row in res.iterrows():
        if 'sigma' in row['Params']:
            _sig = row['Params']['sigma']*np.sqrt(row['metadata.dt'])
            sigma_err = sigma - _sig
            sigma_rerr = abs(sigma_err) / min([abs(sigma), abs(_sig)])
            if abs(sigma_err) > sigma_atol or sigma_rerr > sigma_rtol:
                feasible_indices.append(n)
    feasible_indices = sorted(feasible_indices)
    if len(feasible_indices) > 0: # not found
        exp_index = feasible_indices[-1]
        Is = res.loc[0].Currents
        fs = res.loc[0].Frequencies
    else:
        warn(
                f"Required Sigma value {sigma} too different from cached Sigmas, recomputing F-I curve..."
            )
        Is = np.linspace(0, 150, 100)
        _, fs = model.compute_fi(
            model.NoisyConnorStevens,
            Is,
            dt=5e-6,
            repeat=50,
            neuron_params={"sigma": sigma / math.sqrt(5e-6)},
            save=cache,
        )
    idx = np.argsort(Is)
    return np.interp(x=spike_rate, xp=fs[idx], fp=Is[idx])


def estimate_affinity(amplitude, steady_state_current, cache=True):
    """Estimate Affinity Value From Steady-State Current"""
    res = olfdata.find('OTP', ['Amplitude', 'Br', 'Dr', 'Peak', 'SS'])
    tmp = res[res.Amplitude == amplitude]
    if len(tmp)==0:
        res = res.iloc[-1]
    else:
        res = tmp.iloc[-1]
    I_ss = res.SS
    brs = res.Br
    drs = res.Dr
    v = res.Amplitude
    DR, BR = np.meshgrid(drs, brs)
    bvds = (BR/DR)*v
    bvds = bvds.ravel()
    I_ss_flat = I_ss.ravel()
    idx = np.argsort(bvds)

    # interpolate first
    bvd_intp = 10**np.linspace(-6,3,1000)
    ss_intp = np.interp(bvd_intp, bvds[idx], I_ss_flat[idx])

    from scipy.optimize import differential_evolution
    hill_f = lambda x, a,b,c,n: b + a*x**n/(x**n+c)
    def inverse_hill_f(y,a,b,c,n, x_ref):
        res = np.atleast_1d(np.power(c*(y-b)/(a-(y-b)), 1./n))
        res[y<b] = x_ref.min()
        res[(y-b) > a] = x_ref.max()
        return res

    def cost(x, aff, ss):
        a,b,c,n = x
        pred = hill_f(aff,a,b,c,n)
        return np.linalg.norm(pred-ss)
    bounds = [(0,100), (0, 100), (0,100), (.5, 2.)]
    diffeq_ss = differential_evolution(cost, bounds, tol=1e-4, args=(bvd_intp, ss_intp), disp=False)

    if diffeq_ss.success:
        bvd_inverse = inverse_hill_f(steady_state_current, *diffeq_ss.x, bvd_intp)
    else:
        bvd_inverse = np.interp(steady_state_current, xp=I_ss_flat[idx], fp=bvds[idx])

    return bvd_inverse / amplitude


def estimate_dissociation(
    amplitude, affinity, peak_current=None, decay_time=0.1, cache=True
):
    if peak_current is None:
        if decay_time is not None:
            return 1.0 / decay_time
        else:
            return 10.0
    else:
        res = olfdata.find('OTP', ['Amplitude', 'Br', 'Dr', 'Peak', 'SS'])
        tmp = res[res.Amplitude == amplitude]
        if len(tmp)==0:
            res = res.iloc[-1]
        else:
            res = tmp.iloc[-1]

        peak_current = np.atleast_1d(peak_current)
        peaks = peak_current.ravel()
        drs = np.zeros_like(peaks)
        for n,peak in enumerate(peaks):
            br_idx, dr_idx = np.unravel_index(np.argmin(np.abs(res.Peak-peak)), res.Peak.shape)
            drs[n] = res.Dr[dr_idx]
        if len(peak_current) == 1:
            return drs[0]
        else:
            return drs.reshape(peak_current.shape)

def estimate_resting_spike_rate(sigma, cache=True):
    """Estimate Resting Spike Rate from Sigma Value"""
    res = olfdata.find('rest', ['ParamValue', 'Frequencies', 'metadata.dt'], lambda df: df.ParamKey == 'sigma')
    sigmas = res.iloc[-1].ParamValue*np.sqrt(res.iloc[-1]['metadata.dt'])
    rest_fs = res.iloc[-1].Frequencies
    return np.interp(sigma, xp=sigmas, fp=rest_fs)

def run(t, waveform, br, dr, sigma):
    pass

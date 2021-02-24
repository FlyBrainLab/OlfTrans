"""Main module of Olfactory Transduction Module

The module deals with estimation procedures to obtain
binding and dissociation rates from peak/steady state spike rates
for given odorant.

The main entry of the module is the `estimate` function.
"""
import math
import numpy as np
from warnings import warn
import typing as tp
from .data import olfdata
from .neurodriver import model
from scipy.interpolate import interp2d
from dataclasses import dataclass

float_array = tp.Union[float, np.ndarray]


@dataclass(frozen=True)
class Estimation:
    """Estimation Result
    """

    sigma: float
    """noise parameter of the NoisyConnorStevens neuron"""

    steady_state_spike_rate: float_array
    """steady-state spike rate.
    This is the part of the input to estimation procedure stored as reference"""

    peak_spike_rate: float_array
    """peak spike rate.
    This is the part of the input to estimation procedure stored as reference"""

    steady_state_current: float_array
    """steady-state current estimated from steady-state spike rate"""

    peak_current: float_array
    """peak current estimated from peak spike rate"""

    affs: float_array
    """affinity values"""

    br: float_array
    """binding rates"""

    dr: float_array
    """dissociation rates"""


def estimate(
    amplitude: float,
    resting_spike_rate: tp.Union[float, np.ndarray],
    steady_state_spike_rate: tp.Union[float, np.ndarray],
    peak_spike_rate: float_array = None,
    decay_time: float = None,
    cache: bool = True,
):
    """Estimation Procedure

    The estimation procedure assumes that the input concentration waveform
    is a step-waveform with an amplitude of `amplitude`. The Peak and
    Steady-State spike rate of the OSN measured is then used to estimate the
    binding and dissociation rates.

    Arguments:
        amplitude: concentration waveform's amplitdue
        resting_spike_rate: resting OSN spike rate
        steady_state_spike_rate: steady-state OSN spike rate under input

    Keyword Arguments:
        peak_spike_rate: peak OSN spike rate under input
        decay_time: Time for OSN spike to settle back to resting after odorant input offset
        cache: whether to save the data generated by the estimation result
    """
    sigma = estimate_sigma(resting_spike_rate)
    ss_current = estimate_current(steady_state_spike_rate, sigma, cache=cache)
    affinity = estimate_affinity(amplitude, ss_current)
    if peak_spike_rate is not None:
        peak_current = estimate_current(peak_spike_rate, sigma)
    else:
        peak_current = None
    dr = estimate_dissociation(
        amplitude,
        affinity,
        peak_current=peak_current,
        decay_time=decay_time,
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


def estimate_sigma(resting_spike_rate: float_array, cache: bool = True) -> float_array:
    """Estimate Sigma Value given Resting Spike Rate"""
    res = olfdata.find(
        "rest",
        ["ParamValue", "Frequencies", "metadata.dt"],
        lambda df: df.ParamKey == "sigma",
    )
    sigmas = res.iloc[-1].ParamValue * np.sqrt(res.iloc[-1]["metadata.dt"])
    rest_fs = res.iloc[-1].Frequencies
    return np.interp(resting_spike_rate, xp=rest_fs, fp=sigmas)


def estimate_current(
    spike_rate: float_array,
    sigma: float,
    sigma_atol: float = 1e-4,
    sigma_rtol: float = 1e-4,
    cache: bool = True,
) -> float_array:
    """Estimate Current Value given Spike Rate

    The current value is estimated from the `spike_rate` using the Frequency-Current (F-I)
    curve that is parameterized by the noise parameter `sigma`. If the F-I curve of the
    provided `sigma` value is not found, the F-I curve is calculated using `sigma`.

    Arguments:
        spike_rate: output OSN spike rate
        sigma: noise parameter of the NoisyConnorStevens Model
            This is used to find the appropriate F-I curve

    Keyword Arguments:
        sigma_atol: absolute tolerence of checking if the required `sigma` is close to previously cached values
        sigma_atol: relative tolerence of checking if the required `sigma` is close to previously cached values
        cache: save results of F-I curve if it is run due to `sigma` mismatch

    Returns:
        estimated currents
    """
    res = olfdata.find("FI", ["Currents", "Frequencies", "Params", "metadata.dt"])
    feasible_indices = []
    for n, row in res.iterrows():
        if "sigma" in row["Params"]:
            _sig = row["Params"]["sigma"] * np.sqrt(row["metadata.dt"])
            sigma_err = sigma - _sig
            sigma_rerr = abs(sigma_err) / min([abs(sigma), abs(_sig)])
            if abs(sigma_err) > sigma_atol or sigma_rerr > sigma_rtol:
                feasible_indices.append(n)
    feasible_indices = sorted(feasible_indices)
    if len(feasible_indices) > 0:
        exp_index = feasible_indices[-1]
        Is = res.loc[exp_index].Currents
        fs = res.loc[exp_index].Frequencies
    else:  # not found
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


def estimate_affinity(
    amplitude: float, steady_state_current: float_array
) -> float_array:
    """Estimate Affinity Value From Steady-State Current

    Arguments:
        amplitude: amplitude of the step odorant concentration waveform (in unit of [ppm])
        steady_state_current: steady-state OTP output current value

    Returns:
        estimated affinity value
    """
    res = olfdata.find("OTP", ["Amplitude", "Br", "Dr", "Peak", "SS"])
    tmp = res[res.Amplitude == amplitude]
    if len(tmp) == 0:
        res = res.iloc[-1]
    else:
        res = tmp.iloc[-1]
    I_ss = res.SS
    brs = res.Br
    drs = res.Dr
    v = res.Amplitude
    DR, BR = np.meshgrid(drs, brs)
    bvds = (BR / DR) * v
    bvds = bvds.ravel()
    I_ss_flat = I_ss.ravel()
    idx = np.argsort(bvds)

    # interpolate first
    bvd_intp = 10 ** np.linspace(-6, 3, 1000)
    ss_intp = np.interp(bvd_intp, bvds[idx], I_ss_flat[idx])

    from scipy.optimize import differential_evolution

    hill_f = lambda x, a, b, c, n: b + a * x ** n / (x ** n + c)

    def inverse_hill_f(y, a, b, c, n, x_ref):
        res = np.atleast_1d(np.power(c * (y - b) / (a - (y - b)), 1.0 / n))
        res[y < b] = x_ref.min()
        res[(y - b) > a] = x_ref.max()
        return res

    def cost(x, aff, ss):
        a, b, c, n = x
        pred = hill_f(aff, a, b, c, n)
        return np.linalg.norm(pred - ss)

    bounds = [(0, 100), (0, 100), (0, 100), (0.5, 2.0)]
    diffeq_ss = differential_evolution(
        cost, bounds, tol=1e-4, args=(bvd_intp, ss_intp), disp=False
    )

    if diffeq_ss.success:
        bvd_inverse = inverse_hill_f(steady_state_current, *diffeq_ss.x, bvd_intp)
    else:
        bvd_inverse = np.interp(steady_state_current, xp=I_ss_flat[idx], fp=bvds[idx])

    return bvd_inverse / amplitude


def estimate_dissociation(
    amplitude: float,
    affinity: float_array,
    peak_current: float_array = None,
    decay_time: float = 0.1,
) -> float_array:
    """Estimate Dissociation Rate

    Arguments:
        amplitude: concentration amplitude
        affinity: affinity values

    Keyword Arguments:
        peak_current: maximum current
        decay_time: settling time after offset of odorant input

    Returns:
        Estimated dissociation rate
    """
    if peak_current is None:
        if decay_time is not None:
            return 1.0 / decay_time
        else:
            return 10.0
    else:
        res = olfdata.find("OTP", ["Amplitude", "Br", "Dr", "Peak", "SS"])
        tmp = res[res.Amplitude == amplitude]
        if len(tmp) == 0:
            res = res.iloc[-1]
        else:
            res = tmp.iloc[-1]

        peak_current = np.atleast_1d(peak_current)
        peaks = peak_current.ravel()
        drs = np.zeros_like(peaks)
        for n, peak in enumerate(peaks):
            br_idx, dr_idx = np.unravel_index(
                np.argmin(np.abs(res.Peak - peak)), res.Peak.shape
            )
            drs[n] = res.Dr[dr_idx]
        if len(peak_current) == 1:
            return drs[0]
        else:
            return drs.reshape(peak_current.shape)


def estimate_resting_spike_rate(sigma: float_array) -> float_array:
    """Estimate Resting Spike Rate from Sigma Value

    Arguments:
        sigma: noise parameter

    Returns:
        estimated resting spike rate
    """
    res = olfdata.find(
        "rest",
        ["ParamValue", "Frequencies", "metadata.dt"],
        lambda df: df.ParamKey == "sigma",
    )
    sigmas = res.iloc[-1].ParamValue * np.sqrt(res.iloc[-1]["metadata.dt"])
    rest_fs = res.iloc[-1].Frequencies
    return np.interp(sigma, xp=sigmas, fp=rest_fs)


def run(t, waveform, br, dr, sigma):
    pass

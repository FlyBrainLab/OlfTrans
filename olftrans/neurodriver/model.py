"""NeuroDriver Models and Utilities

Examples:
    1. Compute F-I 
    >>> from olftrans.neurodriver import model
    >>> import numpy as np
    >>> dt = 5e-6
    >>> repeat = 50
    >>> Is = np.linspace(0,150,150)
    >>> _, fs = model.compute_fi(model.NoisyConnorStevens, Is, dt=dt, repeat=repeat, save=True)
    
    2. Compute Resting Spike Rate
    >>> from olftrans.neurodriver import model
    >>> import numpy as np
    >>> dt = 5e-6
    >>> repeat = 50
    >>> sigmas = np.linspace(0,0.005,150)
    >>> _, rest_fs = model.compute_resting(model.NoisyConnorStevens, 'sigma', sigmas/np.sqrt(dt), dt=dt, repeat=repeat, save=True)

    3. Compute Peak and SS Currents of OTP
    >>> from olftrans.neurodriver import model
    >>> import numpy as np
    >>> dt, amplitude = 5e-6, 100.
    >>> br_s = np.linspace(1e-2, 1000., 50)
    >>> dr_s = np.linspace(1e-2, 1000., 50)
    >>> _, _, I_ss, I_peak = model.compute_peak_ss_I(br_s, dr_s, dt=dt, amplitude=amplitude, save=True)
"""
from collections import OrderedDict
from neuroballad.models.element import Element
import numpy as np
import copy
import networkx as nx
import typing as tp
import os
from olftrans import ROOTDIR, DATADIR
from scipy.signal import savgol_filter
from . import NDComponents


class Model(Element):
    """NeuroBallad Element that also wraps the underlying NDComponent"""

    _ndcomp = None


class OTP(Model):
    """
    Odorant Transduction Process
    """

    element_class = "neuron"
    states = OrderedDict(
        [("v", 0.0), ("uh", 0.0), ("duh", 0.0), ("x1", 0.0), ("x2", 0.0), ("x3", 0.0)]
    )

    params = dict(
        br=1.0,
        gamma=0.13827484362015477,
        dr=10.06577381490841,
        c1=0.02159722808408414,
        a2=199.57381809612792,
        b2=51.886883149283406,
        a3=2.443964363230107,
        b3=0.9236173421313049,
        k23=9593.91481121941,
        CCR=0.06590587362782163,
        ICR=91.15901333340182,
        L=0.8,
        W=45.0,
    )
    _ndcomp = NDComponents.OTP


class NoisyConnorStevens(Model):
    """
    Noisy Connor-Stevens Neuron Model

    F-I curve is controlled by `sigma` parameter

    Note:
        `sigma` value should be scaled by `sqrt(dt)` as `sigma/sqrt(dt)`
        where `sigma` is the standard deviation of the Brownian Motion
    """

    states = dict(n=0.0, m=0.0, h=1.0, a=1.0, b=1.0, v1=-60.0, v2=-60.0, refactory=0.0)

    params = dict(
        ms=-5.3,
        ns=-4.3,
        hs=-12.0,
        gNa=120.0,
        gK=20.0,
        gL=0.3,
        ga=47.7,
        ENa=55.0,
        EK=-72.0,
        EL=-17.0,
        Ea=-75.0,
        sigma=2.05,
        refperiod=1.0,
    )
    _ndcomp = NDComponents.NoisyConnorStevens


def compute_fi(
    NeuronModel: Model,
    Is: np.ndarray,
    repeat: int = 1,
    input_var: str = "I",
    spike_var: str = "spike_state",
    dur: float = 2.0,
    start: float = 0.5,
    dt: float = 1e-5,
    neuron_params: dict = None,
    save: bool = True,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Compute Frequency-Current relationship of Neuron

    Note:
        if `save==True`, a dictionary with name `f"{NeuronModel.__name__}_FI.npz"`
        is saved to `olftrans.DATADIR` with the following attributes:

            - `I`: Is
            - `f`: spike_rates
            - `params`: params, which is the default params of the model updated by `neuron_params`
            - `metadata`: a dictionary recording dt,dur,start,stop,repeat

    Returns:
        Is: 1d array of Currents
        spike_rates: 1d array Spiking Frequencies, dimension matches param_values

    Examples:
        Basic Usage:
        >>> from olftrans.neurodriver.model import NoisyConnorStevens
        >>> Is = np.linspace(0., 150, 100)
        >>> dt = 1e-5
        >>> _, fs = compute_fi(NoisyConnorStevens, Is, repeat=50, dt=dt, neuron_params={'sigma': 0.005/np.sqrt(dt)})

        We can look for the input current value from spike rate
        >>> target_spike_rate = 150.  # [Hz]
        >>> target_I = np.interp(x=target_spike_rate, xp=fs, fp=fs)
    """
    from neurokernel.LPU.LPU import LPU
    from neurokernel.LPU.InputProcessors.StepInputProcessor import StepInputProcessor
    from neurokernel.LPU.OutputProcessors.OutputRecorder import OutputRecorder

    stop = dur
    t = np.arange(0.0, dur, dt)
    clsname = NeuronModel.__name__
    Is = np.atleast_1d(Is)
    neuron_params = neuron_params or {}
    params = copy.deepcopy(NeuronModel.params)
    params.update(neuron_params)

    G = nx.MultiDiGraph()
    csn_ids = np.empty((len(Is), repeat), dtype=object)
    for n_I, _I in enumerate(Is):
        for r in range(repeat):
            _id = f"{clsname}-I{n_I}-{r}"
            G.add_node(_id, **{"label": _id, "class": clsname}, **params)
            csn_ids[n_I, r] = _id

    fi = StepInputProcessor(
        variable=input_var,
        uids=csn_ids.ravel().astype(str),
        val=np.repeat(Is, repeat),
        start=start,
        stop=stop,
    )
    fo = OutputRecorder([(spike_var, None)])

    lpu = LPU(
        dt,
        "obj",
        G,
        device=0,
        id=f"F-I {clsname}",
        input_processors=[fi],
        output_processors=[fo],
        debug=False,
        manager=False,
        extra_comps=[NeuronModel._ndcomp],
    )
    lpu.run(steps=len(t))

    Nspikes = np.zeros((len(Is), repeat))
    spikes = fo.get_output(var="spike_state")
    for n_I, _I in enumerate(Is):
        for r in range(repeat):
            _id = f"{clsname}-I{n_I}-{r}"
            Nspikes[n_I, r] = np.sum(
                np.logical_and(
                    spikes[_id]["data"] >= start, spikes[_id]["data"] <= stop
                )
            )
    spike_rates = Nspikes.mean(-1) / (stop - start)

    if save:
        fname = f"{clsname}_FI"
        np.savez(
            os.path.join(DATADIR, fname),
            I=Is,
            f=spike_rates,
            params=params,
            metadata=dict(dt=dt, dur=dur, start=start, stop=stop, repeat=repeat),
        )
    return Is, spike_rates


def compute_peak_ss_I(
    br_s: np.ndarray,
    dr_s: np.ndarray,
    dt: float = 1e-5,
    dur: float = 2.0,
    start: float = 0.5,
    save: bool = True,
    amplitude: float = 100.0,
) -> np.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Peak and Steady-State Current output of OTP Model

    Note:
        if `save==True`, a dictionary with name `"OTP_peak_ss.npz"`
        is saved to `olftrans.DATADIR` with the following attributes:

            - `br` : br_s
            - `dr` : dr_s
            - `ss` : I_ss
            - `peak` : I_peak
            - `amplitude` : amplitude
            - `metadata`: a dictionary recording dt,dur,start,stop,repeat

    Returns:
        br_s: 1d array of binding rate
        dr_s: 1d array of dissociation rate
        I_ss: 2d array of resultant steady-state currents
        I_peak: 2d array of resultant peak currents

    Examples:
        Basic Usage:
        >>> br_s = np.linspace(1e-1, 100., 100)
        >>> dr_s = np.linspace(1e-1, 100., 100)
        >>> _, _, I_ss, I_peak = compute_peak_ss_I(br_s, dr_s, save=True)

        Plotting Steady-State Current against affinity:
        >>> DR, BR = np.meshgrid(dr_s, br_s)
        >>> plt.plot((BR/DR).ravel(), I_ss.ravel()) # steady-state current is only dependent on dissociation

        We can also look for the affinity value from steady-state Current output
        >>> target_I = 50.  # [uA]
        >>> target_aff = np.interp(x=target_I, xp=I_ss.ravel(), fp=(BR/DR).ravel())
    """
    from neurokernel.LPU.LPU import LPU
    from neurokernel.LPU.InputProcessors.StepInputProcessor import StepInputProcessor
    from neurokernel.LPU.OutputProcessors.OutputRecorder import OutputRecorder

    stop = dur
    t = np.arange(0, dur, dt)
    G = nx.MultiDiGraph()
    otp_ids = np.empty((len(br_s), len(dr_s)), dtype=object)
    for n_b, _br in enumerate(br_s):
        for n_d, _dr in enumerate(br_s):
            _id = f"OTP-B{n_b}-D{n_d}"
            _params = copy.deepcopy(OTP.params)
            _params.update(dict(br=_br, dr=_dr))
            G.add_node(_id, **{"label": _id, "class": "OTP"}, **_params)
            otp_ids[n_b, n_d] = _id

    fi = StepInputProcessor(
        variable="conc",
        uids=otp_ids.ravel().astype(str),
        val=amplitude,
        start=start,
        stop=stop,
    )
    fo = OutputRecorder([("I", None)])

    lpu = LPU(
        dt,
        "obj",
        G,
        device=0,
        id="OTP Currents",
        input_processors=[fi],
        output_processors=[fo],
        debug=False,
        manager=False,
        extra_comps=[OTP._ndcomp],
    )
    lpu.run(steps=len(t))
    I_ss = np.zeros((len(br_s), len(dr_s)))
    I_peak = np.zeros((len(br_s), len(dr_s)))
    Is = fo.get_output(var="I")
    for n_b, _br in enumerate(br_s):
        for n_d, _dr in enumerate(br_s):
            _id = f"OTP-B{n_b}-D{n_d}"
            I_ss[n_b, n_d] = Is[_id]["data"][-1]
            I_peak[n_b, n_d] = Is[_id]["data"].max()
    if save:
        fname = f"OTP_peak_ss"
        np.savez(
            os.path.join(DATADIR, fname),
            br=br_s,
            dr=dr_s,
            ss=I_ss,
            peak=I_peak,
            amplitude=amplitude,
            metadata=dict(dt=dt, dur=dur, start=start, stop=stop),
        )
    return br_s, dr_s, I_ss, I_peak


def compute_resting(
    NeuronModel: Model,
    param_key: str,
    param_values: np.ndarray,
    repeat: int = 1,
    input_var: str = "I",
    spike_var: str = "spike_state",
    dur: float = 2.0,
    dt: float = 1e-5,
    save: bool = True,
    smoothen: bool = True,
    savgol_window: int = 15,
    savgol_order: int = 3,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Compute Resting Spike Rate of a Neuron as Parameter varies

    Arguments:
        NeuronModel: Model to be used to compute Resting Spike Rate
        param_key:

    Note:
        if `save==True`, a dictionary with name `f"{NeuronModel.__name__}_resting.npz"`
        is saved to `olftrans.DATADIR` with the following attributes:

            - `param_key`: param_key
            - `param_values`: param_values
            - `f`: spike_rates
            - `metadata`: a dictionary recording dt,dur,start,stop,repeat

    Returns:
        param_values: 1d array of param_values
        spike_rates: 1d array Spiking Frequencies, dimension matches param_values

    Examples:
        Basic Usage:
        >>> from olftrans.neurodriver.model import NoisyConnorStevens
        >>> sigmas = np.linspace(0., 0.005, 100)
        >>> _, fs = compute_resting(NoisyConnorStevens, 'sigma', sigmas, repeat=50)

        We can look for the parameter value from resting spike rate
        >>> target_resting = 8.  # [Hz]
        >>> target_sigma = np.interp(x=target_resting, xp=fs, fp=sigmas)
    """
    from neurokernel.LPU.LPU import LPU
    from neurokernel.LPU.InputProcessors.StepInputProcessor import StepInputProcessor
    from neurokernel.LPU.OutputProcessors.OutputRecorder import OutputRecorder

    start, stop = 0.0, dur
    t = np.arange(0.0, dur, dt)
    clsname = NeuronModel.__name__

    param_values = np.atleast_1d(param_values)

    G = nx.MultiDiGraph()
    csn_ids = np.empty((len(param_values), repeat), dtype=object)
    for n_p, val in enumerate(param_values):
        params = copy.deepcopy(NeuronModel.params)
        params.update({param_key: val})
        for r in range(repeat):
            _id = f"{clsname}-P{n_p}-{r}"
            G.add_node(_id, **{"label": _id, "class": clsname}, **params)
            csn_ids[n_p, r] = _id

    fi = StepInputProcessor(
        variable=input_var,
        uids=csn_ids.ravel().astype(str),
        val=0.0,
        start=start,
        stop=stop,
    )
    fo = OutputRecorder([(spike_var, None)])

    lpu = LPU(
        dt,
        "obj",
        G,
        device=0,
        id=f"Resting Spike Rate {clsname} - Against {param_key}",
        input_processors=[fi],
        output_processors=[fo],
        debug=False,
        manager=False,
        extra_comps=[NeuronModel._ndcomp],
    )
    lpu.run(steps=len(t))

    Nspikes = np.zeros((len(param_values), repeat))
    spikes = fo.get_output(var="spike_state")
    for n_p, val in enumerate(param_values):
        for r in range(repeat):
            _id = f"{clsname}-P{n_p}-{r}"
            Nspikes[n_p, r] = np.sum(
                np.logical_and(
                    spikes[_id]["data"] >= start, spikes[_id]["data"] <= stop
                )
            )
    spike_rates = Nspikes.mean(-1) / (stop - start)

    if smoothen:
        spike_rates = savgol_filter(spike_rates, savgol_window, savgol_order)

    if save:
        fname = f"{clsname}_resting"
        np.savez(
            os.path.join(DATADIR, fname),
            param_key=param_key,
            param_values=param_values,
            f=spike_rates,
            metadata=dict(dt=dt, dur=dur, start=start, stop=stop, repeat=repeat),
        )
    return param_values, spike_rates

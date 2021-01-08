import os
import numpy as np
from olftrans import DATADIR
from collections import namedtuple


def get_data():
    fi = np.load(os.path.join(DATADIR, "NoisyConnorStevens_FI.npz"), allow_pickle=True)
    resting = np.load(
        os.path.join(DATADIR, "NoisyConnorStevens_resting.npz"), allow_pickle=True
    )
    CSNData = namedtuple("CSNData", ["sigma", "I", "f", "rest_sigma", "rest_f"])
    dt = resting["metadata"].item()["dt"]
    return CSNData(
        sigma=fi["params"].item()["sigma"] * np.sqrt(dt),
        rest_sigma=resting["param_values"] * np.sqrt(dt),
        I=fi["I"],
        f=fi["f"],
        rest_f=resting["f"],
    )


DATA = get_data()

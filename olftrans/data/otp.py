import os
import numpy as np
from olftrans import DATADIR
from collections import namedtuple

def get_data():
    data = np.load(os.path.join(DATADIR, 'OTP_peak_ss.npz'), allow_pickle=True)
    OTPData = namedtuple('OTPData', ['br', 'dr', 'ss', 'peak', 'amplitude'])
    return OTPData(
        br=data['br'],dr=data['dr'],ss=data['ss'], peak=data['peak'], amplitude=data['amplitude'].item()
    )

DATA = get_data()
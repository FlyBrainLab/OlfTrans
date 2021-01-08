import os
import pandas as pd
from olftrans import DATADIR

def get_data():
    df = pd.read_csv(os.path.join(DATADIR, 'Kreher_2005.csv'), index_col=0, header=0)
    df.columns = [k.split('Or')[1] for k in df.columns]
    df.columns.name = 'OR'
    df[:-1] = df[:-1] + df.iloc[-1]
    df[df<0] = 0.
    return df

DATA = get_data()
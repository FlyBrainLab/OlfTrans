import os
import pandas as pd
from olftrans import DATADIR

def get_data():
    df = pd.read_csv(os.path.join(DATADIR, 'Hallem_Carlson_2006.csv'), index_col=0, header=1)
    df[:-1] = df[:-1] + df.iloc[-1]
    df[df<0] = 0.
    df.columns.name = 'OR'
    df.index.name = None
    return df

DATA = get_data()
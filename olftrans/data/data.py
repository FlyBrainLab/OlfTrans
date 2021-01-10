# pylint:disable=no-member
import os
import shutil
import datetime
import numpy as np
from olftrans import DATADIR
import pandas as pd
import typing as tp
from dataclasses import dataclass, field, asdict


@dataclass
class DataMetadata:
    dt: float
    dur: float
    start: float
    stop: float
    savgol_window: int = None
    savgol_order: int = None


@dataclass
class DataRest:
    Model: str
    ParamKey: str
    ParamValue: np.ndarray
    Smoothen: bool
    Frequencies: np.ndarray
    InputVar: str
    SpikeVar: str
    Params: dict
    Repeats: int


@dataclass
class DataFI:
    Model: str
    Currents: np.ndarray
    Frequencies: np.ndarray
    InputVar: str
    SpikeVar: str
    Params: dict
    Repeats: int


@dataclass
class DataOTP:
    Model: str
    Amplitude: float
    Br: np.ndarray
    Dr: np.ndarray
    Peak: np.ndarray
    SS: np.ndarray


def _rm_rf(path):
    """Remove Folder"""
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


class Data:
    def __init__(self):
        self.CACHEDIR = os.path.join(DATADIR, "npz_cache")
        self.fnames = dict(
            REST=os.path.join(self.CACHEDIR, f"records_rest.json"),
            FI=os.path.join(self.CACHEDIR, f"records_fi.json"),
            OTP=os.path.join(self.CACHEDIR, f"records_otp.json"),
        )
        self.dfs = dict(REST=None, FI=None, OTP=None)
        self.setup(False)

    def setup(self, clean=False):
        if not os.path.exists(self.CACHEDIR):
            os.makedirs(self.CACHEDIR)
        if clean:
            confirm = "unknown"
            while confirm.lower() not in ["y", "n", "yes", "no"]:
                confirm = input(
                    f"WARNING: setup(clean) with remove everything in cache folder, proceed? [n]"
                )
                if confirm == "":
                    # assume as n
                    break
                elif confirm.lower() in ["y", "yes"]:
                    _rm_rf(self.CACHEDIR)
                    os.makedirs(self.CACHEDIR)
                elif confirm.lower() in ["n", "no"]:
                    break

        for key, DClass in zip(["REST", "FI", "OTP"], [DataRest, DataFI, DataOTP]):
            df = None
            if os.path.exists(self.fnames[key]):
                df = pd.read_json(self.fnames[key], orient="index")
            if df is None or df.empty or not os.path.exists(self.fnames[key]):
                columns = (
                    ["Time", "FileName"]
                    + list(DClass.__dataclass_fields__.keys())
                    + [
                        f"metadata.{k}"
                        for k in DataMetadata.__dataclass_fields__.keys()
                    ]
                )
                df = pd.DataFrame(columns=columns)
                df.to_json(self.fnames[key], orient="index")

            # convert list and tuple to numpy array
            df = df.applymap(
                lambda x: np.asarray(x) if isinstance(x, (list, tuple)) else x
            )
            self.dfs[key] = df

    def save(
        self, data_type: tp.Union["otp", "rest", "fi"], data, metadata, fname=None
    ):
        if data_type.upper() not in ["FI", "REST", "OTP"]:
            raise ValueError(
                f"data_type {data_type} not understood, must be OTP/REST/FI"
            )
        data_type = data_type.upper()
        time_stamp = datetime.datetime.now().isoformat()
        if fname is None:
            time_saveable = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            fname = f"{data_type}_{time_saveable}"
        if isinstance(data, (DataFI, DataOTP, DataRest)):
            data = asdict(data)
        data.update({"Time": time_stamp, "FileName": fname})
        if isinstance(metadata, DataMetadata):
            metadata = asdict(metadata)
        data.update({f"metadata.{key}": val for key, val in metadata.items()})
        self.dfs[data_type] = self.dfs[data_type].append(data, ignore_index=True)
        self.dfs[data_type].to_json(self.fnames[data_type], orient="index")
        np.savez_compressed(os.path.join(self.CACHEDIR, fname), **data)

    def find(
        self,
        data_type: tp.Union["otp", "rest", "fi"],
        key: str,
        loc_args: tp.Callable = None,
    ):
        if data_type.upper() not in ["FI", "REST", "OTP"]:
            raise ValueError(
                f"data_type {data_type} not understood, must be OTP/REST/FI"
            )
        data_type = data_type.upper()
        df = self.dfs[data_type]
        if loc_args is None:
            return df[key]
        else:
            return df.loc[loc_args(df)][key]


olfdata = Data()

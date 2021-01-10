"""FlyBrainLab compatible module

Classes:
    Config: configuration of FBL Module
    FBL: FlyBrainLab-compatible module to be consumed by other FBL packages

Attributes:
    LARVA: an instance of FBL class that is loaded with Larva (Kreher2005) data
    ADULT: an instance of FBL class that is loaded with Adult (HallemCarlson20016) data
"""
import os
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
import typing as tp
import copy
import pandas as pd
from neurokernel.LPU.NDComponents.NDComponent import NDComponent
from .neurodriver import NDComponents as ndcomp
from .neurodriver import model
from . import data
from .olftrans import estimate_resting_spike_rate, estimate_sigma, estimate
from warnings import warn


@dataclass
class Config:
    """Configuration for FlyBrainLab-compatible Module

    Attributes:
        NR: Number of Receptor Types
        NO: Number of OSNs per Receptor Type
        affs: Affinity Values
        drs: Dissociation Rates
        receptor_names: Name of receptors of length NR
        resting: Resting OSN Spike Rates [Hz]
        sigma: NoisyConnorStevens Noise Standard Deviation
    """

    NR: int = field(init=False)
    NO: tp.Iterable[int]
    affs: tp.Iterable[float]
    drs: tp.Iterable[float] = None
    receptor_names: tp.Iterable[str] = None
    resting: float = None
    sigma: float = None

    def __post_init__(self):
        self.affs = np.asarray(self.affs)
        self.NR = len(self.affs)
        if np.isscalar(self.NO):
            self.NO = np.full((self.NR,), self.NO, dtype=int)
        else:
            assert (
                len(self.NO) == self.NR
            ), f"If `NO` is iterable, it has to have length same as affs."

        if self.receptor_names is None:
            self.receptor_names = [f"Or{r}" for r in range(self.NR)]
        else:
            self.receptor_names = np.asarray(self.receptor_names)
            assert (
                len(self.receptor_names) == self.NR
            ), f"If `receptor_names` is specified, it needs to have length the same as affs."
        if self.drs is None:
            self.drs = np.full((self.NR,), 10.0)
        elif np.isscalar(self.drs):
            self.drs = np.full((self.NR,), self.drs)
        else:
            self.drs = np.asarray(self.drs)
            assert (
                len(self.drs) == self.NR
            ), f"If Dissociation rate (dr) is specified as iterable, it needs to have length the same as affs."

        assert not all(
            [v is None for v in [self.resting, self.sigma]]
        ), "Resting and Sigma cannot both be None"
        if self.resting is None:
            self.resting = estimate_resting_spike_rate(self.sigma)
        elif self.sigma is None:
            self.sigma = estimate_sigma(self.resting)


@dataclass
class FBL:
    """FlyBrainLab-compatible Module

    Attributes:
        graph: networkx graph describing the executable circuit
        inputs: input variable and uids dictionary
        outputs: input variable and uids dictionary
        extra_comps: list of neurodriver extra components
        config: configuration
        affinities: a pandas dataframe with affinities saved as reference
            - index: odorants
            - columns: receptor names
    """

    graph: nx.MultiDiGraph
    inputs: dict
    outputs: dict
    extra_comps: tp.List[NDComponent] = field(
        default_factory=lambda: [ndcomp.OTP, ndcomp.NoisyConnorStevens]
    )
    config: Config = None
    affinities: pd.DataFrame = field(default=None, init=None)

    @classmethod
    def create_from_config(cls, cfg: Config):
        """Create Instance from Config

        Arguments:
            cfg: Config instance that specifies the configuration of the module

        Returns:
            A new FBL instance
        """
        G = nx.MultiDiGraph()
        bsg_params = copy.deepcopy(model.NoisyConnorStevens.params)
        bsg_params.update(sigma=cfg.sigma)
        otp_uids = []
        bsg_uids = []
        for n, (_or, _aff, _dr) in enumerate(
            zip(cfg.receptor_names, cfg.affs, cfg.drs)
        ):
            _br = _aff * _dr
            otp_params = copy.deepcopy(model.OTP.params)
            otp_params.update(br=_br, dr=_dr)
            for o in range(cfg.NO[n]):
                otp_id = f"OSN-OTP-{_or}-O{o}"
                bsg_id = f"OSN-BSG-{_or}-O{o}"
                G.add_node(
                    otp_id,
                    **{
                        "label": otp_id,
                        "class": "OTP",
                        "_receptor": _or,
                        "_repeat_idx": o,
                    },
                    **otp_params,
                )
                G.add_node(
                    bsg_id,
                    **{
                        "label": bsg_id,
                        "class": "NoisyConnorStevens",
                        "_receptor": _or,
                        "_repeat_idx": o,
                    },
                    **bsg_params,
                )
                otp_uids.append(otp_id)
                bsg_uids.append(bsg_id)
        otp_uids = np.asarray(otp_uids, dtype="str")
        bsg_uids = np.asarray(bsg_uids, dtype="str")
        inputs = {"conc": otp_uids}
        outputs = {"V": bsg_uids, "spike_state": bsg_uids}
        return cls(graph=G, inputs=inputs, outputs=outputs, config=cfg)

    def __post_init__(self):
        """Parse config from graph if not specified"""
        if self.config is None:
            self.config = FBL.get_config(self)

    @classmethod
    def get_config(cls, fbl) -> Config:
        """Parse Config from given FBL instance"""
        import pandas as pd

        df = pd.DataFrame.from_dict(dict(fbl.graph.nodes(data=True)), orient="index")
        df_otp = df[df["class"] == "OTP"]
        df_ors = df_otp[["_receptor", "br", "dr"]]
        df_ors = df_ors.drop_duplicates()
        df_ors.loc[:, "aff"] = df_ors["br"] / df_ors["dr"]
        df_ors = df_ors.set_index("_receptor")
        sr_repeat = df_otp["_receptor"].value_counts()
        df_ors.loc[:, "repeat"] = sr_repeat
        df_bsg = df[df["class"] == "NoisyConnorStevens"]
        sr_sigma = df_bsg.sigma.value_counts()
        if len(sr_sigma) > 1:
            warn(
                "get_config only supports globally unique sigma values, taking the most common value"
            )
        sigma = sr_sigma.iloc[0]
        return Config(
            NO=df_ors.repeat.values,
            affs=df_ors.aff.values,
            drs=df_ors.dr.values,
            receptor_names=df_ors.index.values,
            sigma=sigma,
        )

    def update_affs(self, affs) -> None:
        """Update Affinities and Change Circuit Accordingly"""
        assert isinstance(affs, dict)

        for _or, _aff in affs.items():
            if _or in self.config.receptor_names:
                idx = list(self.config.receptor_names).index(_or)
                self.config.affs[idx] = _aff
                # get dr
                # compute br
                # update br and dr for given receptor

            else:
                warn(
                    f"Affinity Value key '{_or}' is not in known receptor names, skipping"
                )
                continue

    def update_graph_attributes(
        self,
        data_dict: dict,
        nodes: tp.Union["otp", "bsg"] = "otp",
        receptor: tp.Iterable[str] = None,
        node_predictive: tp.Callable[[nx.classes.reportviews.NodeView], bool] = None,
    ) -> None:
        """Update Attributes of the graph

        Arguments:
            data_dict: a dictionary of {attr: value}

        Keyword Arguments:
            nodes: nodes to update, 'otp' or 'bsg'
            receptor: filter nodes with receptor
            node_predictive: additional filtering of nodes from `nx.nodes` call

        Example:
            >>> fbl.update_graph_attributes({'sigma':1.}, nodes='bsg', receptor=None)
        """
        if node_predictive is None:
            node_predictive = lambda node_id, data: True
        if nodes.lower() == "otp":
            clsname = "OTP"
        elif nodes.lower() == "bsg":
            clsname = "NoisyConnorStevens"
        else:
            raise ValueError("nodes need to be 'otp' or 'bsg'")
        node_uids = [
            key
            for key, val in self.graph.nodes(data=True)
            if val["_receptor"] == receptor
            and val["class"] == clsname
            and node_predictive(key, val)
        ]
        update_dict = {_id: data_dict for _id in node_uids}
        nx.set_node_attributes(self.graph, update_dict)

    def simulate(
        self,
        t: np.ndarray,
        inputs: tp.Any,
        record_var_list: tp.Iterable[tp.Tuple[str, tp.Iterable]] = None,
        sample_interval: int = 1,
    ) -> tp.Tuple["FileInput", "FileOutput", "LPU"]:
        """
        Update Affinities and Change Circuit Accordingly

        Arguments:
            t: input time array
            inputs: input data
                - if is `BaseInputProcessor` instance, passed to LPU directly
                - if is dictionary, passed to ArrayInputProcessor if is compatible

        Keyword Argumnets:
            record_var_list: [(var, uids)]
            sample_interval: interval at which output is recorded

        Returns:
            fi: Input Processor
            fo: Output Processor
            lpu: LPU instance
        """
        from neurokernel.LPU.LPU import LPU
        from neurokernel.LPU.InputProcessors.BaseInputProcessor import (
            BaseInputProcessor,
        )
        from neurokernel.LPU.InputProcessors.ArrayInputProcessor import (
            ArrayInputProcessor,
        )
        from neurokernel.LPU.OutputProcessors.OutputRecorder import OutputRecorder

        dt = t[1] - t[0]
        if isinstance(inputs, BaseInputProcessor):
            fi = inputs
        elif isinstance(inputs, dict):
            for data in inputs.values():
                assert "uids" in data
                assert "data" in data
                assert isinstance(data["data"], np.ndarray)
            fi = ArrayInputProcessor(inputs)
        else:
            raise ValueError("Input not understood")
        fo = OutputRecorder(record_var_list, sample_interval=sample_interval)
        lpu = LPU(
            dt,
            "obj",
            self.graph,
            device=0,
            id=f"OlfTrans",
            input_processors=[fi],
            output_processors=[fo],
            debug=False,
            manager=False,
            extra_comps=self.extra_comps,
        )
        lpu.run(steps=len(t))
        return fi, fo, lpu


def load_adult_affinities(cfg):
    """Load HallemCarlson Spike Data and Parse to Affinities"""
    df = data.HallemCarlson.DATA
    df = df[~df.isna()]
    est = estimate(100.0, cfg.resting, df.values, decay_time=0.1, cache=True)
    df_aff = df.copy()
    df_aff[~df.isna()] = est.affs
    return df_aff


def load_larva_affinities(cfg):
    """Load HallemCarlson Spike Data and Parse to Affinities"""
    df = data.Kreher.DATA
    df = df[~df.isna()]
    est = estimate(100.0, cfg.resting, df.values, decay_time=0.1, cache=True)
    df_aff = df.copy()
    df_aff[~df.isna()] = est.affs
    return df_aff


larva_cfg = Config(
    affs=np.zeros((21,)),
    NO=1,
    drs=10.0,
    resting=8.0,
)
LARVA = FBL.create_from_config(larva_cfg)
LARVA.affinities = load_larva_affinities(larva_cfg)

adult_cfg = Config(
    affs=np.zeros((51,)),
    NO=50,
    drs=10.0,
    resting=8.0,
)
ADULT = FBL.create_from_config(adult_cfg)
ADULT.affinities = load_adult_affinities(adult_cfg)

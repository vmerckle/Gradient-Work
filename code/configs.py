import os
from types import SimpleNamespace
import pickle

import numpy as np
import torch
import torchvision

#torch.use_deterministic_algorithms(True)
#gpudevice = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

from utils import *
import configsGenData, configsGenModel

default = {
        "typefloat": "float32",
        "threadcount": 1,
        "device": "cpu",
        "seed": 1,
        "algo": "proxpoint",
        "algoD": {
            "dist": "frobenius",
            "inneriter": 100,
            "gamma": 1e-1,
            "recordinner": True,
            "recordinnerlayers": False,
            "LRdecay": 1.0,
            "onlyTrainFirstLayer": True,
            "opti": "AdamW",
            "batched": False,
            "batch_size": 2,
            "optiD": {
                "momentum":0.95,
                "weight_decay": 0,
                "lr": 1e-3,
            },
        },
        "data": "random",
        "dataD": {
            "seed": 2,
            "sampling": "uniform",
            "Ynoise": 0,
            "d": 10,
            "n": 11,
            },
        "init": "normal",
        "initD": {
            "seed": 3,
            "onlypositives": False,
            "scale": 1e-2,
            "m": 12,
            },
    }

def applyconfig(D):
    X, Y, Xb = configsGenData.genData(D)
    D["dataD"]["n"], D["dataD"]["d"] = X.shape
    ly1, ly2 = configsGenModel.genModel(D)
    opti = loadOpti(D)
    D.update({"X":X, "Y":Y, "Xb":Xb})
    opti.load(X, Y, ly1, ly2)
    return opti

def loadOpti(D):
    x = SimpleNamespace(**D)
    torch.set_num_threads(x.threadcount)
    if x.typefloat == "float32":
        dtype = torch.float32
    elif x.typefloat == "float64":
        dtype = torch.float64
    else:
        raise Exception("wrong typefloat", x.typefloat)

    algo = x.algo
    if algo == "GD":
        from algo_GD_torch import torch_descent
        opti = torch_descent(x.algoD, device=x.device)
    elif algo == "JKO":
        if x.proxf == "cvxpy":
            from jko_proxf_cvxpy import jko_cvxpy
            opti = jko_cvxpy(x.algoD)
        elif x.proxf == "scipy":
            from jko_proxf_scipy import jko_scipy
            opti = jko_scipy(x.algoD)
        elif x.proxf == "pytorch":
            from jko_proxf_pytorch import jko_pytorch
            opti = jko_pytorch(x.algoD)
        else:
            raise Exception(f"config bad proxf='{x.proxf}' choice")
    elif algo == "proxpoint":
        from algo_prox import proxpoint
        opti = proxpoint(x.algoD, dtype=dtype, device=x.device)
    else:
        raise Exception("config bad algo choice")

    return opti




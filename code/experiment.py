#!/usr/bin/python

import argparse
import time
import os.path
import os
import pickle
from rich import print
from inspect import getmembers, isfunction, isclass
import datetime

# my libraries
from utils import *
from runner import runner
import postprocess
import configs

def debug(args):
    expname = "debug"
    folder = f"data/{expname}"
    config = configs.default
    config["expdesc"] = args.d
    config["algo"] = "GD"
    config["algo"] = "proxpoint"
    config["algoD"]["recordinnerlayers"] = True
    config["algoD"]["batched"] = False
    config["algoD"]["dist"] = "wasser"
    config["dataD"]["d"] = 2
    debugstop = lambda D, opti, num, timestart : num > 5
    runner(config, expname, stopper=debugstop)

def mnist(args):
    expname = "mnist_regression_fit"
    folder = f"data/{expname}"
    config = configs.default
    config["expdesc"] = args.d
    #config["proxD"]["recordinnerlayers"] = True
    config["algo"] = "GD"
    config["algoD"] = { 
                        "momentum":0.95,
                        "opti": "AdamW",
                        "beta": 0,
                        "lr": 1e-3,
                        "onlyTrainFirstLayer": True,
                        }
    config["m"] = 100
    config["datatype"] = "mnist"
    #config["device"] = "cpu"
    config["device"] = "cuda"
    debugstop = lambda D, opti, num, timestart : False
    logger = lambda D, opti, num : None
    runner(config, expname, stopper=debugstop, logger=logger)

def quickexp(args):
    expname = "quick_wasser"
    folder = f"data/{expname}"
    config = {
            "typefloat": "float32",
            "threadcount": 1,
            "device": "cpu",
            "seed": 1,
            "algo": "proxpoint",
            "algoD": {
                "dist": "frobenius",
                "inneriter": 500,
                "gamma": 1e-2,
                "recordinner": True,
                "recordinnerlayers": False,
                "onlyTrainFirstLayer": True,
                "opti": "AdamW",
                "batched": False,
                "batch_size": 2,
                "LRdecay": 0.99993,
                "optiD": {
                    "momentum":0.99,
                    "weight_decay": 0,
                    "lr": 1e-4,
                },
            },
            "data": "random",
            "dataD": {
                "seed": 3,
                "sampling": "normal",
                "eps": 0,
                "d": 2,
                "n": 70,
                },
            "init": "normal",
            "initD": {
                "seed": 3,
                "onlypositives": False,
                "scale": 1e-3,
                "m": 1000,
                },
        }
    runner(config, expname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("dist", type=int)
    parser.add_argument( "--debug", action="store_true")
    parser.add_argument( "-d")
    args = parser.parse_args()
    if args.debug:
        debug(args)
    else:
        quickexp(args)

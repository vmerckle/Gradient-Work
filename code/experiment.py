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


def shouldstop(X1, opti, num):
    start = X1["timestart"]
    step = num > 7
    timec =  time.time() - start > 1
    loss = opti.loss() < 1e-4

    return step 
    return loss
    return timec or loss
    return timec or step 

def debug(args):
    expname = "debug"
    folder = f"data/{expname}"
    config = configs.default
    config["expdesc"] = args.d
    config["algoD"]["recordinnerlayers"] = True
    config["algoD"]["dist"] = "wasser"
    debugstop = lambda D, opti, num : num > 5
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
    debugstop = lambda D, opti, num : False
    logger = lambda D, opti, num : None
    runner(config, expname, stopper=debugstop, logger=logger)

def quickexp(args):
    expname = "quick wasser"
    folder = f"data/{expname}"
    config = {
            "NNseed": 4,
            "dataseed": 4,
            "typefloat": "float32",
            "threadcount": 1,
            "device": "cuda",
            "algo": "proxpoint",
            "algoD": {"dist": "wasser",
                      "inneriter": 200,
                      "gamma": 1e-1,
                      "recordinner": True,
                      "recordinnerlayers": False,
                      "momentum":0.95,
                      "opti": "AdamW",
                      "beta": 0,
                      "lr": 1e-4,
                      "innerlr": 1e-4,
                      "onlyTrainFirstLayer": True,
                      },
            "datatype": "random",
            "Xsampling": "uniform",
            "onlypositives": False,
            "Ynoise": 0,
            "beta": 0,
            "scale": 1e-2,
            "m": 300,
            "d": 300,
            "n": 3000,
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

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
    config["proxD"]["recordinnerlayers"] = True
    debugstop = lambda D, opti, num : num > 10
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
                        "opti": "Adadelta",
                        "beta": 0,
                        "lr": 1e-3,
                        "onlyTrainFirstLayer": True,
                        }
    config["m"] = 100
    config["datatype"] = "mnist"
    config["device"] = "cuda"
    config["device"] = "cpu"
    debugstop = lambda D, opti, num : False
    logger = lambda D, opti, num : None
    runner(config, expname, stopper=debugstop, logger=logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("dist", type=int)
    parser.add_argument( "--debug", action="store_true")
    parser.add_argument( "-d")
    args = parser.parse_args()
    if args.debug:
        debug(args)
    else:
        mnist(args)

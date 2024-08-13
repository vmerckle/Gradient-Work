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

def debug():
    expname = "debug"
    folder = f"data/{expname}"
    config = configs.default
    debugstop = lambda D, opti, num : num > 10
    runner(config, expname, stopper=debugstop)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("dist", type=int)
    parser.add_argument( "--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        debug()
    else:
        pass

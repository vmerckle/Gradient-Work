import argparse
import time
import os.path
import pickle
from rich import print
from inspect import getmembers, isfunction, isclass
import datetime

# my libraries
from utils import *
import runner
import postprocess
import configs

from lightmain import *

def shouldstop(X1, opti, num, start):
    step = num > 1
    timec =  time.time() - start > 1
    loss = opti.loss() < 0.4

    return step 
    return timec or step 
    return timec or loss

def expe1():
    config = "Normal"
    folder = "datatut"
    dists = ["frobenius", "wasser"]
    i = 100
    for dist in dists:
        #for i in [10,100, 1000, 10000, 50000]:
        for g in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            update = {"inneriter":i,
                      "gamma":g,
                      "proxdist": dist}
            runexperiment(config, folder, update, shouldstop)

def expe2():
    config = "Normal"
    folder = "dataot"
    dists = ["frobenius", "wasser"]
    i = 100
    for dist in dists:
        #for i in [10,100, 1000, 10000, 50000]:
        for g in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            update = {"inneriter":i,
                      "gamma":g,
                      "proxdist": dist}
            runexperiment(config, folder, update, shouldstop)

def expe3(dist):
    config = "Normal"
    folder = "datatest"
    dists = ["frobenius", "wasser"]
    i = 100000
    update = {"inneriter":i,
              "gamma":1.,
              "proxdist": dists[dist]}
    runexperiment(config, folder, update, shouldstop)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=int)
    args = parser.parse_args()
    expe3(args.dist)

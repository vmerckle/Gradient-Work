import argparse
import time
import os.path
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

def expe4(dist):
    config = "Normal"
    folder = "datatest"
    dists = ["frobenius", "wasser"]
    i = 100
    update = {"inneriter":i,
              "gamma":10.,
              "proxdist": dists[dist]}
    runexperiment(config, folder, update, shouldstop)

def expe5(dist):
    config = "Normal"
    folder = "23julyc"
    folder = "datatest"
    dists = ["frobenius", "wasser"]
    i = 100000
    NNseed = int(time.time())
    NNseed = 1339
    dataseed = 1339
    device = "cuda"
    device = "cpu"
    
    lr = 1e-3
    if dist == 2:
        algo = "GD"
        proxdist="no"
    else:
        algo = "proxpoint"
        proxdist = dists[dist]
    threadcount = 1
    print(f"seed used: NNseed:{NNseed} data:{dataseed}")
    update = {"inneriter":i,
              "gamma":1.,
              "datatype":"random",
              "device":device,
              "threadcount":threadcount,
              "algo": algo,
              "dataseed": dataseed,
              "NNseed": NNseed,
              "lr": lr,
              "d":10,
              "m":100,
              "n":1000,
              "proxdist": proxdist}
    runexperiment(config, folder, update, shouldstop)

def debug(dist):
    config = "Normal"
    folder = "datatest"
    dists = ["frobenius", "wasser"]
    i = 1000
    seed = int(time.time())
    seed = 1721715896
    print(f"seed used: {seed}")
    update = {"inneriter":i,
              "gamma":10.,
              "datatype":"random",
              "seed":seed,
              "d":7,
              "m":20,
              "n":10,
              "proxdist": dists[dist]}
    runexperiment(config, folder, update, shouldstop)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=int)
    args = parser.parse_args()
    runner(configs.default, "datatestnew", stopper=shouldstop)
    #debug(1)

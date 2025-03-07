import argparse
import time
import os.path
from inspect import getmembers, isfunction, isclass
import datetime
import pickle
import copy

from rich import print

# mylibs
from utils import *
import runner
import postprocess
import configs


def save(fname, res):
    with open(fname, "wb") as f:
        pickle.dump(res, f)

def dontstop(D, opti, num, start): # never stops
    return False

def default_logger(D, opti, num): # logs layers for every iterations
    if "iter" not in D:
        D["iter"] = {}

    D["iter"][num] = opti.params()

def default_printer(D, opti, num): # print every 0.1secs
    if "_lastprint" not in D:
        lastprint = 0
    else:
        lastprint = D["_lastprint"]

    if time.time() - lastprint > 0.1:
        print("it=", num, "loss=", opti.loss())
        D["_lastprint"] = time.time()


def default_saver(D, num, folder): # saves after 10 minutes
    if D["data"] == "mnist": # don't save anything if MNIST.
        D["X"] = "mnist"
        D["Xb"] =  "mnist"
        D["Y"] =  "mnist"
    if "lastsave" not in D:
        D["_lastsave"] = time.time()
    lastsave = D["_lastsave"]

    if time.time() - lastsave > 60*10:
        D["_lastsave"] = time.time()
        return True
    return False


# loop of log, print, save, opti.step()
def runner(D, folder, stopper=dontstop, logger=default_logger, printer=default_printer, saver=default_saver):
    D = copy.deepcopy(D)
    if not os.path.exists("data"):
        os.mkdir(f"data")
    if not os.path.exists(f"data/{folder}"):
        os.mkdir(f"data/{folder}")
    D["timestamp"] = int(datetime.datetime.now().timestamp()*1000)
    D["timestart"] = time.time() # easier to use
    filename = f"data/{folder}/{D['timestamp']}.pkl"
    num = 0

    opti = configs.applyconfig(D) # updates D
    printer(D, opti, num)
    logger(D, opti, num) # print and log first step to file

    try:
        while True:
            if stopper(D, opti, num, D["timestart"]):
                break

            try:
                opti.step()
            except KeyboardInterrupt:
                print("Inside step: Normal interrupt at num=", num)
                break
            num += 1
            logger(D, opti, num)
            printer(D, opti, num)
            if saver(D, num, folder):
                D["numsteps"] = num
                D["timetaken"]= time.time() - D["timestart"]
                save(filename, D)
                print("Saved", "TS=", D["timestamp"], filename, )

    except KeyboardInterrupt:
        print("Normal interrupt at num=", num)

    D["numsteps"] = num
    D["timetaken"]= time.time() - D["timestart"]
    save(filename, D)
    print("Saved", "TS=", D["timestamp"], filename, )

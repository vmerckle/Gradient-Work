import argparse
import time
import os.path
import pickle
from rich import print
from inspect import getmembers, isfunction, isclass
import datetime
# mylibs
from utils import *
import runner
import postprocess
import configs


def save(fname, res):
    with open(fname, "wb") as f:
        pickle.dump(res, f)
    print("Saved", fname)

def dontstop(X1, opti, num, start): # never stops
    return False

def default_logger(X1, opti, num): # logs layers for every iterations
    if "iter" not in X1:
        X1["iter"] = {}

    nly1, nly2 = opti.params()
    X1["iter"][num] = { "ly1": nly1,
                       "ly2": nly2
                       }

def default_printer(X1, opti, num): # print every 0.1secs
    if "_lastprint" not in X1:
        lastprint = 0
    else:
        lastprint = X1["_lastprint"]

    if time.time() - lastprint > 0.1:
        print("it=", num, "loss=", opti.loss())
        X1["_lastprint"] = time.time()


def default_saver(X1, num, folder): # saves after 10 minutes
    if "lastsave" not in X1:
        X1["_lastsave"] = time.time()
    lastsave = X1["_lastsave"]

    if time.time() - lastsave > 60*10:
        X1["_lastsave"] = time.time()
        return True
    return False


# loop of log, print, save, opti.step()
def runner(config, folder, stopper=dontstop, logger=default_logger, printer=default_printer, saver=default_saver):
    X = config
    X["timestamp"] = int(datetime.datetime.now().timestamp()*1000)
    X["timestart"] = time.time() # easier to use
    filename = f"data/{folder}/{X['timestamp']}.pkl"
    num = 0

    opti, ly1, ly2 = configs.applyconfig(X) # update X
    printer(X, opti, num)

    try:
        while True:
            if stopper(X, opti, num):
                break
            num += 1

            try:
                opti.step()
            except KeyboardInterrupt:
                print("Inside step: Normal interrupt at num=", num)
                break

            logger(X, opti, num)
            printer(X, opti, num)
            if saver(X, num, folder):
                save(filename, X)

    except KeyboardInterrupt:
        print("Normal interrupt at num=", num)

    X["timetaken"]= time.time() - X["timestart"]
    save(filename, X)


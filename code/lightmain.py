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

def save(fname, res):
    with open(fname, "wb") as f:
        pickle.dump(res, f)
    print("Saved", fname)

configD = {n[6:]:f for n,f in getmembers(configs, isfunction) if len(n) > 6 and n[:6] == "Config"}

def dontstop(X1, opti, num, start):
    return False

def runexperiment(config, folder, shouldstop=dontstop):
    file = f"{int(datetime.datetime.now().timestamp())}"
    X = config
    opti, ly1, ly2 = configs.applyconfig(X)
    num = 0
    lastsave = time.time()
    lastprint = 0
    start = time.time()
    print("it=", num, "loss=", opti.loss())
    try:
        while True:
            if shouldstop(X1, opti, num, start):
                break
            num += 1

            try:
                opti.step()
            except KeyboardInterrupt:
                print("Inside step: Normal interrupt at num=", num)
                break

            nly1, nly2 = opti.params()
            X["lly1"].append(nly1)
            X["lly2"].append(nly2)

            if time.time() - lastprint > 0.1:
                print("it=", num, "loss=", opti.loss())
                lastprint = time.time()

            # save every 10 minutes
            if time.time() - lastsave > 60*10:
                lastsave = time.time()
                X.update({"lly1":lly1, "lly2":lly2,"steps":len(X2["lly1"]), "timetaken":time.time()-start})
                save(f"data/{folder}/{file}.pkl", X1)

    except KeyboardInterrupt:
        print("Normal interrupt at num=", num)
    #except Exception as e:
        #print("big error:", e)

    X1.update({"lly1":lly1, "lly2":lly2,"steps":len(X2["lly1"]), "timetaken":time.time()-start})
    save(f"data/{folder}/{file}.pkl", X1)

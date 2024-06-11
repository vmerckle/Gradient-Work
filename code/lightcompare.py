import argparse
import os.path
import os
import pickle
from rich import print
from inspect import getmembers, isfunction, isclass
import datetime

# my libraries
from utils import *
import runner
import postprocess
import configs

def loadts(timestamps):
    datalist = {} # generate timestamps->files
    for fname in next(os.walk(args.folder))[2]:
        config_name, timestamp, step = fname.split("-")
        timestamp = int(timestamp)
        step = step.split(".")[0]
        if timestamp in datalist:
            datalist[timestamp][step] = fname
        else:
            datalist[timestamp] = {step:fname, 'config':config_name}

    Xs = []
    for ts in timestamps:
        if ts not in datalist:
            raise Exception("timestamp not found", ts)
        x = datalist[ts]
        with open(f"{args.folder}/{x['meta']}", "rb") as f:
            X4 = pickle.load(f)
        X1 = configD[x['config']]()
        with open(f"{args.folder}/{x['descent']}", "rb") as f:
            X2 = pickle.load(f)
        with open(f"{args.folder}/{x['postprocess']}", "rb") as f:
            X3 = pickle.load(f)
        Xs.append(X1|X2|X3|X4)
    return Xs

    #print(f"{x['config']} {ts} ({timestring(ts)} ago) {X4['steps']} steps, m={X1['m']}, n={X1['n']}, {X1['algo']}-{X1['proxdist']}, g={X1['gamma']}, inner={X1['inneriter']}")

def comparetwo(ts1, ts2):
    Xs = loadts([ts1, ts2])
    minstep = min(Xs[0]["steps"], Xs[1]["steps"])
    ly1, ly2 = Xs[0]["lly1"], Xs[1]["lly1"]
    print(minstep)
    print(type(ly1))

if __name__ == '__main__':
    configD = {n[6:]:f for n,f in getmembers(configs, isfunction) if len(n) > 6 and n[:6] == "Config"}

    parser = argparse.ArgumentParser()
    parser.add_argument("ts1", type=int)
    parser.add_argument("ts2", type=int)
    parser.add_argument("-f", "--folder", help="folder name", default="data")
    args = parser.parse_args()

    comparetwo(args.ts1, args.ts2)



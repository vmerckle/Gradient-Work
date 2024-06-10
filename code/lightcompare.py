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

def dosave(fname, res):
    with open(fname, "wb") as f:
        pickle.dump(args, f)
    print("Saved", fname)
    return res

def gethours(timestamp):
    delta = datetime.datetime.now() - datetime.datetime.fromtimestamp(float(timestamp))
    return delta.total_seconds() / 3600

def timestring(timestamp):
    delta = datetime.datetime.now() - datetime.datetime.fromtimestamp(float(timestamp))
    s = delta.total_seconds()
    d = int(s//(60*60*24)) # i know it's available in the timedelta object
    h = int(s//(60*60) - d*24) # shorter
    m = int(s//60 - h*60)
    s = int(s%60)
    ss = ""
    if d > 0:
        return f"{d}d {h}h"
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

if __name__ == '__main__':
    configD = {n[6:]:f for n,f in getmembers(configs, isfunction) if len(n) > 6 and n[:6] == "Config"}

    parser = argparse.ArgumentParser()
    parser.add_argument("ts1", type=int)
    parser.add_argument("ts2", type=int)
    parser.add_argument("-f", "--folder", help="folder name", default="data")
    args = parser.parse_args()
    ts1 = args.ts1
    ts2 = args.ts2

    datalist = {}
    for fname in next(os.walk(args.folder))[2]:
        config_name, timestamp, step = fname.split("-")
        timestamp = int(timestamp)
        step = step.split(".")[0]
        if timestamp in datalist:
            datalist[timestamp][step] = fname
        else:
            datalist[timestamp] = {step:fname, 'config':config_name}

    if not ts1 in datalist or not ts2 in datalist:
        print("Timestamps not found in", args.folder)

    Xs = []
    for ts in [ts1, ts2]:
        x = datalist[ts]
        with open(f"{args.folder}/{x['meta']}", "rb") as f:
            X4 = pickle.load(f)
        X1 = configD[x['config']]()
        print(f"{x['config']} {ts} ({timestring(ts)} ago) {X4['steps']} steps, m={X1['m']}, n={X1['n']}, {X1['algo']}-{X1['proxdist']}, g={X1['gamma']}, inner={X1['inneriter']}")
        with open(f"{args.folder}/{x['descent']}", "rb") as f:
            X2 = pickle.load(f)
        with open(f"{args.folder}/{x['postprocess']}", "rb") as f:
            X3 = pickle.load(f)
        Xs.append(X1|X2|X3|X4)

    minstep = min(Xs[0]["steps"], Xs[1]["steps"])
    print(minstep)

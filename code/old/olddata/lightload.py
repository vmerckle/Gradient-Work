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


def movetrash(x):
    present = [v for v in ['meta', 'setup', 'postprocess', 'descent'] if v in x]
    for v in present:
        fname = x[v]
        os.rename(f"data/{args.folder}/{fname}", f".trash/{fname}")
        print(f"moved {fname} to .trash")

if __name__ == '__main__':
    configD = {n[6:]:f for n,f in getmembers(configs, isfunction) if len(n) > 6 and n[:6] == "Config"}

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="folder name", default="data")
    parser.add_argument("--pp", help="load postprocess", action="store_true")
    parser.add_argument("--cleanup", help="delete missing and too short exp", action="store_true")
    args = parser.parse_args()
    cleanup = args.cleanup

    datalist = {}
    for fname in next(os.walk(f"data/{args.folder}"))[2]:
        config_name, timestamp, step = fname.split("-")
        timestamp = int(timestamp)
        step = step.split(".")[0]
        if timestamp in datalist:
            datalist[timestamp][step] = fname
        else:
            datalist[timestamp] = {step:fname, 'config':config_name}
    tss = [ts for ts in datalist]
    tss.sort()
    for ts in tss:
        x = datalist[ts]
        missing = [v for v in ['meta', 'setup', 'postprocess', 'descent'] if not v in x]
        if len(missing) > 0:
            print(f"{x['config']} {ts} ({hoursago(ts)} ago) is missing {missing}")
            if cleanup:
                movetrash(x)
            continue
        with open(f"data/{args.folder}/{x['meta']}", "rb") as f:
            X4 = pickle.load(f)
        with open(f"data/{args.folder}/{x['setup']}", "rb") as f:
            X1 = pickle.load(f)

        if not args.pp:
            print(f"{x['config']} {ts} ({hoursago(ts)} ago) {X4['steps']} steps({deltatimestring(X4['timetaken'])}), m={X1['m']}, n={X1['n']}, {X1['algo']}-{X1['proxdist']}, g={X1['gamma']}, inner={X1['inneriter']}")
        else:
            Xbis = configs.applyconfig(X1)
            with open(f"data/{args.folder}/{x['descent']}", "rb") as f:
                X2 = pickle.load(f)
            with open(f"data/{args.folder}/{x['postprocess']}", "rb") as f:
                X3 = pickle.load(f)
            X = X1|Xbis|X2|X3|X4
            if "iterdata" not in X:
                if X4['steps'] > 10000:
                    print(f"{x['config']} {ts} ({hoursago(ts)} ago) {X4['steps']} steps({deltatimestring(X4['timetaken'])}), m={X1['m']}, n={X1['n']}, {X1['algo']}-{X1['proxdist']}, g={X1['gamma']}, inner={X1['inneriter']}")
                    continue
                X  = X|postprocess.simplecalcs(X)
            finalloss = X["iterdata"][-1]["loss"]
            print(f"{x['config']} {ts} ({hoursago(ts)} ago) {X4['steps']} steps({deltatimestring(X4['timetaken'])}) endloss={finalloss:.1E}, m={X1['m']}, n={X1['n']}, {X1['algo']}-{X1['proxdist']}, g={X1['gamma']}, inner={X1['inneriter']}")

        if X4['timetaken'] < 120:
            if cleanup:
                movetrash(x)

    #if (args.keepfirst or args.keepsecond) and os.path.isfile(stepname):
    #    print(f"Loading '{stepname}'") 
    #        myconfig = configDict[local_args.config]
    #        X1 = myconfig(local_args)
    #    print(f"Loaded config='{local_args.config}'")

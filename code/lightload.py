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


if __name__ == '__main__':
    configD = {n[6:]:f for n,f in getmembers(configs, isfunction) if len(n) > 6 and n[:6] == "Config"}

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", help="folder name", default="data")
    args = parser.parse_args()

    datalist = {}
    for fname in next(os.walk(args.folder))[2]:
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
            continue

        with open(f"{args.folder}/{x['meta']}", "rb") as f:
            X4 = pickle.load(f)

        with open(f"{args.folder}/{x['setup']}", "rb") as f:
            X1 = pickle.load(f)

        print(f"{x['config']} {ts} ({hoursago(ts)} ago) {X4['steps']} steps({deltatimestring(X4['timetaken'])}), m={X1['m']}, n={X1['n']}, {X1['algo']}-{X1['proxdist']}, g={X1['gamma']}, inner={X1['inneriter']}")

    #if (args.keepfirst or args.keepsecond) and os.path.isfile(stepname):
    #    print(f"Loading '{stepname}'") 
    #        myconfig = configDict[local_args.config]
    #        X1 = myconfig(local_args)
    #    print(f"Loaded config='{local_args.config}'")

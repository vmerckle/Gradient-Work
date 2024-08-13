#!/usr/bin/env python

import argparse
import os.path
import os
import pickle
import datetime

from rich import print

# mylibs
from utils import *
import runner
import postprocess
import configs

# os.rename(f"{args.folder}/{fname}", f".trash/{fname}")

def listfiles(folder):
    # get all timestamps used
    l = []
    for f in next(os.walk(folder))[2]:
        ts, ext = f.split(".")
        l.append(int(ts))
    l.sort()
    return [f"{folder}/{ts}.pkl" for ts in l]

def listTS(folder):
    # get all timestamps used
    l = []
    for f in next(os.walk(folder))[2]:
        ts, ext = f.split(".")
        l.append(int(ts))
    l.sort()
    return l

def listD(folder):
    for fname in listfiles(folder):
        with open(fname, "rb") as f:
            yield pickle.load(f)

def printD(D):
    spent = f"{deltatimestring(D['timetaken'])}"
    ts = D["timestamp"]
    print(f"{ts} ({hoursago(ts)} ago) {D['numsteps']} steps({spent})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="folder name")
    args = parser.parse_args()
    folder = f"{args.folder}"

    for D in listD(folder):
        printD(D)

    #print(f"{ts} ({hoursago(ts)} ago) {len(D['lly1'])} steps({deltatimestring(D['timetaken'])}) endloss={finalloss:.1E}, m={D['m']}, n={D['n']}, {D['algo']}-{D['proxdist']}, g={D['gamma']}, inner={D['inneriter']}")

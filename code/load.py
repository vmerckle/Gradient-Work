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

# os.rename(f"data/{args.folder}/{fname}", f".trash/{fname}")

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

def listX(folder):
    for fname in listfiles(folder):
        with open(fname, "rb") as f:
            yield pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="folder name")
    args = parser.parse_args()
    folder = f"data/{args.folder}"

    for X in listX(folder):
        spent = f"{deltatimestring(X['timetaken'])}"
        ts = X["timestamp"]
        print(f"{ts} ({hoursago(ts)} ago) {len(X['lly1'])} steps({spent})")

    #print(f"{ts} ({hoursago(ts)} ago) {len(X['lly1'])} steps({deltatimestring(X['timetaken'])}) endloss={finalloss:.1E}, m={X['m']}, n={X['n']}, {X['algo']}-{X['proxdist']}, g={X['gamma']}, inner={X['inneriter']}")

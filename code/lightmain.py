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

def runexperiment(config, folder, update={}, shouldstop=dontstop):
    file = f"{config}-{int(datetime.datetime.now().timestamp())}"

    X1 = configD[config]()
    X1.update(update)
    save(f"{folder}/{file}-setup.pkl", X1)
    X1.update(configs.loadalgo(X1))

    lly1, lly2 = [X1["ly1"]], [X1["ly2"]]
    opti = X1["opti"]
    num = 0
    lastsave = time.time()
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
            lly1.append(nly1)
            lly2.append(nly2)
            print("it=", num, "loss=", opti.loss())
            if time.time() - lastsave > 60*10: # save every 10minutes
                X2 = {"lly1":lly1, "lly2":lly2}
                save(f"{folder}/{file}-descent.pkl", X2)
                lastsave = 0

    except KeyboardInterrupt:
        print("Normal interrupt at num=", num)
    #except Exception as e:
        #print("big error:", e)

    X2 = {"lly1":lly1, "lly2":lly2}
    save(f"{folder}/{file}-descent.pkl", X2)
    X3 = postprocess.simplecalcs(X2|X1)
    save(f"{folder}/{file}-postprocess.pkl", X3)
    save(f"{folder}/{file}-meta.pkl", {"config": config, "steps":len(X2["lly1"]), "timetaken":time.time()-start})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config name", default=None, choices=configD.keys())
    #parser.add_argument("--folder", help="folder name", default="data")
    folder = "data"
    args = parser.parse_args()
    #if args.name is None:
        #args.name = f"{args.config}-{int(datetime.datetime.now().timestamp())}"
    
    runexperiment(args.config, folder)

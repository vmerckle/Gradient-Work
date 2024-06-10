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

if __name__ == '__main__':
    configD = {n[6:]:f for n,f in getmembers(configs, isfunction) if len(n) > 6 and n[:6] == "Config"}

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="output name", default=None)
    parser.add_argument("-c", "--config", help="config name", default=None, choices=configD.keys())
    #parser.add_argument("--folder", help="folder name", default="data")
    folder = "data2"
    args = parser.parse_args()
    if args.name is None:
        args.name = f"{args.config}-{int(datetime.datetime.now().timestamp())}"

    X1 = configD[args.config]()
    save(f"{folder}/{args.name}-setup.pkl", X1)
    X1.update(configs.loadalgo(X1))

    lly1, lly2 = [X1["ly1"]], [X1["ly2"]]
    opti = X1["opti"]
    maxstep = X1["steps"]
    num = 0
    lastsave = time.time()
    start = time.time()
    print("it=", num, "loss=", opti.loss())
    try:
        while True:
            if maxstep != -1 and num >= maxstep:
                break
            num += 1
            opti.step()
            nly1, nly2 = opti.params()
            lly1.append(nly1)
            lly2.append(nly2)
            l = opti.loss()
            print("it=", num, "loss=", l)
            if time.time() - lastsave > 60*10:
                X2 = {"lly1":lly1, "lly2":lly2}
                save(f"{folder}/{args.name}-descent.pkl", X2)
                lastsave = 0

    except KeyboardInterrupt:
        print("Normal interrupt at num=", num)
    except Exception as e:
        print(e)

    X2 = {"lly1":lly1, "lly2":lly2}
    save(f"{folder}/{args.name}-descent.pkl", X2)
    X3 = postprocess.simplecalcs(X2|X1)
    save(f"{folder}/{args.name}-postprocess.pkl", X3)
    save(f"{folder}/{args.name}-meta.pkl", {"config": args.config, "steps":len(X2["lly1"]), "timetaken":time.time()-start})

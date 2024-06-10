import numpy as np
import time

import argparse
import os.path
import sys
import pickle

import matplotlib.pyplot as plt
import matplotlib.animation as animation

#from rich.traceback import install #rich as default traceback handler
from rich.progress import track
from rich.progress import Progress
from rich.columns import Columns
from rich.table import Column
from rich.table import Table

from rich.logging import RichHandler
from rich import print
from rich.pretty import pprint
from rich.prompt import Confirm
from rich.prompt import IntPrompt

# my libraries
import runner
import postprocess
import animations
import configs
from utils import *

if __name__ == '__main__':
    #animD = {n[4:]:f for n,f in getmembers(animations, isclass) if len(n) > 4 and n[:4] == "Anim"}
    animationDict = {"output+neurons":animations.NiceAnim,
                 "wasser": animations.WasserNiceAnim,
                 "dataspace": animations.LessNiceAnim}
    configDict = {"config2d_new": configs.Config2DNew,
                  "config2d_new_grid": configs.Config2DNew_grid,
                  "config2d_new_grid_wasser": configs.Config2DNew_grid_wasser,
                  "config2d_new_grid_wasser_ex": configs.Config2DNew_grid_wasser_ex,
                  "config1d_new": configs.Config1DNew}

    parser = argparse.ArgumentParser()
    parser.add_argument( "--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="seed")
    parser.add_argument("--config", help="config name", default=None, choices=configDict.keys())
    parser.add_argument("--name", help="output name", default="out_new")
    parser.add_argument("--nowrite", help="do not write anything to disk", action="store_true")
    parser.add_argument("-k", "--keepfirst", help="Only redo postprocess and animation", action="store_true")
    parser.add_argument("-r", "--keepsecond", help="Only redo animation", action="store_true")
    parser.add_argument("--noanim", action="store_true", help="do not show the end animation")
    parser.add_argument("--runanim", action="store_true", help="show a real time animation, enables option 'run' as well")
    parser.add_argument("--anim_choice", default="output+neurons", choices=animationDict.keys(), help="what animation")
    parser.add_argument("--movie", help="save movie", action="store_true")
    parser.add_argument("--save_movie", help="save movie", action="store_true")
    parser.add_argument("--fps", type=int, default=10, help="movie fps")
    parser.add_argument("--skiptoseconds", default=-1, type=float, help="maximum time in seconds, will skip frame to match")
    args = parser.parse_args()

    code = f"{args.name}" # filename used
    myanim = animationDict[args.anim_choice]

    ####### Load the setup ####### 
    stepname = f"data/settings_{code}.pkl"
    if (args.keepfirst or args.keepsecond) and os.path.isfile(stepname):
        print(f"Loading '{stepname}'") 
        with open(stepname, "rb") as f:
            local_args = pickle.load(f)
            myconfig = configDict[local_args.config]
            X1 = myconfig(local_args)
        print(f"Loaded config='{local_args.config}'")
    else:
        print(f"Overwriting '{stepname}'")
        if args.config is None:
            cl = list(configDict)
            for i, c in enumerate(cl):
                print(f"{i+1}\t'{c}'")
            while True:
                num = IntPrompt.ask(f"Enter a number between 1 and {len(cl)}")
                if num >= 1 and num <= len(cl):
                    args.config = cl[num-1]
                    break
        myconfig = configDict[args.config]
        X1 = myconfig(args)
        with open(stepname, "wb") as f:
            pickle.dump(args, f)

    ####### Execute the algorithm #######
    stepname = f"data/descent_{code}.pkl"
    if (args.keepfirst or args.keepsecond) and os.path.isfile(stepname):
        print(f"Loading '{stepname}'")
        with open(stepname, "rb") as f:
            X2 = pickle.load(f)
        print(f"Loaded {len(X2['lly1'])} steps")
    else:
        print(f"Overwriting '{stepname}'")
        if args.runanim:
            X2 = runner.simpleAnim(X1, myanim=myanim)
            #X2 = runner.animationRun(X1, myanim=myanim)
        else:
            X2 = runner.simpleRun(X1)
        if not args.nowrite:
            with open(stepname, "wb") as f:
                pickle.dump(X2, f)

    if "wasserstats" in X2:
        animations.niceplots(X2["wasserstats"])

    #assert False
    ####### Apply postprocess to iteration data ####### 
    stepname = f"data/postprocess_{code}.pkl"
    if args.keepfirst and args.keepsecond and os.path.isfile(stepname):
        print(f"Loading '{stepname}'")
        with open(stepname, "rb") as f:
            X3 = pickle.load(f)
    else:
        if not args.nowrite:
            print(f"Overwriting '{stepname}'")
            X3 = postprocess.simplecalcs(X2|X1)
            with open(stepname, "wb") as f:
                pickle.dump(X3, f)

    if not (args.movie or args.save_movie):
        sys.exit(0)
    print(f"'{args.anim_choice}' animation requested")

    ####### Apply postprocess to iteration data ####### 

    XXX = X1|X2|X3
    nframe = len(XXX["iterdata"])
    if args.skiptoseconds != -1:
        skipv = nframe/args.fps/args.skiptoseconds
        l = [i for i in range(0, nframe, int(skipv+0.99))]
        if (nframe-1) not in l:
            l.append(nframe-1)
    else:
        l = list(range(0, nframe))

    print("Animation setup..")
    fig = plt.figure(figsize=(10,10))
    animobj = myanim(fig, X1|X2|X3, frames=l)
    ani = animobj.getAnim(interval=1000/args.fps, blit=True)

    #a = 2
    #animobj.ax.set_xlim(-a, a)
    #animobj.ax.set_ylim(-a, a)

    if args.save_movie:
        print("Saving animation")
        writer = animation.FFMpegWriter(fps=args.fps)#, bitrate=1800)
        name = f"outputs/{code}_movie.gif"
        ani.save(name, writer=writer)
        print(f"saved as {name}")
    else:
        plt.show()

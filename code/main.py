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
    animationDict = {"output+neurons":animations.NiceAnim,
                 "dataspace": animations.LessNiceAnim,
                 "dataspaceb": animations.LessNiceAnim}
    configDict = {"config2d_new": configs.Config2DNew,
                  "config2d_new_grid": configs.Config2DNew_grid,
                  "config2d_new_grid_wasser": configs.Config2DNew_grid_wasser,
                  "config1d_new": configs.Config1DNew}

    parser = argparse.ArgumentParser()
    parser.add_argument( "--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="seed")
    parser.add_argument("--config", help="config name", default=None, choices=configDict.keys())
    parser.add_argument("--name", help="output name", default="out_new")
    parser.add_argument("-k", "--keepfirst", help="Only redo postprocess and animation", action="store_true")
    parser.add_argument("-r", "--keepsecond", help="Only redo animation", action="store_true")
    parser.add_argument("--noanim", action="store_true", help="do not show the end animation")
    parser.add_argument("--runanim", action="store_true", help="show a real time animation, enables option 'run' as well")
    parser.add_argument("--anim_choice", default="output+neurons", choices=animationDict.keys(), help="what animation")
    parser.add_argument("--movie", help="save movie", action="store_true")
    parser.add_argument("--fps", type=int, default=10, help="movie fps")
    parser.add_argument("--skiptoseconds", default=10, type=float, help="maximum time in seconds, will skip frame to match")
    #parser.add_argument("--scaleinit", default=None, type=float, help="scalar factor to weight matrix")
    #parser.add_argument("--algo", default=None, choices=["torch", "jko", "jkocvx"])
    #parser.add_argument("--proxf", default=None, choices=["scipy", "torch", "cvxpy"], help="algo=jko, how to compute the prox")
    #parser.add_argument("--jkosteps", default=None, type=int, help="algo=jko, number of internal iterations")
    #parser.add_argument("--jkogamma", default=None, type=float, help="algo=jko, float")
    #parser.add_argument("--jkotau", default=None, type=float, help="algo=jko, float")
    #parser.add_argument("--adamlr", default=None, type=float, help="algo=jko, proxf=torch, learning rate for gradient descent")
    #parser.add_argument("-lr", type=float, default=None, help="algo='torch', learning rate")
    args = parser.parse_args()

    code = f"{args.name}" # filename used
    myanim = animationDict[args.anim_choice]

    ####### Load the setup ####### 
    stepname = f"data/settings_{code}.pkl"
    if args.keepfirst or args.keepsecond and os.path.isfile(stepname):
        with open(stepname, "rb") as f:
            local_args = pickle.load(f)
            print(f"Loading '{stepname}' - config='{local_args.config}'")
            myconfig = configDict[local_args.config]
            X1 = myconfig(local_args)
    else:
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
        print(f"Overwriting '{stepname}'")
        X1 = myconfig(args)
        with open(stepname, "wb") as f:
            pickle.dump(args, f)

    ####### Execute the algorithm ####### 
    stepname = f"data/descent_{code}.pkl"
    if (args.keepfirst or args.keepsecond) and os.path.isfile(stepname):
        with open(stepname, "rb") as f:
            print(f"Loading '{stepname}'")
            X2 = pickle.load(f)
    else:
        print(f"Overwriting '{stepname}'")
        if args.runanim:
            X2 = runner.animationRun(X1, myanim=myanim)
        elif args.run:
            X2 = runner.simpleRun(X1)
        else:
            sys.exit(0)
        with open(stepname, "wb") as f:
            pickle.dump(X2, f)


    ####### Apply postprocess to iteration data ####### 
    stepname = f"data/postprocess_{code}.pkl"
    if args.keepfirst and args.keepsecond and os.path.isfile(stepname):
        with open(stepname, "rb") as f:
            print(f"Loading '{stepname}'")
            X3 = pickle.load(f)
    else:
        print(f"Overwriting '{stepname}'")
        X3 = postprocess.simplecalcs(X2|X1)
        with open(stepname, "wb") as f:
            pickle.dump(X3, f)

    if not args.movie:
        sys.exit(0)
    print(f"'{args.anim_choice}' animation requested")

    ####### Apply postprocess to iteration data ####### 

    XXX = X1|X2|X3
    nframe = len(XXX["iterdata"])
    skipv = nframe/args.fps/args.skiptoseconds
    l = [i for i in range(0, nframe, int(skipv+0.99))]
    if (nframe-1) not in l:
        l.append(nframe-1)
    #l = list(range(0, 70))
    print("Animation setup..")
    fig = plt.figure(figsize=(10,10))
    animobj = myanim(fig, X1|X2|X3, frames=l)
    ani = animobj.getAnim(1)
    animobj.ax.set_xlim(-1.2, 1.2)
    animobj.ax.set_ylim(-1.2, 1.2)
    animobj.ax.set_xlim(-2.2, 2.2)
    animobj.ax.set_ylim(-0.2, 2.4)

    # todo implement some frame skipping
    if args.movie:
        print("Saving animation")
        writer = animation.FFMpegWriter(fps=args.fps)#, bitrate=1800)
        name = f"outputs/{code}_movie.gif"
        ani.save(name, writer=writer)
        print(f"saved as {name}")
    else:
        plt.show()

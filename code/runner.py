import numpy as np
import time

import argparse
import os.path
import sys
import pickle

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# for real time anim
import postprocess
import animations

from utils import *

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

# run without animation
def simpleRun(X):
    steps = X["steps"] if "steps" in X else -1
    data = X
    lly1, lly2 = [X["ly1"]], [X["ly2"]]
    d, m = lly1[-1].shape
    opti = X["opti"]
    X, Y = X["X"], X["Y"] # no comment
    num = 0
    bestloss = opti.loss()
    print("it=", num, "loss=", opti.loss())
    #print("layer1", ",".join([f"{x:.2f}" for x in lly1[-1].flatten()]))
    #print("layer2", ",".join([f"{x:.2f}" for x in lly2[-1].flatten()]))
    try:
        while True:
            if steps != -1 and num >= steps:
                break
            opti.step()
            nly1, nly2 = opti.params()
            lly1.append(nly1)
            lly2.append(nly2)
            l = opti.loss()
            if m <= 20 and 1:
                #print("layer2", ",".join([f"{x:.2f}" for x in lly2[-1].flatten()]))
                # print("layer2", ",".join([f"{x:.2f}" for x in lly2[-1].flatten()]), f"loss: {l:.4f}, sum {np.sum(lly2[-1]):.4f}")
                pass
            else:
                # print("layer2", ",".join([f"{x:.2f}" for x in lly2[-1].flatten()]), f"loss: {l:.4f}, sum {np.sum(lly2[-1]):.4f}")
                #print(f"{num}: loss: {l:.4f}, sum {np.sum(lly2[-1]):.4f}")
                pass
            print(f"loss: {l}")
            num += 1
            if l < bestloss:
                bestloss = l
            if l/bestloss > 10:
                pass
                #print(".. completely diverged.")
                #assert False
    except KeyboardInterrupt:
        print("Normal interrupt at num=", num)
    #except Exception as e:
    #   print("Something went really wrong at num=", num)
    #   print(e)

    return {"lly1":lly1, "lly2":lly2}

# plot live animation
def animationRun(X, myanim):
    data = X
    lly1, lly2 = [X["ly1"]], [X["ly2"]]
    opti = X["opti"]
    X, Y = X["X"], X["Y"] # no comment
    n, d = X.shape
    num = 0

    print("Animation setup..")
    fig = plt.figure(figsize=(10,4))
    Xoutb = np.linspace(-4,4, 1000)
    if d == 2:
        Xout = add_bias(Xoutb)
    elif d == 1:
        Xout = Xoutb[:, None]
    animobj = myanim(fig, data|{"Xout":Xoutb}, runanim=True)
    already = [False] # see comment about i=0
    bestloss = [opti.loss()]
    def update_ok(i):
        if i == 0: # for some reason, this function is called 4 times with i=0
            if not already[0]:
                print("it=", i, "loss=", opti.loss())
                already[0] = True

            nly1, nly2 = opti.params()
            di = postprocess.NNtoIter(X, Y, Xout, nly1, nly2, run=True)
            p = opti.p
            di["p"] = p
            return animobj.update_aux(di, i) # so we simply don't do the step
        opti.step()
        nly1, nly2 = opti.params()
        #print(nly1, nly2)
        di = postprocess.NNtoIter(X, Y, Xout, nly1, nly2, run=True)
        p = opti.p
        di["p"] = p

        lly1.append(nly1)
        lly2.append(nly2)
        print("it=", i, "loss=", opti.loss())
        l = opti.loss()
        if l < bestloss[0]:
            bestloss[0] = l
        if abs(np.sum(p) - 1) > 1e-1 and False:
            print("we probably diverged here.")
            print("psum:", np.sum(p))
        #if l/bestloss[0] > 10: # jko we don't care
            #print(".. completely diverged.")
             #assert False
        return animobj.update_aux(di, i)

    try:
        ani = animation.FuncAnimation(fig, update_ok, frames=list(range(100000)), blit=True, interval=1)
        animobj.ax.set_xlim(-1.2, 1.2)
        animobj.ax.set_ylim(-1.2, 1.2)
        animobj.ax.set_xlim(-4.2, 4.2)
        animobj.ax.set_ylim(-0.2, 2.4)
        plt.show()
    except KeyboardInterrupt:
        print("Normal interrupt at num=", num)
    print("say something?")
    #except Exception as e:
    #   print("Something went really wrong at num=", num)
    #   print(e)

    return {"lly1":lly1, "lly2":lly2}



import numpy as np
import time
import sys
from types import SimpleNamespace
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils import *
import postprocess
import animations # real time anim
import configs

# plot live animation
def animationRun(setupDict, myanim):
    setup = SimpleNamespace(**setupDict)
    lly1, lly2 = [setup.ly1], [setup.ly2]
    opti = setup.opti
    X, Y = setup.X, setup.Y
    d, m = setup.ly1.shape
    n, d = X.shape

    print("Animation setup..")
    fig = plt.figure(figsize=(10,4))
    Xoutb = np.linspace(-4,4, 1000)
    if d == 2:
        Xout = add_bias(Xoutb)
    elif d == 1:
        Xout = Xoutb[:, None]
    animobj = myanim(fig, setupDict|{"Xout":Xoutb}, runanim=True)
    already = [False] # see comment about i=0
    bestloss = [opti.loss()]
    def update_ok(i):
        if i == 0: # for some reason, this function is called 4 times with i=0
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
        plt.show()
    except KeyboardInterrupt:
        print("Keyboard interrupt")

    return {"lly1":lly1, "lly2":lly2}

# plot live animation
def simpleAnim(setupDict, myanim):
    setup = SimpleNamespace(**setupDict)
    lly1, lly2 = [setup.ly1], [setup.ly2]
    opti = setup.opti
    X, Y = setup.X, setup.Y
    d, m = setup.ly1.shape
    n, d = X.shape

    fig = plt.figure(figsize=(10,4))
    Xoutb = np.linspace(-4,4, 1000)
    Xout = add_bias(Xoutb)
    animobj = myanim(fig, setupDict|{"Xout":Xoutb}, runanim=True)
    #wasserstats = {"obj":[], "wasser":[], "ldeuxdist":[]}
    def update_ok(i):
        if i == 0: # for some reason, this function is called 4 times with i=0
            nly1, nly2 = opti.params()
            di = postprocess.NNtoIter(X, Y, Xout, nly1, nly2, run=True)
            return animobj.update_aux(di, i) # so we simply don't do the step
        opti.step()
        nly1, nly2 = opti.params()
        #wasserstats["obj"].append(opti.objectif)
        #wasserstats["wasser"].append(opti.wasserdist)
        #wasserstats["ldeuxdist"].append(opti.ldeuxdist)
        di = postprocess.NNtoIter(X, Y, Xout, nly1, nly2, run=True)

        lly1.append(nly1)
        lly2.append(nly2)
        print("it=", i, "loss=", opti.loss())
        return animobj.update_aux(di, i)

    try:
        ani = animation.FuncAnimation(fig, update_ok, frames=list(range(100000)), blit=True, interval=1)
        plt.show()
    except KeyboardInterrupt:
        print("Keyboard interrupt")

    return {"lly1":lly1, "lly2":lly2}
    #return {"lly1":lly1, "lly2":lly2, "wasserstats":wasserstats}

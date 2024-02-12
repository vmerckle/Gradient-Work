import numpy as np
import time

import argparse
import os.path
import sys
import pickle

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import animations
import configs
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

# compute plot information from data and layers
# used in live and post animation
def NNtoIter(Xt, Yt, allX, ly1, ly2, run=False):
    d, m = ly1.shape
    if d == 2:
        #lnorm = np.abs(ly1[1,:].flatten() * ly2.flatten()) # -> slope of the ReLU unit
        #lnorm =np.abs(ly2.flatten()) # -> similar alpha = similar speed. But the first layer's norm also matter...
        lspeed = 1/(1e-10+np.linalg.norm(ly1, ord=2, axis=0)) * np.abs(ly2.flatten())
        lslope = np.abs(ly1[1,:].flatten() * ly2.flatten())
        lnorm = np.linalg.norm(ly1, ord=2, axis=0) * ly2.flatten() # -> mult or addition, just an "idea" of how big the neuron is
        lsize = 1/(lspeed+1) + (40 if run else 0)# slower = bigger
        lsize = ly2.flatten() # size=slope
        lnorm = lslope
        lact = np.array([-w2/w1 for w1, w2 in ly1.T])
        Yout = np_2layers(allX, ly1, ly2)
        Yhat = np_2layers(Xt, ly1, ly2)
        signedE = Yhat-Yt
        loss = MSEloss(Yhat, Yt)
        pdirecs = []
        motifs = getMotifNow(Xt, ly1)
        for m in motifs:
            w1, w2 = np.sum(np.atleast_2d(m).T*Xt * signedE , axis=0) # n,d * 10, * 10, =sum> 2
            #print((np.sum(Xt * signedE * m, axis=0)).shape)
            #print(-w2/w1)
            if w1 != 0:
                pdirecs.append(-w2/w1)
                pdirecs.append(w2/w1)
        return {"ly1": ly1, "ly2": ly2, "lact": lact, "lnorm": lnorm, "loss": loss, "Yout": Yout, "lsize": lsize, "signedE":signedE, "pdirecs":np.array(pdirecs)}
    elif d == 1:
        lsize = ly2.flatten() # size=slope
        lnorm = ly1[0, :].flatten()
        lact = np.zeros_like(ly1[0])
        Yout = np_2layers(allX, ly1, ly2)
        Yhat = np_2layers(Xt, ly1, ly2)
        signedE = Yhat-Yt
        loss = MSEloss(Yhat, Yt)
        return {"ly1": ly1, "ly2": ly2, "lact": lact, "lnorm": lnorm, "loss": loss, "Yout": Yout, "lsize": lsize}
    else:
        raise Exception("d>3 has no animation yet")


# run without animation
def simpleRun(X):
    steps = X["steps"]
    data = X
    lly1, lly2 = [X["ly1"]], [X["ly2"]]
    d, m = lly1[-1].shape
    opti = X["opti"]
    X, Y = X["X"], X["Y"] # no comment
    num = 0
    bestloss = opti.loss()
    print("it=", num, "loss=", opti.loss())
    print("layer1", ",".join([f"{x:.2f}" for x in lly1[-1].flatten()]))
    print("layer2", ",".join([f"{x:.2f}" for x in lly2[-1].flatten()]))
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
                print("layer2", ",".join([f"{x:.2f}" for x in lly2[-1].flatten()]))
                #print("layer2", ",".join([f"{x:.2f}" for x in lly2[-1].flatten()]), f"loss: {l:.4f}, sum {np.sum(lly2[-1]):.4f}")
            else:
                print(f"{num}: loss: {l:.4f}, sum {np.sum(lly2[-1]):.4f}")
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
            di = NNtoIter(X, Y, Xout, nly1, nly2, run=True)
            p = opti.p
            di["p"] = p
            return animobj.update_aux(di, i) # so we simply don't do the step
        opti.step()
        nly1, nly2 = opti.params()
        #print(nly1, nly2)
        di = NNtoIter(X, Y, Xout, nly1, nly2, run=True)
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

def simplecalcs(X):
    X, Y, lly1, lly2, rng = [X[x] for x in ["X", "Y", "lly1", "lly2", "rng"]]
    d = X.shape[1]
    allXb = np.linspace(-4,4, 1000)
    if d == 2:
        allX = add_bias(allXb)
    elif d == 1:
        allX = allXb[:, None]
    iterdata = [NNtoIter(X, Y, allX, lly1[i], lly2[i]) for i in range(len(lly1))]
    normData(iterdata, "lnorm", 0, 1) 
    normData(iterdata, "lsize", 1, 100)
    return {"Xout": allXb, "iterdata": iterdata}

if __name__ == '__main__':
    animationDict = {"output+neurons":animations.NiceAnim,
                 "dataspace": animations.LessNiceAnim,
                 "dataspaceb": animations.LessNiceAnim}
    configDict = {"config2d_new": configs.Config2DNew,
                  "config2d_new_grid": configs.Config2DNew_grid,
                  "config1d_new": configs.Config1DNew}

    parser = argparse.ArgumentParser()
    parser.add_argument( "--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="seed")
    parser.add_argument("--config", help="config name", default=None, choices=configDict.keys())
    parser.add_argument("-o", "--output", help="output name", default="out_new")
    parser.add_argument("-k", "--keepfirst", help="reload descent", action="store_true")
    parser.add_argument("-r", "--keepsecond", help="reload descent & postprocess", action="store_true")
    parser.add_argument("--run", action="store_true", help="ignore steps number and run until interrupted")
    parser.add_argument("--steps", default=-1, type=int, help="how many iterations to optimize the network")
    parser.add_argument("--noanim", action="store_true", help="do not show the end animation")
    parser.add_argument("--runanim", action="store_true", help="show a real time animation, enables option 'run' as well")
    parser.add_argument("--anim", default="output+neurons", choices=animationDict.keys(), help="what animation")
    parser.add_argument("--movie", help="save movie", action="store_true")
    parser.add_argument("--movieout", help="output movie name", default="out_new")
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
    code = args.output
    if code != "out_new" and args.movieout == "out_new":
        codemov = code
    else:
        codemov = args.movieout
    myanim = animationDict[args.anim]



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

    stepname = f"data/descent_{code}.pkl"
    if (args.keepfirst or args.keepsecond) and os.path.isfile(stepname):
        with open(stepname, "rb") as f:
            print(f"Loading '{stepname}'")
            X2 = pickle.load(f)
    else:
        print(f"Overwriting '{stepname}'")
        if args.runanim:
            X2 = animationRun(X1, myanim=myanim)
        elif args.run:
            X2 = simpleRun(X1)
        else:
            sys.exit(0)
        with open(stepname, "wb") as f:
            pickle.dump(X2, f)


    stepname = f"data/postprocess_{code}.pkl"
    if args.keepfirst and args.keepsecond and os.path.isfile(stepname):
        with open(stepname, "rb") as f:
            print(f"Loading '{stepname}'")
            X3 = pickle.load(f)
    else:
        print(f"Overwriting '{stepname}'")
        X3 = simplecalcs(X2|X1)
        with open(stepname, "wb") as f:
            pickle.dump(X3, f)

    if args.noanim and not args.movie:
        print("No animation requested")
        sys.exit(0)

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
        name = f"outputs/{codemov}_movie.gif"
        ani.save(name, writer=writer)
        print(f"saved as {name}")
    else:
        plt.show()

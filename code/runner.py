import numpy as np
import cvxpy as cp
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib.markers import MarkerStyle 
# https://stackoverflow.com/questions/58089375/question-about-changing-marker-of-scatter-during-animation
# the only god damn correct answer, all others will have bugs.

import torch
from numpy.random import default_rng

import argparse
import os.path
import sys
import pickle
#import pandas as pd

from utils import *
from torch_descent import torch_descent
from jko_descent import jko_descent

class LessNiceAnim:
    def __init__(self, fig, data, runanim=False):
        # load and check data
        self.Xb, self.Y, self.X = data["Xb"], data["Y"][:, 0], data["X"]
        self.Xout = data["Xout"]
        self.n, self.d = self.X.shape
        self.m = data["ly1"].shape[1]
        assert self.Y.ndim == 1 and len(self.Y) == self.n
        if runanim:
            self.D = None
        else:
            self.D = data["iterdata"]
            self.Nframe = len(self.D)
            # sanity checks on data, looks useless but this is python
            assert np.all([di["ly1"].shape == (self.d, self.m) for di in self.D])
            assert np.all([di["ly2"].shape == (self.m, 1) for di in self.D])
        # plotting setup
        self.fig = fig
        self.ax = self.fig.add_subplot()
        self.ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        self.ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        self.ax.grid(True, alpha=0.2)
        self.ax.scatter(self.Xb, np.ones_like(self.Xb), marker="x", color="red", s=60, alpha=0.5)
        self.signedE = self.ax.scatter([], [], marker="*", color="blue", s=60, alpha=1)
        ## beautiful lines don't touch
        s = u'$\u2193$'
        self.fun_marker = MarkerStyle(s).get_path().transformed(MarkerStyle(s).get_transform())
        self.pos_marker = MarkerStyle('>').get_path().transformed(MarkerStyle('>').get_transform())
        self.neg_marker = MarkerStyle('<').get_path().transformed(MarkerStyle('<').get_transform())
        # artist engaged in animation
        self.text = self.ax.text(0.05, 0.9, f"start frame", transform=self.ax.transAxes)
        self.outline, = self.ax.plot([], [])
        self.allscat = self.ax.scatter([], [], marker='o')
        # administrative stuff
        self.starttime = time.time()
        self.verbose = False

    def update_aux(self, di, frame):
        ly1, ly2 = di["ly1"], di["ly2"]
        lact, lnorm = di["lact"], di["lnorm"]
        loss, Yout = di["loss"], di["Yout"]
        lsize = di["lsize"]
        #ly1g, ly2g = di["ly1g"], di["ly2g"]

        xl, yl, mks = np.zeros(self.m), np.zeros(self.m), []
        sizes, colors = lsize*4, []
        for i in range(self.m):
            w1, w2 = ly1.T[i]
            alpha, = ly2[i]
            xl[i], yl[i] = w1*alpha, w2*alpha
            if alpha > 0:
                colors.append("green")
            else:
                colors.append("red")
            d = self.fun_marker.transformed(mpl.transforms.Affine2D().rotate_deg(np.arctan(w2/w1)*360/2/np.pi+90))
            mks.append(d)

        self.allscat.set_offsets(np.column_stack((xl, yl)))
        self.allscat.set_paths(mks)
        self.allscat.set_sizes(sizes)
        self.allscat.set_color(colors)
        #self.signedE.set_offsets(np.column_stack((self.Xb, di["signedE"])))
        #self.pdirecs.set_offsets(np.column_stack((di["pdirecs"], -0.1*np.ones(di["pdirecs"].shape))))
        #set_alpha : same way 0-1
        self.text.set_text(f"frame {frame}, loss {loss}")
        #self.outline.set_data(self.Xout, Yout)
        return self.text, self.allscat

    def update(self, frame):
        di = self.D[frame]
        return self.update_aux(di, frame)

    def getAnim(self, interval, blit=True):
        return animation.FuncAnimation(self.fig, self.update, frames=list(range(self.Nframe)), blit=blit, interval=interval)

class NiceAnim:
    def __init__(self, fig, data, runanim=False):
        self.fig = fig
        # load data
        self.Xb, self.Y, self.X = data["Xb"], data["Y"][:, 0], data["X"]
        self.Xout = data["Xout"]
        self.n, self.d = self.X.shape
        self.m = data["ly1"].shape[1]
        assert self.Y.ndim == 1 and len(self.Y) == self.n
        if runanim:
            self.D = None
        else:
            self.D = data["iterdata"]
            self.Nframe = len(self.D)
            # sanity checks on data, looks useless but this is python
            assert np.all([di["ly1"].shape == (self.d, self.m) for di in self.D])
            assert np.all([di["ly2"].shape == (self.m, 1) for di in self.D])
        # plotting setup
        self.ax = self.fig.add_subplot()
        self.ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        self.ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        self.ax.grid(True, alpha=0.2)
        self.ax.scatter(self.Xb, self.Y, marker="x", color="red", s=40, alpha=1)
        ## not very useful but works
        #self.signedE = self.ax.scatter([], [], marker="*", color="blue", s=60, alpha=1)
        #self.pdirecs= self.ax.scatter([], [], marker="^", color="blue", s=60, alpha=1)
        ## 
        ## beautiful lines don't touch
        self.pos_marker = MarkerStyle('>').get_path().transformed(MarkerStyle('>').get_transform())
        self.neg_marker = MarkerStyle('<').get_path().transformed(MarkerStyle('<').get_transform())
        # artist engaged in animation
        self.text = self.ax.text(0.05, 0.9, f"start frame", transform=self.ax.transAxes)
        self.outline, = self.ax.plot([], [])
        self.allscat = self.ax.scatter([], [], marker='o')
        # administrative stuff
        self.starttime = time.time()
        self.verbose = False

    def update_aux(self, di, frame):
        ly1, ly2 = di["ly1"], di["ly2"]
        lact, lnorm = di["lact"], di["lnorm"]
        loss, Yout = di["loss"], di["Yout"]
        lsize = di["lsize"]
        #ly1g, ly2g = di["ly1g"], di["ly2g"]

        xl, yl, mks = np.zeros(self.m), np.zeros(self.m), []
        sizes, colors = lsize, []
        for i in range(self.m):
            w1, w2 = ly1.T[i]
            alpha, = ly2[i]
            xl[i], yl[i] = lact[i], lnorm[i]
            if alpha > 0:
                colors.append("green")
            else:
                colors.append("red")
            if w1 > 0:
                mks.append(self.pos_marker)
            else:
                mks.append(self.neg_marker)

        self.allscat.set_offsets(np.column_stack((xl, yl)))
        self.allscat.set_paths(mks)
        self.allscat.set_sizes(sizes)
        self.allscat.set_color(colors)
        ## not very useful but works
        #self.signedE.set_offsets(np.column_stack((self.Xb, di["signedE"])))
        #self.pdirecs.set_offsets(np.column_stack((di["pdirecs"], -0.1*np.ones(di["pdirecs"].shape))))
        ## 
        #set_alpha : same way 0-1
        self.text.set_text(f"frame {frame}, loss {loss}")
        self.outline.set_data(self.Xout, Yout)
        return self.outline, self.text, self.allscat#, self.signedE, self.pdirecs

    def update(self, frame):
        di = self.D[frame]
        return self.update_aux(di, frame)

    def getAnim(self, interval, blit=True):
        return animation.FuncAnimation(self.fig, self.update, frames=list(range(self.Nframe)), blit=blit, interval=interval)

def NNtoIter(Xt, Yt, allX, ly1, ly2, run=False):
    #lnorm = np.abs(ly1[1,:].flatten() * ly2.flatten()) # -> slope of the ReLU unit
    #lnorm =np.abs(ly2.flatten()) # -> similar alpha = similar speed. But the first layer's norm also matter...
    lspeed = 1/np.linalg.norm(ly1, ord=2, axis=0) * np.abs(ly2.flatten())
    lslope = np.abs(ly1[1,:].flatten() * ly2.flatten())
    lsize = 1/(lspeed+1) + (40 if run else 0)# slower = bigger
    lnorm = lslope
    lnorm = np.linalg.norm(ly1, ord=2, axis=0) * ly2.flatten() # -> mult or addition, just an "idea" of how big the neuron is
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


def simplerun(X):
    lly1, lly2 = [X["ly1"]], [X["ly2"]]
    opti, Nstep = X["opti"], X["Nstep"]
    #print(lly2[0])
    for num in range(Nstep-1):
        opti.step()
        nly1, nly2 = opti.params()
        if num%1 == 0:
            #print(nly2)
            #print("ly1 dist", np.linalg.norm(lly1[0]-nly1)) #indeed, it's 0
            print("ly2 dist", np.linalg.norm(lly2[0]-nly2))
            lly1.append(nly1)
            lly2.append(nly2)
            print("loss=", opti.loss())

    return {"lly1":lly1, "lly2":lly2}

def simpleCrun(X):
    data = X
    lly1, lly2 = [X["ly1"]], [X["ly2"]]
    opti = X["opti"]
    X, Y = X["X"], X["Y"] # no comment
    num = 0
    bestloss = opti.loss()
    print("it=", num, "loss=", opti.loss())
    try:
        while True:
            opti.step()
            nly1, nly2 = opti.params()
            lly1.append(nly1)
            lly2.append(nly2)
            num += 1
            print("it=", num, "loss=", opti.loss())
            l = opti.loss()
            if l < bestloss:
                bestloss = l
            if l/bestloss > 10:
                print(".. completely diverged.")
                #assert False
    except KeyboardInterrupt:
        print("Normal interrupt at num=", num)
    print("say something?")
    #except Exception as e:
    #   print("Something went really wrong at num=", num)
    #   print(e)

    return {"lly1":lly1, "lly2":lly2}


def simpleArun(X, myanim):
    data = X
    lly1, lly2 = [X["ly1"]], [X["ly2"]]
    opti = X["opti"]
    X, Y = X["X"], X["Y"] # no comment
    num = 0

    print("Animation setup..")
    fig = plt.figure(figsize=(10,4))
    Xoutb = np.linspace(-2,2, 1000)
    Xout = add_bias(Xoutb)
    animobj = myanim(fig, data|{"Xout":Xoutb}, runanim=True)
    already = [False] # see comment about i=0
    bestloss = [opti.loss()]
    def update_ok(i):
        if i == 0: # for some reason, this function is called 4 times with i=0
            if not already[0]:
                print("it=", i, "loss=", opti.loss())
                already[0] = True
            di = NNtoIter(X, Y, Xout, lly1[0], lly2[0], run=True)
            return animobj.update_aux(di, i) # so we simply don't do the step
        opti.step()
        nly1, nly2 = opti.params()
        di = NNtoIter(X, Y, Xout, nly1, nly2, run=True)
        lly1.append(nly1)
        lly2.append(nly2)
        print("it=", i, "loss=", opti.loss())
        l = opti.loss()
        if l < bestloss[0]:
            bestloss[0] = l
        if l/bestloss[0] > 10:
            print(".. completely diverged.")
             #assert False
        return animobj.update_aux(di, i)

    try:
        ani = animation.FuncAnimation(fig, update_ok, frames=list(range(1000)), blit=True, interval=1)
        animobj.ax.set_xlim(-1.2, 1.2)
        animobj.ax.set_ylim(-1.2, 1.2)
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
    allXb = np.linspace(-2,2, 1000)
    allX = add_bias(allXb)
    iterdata = [NNtoIter(X, Y, allX, lly1[i], lly2[i]) for i in range(len(lly1))]
    normData(iterdata, "lnorm", 0, 1)
    normData(iterdata, "lsize", 10, 70)
    return {"Xout": allXb, "iterdata": iterdata}


def getInput2(args):
    seed = 4
    torch.use_deterministic_algorithms(True)
    gpudevice = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if gpudevice != "cpu":
        print(f"Found a gpu: {gpudevice}..")
    device = "cpu"

    m, d, n = 10, 2, 20
    lr, beta = args.lr, 0
    scaling = args.scaleinit
    Nstep = args.steps

    rng = np.random.default_rng(seed) # do not use np.random, see https://numpy.org/doc/stable/reference/random/generator.html#distributions
    #X, Y = add_bias(Xb), np.sin(10*Xb)*Xb*0.3
    Xb = np.array([0.2, 0.4, 0.6])[:, None]
    X, Y = add_bias(Xb), np.array([0.3, 0.6, 0.7])[:, None]
    Xb = np.linspace(-1, 1, n)[:, None]
    #X, Y = add_bias(Xb), np.sin(Xb-np.pi/2)+1 # convex
    X, Y = add_bias(Xb), np.sin(Xb*4-np.pi/2) # convex

    ly1 = rng.uniform(-scaling, scaling, size=(d, m))
    ly2 = rng.uniform(-scaling, scaling, size=(m, 1))
    ly2 = rng.uniform(0, scaling, size=(m, 1))
    ly1 = np.array([[2, 0.5], [1, 0.5], [-1.2, 0.5], [1, 0.1]]).T*scaling
    ly1 = np.array([[1, 0.01]]).T*scaling
    m = 100
    ly1 = rng.uniform(-1, 1, size=(d, m))
    ly1 = ly1 / np.linalg.norm(ly1, axis=0)
    scalars = 3
    left, right = 0.01, 1
    scales = np.array([np.linspace([left]*m, [right]*m, scalars).T, np.linspace([left]*m, [right]*m, scalars).T])
    ly1 = (scales.T * ly1.T).reshape((m*scalars, d)).T
    ly2 = np.ones((len(ly1.T), 1))*scaling
    # double the number of neurons to allow for negative neurons..
    ly1 = np.concatenate((ly1, ly1*1.0), axis=1)
    ly2 = np.concatenate((ly2, ly2*(-1.0)), axis=0)
    if args.algo == "torch":
        opti = torch_descent(device=device, algo="gd")
        opti.load(X, Y, ly1, ly2, lr, beta)
    elif args.algo == "jko":
        opti = jko_descent(interiter=args.jkosteps, gamma=args.jkogamma, tau=args.jkotau, verb=args.verbose, proxf=args.proxf, adamlr=args.adamlr)
        opti.load(X, Y, ly1, ly2, lr, beta)
        optit = torch_descent(device=device)
        optit.load(X, Y, ly1, ly2, lr, beta)
        print(f"jkoloss:{opti.loss()} vs torchloss {optit.loss()}")

    return {"seed": seed,
            "gpudevice": gpudevice,
            "device": device,
            "m": m,
            "d": d,
            "n": n,
            "lr": lr,
            "beta": beta,
            "rng": rng,
            "Nstep": Nstep,
            "scaling": scaling,
            "Xb": Xb,
            "X": X,
            "Y": Y,
            "ly1": ly1,
            "ly2": ly2,
            "opti": opti,
            }

def getInput(args):
    seed = 4
    torch.use_deterministic_algorithms(True)
    gpudevice = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if gpudevice != "cpu":
        print(f"Found a gpu: {gpudevice}..")
    device = "cpu"

    m, d, n = 10, 2, 4
    lr, beta = args.lr, 0
    scaling = args.scaleinit
    Nstep = args.steps

    rng = np.random.default_rng(seed) # do not use np.random, see https://numpy.org/doc/stable/reference/random/generator.html#distributions
    Xb = np.linspace(-1, 1, n)[:, None]
    #X, Y = add_bias(Xb), np.sin(10*Xb)*Xb*0.3
    Xb = np.array([0.2, 0.4, 0.6])[:, None]
    X, Y = add_bias(Xb), np.array([0.5, 0.6, 0.7])[:, None]
    X, Y = add_bias(Xb), Xb*1

    ly1 = rng.uniform(-scaling, scaling, size=(d, m))
    ly2 = rng.uniform(-scaling, scaling, size=(m, 1))
    ly2 = rng.uniform(0, scaling, size=(m, 1))
    ly1 = np.array([[2, 0.5], [0.002, 0.0005], [0.002, 0.0005]]).T*scaling
    scales = np.linspace(0.01, 5, 5)
    ly2 = np.ones((len(ly1.T), 1))*scaling
    if args.algo == "torch":
        opti = torch_descent(device=device, algo="gd")
        opti.load(X, Y, ly1, ly2, lr, beta)
    elif args.algo == "jko":
        opti = jko_descent(interiter=args.jkosteps, gamma=args.jkogamma, tau=args.jkotau, verb=args.verbose, proxf=args.proxf, adamlr=args.adamlr)
        opti.load(X, Y, ly1, ly2, lr, beta)
        optit = torch_descent(device=device)
        optit.load(X, Y, ly1, ly2, lr, beta)
        print(f"jkoloss:{opti.loss()} vs torchloss {optit.loss()}")
    return {"seed": seed,
            "gpudevice": gpudevice,
            "device": device,
            "m": m,
            "d": d,
            "n": n,
            "lr": lr,
            "beta": beta,
            "rng": rng,
            "Nstep": Nstep,
            "scaling": scaling,
            "Xb": Xb,
            "X": X,
            "Y": Y,
            "ly1": ly1,
            "ly2": ly2,
            "opti": opti,
            }

if __name__ == '__main__':
    animationDict = {"output+neurons":NiceAnim,
                 "dataspace": LessNiceAnim,
                 }

    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--keepfirst", help="keep first step", action="store_true")
    parser.add_argument("-r", "--keepsecond", help="keep second step", action="store_true")
    parser.add_argument( "--run", action="store_true", help="ignore steps number and run until interrupted")
    parser.add_argument("--steps", default=10, type=int, help="how many iterations to optimize the network")
    parser.add_argument( "--noanim", action="store_true", help="do not show the end animation")
    parser.add_argument( "--runanim", action="store_true", help="show a real time animation, enables option 'run' as well")
    parser.add_argument( "--anim", default="output+neurons", choices=animationDict.keys(), help="what animation")
    parser.add_argument("-m", "--movie", help="save movie", action="store_true")
    parser.add_argument("--fps", type=int, default=10, help="movie fps")
    parser.add_argument("--movieout", help="output movie name", default="out_new")
    parser.add_argument("-o", "--output", help="output name", default="out_new")
    parser.add_argument( "--verbose", action="store_true")
    parser.add_argument("--scaleinit", default=1e-3, type=float, help="scalar factor to weight matrix")
    parser.add_argument("--algo", default="torch", choices=["torch", "jko"])
    parser.add_argument("-lr", type=float, default=1e-3, help="learning rate for algo='torch'")
    parser.add_argument("--jkosteps", default=10, type=int, help="algo=jko, number of internal iterations")
    parser.add_argument("--jkogamma", default=1, type=float, help="algo=jko, float")
    parser.add_argument("--jkotau", default=1, type=float, help="algo=jko, float")
    parser.add_argument("--proxf", default="scipy", choices=["scipy", "torch"], help="algo=jko, how to compute the prox")
    parser.add_argument("--adamlr", default=1e-3, type=float, help="algo=jko, proxf=torch, learning rate for gradient descent")
    args = parser.parse_args()
    code = args.output
    if code != "out_new" and args.movieout == "out_new":
        codemov = code
    else:
        codemov = args.movieout


    myanim = animationDict[args.anim]

    stepname = f".settings_{code}.pkl"
    if args.keepfirst or args.keepsecond and os.path.isfile(stepname):
        with open(stepname, "rb") as f:
            print(f"Loading {stepname}")
            local_args = pickle.load(f)
            X1 = getInput2(local_args)
    else:
        print(f"Overwriting {stepname}")
        X1 = getInput2(args)
        with open(stepname, "wb") as f:
            pickle.dump(args, f)

    stepname = f".layer_{code}.pkl"
    if (args.keepfirst or args.keepsecond) and os.path.isfile(stepname):
        with open(stepname, "rb") as f:
            print(f"Loading {stepname}")
            X2 = pickle.load(f)
    else:
        print(f"Overwriting {stepname}")
        if args.runanim:
            X2 = simpleArun(X1, myanim=myanim)
        elif args.run:
            X2 = simpleCrun(X1)
        else:
            X2 = simplerun(X1)
        with open(stepname, "wb") as f:
            pickle.dump(X2, f)


    stepname = f".calcs_{code}.pkl"
    if args.keepfirst and args.keepsecond and os.path.isfile(stepname):
        with open(stepname, "rb") as f:
            print(f"Loading {stepname}")
            X3 = pickle.load(f)
    else:
        print(f"Overwriting {stepname}")
        X3 = simplecalcs(X2|X1)
        with open(stepname, "wb") as f:
            pickle.dump(X3, f)

    if args.noanim and not args.movie:
        print("No animation requested")
        sys.exit(0)

    print("Animation setup..")
    fig = plt.figure(figsize=(10,10))
    animobj = myanim(fig, X1|X2|X3)
    ani = animobj.getAnim(1)
    animobj.ax.set_xlim(-1.2, 1.2)
    animobj.ax.set_ylim(-1.2, 1.2)

    if args.movie:
        print("Saving animation")
        writer = animation.FFMpegWriter(fps=args.fps)#, bitrate=1800)
        ani.save(f"{codemov}_movie.gif", writer=writer)
        print(f"saved as {codemov}_movie.mp4")
    else:
        plt.show()

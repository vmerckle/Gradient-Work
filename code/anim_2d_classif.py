import numpy as np
import cvxpy as cp
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable # nicer colobar
from matplotlib.markers import MarkerStyle 
# https://stackoverflow.com/questions/58089375/question-about-changing-marker-of-scatter-during-animation
# the only god damn correct answer, all others will have bugs.

# plt.rcParams['text.usetex'] = True # if needed, needed.

import torch
from numpy.random import default_rng

import argparse
import os.path
import sys
import pickle
#import pandas as pd

from utils import *
from torch_descent import torch_descent

class NiceAnim:
    def __init__(self, fig, data):
        self.fig = fig
        # load data
        self.Xb, self.Y, self.X = data["Xb"], data["Y"][:, 0], data["X"]
        self.Xout, self.Nout = data["Xout"], data["Nout"]
        self.outleft, self.outright = data["leftout"], data["rightout"]
        self.D = data["iterdata"]
        self.xboth = data["xboth"]

        self.Nframe = len(self.D)
        # sanity checks on data, looks useless but this is python
        self.n, self.d = self.X.shape
        self.m = self.D[0]["ly1"].shape[1]
        assert self.Y.ndim == 1 and len(Y) == self.n
        assert np.all([di["ly1"].shape == (self.d, self.m) for di in self.D])
        assert np.all([di["ly2"].shape == (self.m, 1) for di in self.D])
        # plotting setup
        self.ax = self.fig.add_subplot()
        self.ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        self.ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        self.ax.grid(True, alpha=0.2)
        self.dat = self.ax.scatter(self.Xb[:, 0], self.Xb[:, 1], marker="x", color=["red" if y < 0 else "cyan" for y in self.Y], s=40, alpha=1)
        Yout = self.D[0]["Yout"]
        self.im = self.ax.imshow(Yout.reshape(self.Nout, self.Nout).T, extent=[self.outleft, self.outright, self.outleft, self.outright], alpha=1.0, cmap=mpl.cm.coolwarm)

        #self.im.set_clim(min(np.min(Yout),-1), max(1, np.max(Yout))) # blit must be disabled for colormap to auto update
        self.im.set_clim(-1.1, 1.1) # blit must be disabled for colormap to auto update
        # artist engaged in animation
        self.text = self.ax.text(0.05, 0.9, f"start frame", transform=self.ax.transAxes)
        # administrative stuff
        self.starttime = time.time()
        self.verbose = False
        self.lines = []
        nb_neurons = len(self.D[0]["slopeY"])
        for yy in self.D[0]["slopeY"]:
            x, = self.ax.plot(self.xboth, yy, color="green", alpha=2/np.sqrt(nb_neurons))
            self.lines.append(x)
        div = make_axes_locatable(self.ax)
        cax = div.append_axes('right', '5%', '5%')
        self.colorbar = fig.colorbar(self.im, cax=cax)
        #self.title = self.ax.set_title('Frame 0')

    def update(self, frame):
        #self.title.set_text(f"Frame {frame}") #blit=false
        di = self.D[frame]
        ly1, ly2 = di["ly1"], di["ly2"]
        loss, Yout = di["loss"], di["Yout"]

        self.text.set_text(f"Frame {frame}, loss {loss}")
        for i, l in enumerate(self.lines):
            l.set_data(di["slopeX"], di["slopeY"][i])
        self.im.set_data(Yout.reshape(self.Nout, self.Nout))#, extent=[-1, 1, -1, 1])
        # self.im.set_clim(np.min(Yout), np.max(Yout)) # blit must be disabled for colormap to auto update
        return self.text, self.im, self.dat, *self.lines

    def getAnim(self, interval, blit=True):
        return animation.FuncAnimation(self.fig, self.update, frames=list(range(self.Nframe)), blit=blit, interval=interval)


def NNtoIter(Xt, Yt, allX, ly1, ly2, xboth):
    #lnorm = np.linalg.norm(ly1, ord=2, axis=0) * np.abs(ly2.flatten()) # -> mult or addition, just an "idea" of how big the neuron is
    #lnorm = np.abs(ly1[1,:].flatten() * ly2.flatten()) # -> slope of the ReLU unit
    #lnorm =np.abs(ly2.flatten()) # -> similar alpha = similar speed. But the first layer's norm also matter...
    #lspeed = 1/np.linalg.norm(ly1, ord=2, axis=0) * np.abs(ly2.flatten())
    lslope = np.abs(ly1[1,:].flatten() * ly2.flatten())
    #lsize = 1/(lspeed+1)# slower = bigger
    #lnorm = lslope
    Yout = np_2layers(allX, ly1, ly2)
    Yhat = np_2layers(Xt, ly1, ly2)
    signedE = Yhat-Yt
    loss = MSEloss(Yhat, Yt)
    both = -(ly1[0]*xboth + ly1[2])/ly1[1]
    #print(ly1.T.flatten(), "->" , ly2.flatten())
    
    return {"ly1": ly1, "ly2": ly2, "loss": loss, "Yout": Yout, "slopeX":xboth, "slopeY":both.T}


def simplerun(opti, Nstep):
    lly1, lly2 = [ly1], [ly2]
    for _ in range(Nstep-1):
        opti.step()
        nly1, nly2 = opti.params()
        lly1.append(nly1)
        lly2.append(nly2)

    return lly1, lly2

def simplecalcs(X, Y, lly1, lly2, rng):
    xboth = np.array([[-1, 1]]).T
    left, right, nb = -1.2, 1.2, 100
    x,y = np.meshgrid(np.linspace(right, left, nb), np.linspace(left, right, nb))
    # inversion(instead of (x,y)) needed, no margin for proof
    # we need the first element to be the top corner (-1, 1), then (-0.5, 1)  etc..
    Xgrid = np.array((y,x)).T
    Xout = Xgrid.reshape(-1, 2)
    allX = add_bias(Xout)
    iterdata = [NNtoIter(X, Y, allX, lly1[i], lly2[i], xboth) for i in range(len(lly1))]
    #normData(iterdata, "lnorm", 0, 1)
    #normData(iterdata, "lsize", 10, 70)
    data = {"Xb": Xb, "Y": Y, "X": X, "Xout": Xout, "Nout":nb, "leftout":left, "rightout":right, "iterdata": iterdata, "xboth":xboth}

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="output name", default="out_classif2D")
    parser.add_argument("-m", "--movie", help="save movie", action="store_true")
    parser.add_argument("-k", "--keepfirst", help="keep first step", action="store_true")
    parser.add_argument("-r", "--keepsecond", help="keep second step", action="store_true")
    parser.add_argument("--algo", default="torch", choices=["torch", "jko"])
    parser.add_argument("--steps", default=10, type=int)
    parser.add_argument("--jkosteps", default=10, type=int)
    parser.add_argument("-lr", type=float, default=1e-3)
    args = parser.parse_args()
    code = args.output

    seed = 9
    torch.use_deterministic_algorithms(True)
    gpudevice = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if gpudevice != "cpu":
        print(f"Found a gpu: {gpudevice}..")
    device = "cpu"

    m, d, n = 1000, 3, 10
    lr, beta = 1e-1, 0
    scaling = np.sqrt(1/m)/10
    Nstep = args.steps

    rng = np.random.default_rng(seed) # do not use np.random, see https://numpy.org/doc/stable/reference/random/generator.html#distributions
    Xb = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]])
    #Xb = np.array([[-0.5, 0.1], [0.5, 0.4], [0.75, 0.7], [1, 1]])
    uno = rng.multivariate_normal([0.2, 0.2], np.array([[1,0],[0,1]])/10, n)
    deuso = rng.multivariate_normal([-0.2, -0.2], np.array([[1,0],[0,1]])/10, n)
    Xb = np.vstack((uno, deuso))
    X, Y = add_bias(Xb), np.vstack((np.ones((n,1)), -np.ones((n,1))))
    #X, Y = add_bias(Xb), np.array([[-1, 0.5, 0.75, 1]]).T
    ly1 = rng.uniform(-scaling, scaling, size=(d, m))
    #ly1 = np.array([[1, 0.01, 0.7], [-1, -0.02, 0.1]]).T
    ly2 = rng.uniform(-scaling, scaling, size=(m, 1))
    #ly2 = np.array([[0.5], [-0.05]])
    print(f"X: {X.shape}, Xb: {Xb.shape}, Y: {Y.shape}")
    print(f"ly1: {ly1.shape}, ly2: {ly2.shape}")

    if args.algo == "torch":
        opti = torch_descent(device=device, algo="adam")
        opti.load(X, Y, ly1, ly2, lr, beta)
    elif args.algo == "jko":
        opti = jko_descent(interiter=args.jkosteps, gamma=0.01, tau=0.1)
        opti.load(X, Y, ly1, ly2, lr, beta)

    stepname = f".layer_{code}.pkl"
    if (args.keepfirst or args.keepsecond) and os.path.isfile(stepname):
        with open(stepname, "rb") as f:
            print(f"Loading {stepname}")
            lly1, lly2 = pickle.load(f)
    else:
        lly1, lly2 = simplerun(opti, Nstep)
        with open(stepname, "wb") as f:
            pickle.dump((lly1, lly2), f)


    stepname = f".calcs_{code}.pkl"
    if args.keepfirst and args.keepsecond and os.path.isfile(stepname):
        with open(stepname, "rb") as f:
            print(f"Loading {stepname}")
            data = pickle.load(f)
    else:
        data = simplecalcs(X, Y, lly1, lly2, rng)
        with open(stepname, "wb") as f:
            pickle.dump(data, f)
    
    print("Animation setup..")
    fig = plt.figure(figsize=(10,10))
    animobj = NiceAnim(fig, data)
    ani = animobj.getAnim(1)
    animobj.ax.set_xlim(-1., 1.)
    animobj.ax.set_ylim(-1., 1.)

    if args.movie:
        print("Saving animation")
        writer = animation.FFMpegWriter(fps=20,bitrate=4000)#, bitrate=1800)
        ani.save(f"{code}_movie.mp4", writer=writer)
    else:
        plt.show()

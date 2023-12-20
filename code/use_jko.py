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
from jko_descent import jko_descent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default=10, type=int)
    parser.add_argument("--jkosteps", default=10, type=int)
    args = parser.parse_args()

    Nsteps = args.steps
    Niter = args.jkosteps
    gamma, tau = 1, 1.0
    seed = 4
    rng = np.random.default_rng(seed)


    left, right, nb = -1.2, 1.0, 10
    x,y = np.meshgrid(np.linspace(left, right, nb), np.linspace(left, right, nb))
    Xgrid = np.array((x,y)).T
    grid = Xgrid.reshape(-1, 2)

    q     = np.ones((len(grid), 1))/len(grid) # Initial distribution
    #q     = np.zeros_like(grid); q[0] = 1.0

    #rng = np.random.default_rng(4)
    ## Data
    #m = 5
    #Xb = rng.normal(size=(m, 1))
    #X = add_bias(Xb)
    #y = Xb*0.4000 #+ 0.01*rng.normal(size=m)

    Xb = np.array([-1, 0])
    X = add_bias(Xb)
    y = np.array([1, 1])

    opti = jko_descent(interiter=Niter, gamma=gamma, tau=tau, verb=True)
    norm = np.linalg.norm(grid, axis=1)
    opti.load(X, y, ly1=None, ly2=None, lr=0, beta=0, grid=grid, p=q)
    print(grid)
    print(q)
    
    plist = [q]
    loss_list = [opti.loss()]
    print(f"0: Loss={opti.loss()}")
    for i in range(Nsteps-1):
        opti.step()
        p = opti.params(gridout=True)[1]
        plist.append(p)
        loss = opti.loss()
        loss_list.append(loss)
        print(f"{i+1}: Loss={loss}, Sum={p.sum():.4f}")


    Xtoplot = []
    for (w1, w2) in grid:
        if w1 == 0:
            Xtoplot.append(0)
        else:
            Xtoplot.append(-w2/w1)

    fig,ax = plt.subplots()
    plt.scatter(Xtoplot,q)

    def updateData(curr):
        ax.clear()
        p = plist[curr]
        print(f"{curr}: Loss={loss_list[curr]}, Sum={p.sum():.4f}")
        plt.scatter(Xtoplot,p)

    #ani = animation.FuncAnimation(fig, updateData, interval=200, frames=Nsteps, repeat=False)
    #writer = animation.FFMpegWriter(fps=10)#, bitrate=1800)
    #ani.save(f"test.mp4", writer=writer)

    #plt.show()

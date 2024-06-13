import argparse
import os.path
import os
import pickle
from rich import print
from inspect import getmembers, isfunction, isclass
import datetime

# my libraries
from utils import *
import runner
import postprocess
import configs
import proxdistance

import torch


import matplotlib.pyplot as plt
import matplotlib as mpl

def loadts(timestamps, folder):
    configD = {n[6:]:f for n,f in getmembers(configs, isfunction) if len(n) > 6 and n[:6] == "Config"}
    datalist = {} # generate timestamps->files
    for fname in next(os.walk(folder))[2]:
        config_name, timestamp, step = fname.split("-")
        timestamp = int(timestamp)
        step = step.split(".")[0]
        if timestamp in datalist:
            datalist[timestamp][step] = fname
        else:
            datalist[timestamp] = {step:fname, 'config':config_name}

    Xs = []
    for ts in timestamps:
        if ts not in datalist:
            raise Exception("timestamp not found", ts)
        x = datalist[ts]
        with open(f"{folder}/{x['setup']}", "rb") as f:
            X1 = pickle.load(f)
        with open(f"{folder}/{x['descent']}", "rb") as f:
            X2 = pickle.load(f)
        with open(f"{folder}/{x['postprocess']}", "rb") as f:
            X3 = pickle.load(f)
        with open(f"{folder}/{x['meta']}", "rb") as f:
            X4 = pickle.load(f)
        Xs.append(X1|X2|X3|X4)
    return Xs

    #print(f"{x['config']} {ts} ({timestring(ts)} ago) {X4['steps']} steps, m={X1['m']}, n={X1['n']}, {X1['algo']}-{X1['proxdist']}, g={X1['gamma']}, inner={X1['inneriter']}")

def comparetwo(ts1, ts2, folder):
    rg = range(2)
    Xs = loadts([ts1, ts2], folder)
    lys = [Xs[i]["lly1"] for i in rg]
    dists = [Xs[i]["proxdist"] for i in rg]
    iters = np.array([Xs[i]["iterdata"] for i in rg])
    objs = [[x["loss"] for x in iters[i]] for i in rg] # iterdata is a list of {}
    gammas = [Xs[i]["gamma"] for i in rg]

    wassers = []
    numerop= []
    euclids = []
    for i in rg: # someone(no one) said no double list comprehensions.. for comprehension
        a = np.array([proxdistance.wasserstein_np(lys[i][j], lys[i][j+1]) for j in range(len(lys[i])-1)])
        wassers.append(a)
        c = np.array([proxdistance.wasserstein_num(lys[i][j], lys[i][j+1]) for j in range(len(lys[i])-1)])
        numerop.append(c)
        b = np.array([proxdistance.frobenius_np(lys[i][j], lys[i][j+1]) for j in range(len(lys[i])-1)])
        euclids.append(b)

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(frameon=False)

    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    #ax.set_ylabel('values', loc='center')
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.grid(True, alpha=0.2)
    ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    for i in rg:
        ax.plot(objs[i], label=f"{dists[i]} Loss(MSE)")
        ax.plot(numerop[i], label=f"{dists[i]} Nb of permut")
        ax2.plot(wassers[i], label=f"{dists[i]}-Distance($W_2^2$)", linestyle="dashed")
        ax2.plot(euclids[i], label=f"{dists[i]}-Distance($F$)", linestyle="dotted")
    ax2.set_yscale("log")
    ax.set_xlabel('step', loc='center')
    ax.set_ylabel('Loss', loc='center')
    ax2.set_ylabel('Norms', loc='center')
    ax.legend(loc="upper left")
    ax2.legend()
    #lines, labels = ax.get_legend_handles_labels()
    #lines2, labels2 = ax2.get_legend_handles_labels()
    #ax2.legend(lines + lines2, labels + labels2, loc=0)
    #plt.legend()
    #plt.savefig(f"{codemov}_plot.png", dpi=400)
    plt.show()

def comparetwoscatt(ts1, ts2, folder):
    rg = range(2)
    Xs = loadts([ts1, ts2], folder)
    lys = [Xs[i]["lly1"] for i in rg]
    ly2s = [Xs[i]["lly2"] for i in rg]
    dists = [Xs[i]["proxdist"] for i in rg]
    iters = np.array([Xs[i]["iterdata"] for i in rg])
    objs = [[x["loss"] for x in iters[i]] for i in rg] # iterdata is a list of {}
    gammas = [Xs[i]["gamma"] for i in rg]

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(frameon=False)

    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    #ax.set_ylabel('values', loc='center')
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.grid(True, alpha=0.2)
    i = 0
    linestyle = ["solid", "dashed"]
    for i in rg:
        nstep = len(lys[i])
        m = len(lys[i][0][0])
        trajsx = np.array([lys[i][k][0] for k in range(nstep)]).T
        trajsy = np.array([lys[i][k][1] for k in range(nstep)]).T # number of neurons x nsteps
        colors = ["C1" if ly2s[i][0][u]>0 else "C0" for u in range(m)]
        colorsc = ["red" if ly2s[i][0][u]>0 else "blue" for u in range(m)]
        ax.scatter(x=lys[i][0][0], y=lys[i][0][1], color=colorsc)
        for (trajx, trajy, color) in zip(trajsx, trajsy, colors):
            ax.plot(trajx, trajy, color=color, linestyle=linestyle[i], alpha=0.5)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("ts1", type=int)
    parser.add_argument("ts2", type=int)
    parser.add_argument("-f", "--folder", help="folder name", default="data")
    args = parser.parse_args()

    comparetwoscatt(args.ts1, args.ts2, args.folder)
    comparetwo(args.ts1, args.ts2, args.folder)
    plt.show()

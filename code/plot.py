#!/usr/bin/env python

import argparse
import os.path
import os
import pickle
import datetime

import numpy as np
from rich import print
import matplotlib.pyplot as plt

from utils import *
import postprocess
import load

def tolist(ld, key):
    # from Dict {10: {"key1": 0.22}, 13: {"key1":0.35}}
    # outputs [10, 13], [0.22, 0.35] for "key1"
    nums = list(ld.keys())
    nums.sort()
    return np.array(nums), np.array([ld[x][key] for x in nums])

def mergeproxfloats(D):
    nums = list(D["iter"].keys())
    nums.sort()
    if nums != list(range(len(nums))):
        print("not every iteration was recorded - doesn't make sense to merge prox loops")
        return None
    keylist = ["obj", "dist", "loss", "lr"]
    s = {}
    for k in keylist:
        s[k] = []
        for i in range(1, len(nums)):
            # works for varying prox durations
            _, l = tolist(D["iter"][i]["innerD"], k)
            s[k].extend(l)
    return list(range(len(s["obj"]))), s

def plotprox(D, pre, show, dpi=100):
    nums, s = mergeproxfloats(D)

    fig = plt.figure(figsize=(19.8,10.8))
    ax = fig.add_subplot(frameon=False)

    #ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    #ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.grid(True, alpha=0.2)
    ax.scatter(nums, s["obj"], label='prox objective', marker='+', alpha=0.2)
    ax.scatter(nums, s["dist"], label='prox distance', marker='+', alpha=0.2)
    ax.plot(nums, s["lr"], label='learning rate', color="grey")
    ax.set_yscale('log')
    #ax.set_ylabel('values', loc='center')
    ax.legend()
    if show:
        plt.show()
    else:
        name = f"{pre}mergedprox_obj_dist.png"
        plt.savefig(name, dpi=dpi)
        print("saved", name)

def plotloss(D, pre, show, dpi):
    nums, lossL = tolist(D["iter"], "loss")

    fig = plt.figure(figsize=(19.8,10.8))
    ax = fig.add_subplot(frameon=False)

    ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.grid(True, alpha=0.2)
    ax.scatter(list(range(len(lossL))), lossL, label="loss", marker='+', alpha=0.8)
    ax.set_yscale('log')
    ax.legend()

    if show:
        plt.show()
    else:
        name = f"{pre}loss.png"
        plt.savefig(name, dpi=dpi)
        print("saved", name)

def plotdata1D(D, pre, show, dpi):
    X, Y = D["Xb"], D["Y"].flatten()
    Xb = D["X"]
    assert X.shape[1] == 1
    X = X.flatten()

    fig = plt.figure(figsize=(19.8,10.8))
    ax = fig.add_subplot(frameon=False)


    ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.grid(True, alpha=0.2)
    ax.scatter(X, Y, label="1D data labelled", marker='+', alpha=1.0, color="red")
    #ax.set_yscale('log')
    ax.legend()

    if show:
        plt.show()
    else:
        name = f"{pre}1Data_labelled.png"
        plt.savefig(name, dpi=dpi)
        print("saved", name)

    fig = plt.figure(figsize=(19.8,10.8))
    ax = fig.add_subplot(frameon=False)


    ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.grid(True, alpha=0.2)
    ax.scatter(Xb[:,0], Xb[:,1], label="data", marker='+', alpha=1.0, color="red")
    #ax.set_yscale('log')
    ax.legend()

    if show:
        plt.show()
    else:
        name = f"{pre}1Data.png"
        plt.savefig(name, dpi=dpi)
        print("saved", name)

def plotneurons1D(D, pre, show, dpi):
    assert D["dataD"]["d"] == 2
    nums = list(D["iter"].keys())
    nums.sort()
    X, Y = D["Xb"], D["Y"].flatten()
    X = X.flatten()
    ly2 = D["iter"][0]["ly2"]
    nbpos = np.sum((ly2 > 0)).item()



    def plotone(i):
        ly1 = D["iter"][i]["ly1"]
        fig = plt.figure(figsize=(19.8,10.8))
        ax = fig.add_subplot(frameon=False)


        ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        ax.grid(True, alpha=0.2)
        ax.scatter(ly1[0,:nbpos], ly1[1,:nbpos], color="green",label="positive neuron", marker='+', alpha=1.0)
        ax.scatter(ly1[0,nbpos:], ly1[1,nbpos:], color="red",label="negative neuron", marker='+', alpha=1.0)
        #ax.set_yscale('log')
        ax.legend()

        if show:
            plt.show()
        else:
            name = f"{pre}neurons_{i}.png"
            plt.savefig(name, dpi=dpi)
            print("saved", name)

    plotone(0)

def plotneurons1D_traj(D, pre, show, dpi):
    assert D["dataD"]["d"] == 2
    X, Y = D["Xb"], D["Y"].flatten()
    X = X.flatten()
    ly2 = D["iter"][0]["ly2"]
    nbpos = np.sum((ly2 > 0)).item()

    nums = list(D["iter"].keys())
    nums.sort()
    lly1 = np.array([D["iter"][i]["ly1"] for i in nums]) # numstep, d, m
    xl, yl = lly1[:,0,:], lly1[:,1,:] # each is numstep, m for each dimension.
    pxl, pyl = xl[:, :nbpos], yl[:, :nbpos]
    nxl, nyl = xl[:, nbpos:], yl[:, nbpos:]

    fig = plt.figure(figsize=(19.8*4,10.8*4))
    ax = fig.add_subplot(frameon=False)

    ax.scatter(pxl[0], pyl[0], color="green", marker='+', label='pos start')
    ax.scatter(nxl[0], nyl[0], color="red", marker='+', label='neg start')
    ax.plot(pxl, pyl, color="green", label="positive neurons", marker='x')
    ax.plot(nxl, nyl, color="red", label="negative neurons", marker='x')


    ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.grid(True, alpha=0.2)
    #ax.set_yscale('log')
    legend_without_duplicate_labels(ax)
    #ax.legend()

    if show:
        plt.show()
    else:
        name = f"{pre}traj_neurons.png"
        plt.savefig(name, dpi=dpi)
        print("saved", name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("folder", help="folder name")
    parser.add_argument("-t", "--timestamp", help="timestamp", type=int)
    parser.add_argument("-s", "--show", help="show plots instead of saving", type=int)
    args = parser.parse_args()
    if args.timestamp is None:
        args.timestamp = load.listAllTS()[-1]
        print("No argument -> taking last timestamped data:", args.timestamp)

    D = load.getTS(args.timestamp)
    pre= f"output/{args.timestamp}_"

    plotparams = {"D": D,
                  "pre": pre,
                  "dpi": 100,
                  "show": args.show}

    #plotdata1D(**plotparams)
    plotneurons1D(**plotparams)
    plotneurons1D_traj(**plotparams)

    #if D["algoD"]["recordinner"]:
    #    plotprox(**plotparams)
    #plotloss(**plotparams)

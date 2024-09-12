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
    keylist = ["obj", "dist", "loss"]
    s = {}
    for k in keylist:
        s[k] = []
        for i in range(1, len(nums)):
            # works for varying prox durations
            _, l = tolist(D["iter"][i]["innerD"], k)
            s[k].extend(l)
    return list(range(len(s["obj"]))), s

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("folder", help="folder name")
    parser.add_argument("-t", "--timestamp", help="timestamp", type=int)
    args = parser.parse_args()
    if args.timestamp is None:
        print("Taking last timestamped data")
        args.timestamp = load.listAllTS()[-1]

    D = load.getTS(args.timestamp)
    nums, lossL = tolist(D["iter"], "loss")
    nums, s = mergeproxfloats(D)


    fig = plt.figure(figsize=(19.8,10.8))
    ax = fig.add_subplot(frameon=False)
    #ax = fig.add_subplot()

    #ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    #ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    #ax.set_ylabel('values', loc='center')
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.grid(True, alpha=0.2)
    #ax.plot(nums, lossL)
    #ax.plot(nums, s["obj"], label='prox objective')
    #ax.plot(nums, s["dist"], label='prox distance')
    ax.scatter(nums, s["obj"], label='prox objective', marker='+', alpha=0.4)
    ax.scatter(nums, s["dist"], label='prox distance', marker='+', alpha=0.4)
    ax.set_yscale('log')
    ax.legend()
    #plt.show()
    plt.savefig("output/mergedprox_obj_dist.png", dpi=300)

    fig = plt.figure(figsize=(19.8,10.8))
    ax = fig.add_subplot(frameon=False)
    #ax = fig.add_subplot()

    #ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    #ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    #ax.set_ylabel('values', loc='center')
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.grid(True, alpha=0.2)
    ax.scatter(list(range(len(lossL))), lossL, label="loss", marker='+', alpha=0.8)
    ax.set_yscale('log')
    ax.legend()
    #plt.show()
    plt.savefig("output/loss.png", dpi=300)

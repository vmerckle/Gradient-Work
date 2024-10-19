import numpy as np
from utils import *

def genModel(D):
    choice = {"grid": grid2dneuron,
              "normal": normalneuron,
              }
    t = D["init"]
    if t not in choice:
        raise Exception(f"{t} is not in {choice.keys()}: wrong init type in config")

    return choice[t](D["dataD"], D["initD"])

def grid2dneuron(dataD, D): # d = 2
    d = dataD["d"]
    assert d == 2

    rng = np.random.default_rng(D["seed"])
    m, s = D["m"], D["scale"]
    t = np.linspace(-s, s, m)
    Xm, Ym = np.meshgrid(t, t)
    # Transform X and Y into 2x(m^2) matrices
    ly1 = np.vstack((Xm.flatten(), Ym.flatten()))
    m = ly1.shape[1]

    if D["onlypositives"]:
        ly2 = np.ones((m, 1))
    else:
        ly2 = np.ones((m, 1)) * np.sign(1-2*rng.random((m, 1)))
    return ly1, ly2

def normalneuron(dataD, D):
    d = dataD["d"] # adapts model's input dim to data.
    m, s = D["m"], D["scale"]

    rng = np.random.default_rng(D["seed"])
    ly1 = rng.standard_normal((d, m))*s
    if D["onlypositives"]:
        ly2 = np.ones((m, 1))
    else:
        pos = min(m-1, max(1, int(rng.binomial(m, 0.5)))) # number of positive neurons
        ly2 = np.vstack((np.ones((pos, 1)), -1*np.ones((m-pos, 1))))
    return ly1, ly2


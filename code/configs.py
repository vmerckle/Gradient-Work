import numpy as np

#import torch
#torch.use_deterministic_algorithms(True)
#gpudevice = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

from utils import *
from types import SimpleNamespace
import torch

default = {
        "seed": 4,
        "typefloat": "float32",
        "threadcount": 1,
        "algo": "proxpoint",
        "proxdist": "frobenius",  # the last assigned value
        "gamma": 1e-2,
        "inneriter": 100,
        "datatype": "random",  # the last assigned value
        "Xsampling": "uniform",  # the last assigned value
        "onlypositives": False,
        "Ynoise": 1e-1,
        "device": "cpu",  # the last assigned value
        "beta": 0,
        "scale": 1e-3,
        "m": 10,
        "d": 10,
        "n": 10,
    }

def applyconfig(X1):
    x = SimpleNamespace(**X1)
    datarng = np.random.default_rng(x.dataseed)
    rng = np.random.default_rng(x.NNseed)
    
    if x.datatype == "linear2d":
        X, Y, Xb = linear2d(x.n, 0.1, 0.1)
    elif x.datatype == "rnglinear":
        X, Y, Xb = rnglinear(datarng, x.n, x.d, x.Xsampling, eps=x.Ynoise)
    elif x.datatype == "sinus":
        X, Y, Xb = sinus2d(x.n)
    elif x.datatype == "random":
        X, Y, Xb = rngrng(datarng, x.n, x.d, x.Xsampling)
    else:
        raise Exception("wrong datatype", x.datatype)

    if x.typefloat == "float32":
        dtype = torch.float32
    elif x.typefloat == "float64":
        dtype = torch.float64
    else:
        raise Exception("wrong typefloat", x.typefloat)

    ly1, ly2 = grid2dneuron(rng, x.m, x.scale, x.onlypositives)
    ly1, ly2 = normalneuron(rng, x.m, x.d, x.scale, x.onlypositives, sortpos=True)

    if x.proxdist == "wasser":
        x.device = "cpu"
        x.threadcount = 1

    torch.set_num_threads(x.threadcount) # 10% perf loss for wasser but only use one core.

    algo = x.algo
    if algo == "GD":
        from algo_GD_torch import torch_descent
        opti = torch_descent("gd", x.lr, device=x.device)
    elif algo == "JKO":
        proxf = x.proxf
        if proxf == "cvxpy":
            from jko_proxf_cvxpy import jko_cvxpy
            opti = jko_cvxpy(interiter=x.jko_inter_maxstep, gamma=x.gamma, tau=x.tau, tol=x.jko_tol)
        elif proxf == "scipy":
            from jko_proxf_scipy import jko_scipy
            opti = jko_scipy(interiter=x.jko_inter_maxstep, gamma=x.gamma, tau=x.tau, tol=x.jko_tol)
        elif proxf == "pytorch":
            from jko_proxf_pytorch import jko_pytorch
            opti = jko_pytorch(interiter=x.jko_inter_maxstep, gamma=x.gamma, tau=x.tau, tol=x.jko_tol)
        else:
            raise Exception("config bad proxf choice")
    elif algo == "proxpoint":
        from algo_prox import proxpoint
        if x.proxdist == "wasser":
            from proxdistance import wasserstein
            proxdist = wasserstein
        elif x.proxdist == "frobenius":
            from proxdistance import frobenius 
            proxdist = frobenius
        elif x.proxdist == "sliced":
            from proxdistance import slicedwasserstein
            proxdist = slicedwasserstein
        else:
            raise Exception("config bad proxdist choice")
        opti = proxpoint(rng=rng, proxdist=proxdist, inneriter=x.inneriter, gamma=x.gamma, dtype=dtype, device=x.device)
    else:
        raise Exception("config bad algo choice")

    opti.load(X, Y, ly1, ly2, x.beta)
    X1.update({"X":X, "Y":Y, "Xb":Xb, "lly1":[ly1], "lly2":[ly2]})
    return opti, ly1, ly2

def sinus2d(n):
    Xb = np.linspace(-0.5, 0.5, n)[:, None]
    X, Y = add_bias(Xb), np.sin(10*Xb-np.pi/2)+0.2*Xb - 0.5
    #X, Y = add_bias(Xb), Xb*0.1+0.1
    return X, Y, Xb

def linear2d(n, a, b):
    Xb = np.linspace(-0.5, 0.5, n)[:, None]
    X, Y = add_bias(Xb), Xb*a+b
    return X, Y, Xb

def rnglinear(rng, n, d, sampling, eps=0):
    if sampling == "uniform":
        Xb = rng.uniform(-0.5, 0.5, (n, d-1))
    elif sampling == "normal":
        Xb = rng.standard_normal((n, d-1))

    b = rng.uniform(-1, 1, d-1)
    noise = rng.uniform(-1, 1, n)*eps
    return add_bias(Xb), (Xb.dot(b) + noise)[:, None], Xb

def rngrng(rng, n, d, sampling):
    if sampling == "uniform":
        Xb = rng.uniform(-0.5, 0.5, (n, d-1))
        Y = rng.uniform(-2, 2, (n, 1))
    elif sampling == "normal":
        Xb = rng.standard_normal((n, d-1))
        Y = rng.normal(0.25, 1, (n, 1))
        Y += rng.normal(-0.25, 1, (n, 1)) # attempt at creating something hard
        Y = rng.uniform(-0.5, 0.5, (n, 1))

    return add_bias(Xb), Y, Xb
def grid2dneuron(rng, m, s, onlypositives):
    t = np.linspace(-s, s, m)
    Xm, Ym = np.meshgrid(t, t)
    # Transform X and Y into 2x(m^2) matrices
    ly1 = np.vstack((Xm.flatten(), Ym.flatten()))
    m = ly1.shape[1]
    if onlypositives:
        ly2 = np.ones((m, 1))
    else:
        ly2 = np.ones((m, 1)) * np.sign(1-2*rng.random((m, 1)))
    return ly1, ly2

def normalneuron(rng, m, d, s, onlypositives, sortpos):
    ly1 = rng.standard_normal((d, m))*s
    if onlypositives:
        ly2 = np.ones((m, 1))
        return ly1, ly2
    else:
        pos = min(m-1, max(1, int(rng.binomial(m, 0.5)))) # number of positive neurons
        ly2 = np.vstack((np.ones((pos, 1)), -1*np.ones((m-pos, 1))))
        return ly1, ly2


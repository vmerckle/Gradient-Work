import os
from types import SimpleNamespace
import pickle

import numpy as np
import torch
import torchvision

#torch.use_deterministic_algorithms(True)
#gpudevice = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

from utils import *

default = {
        "NNseed": 4,
        "dataseed": 4,
        "typefloat": "float32",
        "threadcount": 1,
        "device": "cpu",
        "algo": "proxpoint",
        "proxF": "scipy",
        "algoD": {"dist": "frobenius",
                  "inneriter": 100,
                  "gamma": 1e-2,
                  "recordinner": True,
                  "recordinnerlayers": False,
                  "momentum":0.95,
                  "opti": "prodigy",
                  "beta": 0,
                  "lr": 1e0,
                  "onlyTrainFirstLayer": True,
                  },
        "datatype": "random",
        "Xsampling": "uniform",
        "onlypositives": False,
        "Ynoise": 1e-1,
        "beta": 0,
        "scale": 1e-3,
        "m": 10,
        "d": 10,
        "n": 10,
    }

def mnist():
    pth = "dataset/mnist.pkl"
    if os.path.exists(pth):
        with open(pth, "rb") as f:
            return pickle.load(f)
    print("load mnist for the first time")
    if not os.path.exists("dataset"):
        os.mkdir(f"dataset")
    
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    # laziest way
    train_dataset = torchvision.datasets.MNIST('dataset/', train=True, download=True, transform=image_transform)
    test_dataset = torchvision.datasets.MNIST('dataset/', train=False, download=True, transform=image_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=60000, 
                                               shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=10000, 
                                              shuffle=False)
    for d,t in train_loader:
        Xb, Y = d, t
    Xb = Xb.view(-1, 784)
    Y = Y.to(torch.float32)[:, None]
    Xb, Y = Xb.numpy(), Y.numpy()
    X = add_bias(Xb)

    with open(pth, "wb") as f:
        pickle.dump((X, Y, Xb), f)
    return X, Y, Xb

def genData(D):
    x = SimpleNamespace(**D)
    datarng = np.random.default_rng(x.dataseed)

    if x.datatype == "linear2d":
        X, Y, Xb = linear2d(x.n, 0.1, 0.1)
    elif x.datatype == "rnglinear":
        X, Y, Xb = rnglinear(datarng, x.n, x.d, x.Xsampling, eps=x.Ynoise)
    elif x.datatype == "sinus":
        X, Y, Xb = sinus2d(x.n)
    elif x.datatype == "random":
        X, Y, Xb = rngrng(datarng, x.n, x.d, x.Xsampling)
    elif x.datatype == "mnist":
        X, Y, Xb = mnist()
        D["n"], D["d"] = X.shape
    else:
        raise Exception("wrong datatype", x.datatype)
    
    return X, Y, Xb

def genModel(D):
    x = SimpleNamespace(**D)
    rng = np.random.default_rng(x.NNseed)
    ly1, ly2 = grid2dneuron(rng, x.m, x.scale, x.onlypositives)
    ly1, ly2 = normalneuron(rng, x.m, x.d, x.scale, x.onlypositives, sortpos=True)
    return ly1, ly2

def loadOpti(D):
    x = SimpleNamespace(**D)
    #if x.proxdist == "wasser":
    #    x.device = "cpu"
    #    x.threadcount = 1
    torch.set_num_threads(x.threadcount) # 10% perf loss for wasser but only use one core.

    if x.typefloat == "float32":
        dtype = torch.float32
    elif x.typefloat == "float64":
        dtype = torch.float64
    else:
        raise Exception("wrong typefloat", x.typefloat)

    algo = x.algo
    if algo == "GD":
        from algo_GD_torch import torch_descent
        opti = torch_descent(x.algoD, device=x.device)
    elif algo == "JKO":
        if x.proxf == "cvxpy":
            from jko_proxf_cvxpy import jko_cvxpy
            opti = jko_cvxpy(x.algoD)
        elif x.proxf == "scipy":
            from jko_proxf_scipy import jko_scipy
            opti = jko_scipy(x.algoD)
        elif x.proxf == "pytorch":
            from jko_proxf_pytorch import jko_pytorch
            opti = jko_pytorch(x.algoD)
        else:
            raise Exception(f"config bad proxf='{x.proxf}' choice")
    elif algo == "proxpoint":
        from algo_prox import proxpoint
        opti = proxpoint(x.algoD, dtype=dtype, device=x.device)
    else:
        raise Exception("config bad algo choice")

    return opti

def applyconfig(D):
    X, Y, Xb = genData(D)
    ly1, ly2 = genModel(D)
    opti = loadOpti(D)
    D.update({"X":X, "Y":Y, "Xb":Xb})
    opti.load(X, Y, ly1, ly2)
    return opti

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


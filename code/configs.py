import numpy as np

#import torch
#torch.use_deterministic_algorithms(True)
#gpudevice = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

from utils import *
from types import SimpleNamespace
import torch

def applyconfig(X1):
    x = SimpleNamespace(**X1)
    rng = np.random.default_rng(x.seed)
    
    if x.datatype == "linear2d":
        X, Y, Xb = linear2d(x.n, 0.1, 0.1)
    elif x.datatype == "rnglinear":
        X, Y, Xb = rnglinear(rng, x.n, x.d, x.Xsampling, eps=x.Ynoise)
    elif x.datatype == "sinus":
        X, Y, Xb = sinus2d(x.n)
    elif x.datatype == "random":
        X, Y, Xb = rngrng(rng, x.n, x.d, x.Xsampling)
    else:
        raise Exception("wrong datatype", x.datatype)

    if x.typefloat == "float32":
        dtype = torch.float32
    elif x.typefloat == "float64":
        dtype = torch.float64
    else:
        raise Exception("wrong typefloat", x.typefloat)

    ly1, ly2 = grid2dneuron(rng, x.m, x.scale, x.onlypositives)
    ly1, ly2 = normalneuron(rng, x.m, x.d, x.scale, x.onlypositives)

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
    return {"opti": opti, "ly1":ly1, "X":X, "Y":Y, "Xb":Xb, "ly2":ly2}

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

def normalneuron(rng, m, d, s, onlypositives):
    ly1 = rng.standard_normal((d, m))*s
    if onlypositives:
        ly2 = np.ones((m, 1))
    else:
        ly2 = np.ones((m, 1)) * np.sign(1-2*rng.random((m, 1)))
    return ly1, ly2

def ConfigNormal():
    seed = 4
    typefloat = "float32"
    threadcount = 1
    device = "cuda"
    device = "cpu"

    algo = "proxpoint"
    proxdist = "wasser"
    proxdist = "sliced"
    proxdist = "frobenius"
    gamma = 1e2
    inneriter = 1000
    datatype = "rnglinear"
    datatype = "sinus"
    datatype = "random"

    Xsampling = "grid"
    Xsampling = "uniform"
    onlypositives = True
    Ynoise = 1e-1

    if proxdist == "wasser":
        device = "cpu"
        threadcount = 1

    #algo = "GD"
    lr= 1e-3*2

    beta = 0
    scale = 1e-2
    m, d, n = 1000, 1000, 1000
    m, d, n = 50, 2, 5
    #m, d, n = 100, 100, 10000

    X1 = dict([(k,v) for k,v in locals().items() if k[:2] != '__'])
    return X1

def _Config2DNew():
    m, d, n = 10, 2, 10
    Xb = np.linspace(-0.5, 0.5, n)[:, None]
    #X, Y = add_bias(Xb), np.sin(Xb-np.pi/2)+1
    X, Y = add_bias(Xb), Xb*0.1+0.1

    newneu = []
    laydeu = []
    for acti in np.linspace(-1, 1, m):
        if rng.integers(0, 2) == 0:
            a = 1
        else:
            a = 1
        b = acti*a
        newneu.append([a, b])
        if rng.integers(0, 2) == 0:
            laydeu.append(1)
        else:
            laydeu.append(1)

    ly1 = np.array(newneu).T
    ly2 = np.ones((len(ly1.T), 1))
    ly2 = np.array(laydeu)[:, None]

    if include_negative_neurons:
        raise Exception("Not implemented")
    # double the number of neurons to allow for negative neurons..
    # ly1 = np.concatenate((ly1, ly1*1.0), axis=1)
    # ly2 = np.concatenate((ly2, ly2*(-1.0)), axis=0)

def _Config1DNew():
    m, d, n = 100, 1, 5
    Xb = np.linspace(0, 1, n)[:, None]
    X, Y = Xb, Xb*0.6

    dist_bet_2pts = 0.1
    ly1 = np.array([c*dist_bet_2pts for c in range(0,m)])[:, None]
    if 0:
        ly1 = np.linspace(0.0, 1, m)[:, None]
    print(ly1.shape)
    ini = 2
    if ini == 1: # dirac
        ly2 = np.zeros((m, 1))
        ly2[-1] = 1
    elif ini == 2: #gauss
        ly2 = np.maximum(np.exp(-(ly1-6)**2*10), 1e-6)
    elif ini == 3: #uniform
        ly2 = np.ones((m, 1))
    ly2 = ly2/np.sum(ly2)
    ly1 = ly1.T

    print(f"Running {algo} (with prox={proxf}) on {m} 2D neurons, startsum = {np.sum(ly2):.1f}")
    X1 = dict([(k,v) for k,v in locals().items() if k[:2] != '__'])
    X1.update(loadalgo(X1))
    return X1
#
#
#    c = 45
#    if c == 1: # old
#        m, d, n = 10, 2, 20
#        #X, Y = add_bias(Xb), np.sin(10*Xb)*Xb*0.3
#        Xb = np.array([0.2, 0.4, 0.6])[:, None]
#        X, Y = add_bias(Xb), np.array([0.3, 0.6, 0.7])[:, None]
#        Xb = np.linspace(-1, 1, n)[:, None]
#        #X, Y = add_bias(Xb), np.sin(Xb-np.pi/2)+1
#        X, Y = add_bias(Xb), np.sin(Xb*4-np.pi/2)
#
#        ly1 = rng.uniform(-scaling, scaling, size=(d, m))
#        ly2 = rng.uniform(-scaling, scaling, size=(m, 1))
#        ly2 = rng.uniform(0, scaling, size=(m, 1))
#        ly1 = np.array([[2, 0.5], [1, 0.5], [-1.2, 0.5], [1, 0.1]]).T*scaling
#        ly1 = np.array([[1, 0.01]]).T*scaling
#    elif c== 2: # working
#        m, d, n = 100, 2, 20
#        Xb = np.linspace(-1, 1, n)[:, None]
#        #X, Y = add_bias(Xb), np.sin(Xb-np.pi/2)+1
#        X, Y = add_bias(Xb), np.sin(Xb*4-np.pi/2)
#
#        ly1 = rng.uniform(-1, 1, size=(d, m))
#        ly1 = ly1 / np.linalg.norm(ly1, axis=0)
#        scalars = 3
#        left, right = 0.01, 1
#        scales = np.array([np.linspace([left]*m, [right]*m, scalars).T, np.linspace([left]*m, [right]*m, scalars).T])
#        ly1 = (scales.T * ly1.T).reshape((m*scalars, d)).T
#        ly2 = np.ones((len(ly1.T), 1))*scaling
#        # double the number of neurons to allow for negative neurons..
#        ly1 = np.concatenate((ly1, ly1*1.0), axis=1)
#        ly2 = np.concatenate((ly2, ly2*(-1.0)), axis=0)
#    elif c == 45: # linear data, neurons uniform by activation
#        m, d, n = 100, 2, 5
#        Xb = np.linspace(0, 2, n)[:, None]
#        #X, Y = add_bias(Xb), np.sin(Xb-np.pi/2)+1
#        X, Y = add_bias(Xb), Xb*0.1+0.1
#
#        newneu = []
#        laydeu = []
#        for acti in np.linspace(-4, 4, m):
#            if rng.integers(0, 2) == 0:
#                a = 1
#            else:
#                a = 1
#            b = (acti+1e-5)*a
#            #a = -b/(acti+1e-5)
#            newneu.append([a, b])
#            if rng.integers(0, 2) == 0:
#                laydeu.append(1)
#            else:
#                laydeu.append(1)
#
#
#        ly1 = np.array(newneu).T
#        ly1 = ly1# / np.linalg.norm(ly1, axis=0)
#        ly2 = np.ones((len(ly1.T), 1))
#        ly2 = np.array(laydeu)[:, None]
#        # double the number of neurons to allow for negative neurons..
#        # ly1 = np.concatenate((ly1, ly1*1.0), axis=1)
#        # ly2 = np.concatenate((ly2, ly2*(-1.0)), axis=0)
#    elif c == 4: # working but simpler
#        m, d, n = 5, 2, 5
#        Xb = np.linspace(-1, 1, n)[:, None]
#        #X, Y = add_bias(Xb), np.sin(Xb-np.pi/2)+1
#        X, Y = add_bias(Xb), Xb*0.5+0.6
#
#        ly1 = rng.uniform(-1, 1, size=(d, m))
#        ly1 = np.array([[0.5, 0.6], [1, 0.6], [-1, 0.3]]).T
#        ly1 = ly1 / np.linalg.norm(ly1, axis=0)
#        ly2 = np.ones((len(ly1.T), 1))
#        # double the number of neurons to allow for negative neurons..
#        # ly1 = np.concatenate((ly1, ly1*1.0), axis=1)
#        # ly2 = np.concatenate((ly2, ly2*(-1.0)), axis=0)
#    elif c == 44: # working but simpler
#        m, d, n = 5, 2, 5
#        Xb = np.linspace(-1, 1, n)[:, None]
#        #X, Y = add_bias(Xb), np.sin(Xb-np.pi/2)+1
#        X, Y = add_bias(Xb), Xb*0.5+0.3
#
#        ly1 = rng.uniform(-1, 1, size=(d, m))
#        ly1 = np.array([[1, 1], [1, 0.6], [-1, 0.3]]).T
#        m = len(ly1.T)
#        ly1 = ly1 / np.linalg.norm(ly1, axis=0)
#        scalars = 3
#        left, right = 0.01, 1
#        scales = np.array([np.linspace([left]*m, [right]*m, scalars).T, np.linspace([left]*m, [right]*m, scalars).T])
#        ly1 = (scales.T * ly1.T).reshape((m*scalars, d)).T
#        ly2 = np.ones((len(ly1.T), 1))*scaling
#        # double the number of neurons to allow for negative neurons..
#        ly1 = np.concatenate((ly1, ly1*1.0), axis=1)
#        ly2 = np.concatenate((ly2, ly2*(-1.0)), axis=0)
#    elif c== 3: # old
#        m, d, n = 10, 2, 20
#        Xb = np.linspace(-1, 1, n)[:, None]
#        #X, Y = add_bias(Xb), np.sin(10*Xb)*Xb*0.3
#        Xb = np.array([0.2, 0.4, 0.6])[:, None]
#        X, Y = add_bias(Xb), np.array([0.5, 0.6, 0.7])[:, None]
#        X, Y = add_bias(Xb), Xb*1
#
#        ly1 = rng.uniform(-scaling, scaling, size=(d, m))
#        ly2 = rng.uniform(-scaling, scaling, size=(m, 1))
#        ly2 = rng.uniform(0, scaling, size=(m, 1))
#        ly1 = np.array([[2, 0.5], [0.002, 0.0005], [0.002, 0.0005]]).T*scaling
#        scales = np.linspace(0.01, 5, 5)
#        ly2 = np.ones((len(ly1.T), 1))*scaling

import numpy as np

#import torch

from utils import *

def getInput2(args):
    seed = args.seed
    #torch.use_deterministic_algorithms(True)
    #gpudevice = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    gpudevice = "cpu"
    if gpudevice != "cpu":
        print(f"Found a gpu: {gpudevice}..")
    device = "cpu"

    lr, beta = args.lr, 0
    scaling = args.scaleinit
    Nstep = args.steps

    rng = np.random.default_rng(seed) # do not use np.random, see https://numpy.org/doc/stable/reference/random/generator.html#distributions

    c = 45
    if c == 1: # old
        m, d, n = 10, 2, 20
        #X, Y = add_bias(Xb), np.sin(10*Xb)*Xb*0.3
        Xb = np.array([0.2, 0.4, 0.6])[:, None]
        X, Y = add_bias(Xb), np.array([0.3, 0.6, 0.7])[:, None]
        Xb = np.linspace(-1, 1, n)[:, None]
        #X, Y = add_bias(Xb), np.sin(Xb-np.pi/2)+1
        X, Y = add_bias(Xb), np.sin(Xb*4-np.pi/2)

        ly1 = rng.uniform(-scaling, scaling, size=(d, m))
        ly2 = rng.uniform(-scaling, scaling, size=(m, 1))
        ly2 = rng.uniform(0, scaling, size=(m, 1))
        ly1 = np.array([[2, 0.5], [1, 0.5], [-1.2, 0.5], [1, 0.1]]).T*scaling
        ly1 = np.array([[1, 0.01]]).T*scaling
    elif c== 2: # working
        m, d, n = 100, 2, 20
        Xb = np.linspace(-1, 1, n)[:, None]
        #X, Y = add_bias(Xb), np.sin(Xb-np.pi/2)+1
        X, Y = add_bias(Xb), np.sin(Xb*4-np.pi/2)

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
    elif c == 45: # linear data, neurons uniform by activation
        m, d, n = 100, 2, 5
        Xb = np.linspace(2, 4, n)[:, None]
        #X, Y = add_bias(Xb), np.sin(Xb-np.pi/2)+1
        X, Y = add_bias(Xb), Xb*0.1+0.1

        newneu = []
        laydeu = []
        for acti in np.linspace(-4, 4, m):
            if rng.integers(0, 2) == 0:
                a = 1
            else:
                a = 1
            b = (acti+1e-5)*a
            #a = -b/(acti+1e-5)
            newneu.append([a, b])
            if rng.integers(0, 2) == 0:
                laydeu.append(1)
            else:
                laydeu.append(1)


        ly1 = np.array(newneu).T
        ly1 = ly1# / np.linalg.norm(ly1, axis=0)
        ly2 = np.ones((len(ly1.T), 1))
        ly2 = np.array(laydeu)[:, None]
        # double the number of neurons to allow for negative neurons..
        # ly1 = np.concatenate((ly1, ly1*1.0), axis=1)
        # ly2 = np.concatenate((ly2, ly2*(-1.0)), axis=0)
    elif c == 4: # working but simpler
        m, d, n = 5, 2, 5
        Xb = np.linspace(-1, 1, n)[:, None]
        #X, Y = add_bias(Xb), np.sin(Xb-np.pi/2)+1
        X, Y = add_bias(Xb), Xb*0.5+0.6

        ly1 = rng.uniform(-1, 1, size=(d, m))
        ly1 = np.array([[0.5, 0.6], [1, 0.6], [-1, 0.3]]).T
        ly1 = ly1 / np.linalg.norm(ly1, axis=0)
        ly2 = np.ones((len(ly1.T), 1))
        # double the number of neurons to allow for negative neurons..
        # ly1 = np.concatenate((ly1, ly1*1.0), axis=1)
        # ly2 = np.concatenate((ly2, ly2*(-1.0)), axis=0)
    elif c == 44: # working but simpler
        m, d, n = 5, 2, 5
        Xb = np.linspace(-1, 1, n)[:, None]
        #X, Y = add_bias(Xb), np.sin(Xb-np.pi/2)+1
        X, Y = add_bias(Xb), Xb*0.5+0.3

        ly1 = rng.uniform(-1, 1, size=(d, m))
        ly1 = np.array([[1, 1], [1, 0.6], [-1, 0.3]]).T
        m = len(ly1.T)
        ly1 = ly1 / np.linalg.norm(ly1, axis=0)
        scalars = 3
        left, right = 0.01, 1
        scales = np.array([np.linspace([left]*m, [right]*m, scalars).T, np.linspace([left]*m, [right]*m, scalars).T])
        ly1 = (scales.T * ly1.T).reshape((m*scalars, d)).T
        ly2 = np.ones((len(ly1.T), 1))*scaling
        # double the number of neurons to allow for negative neurons..
        ly1 = np.concatenate((ly1, ly1*1.0), axis=1)
        ly2 = np.concatenate((ly2, ly2*(-1.0)), axis=0)
    elif c== 3: # old
        m, d, n = 10, 2, 20
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
        from torch_descent import torch_descent
        opti = torch_descent(device=device, algo="gd")
        opti.load(X, Y, ly1, ly2, lr, beta)
    elif args.algo == "jko":
        from jko_descent import jko_descent
        opti = jko_descent(interiter=args.jkosteps, gamma=args.jkogamma, tau=args.jkotau, verb=args.verbose, proxf=args.proxf, adamlr=args.adamlr)
        opti.load(X, Y, ly1, ly2, lr, beta)
    elif args.algo == "jkocvx":
        from jko_cvxpy import jko_cvxpy
        opti = jko_cvxpy(interiter=args.jkosteps, gamma=args.jkogamma, tau=args.jkotau, verb=args.verbose, proxf=args.proxf, adamlr=args.adamlr)
        opti.load(X, Y, ly1, ly2, lr, beta)

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

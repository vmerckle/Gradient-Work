import numpy as np

#import torch
#torch.use_deterministic_algorithms(True)
#gpudevice = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

from utils import *


def Config2DNew_grid_wasser(args):
    seed = 4
    gpudevice = "cpu"
    device = "cpu"
    rng = np.random.default_rng(seed) # do not use np.random, see https://numpy.org/doc/stable/reference/random/generator.html#distributions

    algo = "GD"
    algo = "JKO"
    proxf = "scipy"
    proxf = "cvxpy"
    algo, proxf = "wasser", "no"
    wasseriter = 10
    wassertau = 1e4
    num_projections = 1000
    gd_lr = 1e-4
    beta = 0
    jko_inter_maxstep = 100
    jko_tol = 1e-6
    cvxpy_tol = 1e-6
    cvxpy_maxit = 100
    coef_KLDIV = 0.01
    gamma = 0.01
    tau = coef_KLDIV*gamma
    include_negative_neurons = False

    m, d, n = 11, 2, 5
    Xb = np.linspace(-0.5, 0.5, n)[:, None]
    #X, Y = add_bias(Xb), np.sin(Xb-np.pi/2)+1
    X, Y = add_bias(Xb), Xb*0.1+0.1


    t = np.linspace(-0.0001, 0.0001, m)
    Xm, Ym = np.meshgrid(t, t)
    # Transform X and Y into 2x(m^2) matrices
    ly1 = np.vstack((Xm.flatten(), Ym.flatten()))
    m = ly1.shape[1]
    ly2 = np.ones((m, 1))

    if include_negative_neurons:
        raise Exception("Not implemented")
    # double the number of neurons to allow for negative neurons..
    # ly1 = np.concatenate((ly1, ly1*1.0), axis=1)
    # ly2 = np.concatenate((ly2, ly2*(-1.0)), axis=0)

    print(f"Running {algo} (with prox={proxf}) on {m} 2D neurons, startsum = {np.sum(ly2):.1f}")

    if algo == "GD":
        from torch_descent import torch_descent
        opti = torch_descent(device=device, algo="gd")
        opti.load(X, Y, ly1, ly2, beta)
    elif algo == "JKO":
        if proxf == "cvxpy":
            from jko_cvxpy import jko_cvxpy
            opti = jko_cvxpy(interiter=jko_inter_maxstep, gamma=gamma, tau=tau, tol=jko_tol, verb=args.verbose)
            opti.load(X, Y, ly1, ly2, beta)
        elif proxf == "scipy":
            from jko_descent import jko_descent
            opti = jko_descent(interiter=jko_inter_maxstep, gamma=gamma, tau=tau, tol=jko_tol, verb=args.verbose)
            opti.load(X, Y, ly1, ly2, beta)
        else:
            raise Exception("config bad proxf choice")
    elif algo == "wasser":
            from wasserstein import wasser
            opti = wasser(wasseriter=wasseriter, tau=wassertau, num_projections=num_projections, verb=args.verbose, adamlr=gd_lr)
            opti.load(X, Y, ly1, ly2, beta)
    else:
        raise Exception("config bad algo choice")

    opti.load(X, Y, ly1, ly2, beta)

    return {"seed": seed,
            "gpudevice": gpudevice,
            "device": device,
            "m": m,
            "d": d,
            "n": n,
            "gd_lr": gd_lr,
            "beta": beta,
            "rng": rng,
            "Xb": Xb,
            "X": X,
            "Y": Y,
            "ly1": ly1,
            "ly2": ly2,
            "opti": opti,
            "steps": args.steps
            }

def Config2DNew_grid(args):
    seed = 4
    gpudevice = "cpu"
    device = "cpu"
    rng = np.random.default_rng(seed) # do not use np.random, see https://numpy.org/doc/stable/reference/random/generator.html#distributions

    algo = "GD"
    algo = "JKO"
    proxf = "scipy"
    proxf = "cvxpy"
    algo, proxf = "wasser", "no"
    wasseriter = 1
    wassertau = 1e3
    gd_lr = 1e-3
    beta = 0
    jko_inter_maxstep = 100
    jko_tol = 1e-6
    cvxpy_tol = 1e-6
    cvxpy_maxit = 100
    coef_KLDIV = 0.01
    gamma = 0.01
    tau = coef_KLDIV*gamma
    include_negative_neurons = False

    m, d, n = 10, 2, 5
    Xb = np.linspace(-0.5, 0.5, n)[:, None]
    #X, Y = add_bias(Xb), np.sin(Xb-np.pi/2)+1
    X, Y = add_bias(Xb), Xb*0.1+0.1


    t = np.linspace(-1, 1, m)
    Xm, Ym = np.meshgrid(t, t)
    # Transform X and Y into 2x(m^2) matrices
    ly1 = np.vstack((Xm.flatten(), Ym.flatten()))
    m = ly1.shape[1]
    ly2 = np.ones((m, 1))/m

    if include_negative_neurons:
        raise Exception("Not implemented")
    # double the number of neurons to allow for negative neurons..
    # ly1 = np.concatenate((ly1, ly1*1.0), axis=1)
    # ly2 = np.concatenate((ly2, ly2*(-1.0)), axis=0)

    print(f"Running {algo} (with prox={proxf}) on {m} 2D neurons, startsum = {np.sum(ly2):.1f}")

    if algo == "GD":
        from torch_descent import torch_descent
        opti = torch_descent(device=device, algo="gd")
        opti.load(X, Y, ly1, ly2, beta)
    elif algo == "JKO":
        if proxf == "cvxpy":
            from jko_cvxpy import jko_cvxpy
            opti = jko_cvxpy(interiter=jko_inter_maxstep, gamma=gamma, tau=tau, tol=jko_tol, verb=args.verbose)
            opti.load(X, Y, ly1, ly2, beta)
        elif proxf == "scipy":
            from jko_descent import jko_descent
            opti = jko_descent(interiter=jko_inter_maxstep, gamma=gamma, tau=tau, tol=jko_tol, verb=args.verbose)
            opti.load(X, Y, ly1, ly2, beta)
        else:
            raise Exception("config bad proxf choice")
    elif algo == "wasser":
            from wasserstein import wasser
            opti = wasser(wasseriter=wasseriter, tau=wassertau, verb=args.verbose, adamlr=gd_lr)
            opti.load(X, Y, ly1, ly2, beta)
    else:
        raise Exception("config bad algo choice")

    opti.load(X, Y, ly1, ly2, beta)

    return {"seed": seed,
            "gpudevice": gpudevice,
            "device": device,
            "m": m,
            "d": d,
            "n": n,
            "gd_lr": gd_lr,
            "beta": beta,
            "rng": rng,
            "Xb": Xb,
            "X": X,
            "Y": Y,
            "ly1": ly1,
            "ly2": ly2,
            "opti": opti,
            "steps": args.steps
            }


def Config2DNew(args):
    seed = 4
    gpudevice = "cpu"
    device = "cpu"
    rng = np.random.default_rng(seed) # do not use np.random, see https://numpy.org/doc/stable/reference/random/generator.html#distributions

    algo = "GD"
    algo = "JKO"
    proxf = "scipy"
    proxf = "cvxpy"
    gd_lr = 1e-3
    beta = 0
    jko_inter_maxstep = 10
    coef_KLDIV = 1.0
    gamma = 0.001
    tau = coef_KLDIV*gamma
    include_negative_neurons = False

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

    if algo == "GD":
        from torch_descent import torch_descent
        opti = torch_descent(device=device, algo="gd")
        opti.load(X, Y, ly1, ly2, beta)
    elif algo == "JKO":
        if proxf == "cvxpy":
            from jko_cvxpy import jko_cvxpy
            opti = jko_cvxpy(interiter=jko_inter_maxstep, gamma=gamma, tau=tau, tol=jko_tol, verb=args.verbose)
            opti.load(X, Y, ly1, ly2, beta)
        elif proxf == "scipy":
            from jko_descent import jko_descent
            opti = jko_descent(interiter=jko_inter_maxstep, gamma=gamma, tau=tau, tol=jko_tol, verb=args.verbose)
            opti.load(X, Y, ly1, ly2, beta)
        else:
            raise Exception("config bad proxf choice")
    elif algo == "wasser":
            from wasserstein import wasser
            opti = wasser(wasseriter=wasseriter, tau=wassertau, verb=args.verbose, adamlr=gd_lr)
            opti.load(X, Y, ly1, ly2, beta)
    else:
        raise Exception("config bad algo choice")

    opti.load(X, Y, ly1, ly2, beta)

    return {"seed": seed,
            "gpudevice": gpudevice,
            "device": device,
            "m": m,
            "d": d,
            "n": n,
            "gd_lr": gd_lr,
            "beta": beta,
            "rng": rng,
            "Xb": Xb,
            "X": X,
            "Y": Y,
            "ly1": ly1,
            "ly2": ly2,
            "opti": opti,
            "steps": args.steps
            }

def Config1DNew(args):
    seed = 4
    gpudevice = "cpu"
    device = "cpu"
    rng = np.random.default_rng(seed) # do not use np.random, see https://numpy.org/doc/stable/reference/random/generator.html#distributions

    algo = "GD"
    algo = "JKO"
    proxf = "scipy"
    proxf = "cvxpy"
    algo, proxf = "wasser", "no"
    wasseriter = 1
    wassertau = 1e3
    gd_lr = 1e-5
    beta = 1e-3
    jko_inter_maxstep = 300
    jko_tol = 1e-6 # 1e-6 to 1e-7
    coef_KLDIV = 0.1
    gamma = 0.1
    tau = coef_KLDIV*gamma
    scaling = 1

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

    print(f"Running {algo} (with prox={proxf}) on {m} 1D neurons, startsum = {np.sum(ly2):.1f}")

    if algo == "GD":
        from torch_descent import torch_descent
        opti = torch_descent(device=device, algo="gd")
        opti.load(X, Y, ly1, ly2, beta)
    elif algo == "JKO":
        if proxf == "cvxpy":
            from jko_cvxpy import jko_cvxpy
            opti = jko_cvxpy(interiter=jko_inter_maxstep, gamma=gamma, tau=tau, tol=jko_tol, verb=args.verbose)
            opti.load(X, Y, ly1, ly2, beta)
        elif proxf == "scipy":
            from jko_descent import jko_descent
            opti = jko_descent(interiter=jko_inter_maxstep, gamma=gamma, tau=tau, tol=jko_tol, verb=args.verbose)
            opti.load(X, Y, ly1, ly2, beta)
        else:
            raise Exception("config bad proxf choice")
    elif algo == "wasser":
            from wasserstein import wasser
            opti = wasser(wasseriter=wasseriter, tau=wassertau, verb=args.verbose, adamlr=gd_lr)
            opti.load(X, Y, ly1, ly2, beta)
    else:
        raise Exception("config bad algo choice")


    return {"seed": seed,
            "gpudevice": gpudevice,
            "device": device,
            "m": m,
            "d": d,
            "n": n,
            "gd_lr": gd_lr,
            "beta": beta,
            "rng": rng,
            "Xb": Xb,
            "X": X,
            "Y": Y,
            "ly1": ly1,
            "ly2": ly2,
            "opti": opti,
            "steps": args.steps
            }

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

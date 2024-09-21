# current models expects data with a bias column

import numpy as np

from utils import *

def genData(D):
    choice = {"linear2d": linear2d,
              "rnglinear": rnglinear,
              "sinus": sinus2d,
              "random": rngrng,
              "mnist": mnist
              }
    t = D["data"]
    if t not in choice:
        raise Exception(f"{t} is not in {choice.keys()}: wrong data type in config")

    return choice[t](D["dataD"])

def sinus2d(D): #only d=2
    n = D["n"]
    Xb = np.linspace(-0.5, 0.5, n)[:, None]
    X, Y = add_bias(Xb), np.sin(10*Xb-np.pi/2)+0.2*Xb - 0.5
    #X, Y = add_bias(Xb), Xb*0.1+0.1
    return X, Y, Xb

def linear2d(D): #d=2
    n = D["n"]
    a, b = 0.2, 0.2
    Xb = np.linspace(-0.5, 0.5, n)[:, None]
    X, Y = add_bias(Xb), Xb*a+b
    return X, Y, Xb

def rnglinear(D):
    rng = np.random.default_rng(D["seed"])
    n, d = D["n"], D["d"]
    sampling, eps = D["sampling"], D["eps"]

    if sampling == "uniform":
        Xb = rng.uniform(-0.5, 0.5, (n, d-1))
    elif sampling == "normal":
        Xb = rng.standard_normal((n, d-1))

    b = rng.uniform(-1, 1, d-1)
    noise = rng.uniform(-1, 1, n)*eps
    return add_bias(Xb), (Xb.dot(b) + noise)[:, None], Xb

def rngrng(D):
    rng = np.random.default_rng(D["seed"])
    n, d = D["n"], D["d"]

    if D["sampling"] == "uniform":
        Xb = rng.uniform(-0.5, 0.5, (n, d-1))
        Y = rng.uniform(-2, 2, (n, 1))
    elif D["sampling"] == "normal":
        Xb = rng.standard_normal((n, d-1))
        Y = rng.normal(0.25, 1, (n, 1))
        Y += rng.normal(-0.25, 1, (n, 1)) # (bad) attempt at creating something harder
        Y = rng.uniform(-0.5, 0.5, (n, 1))

    return add_bias(Xb), Y, Xb

def mnist(D):
    # if not yet done, downloads MNIST using pytorch and save numpy version to file.
    pth = "dataset/mnist_test.pkl"
    if D["train"]:
        pth = "dataset/mnist_train.pkl"

    if os.path.exists(pth):
        with open(pth, "rb") as f:
            return pickle.load(f)

    import torch
    import torchvision
    print("load mnist for the first time, train={D['train']}")
    if not os.path.exists("dataset"):
        os.mkdir(f"dataset")
    
    # no transform
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    if D["train"]:
        dataset = torchvision.datasets.MNIST('dataset/', train=True, download=True, transform=image_transform)
    else:
        dataset = torchvision.datasets.MNIST('dataset/', train=False, download=True, transform=image_transform)

    for x,y in dataset:
        Y.append(y)
        X.append(x)

    Y = torch.tensor(Y)
    X = torch.cat(X) # only works because one item is 1,28,28 and this gets rid of the dimension 1,
    Xb = X.view(-1, 784) # no convolution: flatten the data
    Y = Y.to(torch.float32)[:, None] # convert classification labels to floats
    Xb, Y = Xb.numpy(), Y.numpy()
    X = add_bias(Xb)

    with open(pth, "wb") as f:
        pickle.dump((X, Y, Xb), f)
    return X, Y, Xb


import numpy as np
import argparse
import sys
from scipy.optimize import minimize
from scipy.special import kl_div
from scipy.spatial import distance

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from utils import *
import torch

N = 30
tau = 0.1
gamma = 0.001
tol = 1e-6
niter = 100
nsteps = 50
r = 0.23

t = np.linspace(0, 1, N)
Y, X = np.meshgrid(t, t)

# Transform X and Y into 2x900 matrices
cm = np.vstack((X.flatten(), Y.flatten()))
mask = (X - 0.5)**2 + (Y - 0.45)**2 >= r**2

dist = distance.pdist(cm.T, metric="sqeuclidean") #sq-uared
d = distance.squareform(dist) # get NxN matrix of all distance diffs
Gibbs = np.exp(-d/gamma)
Gibbs = Gibbs * mask.flatten()

def K(p):
    res = np.dot(Gibbs,p.flatten())
    return res.reshape(N, N)


gaussian = lambda m, s: np.exp(-((X - m[0])**2 + (Y - m[1])**2) / (2 * s**2))
mask = np.double((X - 0.5)**2 + (Y - 0.45)**2 >= r**2)
p0 = gaussian([0.5, 0.9], 0.14)
p0 = p0 * mask

normalize = lambda x: x / np.sum(x)
doclamp = 0.7
f = lambda u: normalize(np.minimum(u, np.max(u) * doclamp))
p0 = f(p0 * mask + 1e-10)

kappa = np.max(p0)
w = 0.5 * Y
proxf = lambda p, sigma: np.minimum(p * np.exp(-sigma * w), kappa)


# Helper function for norm
mynorm = lambda x: np.linalg.norm(x.flatten())

q = p0
p = p0
p_list = [p]
for it in range(nsteps - 1):
    q = p

    Constr = [[], []]
    b = np.ones_like(w)
    kb = K(b)
    for i in range(niter):
        p = proxf(kb, tau / gamma)
        a = p / kb
        ka = K(a)
        Constr[1].append(mynorm(b * ka - q) / mynorm(q))
        b = q / ka
        kb = K(b)
        Constr[0].append(mynorm(a * kb - p) / mynorm(q))
        if Constr[0][-1] > 1000:
            print("abort")
            sys.exit(0)
        if Constr[0][-1] < tol and Constr[1][-1] < tol:
            break
    p_list.append(p)


fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    ax.imshow(p_list[frame], cmap='gray')
    ax.set_title(f'Frame {frame}/{len(p_list)}')

ani = FuncAnimation(fig, update, frames=len(p_list), interval=100)
ani.save("outputs/disk_py.gif", writer=animation.FFMpegWriter())#, bitrate=1800)
plt.show()

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

# standalone file to reproduce the disk experience in Gpeyre's paper

# tua=0.1, gamma = 0.001 is nice and need 50nsteps to cvg.
N = 30
tau = 0.003
gamma = 0.1
tau = 0.1
gamma = 0.003

tau = 0.8;
gamma = 0.008;
tol = 1e-6
niter = 100
nsteps = 10
r = 0.23

t = np.linspace(0, 1, N)
Y, X = np.meshgrid(t, t)

# Transform X and Y into 2x900 matrices
cm = np.vstack((X.flatten(), Y.flatten()))
mask = (X - 0.5)**2 + (Y - 0.45)**2 >= r**2

dist = distance.pdist(cm.T, metric="sqeuclidean") #sq-uared
d = distance.squareform(dist) # get NxN matrix of all distance diffs
Gibbs = np.exp(-d/gamma)
#print(Gibbs.shape, mask.flatten().shape)
#Gibbs = Gibbs * mask.flatten()
Gibbs = (Gibbs.T * mask.flatten()).T
print("almost zero: in mask", np.sum(mask==0))
print("almost zero in gibbs:", np.sum(Gibbs==0))

def K(p):

    res = np.dot(Gibbs,p.flatten())
    return res.reshape(N, N)

print([f"{x:.3f}" for x in Gibbs[15][0:10]])

gaussian = lambda m, s: np.exp(-((X - m[0])**2 + (Y - m[1])**2) / (2 * s**2))
mask = np.double((X - 0.5)**2 + (Y - 0.45)**2 >= r**2)
p0 = gaussian([0.5, 0.9], 0.14)
p0 = p0 * mask

normalize = lambda x: x / np.sum(x)
doclamp = 0.7
f = lambda u: normalize(np.minimum(u, np.max(u) * doclamp))
p0 = f(p0 * mask + 1e-100)

kappa = np.max(p0)
w = 0.5 * Y
proxf = lambda p, sigma: np.minimum(p * np.exp(-sigma * w), kappa)
proxf_force = lambda p, sigma: np.minimum(p * np.exp(-sigma * w), kappa) * mask

realf = lambda p: np.dot(w.flatten(), p)
def proxfr(q, sigma):

    step = sigma
    #print(step)
    def objective(p):
        # annoying facts: p=(N,) q=(N, 1)
        return realf(p.flatten())+ kl_div(p.flatten(),q.flatten()).sum()*step

    bounds = [(0, None) for _ in q.flatten()] # positivity
    res = minimize(objective , q.flatten(), bounds=bounds)
    return res.x.reshape((30,30))


# Helper function for norm
mynorm = lambda x: np.linalg.norm(x.flatten())

q = p0
p = p0
p_list = [p0]
for it in range(nsteps - 1):
    q = p
    print("almost zero:", np.sum(p<=1e-5))

    Constr = [[], []]
    b = np.ones_like(w)
    kb = K(b)
    for i in range(niter):
        p = proxf(kb, tau / gamma)
        #print("almost zero:", np.sum(p<=1e-5))
        print(it, i, f"psum: {p.sum():.3f}", f"loss: {realf(p.flatten()):.3f}")
        a = p / (kb+1e-10)
        ka = K(a)
        Constr[1].append(mynorm(b * ka - q) / mynorm(q))
        b = q / (ka+1e-10)
        kb = K(b)
        Constr[0].append(mynorm(a * kb - p) / mynorm(q))
        if Constr[0][-1] > 1000:
            print("abort")
            #sys.exit(0)
        if Constr[0][-1] < tol and Constr[1][-1] < tol:
            break
        #print(np.sum(p))
    p_list.append(p)

fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    ax.imshow(p_list[frame], cmap='gray')
    #ax.imshow(p_list[frame]*(p_list[frame]>1e-3), cmap='gray')
    ax.set_title(f'Frame {frame}/{len(p_list)}')

ani = FuncAnimation(fig, update, frames=len(p_list), interval=100)
ani.save("outputs/disk_py.gif", writer=animation.FFMpegWriter())#, bitrate=1800)
plt.show()

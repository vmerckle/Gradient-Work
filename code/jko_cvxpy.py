import numpy as np
import argparse
import sys
from scipy.spatial import distance
from scipy.special import kl_div
import cvxpy as cp

from numpy.random import default_rng

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import *

class jko_cvxpy:
    def __init__(self, gamma=1, tau=1, interiter=10, verb=False, tol=1e-6, dtype=torch.float32, device="cpu", adamlr=1e-3, proxf="scipy"):
        self.gamma = gamma
        self.tau = tau
        self.tol = tol
        self.verb = verb
        self.interiter = interiter
        self.dtype = dtype  # float16 does a runtime error in pred
        self.device = device
        self.adamlr = adamlr
        self.mynorm = lambda x: np.linalg.norm(x.flatten())
        if proxf != "cvxpy":
            print(f"{proxf} proxf has been asked for jko_cvxpy, but we will use cvxpy..")

    def load(self, X, Y, ly1, ly2, lr=0, beta=0, grid=None, p=None):
        self.X = X
        self.Y = Y
        self.psigns = np.sign(ly2)
        ly2 = np.abs(ly2)
        self.n, self.d = X.shape
        _, self.m = ly1.shape

        if grid is None or p is None:
            norm = ly2.sum()
            self.normori = norm
            #self.grid = ly1.T*norm
            self.grid = ly1.T#*ly2
            self.p = np.ones_like(ly2)
            #self.p = np.ones_like(ly2)/len(ly2)
            self.p = np.zeros_like(ly2)/len(ly2)
            #self.p = np.zeros_like(ly2)/len(ly2)+1e-8
            
            self.p[25] = 0.2
        else:
            self.normori = 1
            self.grid = grid
            self.p = p

        self.K = getKernel(self.grid, self.gamma) # maybe psigns, maybe not TODO

        xwi = self.X @ ly1
        xwir = (xwi >0) * xwi # n,m

        self.pvar = cp.Variable(self.m)
        self.qparam = cp.Parameter(self.m, pos=True)
        Yhat = 0
        for xwiri, sgn, pi in zip(xwir.T, self.psigns.flatten(), self.pvar):
            Yhat += xwiri * sgn * pi

        Yhat = cp.reshape(Yhat, (self.n, 1))
        self.residuals = cp.quad_over_lin(Yhat-self.Y, self.n)
        self.regularization = cp.sum(cp.kl_div(self.pvar, self.qparam))
        constraints = [self.pvar >= 1e-8]
        obj = cp.Minimize(self.residuals + self.tau/self.gamma * self.regularization)
        self.problem = cp.Problem(obj, constraints)


    def step(self):
        q = self.p
        qnorm = self.mynorm(q)
        a, b = (np.ones_like(self.p) for _ in range(2))

        # this seems to remove a few early useless iterations
        kb = self.K(b)

        #los,kl = self.f(self.p), kl_div(kb.flatten(),self.p.flatten()).sum()*self.tau/self.gamma
        print("current sum:", np.sum(self.p))
        #print(self.p)


        for i in range(self.interiter):
            print(f".", end="", flush=True)

            self.qparam.value = kb.flatten()
            verb = False
            for _ in range(1):
                try:
                    self.problem.solve(verbose=verb)
                    self.p = self.pvar.value[:, None]
                    break
                except cp.error.SolverError as e:
                    #print(e)
                    #verb = True
                    print("X", end="", flush=True)
            if np.any(self.p <= 0):
                print("some are negs..")
            #print(f"error={self.res.value:.9f}, kldiv={self.reg.value:.9f}, kldiv*step = {self.reg.value*self.tau/self.gamma:.4f}")

            a = self.p / kb
            ka = self.K(a)
            ConstrEven = self.mynorm(b * ka - q) / qnorm
            b = q / ka
            kb = self.K(b)
            ConstrOdd = self.mynorm(a * kb - self.p) / qnorm
           
            if ConstrOdd < self.tol and ConstrEven < self.tol:
                print(f"early exit after {i} iterations")
                return

    def params(self, regopti=False, oneway=False):
        #ly1 = self.grid * np.sqrt(self.p)
        #ly2 = np.sqrt(self.p)
        if regopti:
            ly1 = self.grid.T/self.normori # some attempt at not modifying the scale
            ly2 = self.p*self.normori * self.psigns
            return ly1, ly2
        elif oneway:
            ly1 = self.grid.T*self.p.flatten()
            ly2 = np.ones_like(self.p)
            return ly1, ly2
        else:
            ly1 = self.grid.T
            return ly1, self.p * self.psigns

    def gridout(self):
        return self.grid, self.p, self.psigns

    def loss(self):
        return self.f(self.p)

    def f(self, p):
        p = p.flatten()
        activ = (self.X @ self.grid.T) > 0 # (n, N)
        effneurons = self.psigns.flatten() * p * self.grid.T # (d,N) 
        out = activ * (self.X @ effneurons) # (n, d) * (d, N) = (n, N)
        yhat = np.sum(out, axis=1)[:, None] # avoid broadcasting..

        return MSEloss(yhat, self.Y, coeff=len(yhat))

# kernel with only distance between activation points
def getKernel(grid, gamma):
    grid = [[-b/a,1] for (a, b) in grid]
    dist = distance.pdist(grid, metric="sqeuclidean") #sq-uared
    d = distance.squareform(dist) # get NxN matrix of all distance diffs
    Gibbs = np.exp(-d/gamma)
    #Gibbs = np.eye(len(grid))+1e-3 # some very uniform movement
    #Gibbs = np.eye(len(grid)) # don't allow movement... basically
    print("[" ,[f"{x:.3f}" for x in Gibbs[0]])

    def aux(p):  # Gibbs Kernel applied to a vector p 
        return np.dot(Gibbs,p)
    return aux



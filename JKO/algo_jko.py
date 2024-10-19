import numpy as np
import argparse
import sys
from scipy.spatial import distance

from utils import *

class JKO:
    def __init__(self, gamma=1, tau=1, interiter=10, verb=False, tol=1e-6, dtype=np.float32):
        self.gamma = gamma
        self.tau = tau
        self.tol = tol
        self.verb = verb
        self.interiter = interiter
        self.dtype = dtype # float16 does a runtime error in pred
        self.device = device
        self.adamlr = adamlr
        self.mynorm = lambda x: np.linalg.norm(x.flatten())

    def loadgrid(self, X, Y, ly1, ly2, grid, p):
        self.X = X
        self.Y = Y
        self.psigns = np.sign(ly2+1e-30) # sign(0) -> 0.. now 1.
        ly2 = np.abs(ly2)
        self.d, self.m  = ly1.shape
        self.n,_  = X.shape

        if grid is None or p is None:
            self.normori = ly2.sum()
            self.grid = ly1.T#*ly2
            self.p = ly2 # important 1.0 otherwise it's an int array
        else:
            self.normori = 1
            self.grid = grid
            self.p = p

    def load(self, X, Y, ly1, ly2, beta=0, grid=None, p=None):
        self.beta = beta
        self.loadgrid(X, Y, ly1, ly2, grid, p)

        self.K = getKernel(self.grid, self.gamma) # maybe psigns, maybe not TODO

    def step(self):
        q = self.p
        qnorm = self.mynorm(q)
        startsum = np.sum(self.p)
        a, b = (np.ones_like(self.p) for _ in range(2))

        stop_proto = 1e-3 # don't accelerate that much in the end
        stop_proto = None

        # this seems to remove a few early useless iterations
        #a = self.p
        #ka = self.K(a)
        #b = q / ka
        kb = self.K(b)

        if 0:
            print("a", np.max(a), np.min(a), np.any(np.isnan(a)))
            print("b = q / ka", np.max(b), np.min(b), np.any(np.isnan(b)))
            print(np.any(np.isnan(kb)))
        eps = 1e-8
        for i in range(self.interiter):
            self.jkotot += 1
            #print(f".", end="", flush=True)
            proxres = self.proxf(kb)
            if proxres is None:
                print("solver failed")
                sys.exit(0)

            if np.any(proxres < 0):
                print(f"{np.sum(proxres < 0)/len(proxres)*100:.2f}% of entries were negatives")
            self.p = np.maximum(proxres, eps)
            psum = np.sum(self.p)
            if stop_proto is not None and abs(startsum-psum) < stop_proto:
                print(f"psum={psum:.4f} exit after {i+1}/{self.interiter} iterations")
                break
            if (psum < 0.9 or psum > 1.1) and i > 10 and 0:
                print(f"{i+1}/{self.interiter}: psum={psum:.2f}")

            # print("selfp", ", ".join([f"{x:.6f}" for x in self.p.flatten()]), f"sum(p)={psum:.3f}")
            a = self.p / kb
            ka = np.maximum(self.K(a), eps)
            ConstrEven = self.mynorm(b * ka - q) / qnorm
            b = q / ka
            kb = np.maximum(self.K(b), eps)
            ConstrOdd = self.mynorm(a * kb - self.p) / qnorm
           
            if ConstrOdd < self.tol and ConstrEven < self.tol:
                #print(f"early exit after {i} iterations")
                break
        if (psum < 0.9 or psum > 1.1) and 1:
            print(f"jko inter exit after {i+1}/{self.interiter} iterations, psum=", psum)
        print(psum)
        #self.p = self.p/np.sum(self.p)

    def params(self, regopti=False, oneway=False):
        #ly1 = self.grid * np.sqrt(self.p)
        #ly2 = np.sqrt(self.p)
        if regopti:
            ly1 = self.grid.T/self.normori # some attempt at not modifying the scale
            ly2 = self.p*self.normori * self.psigns
        elif oneway:
            ly1 = self.grid.T*self.p.flatten()
            ly2 = np.ones_like(self.p)
        else:
            #print(self.p*self.psigns)
            ly1 = self.grid.T
            ly2 = self.p * self.psigns
        
        return {"ly1": ly1,
                "ly2": ly2} # otherwise it's just a pointer...

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
    
    def proxf(self):
        raise Exception("To implement")

# kernel 
def getKernel(grid, gamma):
    #grid = [[-b/a,1] for (a, b) in grid] # with only distance between activation points
    dist = distance.pdist(grid, metric="sqeuclidean") #sq-uared
    d = distance.squareform(dist) # get NxN matrix of all distance diffs
    # Gibbs = np.exp(-d/gamma)  
    # Gibbs = Gibbs * (Gibbs > 1e-6) # this doesn't not get rid of cvg problems.
    Gibbs = np.maximum(np.exp(-d/gamma), 1e-6) # lower than 1e-7 will make solvers not converge
    #Gibbs = Gibbs/np.median(Gibbs) # not like this
    #Gibbs = np.eye(len(grid))+1e-3 # some very uniform movement
    #Gibbs = np.eye(len(grid)) # don't allow movement... basically
    print("GibbsK", ", ".join([f"{x:.1E}" for x in Gibbs[0][0:10]]))

    def aux(p):  # Gibbs Kernel applied to a vector p 
        return np.dot(Gibbs,p)
    return aux

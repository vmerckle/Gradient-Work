import numpy as np
import cvxpy as cp

from scipy.spatial import distance
from utils import *

class jko_cvxpy:
    def __init__(self, gamma=1, tau=1, interiter=10, verb=False, tol=1e-6, dtype=np.float32, device="cpu", adamlr=1e-3, proxf="cvxpy"):
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
            #self.p = np.zeros_like(ly2)
            self.p = np.zeros_like(ly2)*1.0 # important 1.0 otherwise it's an int array
            self.p = self.p + 1e-10 # important 1.0 otherwise it's an int array
            # and if you do intarray[0] = 0.2, it will truncate to 0.
            #ss = [3, 49, 400, 700, 900]
            #for sss in ss:
            #    self.p[sss] = 0.2
            self.p = np.zeros_like(ly2)*1.0 # important 1.0 otherwise it's an int array
            self.p = self.p + 1e-10 # important 1.0 otherwise it's an int array
            self.p[50] = 0.1
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
        #constraints = [self.pvar >= 1e-10]
        constraints = []
        obj = cp.Minimize(self.residuals + self.tau/self.gamma * self.regularization)
        self.problem = cp.Problem(obj, constraints)


    def step(self):
        q = self.p
        qnorm = self.mynorm(q)
        a, b = (np.ones_like(self.p) for _ in range(2))

        kb = self.K(b)

        a = self.p
        ka = self.K(a)
        b = q / ka
        kb = self.K(b)
        # this seems to remove a few early useless iterations
        if 0:
            a = self.p / self.K(b)
            b = q / self.K(a)
            kb = self.K(b)
            print("a", np.max(a), np.min(a), np.any(np.isnan(a)))
            print("b = q / ka", np.max(b), np.min(b), np.any(np.isnan(b)))
            print(np.any(np.isnan(kb)))

        #los,kl = self.f(self.p), kl_div(kb.flatten(),self.p.flatten()).sum()*self.tau/self.gamma
        print("current sum:", np.sum(self.p))
        #print(self.p)

        for i in range(self.interiter):
            print(f".", end="", flush=True)

            if np.any(np.isnan(kb)):
                print("EXIT: kb is", np.sum(np.isnan(kb))/len(kb), "nan")
            if np.any(kb<0):
                print("EXIT: kb is", np.sum(kb<0)/len(kb), "neg")
            self.qparam.value = kb.flatten()
            verb = False 
            for _ in range(1):
                try:
                    #self.problem.solve(verbose=verb, solver=cp.ECOS_BB, max_iters=50)
                    self.problem.solve(verbose=verb, solver=cp.ECOS, max_iters=100)
                    self.p = self.pvar.value[:, None]
                    break
                except cp.error.SolverError as e:
                    #print(e)
                    #verb = True
                    print("X", end="", flush=True)
            if np.any(self.p <= 0):
                print("some are negs..")
            #print(f"error={self.res.value:.9f}, kldiv={self.reg.value:.9f}, kldiv*step = {self.reg.value*self.tau/self.gamma:.4f}")

            psum = np.sum(self.pvar.value)
            print("psum=", psum)
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
    #grid = [[-b/a,1] for (a, b) in grid]
    dist = distance.pdist(grid, metric="sqeuclidean") #sq-uared
    d = distance.squareform(dist) # get NxN matrix of all distance diffs
    Gibbs = np.exp(-d/gamma)
    #Gibbs = np.eye(len(grid))+1e-3 # some very uniform movement
    #Gibbs = np.eye(len(grid)) # don't allow movement... basically
    print([f"{x:.3f}" for x in Gibbs[0][0:10]])

    def aux(p):  # Gibbs Kernel applied to a vector p 
        return np.dot(Gibbs,p)
    return aux



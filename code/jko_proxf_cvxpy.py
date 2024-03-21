import numpy as np
import cvxpy as cp

from scipy.spatial import distance
from utils import *

import jko

class jko_cvxpy(jko.JKO):
    def __init__(self, gamma=1, tau=1, interiter=10, verb=False, tol=1e-6, cvxpy_tol=1e-6, cvxpy_maxit=100, dtype=np.float32):
        self.gamma = gamma
        self.tau = tau
        self.tol = tol
        self.cvxpy_tol = cvxpy_tol
        self.cvxpy_maxit = cvxpy_maxit
        self.verb = verb
        self.interiter = interiter
        self.dtype = dtype  # float16 does a runtime error in pred
        self.mynorm = lambda x: np.linalg.norm(x.flatten())
        self.jkotot = 0

    def load(self, X, Y, ly1, ly2, beta=0, grid=None, p=None):
        self.loadgrid(X, Y, ly1, ly2, grid, p)
        
        self.K = jko.getKernel(self.grid, self.gamma) # maybe psigns, maybe not TODO
        self.beta = beta

        xwi = self.X @ ly1
        xwir = (xwi >0) * xwi # n,m

        self.pvar = cp.Variable(self.m)
        self.qparam = cp.Parameter(self.m, pos=True)
        Yhat = 0
        for xwiri, sgn, pi in zip(xwir.T, self.psigns.flatten(), self.pvar):
            Yhat += xwiri * sgn * pi

        Yhat = cp.reshape(Yhat, (self.n, 1))
        self.residuals = cp.quad_over_lin(Yhat-self.Y, self.n)
        self.kl_regu = cp.sum(cp.kl_div(self.pvar, self.qparam))
        self.weight_regu = cp.sum(self.pvar ** 2)
        #self.l1norm = cp.sum(cp.abs(self.pvar)) # always one ofc..
        constraints = []
        constraints = [self.pvar >= 1e-6]
        obj = cp.Minimize(self.residuals + self.beta * self.weight_regu + self.tau/self.gamma * self.kl_regu)
        self.problem = cp.Problem(obj, constraints)

    def proxf(self, kb):
        self.qparam.value = kb.flatten()
        verb = True
        verb = False
        try:
            #self.problem.solve(verbose=verb, solver=cp.ECOS_BB, max_iters=50)
            # incredible but ECOS might return None when it runs out of iterations
            eps = self.cvxpy_tol # def 1e-8, 1e-5 seems nice with a big maxiters
            epsb = 5e-5 # def 5e-5, only change warning
            epsc = 1e-4 # def 1e-4
            self.problem.solve(verbose=verb, solver=cp.ECOS, max_iters=self.cvxpy_maxit, abstol=eps, reltol=eps, feastol=eps, abstol_inacc=epsb, reltol_inacc=epsb, feastol_inacc=epsc)
            #self.problem.solve(verbose=verb, solver=cp.ECOS, max_iters=100)
        except cp.error.SolverError as e:
                    #'info': {'exitFlag': 0, 'pcost': -1052648.926085804, 'dcost': -1052648.925960286, 'pres': 4.7189265712903e-09, 'dres': 7.554818405757892e-10, 'pinf': 0.0, 'dinf': 0.0, 'pinfres': nan, 'dinfres': 1.116626843860054e-05, 'gap': 1.2996111340072398e-13, 'relgap': 1.234610231200013e-19, 'r0': 1e-08, 'iter': 23, 'mi_iter': -1, 'infostring': 'Optimal solution found', 'timing': {'runtime': 0.000631987, 'tsetup': 3.4505e-05, 'tsolve': 0.000597482}, 'numerr': 0}}
            #print(e)
            # when it really 
            pass
        infos = self.problem.solver_stats.extra_stats["info"]
        msg = infos["infostring"]
        flag = infos["exitFlag"]
        pres = infos["pres"]
        itern = infos["iter"]
        if flag < 0 or 0:
            print(f"{msg}:{flag}: {pres} in {itern} iters")
        #print(f"res:{self.residuals.value:.2E} vs kld:{self.kl_regu.value:.2E}(*{self.tau/self.gamma:.4f}) vs wed:{self.weight_regu.value:.2E}(*{self.beta:.4f})")
        return self.pvar.value[:, None]

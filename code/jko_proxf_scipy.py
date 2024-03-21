import numpy as np
import argparse
import sys
from scipy.optimize import minimize
from scipy.special import kl_div
from scipy.spatial import distance
from utils import *

import jko

"""
with proxf implemented
    - with scipy

results: scipy is slow
"""

class jko_scipy(jko.JKO):
    def __init__(self, gamma=1, tau=1, interiter=10, verb=False, tol=1e-6, dtype=torch.float32, device="cpu", adamlr=1e-3):
        self.gamma = gamma
        self.tau = tau
        self.tol = tol
        self.verb = verb
        self.interiter = interiter
        self.dtype = dtype  # float16 does a runtime error in pred
        self.device = device
        self.adamlr = adamlr
        self.mynorm = lambda x: np.linalg.norm(x.flatten())
        self.jkotot = 0

        self.proxf = self.prox_f_scipy

    def prox_f_scipy(self, q):
        step = self.tau/self.gamma
        #print(step)
        def objective(p):
            # annoying facts: p=(N,) q=(N, 1)
            return self.f(p)+ kl_div(p,q.flatten()).sum()*step
        bounds = [(0, None) for _ in q] # positivity
        res = minimize(objective , q.flatten(), bounds=bounds)
        return res.x.reshape(-1,1)

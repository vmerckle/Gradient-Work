import numpy as np
import cvxpy as cp

from utils import *

class cvx_solve:
    def __init__(self):
        pass

    def load(self, X, Y, ly1, ly2, lr, beta):
        self.lr, self.beta = lr, beta
        self.n, self.d = X.shape
        self.m = ly1.shape[1]
        self.X = X
        self.Y = Y
        self.ly1 = ly1
        self.ly2 = ly2

    def step(self):
        wly1, wly2 = self.ly1 * np.abs(np.squeeze(self.ly2)), np.sign(self.ly2)

        def cvxpb(G, signs, X, Y):
            n, d = X.shape
            m = len(G)
            bet = cp.Parameter(nonneg=True)
            W = cp.Variable((m, d))
            Yhat = 0
            for i, (si, sgn) in enumerate(zip(G, signs)):
                Yhat += sgn*np.diag(si) @ X @ W[i]
            Yhat = cp.reshape(Yhat, (n, 1))
            constraints = []
            for i, si in enumerate(G):
                constraints.append((2*np.diag(si) - np.eye(n)) @ X @ W[i] >= 0)

            residuals = cp.quad_over_lin(Yhat-Y, n)
            regularization = cp.sum(cp.norm(W, 2, axis=1))
            obj = cp.Minimize(residuals + bet*regularization)
            return cp.Problem(obj, constraints), (W, bet)

        G = (self.X @ wly1 > 0).T
        problem, (W, bet) = cvxpb(G, wly2, self.X, self.Y)
        bet.value = self.beta
        problem.solve()

        wly1 = W.value.T
        norm = np.linalg.norm(wly1, ord=2, axis=0)
        wly2 = wly2.flatten()*norm
        self.ly1, self.ly2 = wly1/norm, wly2[:, None]

    def params(self):
        return {"ly1": self.ly1,
                "ly2": self.ly2} # otherwise it's just a pointer...

    def grads(self):
        return np.zeros(self.ly1.shape), np.zeros(self.ly2.shape)

    def pred(self, X):
        return np.maximum(X@self.ly1, 0) @ self.ly2

    def loss(self, beta = None):
        if beta is None:
            beta = self.beta
        return np.square(self.pred(self.X)-self.Y).sum()/self.n + beta*np.sum((np.linalg.norm(self.ly1, ord=2, axis=0)+ np.linalg.norm(self.ly2, ord=2, axis=0)))

import numpy as np
from scipy.optimize import minimize
from scipy.special import kl_div

def prox_f_scipy(q,step,grid,X,y):
    """
    Solving for F with CVXPY?
    """

    def objective(p):
        return f(p,grid,X,y) + step*dkl(p,q)

    bounds = [(0, None) for _ in q]
    res = minimize(objective , q, bounds=bounds)

    return res.x.reshape(-1,1)


def f(p,grid,X,y):

    m,d = X.shape
    s = grid.shape
    
    n = p.size

    l = 0.0

    for i in range(m): # data points
        out = 0.0
        for j in range(n):  # neurons
            contrib = 0.0
            for k in range(d): # dimension
                contrib += X[i,k]*grid[j,k]
            
            if contrib < 0.0: # ReLU gate
                contrib = 0.0

            out += contrib*p[j]  

        l += loss(out,y[i])

    return l


def loss(x,y):
    return np.linalg.norm(x-y)**2

def dkl(p,q):
    n = p.size 

    d = 0.0
    for i in range(n):
        d += kl_div(p[i],q[i][0])

    return d

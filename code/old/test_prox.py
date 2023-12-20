import numpy as np


from base.prox import prox_f_scipy



d = 1

N     = 5 # Number of points
grid  = np.linspace(0,4,N).reshape(-1,1)

q     = np.ones_like(grid)/N # Initial distribution
#q     = np.zeros_like(grid); q[0] = 1.0


## Data
m = 3
X = np.random.randn(m,1)
y = X*1.5 + 0.01*np.random.randn(m)

step = 1.0 # Prox step

prox_f = lambda q,step : prox_f_scipy(q,step,grid,X,y)


u = prox_f(q,step)

print(u)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from base.jko_step import perform_jko_stepping
from base.prox import prox_f_scipy

gamma = 1# Entropy regularization
tau   = 1# JKO step 

Niter = 5 # Internal iterations of Sinkhorn
Nsteps= 2 # JKO Steps

N     = 5 # Number of points
grid  = np.linspace(-2,2,N).reshape(-1,1)

#left, right, nb = -1.0, 1.0, 3
#x,y = np.meshgrid(np.linspace(left, right, nb), np.linspace(left, right, nb))
#Xgrid = np.array((x,y)).T
#grid = Xgrid.reshape(-1, 2)

diff  = grid - grid.T
C     = diff**2   # transport costs


Gibbs = np.exp(-C/gamma)
print("gibshape", Gibbs.shape)

def K(p):  # Gibbs Kernel applied to a vector p 
    print((np.dot(Gibbs,p)).shape, "K shape")
    return np.dot(Gibbs,p)

q     = np.ones(grid.shape)/N # Initial distribution
#q     = np.zeros_like(grid); q[0] = 1.0


rng = np.random.default_rng(4)
## Data
m = 5
X = rng.normal(size=(m, 1))
y = X*0.4000 #+ 0.01*rng.normal(size=m)


prox_f = lambda q,step : prox_f_scipy(q,step,grid,X,y)


options = dict()
options["niter"] = Niter # Internal iterations of Sinkhorn
options["nsteps"] = Nsteps # JKO Steps
options["verb"] = True



p, Constr, p_list = perform_jko_stepping(K, q, gamma, tau, prox_f, options)

print("p", p)

fig,ax = plt.subplots()
plt.scatter(grid,q)


def updateData(curr):
    #ax.clear()
    print(curr, np.sum(grid*p_list[curr]))
    plt.scatter(grid,p_list[curr])

simulation = animation.FuncAnimation(fig, updateData, interval=100, frames=Nsteps, repeat=False)

plt.show()

print(grid)
print(p_list[-1])
print(grid*p_list[-1])
print(np.sum(grid*p_list[-1]))

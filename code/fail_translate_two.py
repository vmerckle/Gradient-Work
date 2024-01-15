import numpy as np
from scipy.sparse import diags, kron, spdiags, eye
from scipy.linalg import cholesky
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import spsolve
from sksparse.cholmod import cholesky
# Simulation of crowd motion through JKO flow.

# Set the size of the matrix
N = 5

# Normalize function
normalize = lambda x: x / np.sum(x)

# Set figure title
t = np.linspace(0, 1, N)
Y, X = np.meshgrid(t, t)

# Gaussian function
gaussian = lambda m, s: np.exp(-((X - m[0])**2 + (Y - m[1])**2) / (2 * s**2))

# Set the size of the matrix
Ne = N**2

# Initial density
p0 = gaussian([0.5, 0.9], 0.14)

# Mask
r = 0.23
mask = np.double((X - 0.5)**2 + (Y - 0.45)**2 >= r**2)

# Attraction potential
w = 0.5 * Y

# Function to normalize and clamp
f = lambda u: normalize(u)
doclamp = 0.7
f = lambda u: normalize(np.minimum(u, np.max(u) * doclamp))
p0 = f(p0 * mask + 1e-10)

# Compute a geodesic metric and geodesic potential
vmin = 0
M = np.zeros((N, N, 2, 2))
M[:, :, 0, 0] = mask + vmin
M[:, :, 1, 1] = mask + vmin

tau = 10
gamma = 0.2
model = 'crowd'

# Load the proximal map.
# Crowd motion clamping
kappa = np.max(p0)
proxf = lambda p, sigma: np.minimum(p * np.exp(-sigma * w), kappa)

# Gibbs kernel
# Anisotropic metric

# Forward finite difference matrix
e = np.ones(Ne)
D = diags([e, -e], [0, 1], (Ne, Ne))
D = D.tocsc()
D[-1, :] = 0

# Spatial derivatives
Dx = kron(D, eye(Ne))
Dy = kron(eye(Ne), D)
Grad = np.vstack((Dx, Dy))

# Metric along the diagonal
diagH = lambda h: spdiags(h.flatten(), [0], Ne**2, Ne**2)
dH = np.block([[diagH(M[:, :, 0, 1]), diagH(M[:, :, 1, 1])],
               [diagH(M[:, :, 0, 0]), diagH(M[:, :, 1, 0])]])

# Laplacian
Delta = Grad.T @ dH @ Grad
# Perform Cholesky factorization of A
Id = eye(Ne**2)

Delta = Delta[0][0]
print(Delta.shape, type(Delta))
print(Id.shape, type(Id))
A = Id + gamma * Delta
R = cholesky(A)


# Function to perform heat iteration using Cholesky factorization
def heat_iter_chol(R, filtIter, u):
    s = u.shape
    u = u.flatten()
    for i in range(filtIter):
        u = np.linalg.solve(R.T, np.linalg.solve(R, u))
    u = u.reshape(s)
    return u

# Blur function using heat iteration
def blur(R, u, filtIter):
    return heat_iter_chol(R,filtIter, u)

K = lambda x: blur(R, x, 5)

# Perform several iterates
tol = 1e-6
niter = 100
nsteps = 20

# Helper function for norm
mynorm = lambda x: np.linalg.norm(x.flatten())

q = p0
p = p0
p_list = [p]
for it in range(nsteps - 1):
    print(it)
    q = p

    Constr = [[], []]
    b = w * 0 + 1
    for i in range(niter):
        p = proxf(K(b), tau / gamma)
        a = p / K(b)
        Constr[1].append(mynorm(b * K(a) - q) / mynorm(q))
        b = q / K(a)
        Constr[0].append(mynorm(a * K(b) - p) / mynorm(q))
        if Constr[0][-1] < tol and Constr[1][-1] < tol:
            break

    p_list.append(p)

# Sanity checks
total_sum = sum([np.sum(x) for x in p_list])
print(f'Total sum of all matrices: {total_sum}')
last_matrix = p_list[-1]
sum_last_matrix = np.sum(last_matrix)
print(f'Sum of the last matrix: {sum_last_matrix}')
sum_last_matrix = np.sum(Constr[0])
print(f'Sum of the constr1: {sum_last_matrix}')
sum_last_matrix = np.sum(Constr[1])
print(f'Sum of the constr2: {sum_last_matrix}')

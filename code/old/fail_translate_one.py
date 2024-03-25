import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# Simulation parameters
N = 30
normalize = lambda x: x / np.sum(x)
t = np.linspace(0, 1, N)
Y, X = np.meshgrid(t, t)
gaussian = lambda m, s: np.exp(-((X - m[0])**2 + (Y - m[1])**2) / (2 * s**2))
rectangle = lambda c, w: np.double(np.maximum(np.abs(X - c[0]) / w[0], np.abs(Y - c[1]) / w[1]) < 1)
disk = lambda m, s: (X - m[0])**2 + (Y - m[1])**2 <= s**2
normalize = lambda x: x / np.sum(x)

# Initializations
p0 = gaussian([0.5, 0.9], 0.14)
r = 0.23
mask = np.double((X - 0.5)**2 + (Y - 0.45)**2 >= r**2)
target = np.round(N * np.array([0.5, 0.03]))
w = 0.5 * Y
vmin = 0
M = np.zeros((N, N, 2, 2))
M[:, :, 0, 0] = mask + vmin
M[:, :, 1, 1] = mask + vmin

# Functions
def proxf(p, sigma):
    kappa = np.max(p)
    return np.minimum(p * np.exp(-sigma * w), kappa)

def blur(u, t, filt_iter):
    R, I = cholesky_factorization(t)
    s = u.shape
    u = u.flatten()
    u = u[I]
    for _ in range(filt_iter):
        u = spsolve(R, spsolve(R.T, u))
    u[I] = u
    return u.reshape(s)

def K(x, gamma, filt_iter):
    return blur(x, gamma, filt_iter)

def heat_iter_chol(R, I, t, t0, filt_iter, u):
    s = u.shape
    u = u.flatten()
    u = u[I]
    for _ in range(filt_iter):
        u = spsolve(R, spsolve(R.T, u))
    u[I] = u
    return u.reshape(s)

def cholesky_factorization(t0, N):

    CholFactor = t
    Id = sp.eye(N*N, N*N)
    A = Id + CholFactor * Delta
    I = np.argsort(A.diagonal())
    R = sp.linalg.cholesky(A[I, :][:, I].tocsc(), lower=True)
    return R, I

# Simulation parameters
tau = 10
gamma = 0.2
model = 'crowd'
kappa = np.max(p0)
proxf = lambda p, sigma: np.minimum(p * np.exp(-sigma * w), kappa)

# Gibbs kernel
N = t.shape[0]
id = sp.eye(N)
e = np.ones(N)
D = sp.diags([e, -e], [0, 1], (N-1, N), format='csr')
# Add a zero at the end of the diagonal
last_row = sp.csr_matrix((1, N))
D = sp.vstack([D, last_row])

Dx = sp.kron(D, sp.eye(N))
Dy = sp.kron(sp.eye(N), D)
Grad = sp.vstack([Dx, Dy])

# Metric along the diagonal
diagH = lambda h: sp.diags(h.flatten(), 0, (N**2, N**2))
dH = sp.vstack([
    sp.hstack([diagH(np.copy(M[:, :, 1, 1])), diagH(np.copy(M[:, :, 0, 1]))]),
    sp.hstack([diagH(np.copy(M[:, :, 1, 0])), diagH(np.copy(M[:, :, 0, 0]))])
])
Delta = Grad.T @ dH @ Grad

# Load the proximal map
CholFactor = gamma
t0 = CholFactor
R, I = cholesky_factorization(t0)
K = lambda x: blur(x, gamma, filt_iter)
tol = 1e-6
niter = 100
nsteps = 20
mynorm = lambda x: np.linalg.norm(x.flatten())

# Main loop
q = p0
p = p0
p_list = [p]
for it in range(1, nsteps):
    print(it)
    q = p

    # Slim try
    if True:
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
total_sum = np.sum([np.sum(x) for x in p_list])
print(f'Total sum of all matrices: {total_sum}')
last_matrix = p_list[-1]
sum_last_matrix = np.sum(last_matrix)
print(f'Sum of the last matrix: {sum_last_matrix}')
sum_last_matrix = np.sum(Constr[0])
print(f'Sum of Constr[0]: {sum_last_matrix}')
sum_last_matrix = np.sum(Constr[1])
print(f'Sum of Constr[1]: {sum_last_matrix}')

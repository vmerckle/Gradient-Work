import torch
import ot
import numpy as np

def wasserstein(x, x_prev, ly2): # x has grad=true
    # old stuff, does WS on whole x
    #M = ot.dist(x.T, x_prev.T, metric='sqeuclidean', p=2)
    #emp = torch.Tensor([])
    #return ot.emd2(emp, emp, M)

    # ASSUMES ly2 is SORTED
    # do a negative and positive neuron split.
    with torch.no_grad():
        nbpos = torch.sum((ly2 > 0)).item()
    Mpos = ot.dist(x.T[:nbpos], x_prev.T[:nbpos], metric='sqeuclidean', p=2)
    Mneg = ot.dist(x.T[nbpos:], x_prev.T[nbpos:], metric='sqeuclidean', p=2)
    emp = torch.Tensor([])
    return ot.emd2(emp, emp, Mpos) +ot.emd2(emp, emp, Mneg)

def wasserstein_np(x, x_prev): # same remarks as torch wasser
    nbpos = np.sum((ly2 > 0)).item()
    Mpos = ot.dist(x.T[:nbpos], x_prev.T[:nbpos], metric='sqeuclidean', p=2)
    Mneg = ot.dist(x.T[nbpos:], x_prev.T[nbpos:], metric='sqeuclidean', p=2)
    emp = np.array([])
    return ot.emd2(emp, emp, Mpos) + ot.emd2(emp, emp, Mneg)

# return the % of particle exchanged
def wasserstein_num(x, x_prev):
    m = len(x[0])
    nbpos = np.sum((ly2 > 0)).item()
    Mpos = ot.dist(x.T[:nbpos], x_prev.T[:nbpos], metric='sqeuclidean', p=2)
    Mneg = ot.dist(x.T[nbpos:], x_prev.T[nbpos:], metric='sqeuclidean', p=2)
    emp = np.array([])
    upos = np.sum(np.abs(ot.emd(emp, emp, Mpos) - np.eye(len(Mpos))/m))
    uneg = np.sum(np.abs(ot.emd(emp, emp, Mneg) - np.eye(len(Mneg))/m))
    return (uneg+upos)/2/m # every permut counts for 2, return a percentage

def frobenius(x, x_prev, ly2):
    d, m = x.shape
    return torch.sum((x - x_prev)**2)/(d*m)

def frobenius_np(x, x_prev):
    d, m = x.shape
    return np.sum((x - x_prev)**2)/(d*m)

def slicedwasserstein(x, x_prev): # x has grad=true
    return ot.sliced_wasserstein_distance(x, x_prev+1e-10, n_projections=20000)
    # this isn't made for stuff that's close together, hence the 1e-7

# custom sliced wasserstein, no idea if it's different from POT.
#def sliced_wasser(x, x_prev):
#    if self.d>1:
#        return sliced_wasserstein_c(x, x_prev, self.num_projections, self.device, p=2)
#    else:
#        return emd1D(x.reshape(1,-1), x_prev.reshape(1,-1), p=2)
#
#def emd1D(self, u_values, v_values, u_weights=None, v_weights=None,p=1, require_sort=True):
#    n = u_values.shape[-1]
#    m = v_values.shape[-1]
#
#    device = self.device
#    dtype = self.dtype
#
#    if u_weights is None:
#        u_weights = torch.full((n,), 1/n, dtype=dtype, device=device)
#
#    if v_weights is None:
#        v_weights = torch.full((m,), 1/m, dtype=dtype, device=device)
#
#    if require_sort:
#        u_values, u_sorter = torch.sort(u_values, -1)
#        v_values, v_sorter = torch.sort(v_values, -1)
#
#        u_weights = u_weights[..., u_sorter]
#        v_weights = v_weights[..., v_sorter]
#
#    zero = torch.zeros(1, dtype=dtype, device=device)
#    
#    u_cdf = torch.cumsum(u_weights, -1)
#    v_cdf = torch.cumsum(v_weights, -1)
#
#    cdf_axis, _ = torch.sort(torch.cat((u_cdf, v_cdf), -1), -1)
#    
#    u_index = torch.searchsorted(u_cdf, cdf_axis)
#    v_index = torch.searchsorted(v_cdf, cdf_axis)
#
#    u_icdf = torch.gather(u_values, -1, u_index.clip(0, n-1))
#    v_icdf = torch.gather(v_values, -1, v_index.clip(0, m-1))
#
#    cdf_axis = torch.nn.functional.pad(cdf_axis, (1, 0))
#    delta = cdf_axis[..., 1:] - cdf_axis[..., :-1]
#
#    if p == 1:
#        return torch.sum(delta * torch.abs(u_icdf - v_icdf), axis=-1)
#    if p == 2:
#        return torch.sum(delta * torch.square(u_icdf - v_icdf), axis=-1)  
#    return torch.sum(delta * torch.pow(torch.abs(u_icdf - v_icdf), p), axis=-1)
#
#def sliced_cost(self, Xs, Xt, projections=None,u_weights=None,v_weights=None,p=1):
#    if projections is not None:
#        Xps = (Xs @ projections).T
#        Xpt = (Xt @ projections).T
#    else:
#        Xps = Xs.T
#        Xpt = Xt.T
#
#    return torch.mean(self.emd1D(Xps,Xpt,
#                       u_weights=u_weights,
#                       v_weights=v_weights,
#                       p=p))
#
#
#def sliced_wasserstein_c(self, Xs, Xt, num_projections, device,
#                       u_weights=None, v_weights=None, p=1):
#    num_features = Xs.shape[1]
#
#    # Random projection directions, shape (num_features, num_projections)
#    projections = self.rng.normal(size=(num_features, num_projections))
#    projections = F.normalize(torch.from_numpy(projections), p=2, dim=0).type(Xs.dtype).to(device)
#
#    return self.sliced_cost(Xs,Xt,projections=projections,
#                       u_weights=u_weights,
#                       v_weights=v_weights,
#                       p=p)

import numpy as np
import argparse
import sys
from scipy.spatial import distance


import torch
import torch.nn.functional as F
from utils import *
from copy import deepcopy
import algo_jko

class wasser(): #just for grid stuff
    def __init__(self, rng, wasseriter=100, tau=1, num_projections=10, verb=False, dtype=torch.float32, device="cpu", adamlr=1e-3):
        self.wasseriter=wasseriter
        self.num_projections = num_projections
        self.tau = tau
        self.verb = verb
        self.adamlr = adamlr
        self.dtype = dtype  # float16 does a runtime error in pred
        self.device = device
        self.rng = rng

    ## c/c https://raw.githubusercontent.com/clbonet/Sliced-Wasserstein_Gradient_Flows/main/Particles/sw.py


    def load(self, X, Y, ly1, ly2, beta=0):
        self.beta = beta
        self.n, self.d = ly1.shape
        self.ly1 = torch.tensor(ly1, dtype=self.dtype, device=self.device, requires_grad=True)
        self.ly2 = torch.tensor(ly2, dtype=self.dtype, device=self.device)
        self.X = torch.tensor(X, dtype=self.dtype, device=self.device)
        self.Y = torch.tensor(Y, dtype=self.dtype, device=self.device)

    def obj(self, Wn):
        out = torch.relu(self.X @ Wn)@self.ly2
        yhat = torch.sum(out, axis=1)
        mseloss = torch.nn.MSELoss()(yhat.flatten(), self.Y.flatten())
        return mseloss 

    def sliced_wasser(self, Wn, x_prev):
        if self.d>1:
            return self.sliced_wasserstein(Wn, x_prev, self.num_projections, self.device, p=2)
        else:
            return self.emd1D(Wn.reshape(1,-1), x_prev.reshape(1,-1), p=2)

    def step(self):
        x_prev = self.ly1.clone().detach()
        from mechanic_pytorch import mechanize # lr magic (rollbacks)
        optimizer = mechanize(torch.optim.SGD)([self.ly1], lr=self.adamlr)
        
        for i in range(self.wasseriter):
            loss = self.obj(self.ly1) + self.tau*self.sliced_wasser(self.ly1, x_prev)
            loss.backward()
            self.lossi = loss.item()
            optimizer.step()
            optimizer.zero_grad()

    def loss(self):
        with torch.no_grad():
            return self.obj(self.ly1).item()

    def params(self):
        ly1 = self.ly1.cpu().detach().numpy()
        ly2 = self.ly2.cpu().detach().numpy()
        return np.array(ly1), np.array(ly2) # otherwise it's just a pointer...


    def emd1D(self, u_values, v_values, u_weights=None, v_weights=None,p=1, require_sort=True):
        n = u_values.shape[-1]
        m = v_values.shape[-1]

        device = self.device
        dtype = self.dtype

        if u_weights is None:
            u_weights = torch.full((n,), 1/n, dtype=dtype, device=device)

        if v_weights is None:
            v_weights = torch.full((m,), 1/m, dtype=dtype, device=device)

        if require_sort:
            u_values, u_sorter = torch.sort(u_values, -1)
            v_values, v_sorter = torch.sort(v_values, -1)

            u_weights = u_weights[..., u_sorter]
            v_weights = v_weights[..., v_sorter]

        zero = torch.zeros(1, dtype=dtype, device=device)
        
        u_cdf = torch.cumsum(u_weights, -1)
        v_cdf = torch.cumsum(v_weights, -1)

        cdf_axis, _ = torch.sort(torch.cat((u_cdf, v_cdf), -1), -1)
        
        u_index = torch.searchsorted(u_cdf, cdf_axis)
        v_index = torch.searchsorted(v_cdf, cdf_axis)

        u_icdf = torch.gather(u_values, -1, u_index.clip(0, n-1))
        v_icdf = torch.gather(v_values, -1, v_index.clip(0, m-1))

        cdf_axis = torch.nn.functional.pad(cdf_axis, (1, 0))
        delta = cdf_axis[..., 1:] - cdf_axis[..., :-1]

        if p == 1:
            return torch.sum(delta * torch.abs(u_icdf - v_icdf), axis=-1)
        if p == 2:
            return torch.sum(delta * torch.square(u_icdf - v_icdf), axis=-1)  
        return torch.sum(delta * torch.pow(torch.abs(u_icdf - v_icdf), p), axis=-1)

    def sliced_cost(self, Xs, Xt, projections=None,u_weights=None,v_weights=None,p=1):
        if projections is not None:
            Xps = (Xs @ projections).T
            Xpt = (Xt @ projections).T
        else:
            Xps = Xs.T
            Xpt = Xt.T

        return torch.mean(self.emd1D(Xps,Xpt,
                           u_weights=u_weights,
                           v_weights=v_weights,
                           p=p))


    def sliced_wasserstein(self, Xs, Xt, num_projections, device,
                           u_weights=None, v_weights=None, p=1):
        num_features = Xs.shape[1]

        # Random projection directions, shape (num_features, num_projections)
        projections = self.rng.normal(size=(num_features, num_projections))
        projections = F.normalize(torch.from_numpy(projections), p=2, dim=0).type(Xs.dtype).to(device)

        return self.sliced_cost(Xs,Xt,projections=projections,
                           u_weights=u_weights,
                           v_weights=v_weights,
                           p=p)

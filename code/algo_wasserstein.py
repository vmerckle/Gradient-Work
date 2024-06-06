import numpy as np

import torch
import ot
import geomloss as poto

from utils import *

class wasser: #just for grid stuff
    def __init__(self, rng, wasseriter=100, tau=1, num_projections=10, verb=False, dtype=torch.float32, device="cpu", adamlr=1e-3):
        self.wasseriter = wasseriter
        self.num_projections = num_projections
        self.tau = tau
        self.verb = verb
        self.adamlr = adamlr
        self.dtype = dtype  # float16 does a runtime error in pred
        self.device = device
        self.rng = rng

    def load(self, X, Y, ly1, ly2, beta=0):
        self.beta = beta
        self.d, self.m = ly1.shape # a mistake was here before, rip mistake.
        self.ly1 = torch.tensor(ly1, dtype=self.dtype, device=self.device, requires_grad=True)
        self.ly2 = torch.tensor(ly2, dtype=self.dtype, device=self.device)
        self.X = torch.tensor(X, dtype=self.dtype, device=self.device)
        self.Y = torch.tensor(Y, dtype=self.dtype, device=self.device)

    def obj(self, Wn):
        out = torch.relu(self.X @ Wn)@self.ly2
        yhat = torch.sum(out, axis=1)
        mseloss = torch.nn.MSELoss()(yhat.flatten(), self.Y.flatten())
        return mseloss 


    #    return poto.SamplesLoss("sinkhorn", p=2, blur=0.01)(x, x_prev)
    #    return ot.sliced_wasserstein_distance(x, x_prev+1e-12, n_projections=200)
    def wasserstein(self, x, x_prev): # x has grad=true
        M = ot.dist(x.T, x_prev.T, metric='sqeuclidean', p=2) # there was a criticla mistake here before
        return ot.emd2(torch.Tensor([]), torch.Tensor([]), M)  # uniform if empty lists are given

    def step(self):
        x_prev = self.ly1.clone().detach()
        from mechanic_pytorch import mechanize # lr magic (rollbacks)
        optimizer = mechanize(torch.optim.SGD)([self.ly1], lr=self.adamlr)
        #optimizer = torch.optim.SGD([self.ly1], lr=self.adamlr)
        
        for i in range(self.wasseriter):
            loss = self.obj(self.ly1) + self.tau*self.wasserstein(self.ly1, x_prev)
            #loss = self.obj(self.ly1) + self.tau*torch.sum((self.ly1 - x_prev)**2)/self.m
            loss.backward()
            self.lossi = loss.item()
            optimizer.step()
            optimizer.zero_grad()
        #print("right?", loss.item())
        with torch.no_grad():
            self.objectif = (self.obj(self.ly1)).item()
            self.wasserdist = (self.wasserstein(self.ly1, x_prev)).item()
            self.ldeuxdist = (torch.sum((self.ly1 - x_prev)**2)/self.m).item()
            #print(f"obj={a}, wasser={b}, tau={self.tau}")
        with torch.no_grad():
            M = ot.emd(torch.Tensor([]), torch.Tensor([]), ot.dist(self.ly1.T, x_prev.T)).detach().numpy()
            a = np.abs(self.wasserstein(self.ly1, x_prev) - torch.sum((self.ly1 - x_prev)**2)/self.m).item()
            b = np.sum(np.abs(np.eye(self.m)/self.m - M))
            print(f"sum(I/m - M)= {b:.3f}, wasser-L2Q = {a:.3E}")
        print("")


    def loss(self):
        with torch.no_grad():
            return self.obj(self.ly1).item()

    def params(self):
        ly1 = self.ly1.cpu().detach().numpy()
        ly2 = self.ly2.cpu().detach().numpy()
        return np.array(ly1), np.array(ly2) # otherwise it's just a pointer...


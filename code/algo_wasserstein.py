import numpy as np
import argparse
import sys
from scipy.spatial import distance


import torch
import torch.nn.functional as F
from utils import *
from copy import deepcopy
import algo_jko

class wasser(algo_jko.JKO): #just for grid stuff
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


    def load(self, X, Y, ly1, ly2, beta=0, grid=None, p=None):
        self.loadgrid(X, Y, ly1, ly2, grid, p)
        self.beta = beta

    def step(self):
        q_in = self.p
        X = torch.tensor(self.X, dtype=self.dtype, device=self.device)
        Y = torch.tensor(self.Y, dtype=self.dtype, device=self.device)
        q = torch.tensor(q_in, dtype=self.dtype, device=self.device)
        #p = torch.tensor(q_in, dtype=self.dtype, device=self.device, requires_grad=True)
        #p = deepcopy(q).requires_grad_(True)
        grid = torch.tensor(self.grid, dtype=self.dtype, device=self.device)
        W = deepcopy(grid.T).requires_grad_(True)
        x_prev = grid.T
        x_k = W

        activ = (X @ grid.T) > 0

        def obj(Wn, verb=False):
            #effneurons = grid.T # (d,N) 
            out = activ * (X @ Wn) # (n, d) * (d, N) = (n, N)
            yhat = torch.sum(out, axis=1)
            mseloss = torch.nn.MSELoss()(yhat.flatten(), Y.flatten())
            return mseloss 

        #optimizer = torch.optim.SGD([W], lr=self.adamlr, weight_decay=self.beta, momentum=0.9)
        optimizer = torch.optim.AdamW([W], lr=self.adamlr, weight_decay=self.beta)
        from mechanic_pytorch import mechanize # lr magic (rollbacks)
        optimizer = mechanize(torch.optim.SGD)([W], lr=self.adamlr)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

        noimprovstop = 20000

        noimprovsince = 0
        biggestnoimprov = 0
        bestloss = 1e10
        bestloss_i = 0
        startloss=0
        lastlr = 1
        bestW = W.data.clone().detach()

        for i in range(self.wasseriter):
            if self.d>1:
                sw = self.sliced_wasserstein(x_k, x_prev, self.num_projections, self.device, p=2)
            else:
                sw = self.emd1D(x_k.reshape(1,-1), x_prev.reshape(1,-1), p=2)

            # loss = 1/self.m * obj(W) + self.tau*sw # not sure about 1/m
            loss = obj(W) + self.tau*sw
            loss.backward()
            lossval = loss.item()
            nowlr = optimizer.param_groups[0]["lr"]
            if abs(lastlr - nowlr) > 1e-10:
                lastlr = nowlr
                print("newlr", nowlr)
            if i == 0:
                startloss=lossval
                print("startloss", lossval)
                bestloss=lossval
            noimprovsince += 1
            #with torch.no_grad():
                #print(f"wgrad norm: {W.grad.norm(2):.7f}")
            if lossval < bestloss:
                #bestW = W.data.clone().detach()
                improvement = (1-lossval/startloss)/startloss
                #print(f"{i}({improvement:.1f}%) .. ", end="", flush=True)

                biggestnoimprov = noimprovsince if noimprovsince > biggestnoimprov else biggestnoimprov
                noimprovsince = 0
                bestloss = lossval
                bestloss_i = i
                # print(f"bestloss at i={bestloss_i}: {bestloss}")
            #else:
            #    with torch.no_grad():
            #        W.data = bestW.clone()
            if noimprovsince > noimprovstop:
                break
            optimizer.step()
            #scheduler.step()
            #scheduler.step(lossval)
            optimizer.zero_grad()
        print("")
        print("final loss", lossval)
        stableprojec_numb = 1 #10 000 is a good number, but slow
        sw = self.sliced_wasserstein(x_k, x_prev, 1, self.device, p=2)
        loss = obj(W) + self.tau*sw
        loss.backward()
        print(f"gradnorm : {W.grad.norm(2):.7f}")
        lossval = loss.item()
        improvement = (1-lossval/startloss)/startloss
        bimprovement = (1-bestloss/startloss)/startloss
        print(f"END(it={i}) bestloss at i={bestloss_i}: {bestloss}, biggest noimprov={biggestnoimprov}, final(stable):{improvement:.2f}%(best(unstable) {bimprovement:.2f}%)")

        self.grid = x_k.detach().numpy().T
        #self.p = (self.p > 0)*self.p
        #self.p = self.p/np.sum(self.p)

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

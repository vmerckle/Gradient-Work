import numpy as np

from tqdm import tqdm
import torch
import ot
#import geomloss as poto
    #    return poto.SamplesLoss("sinkhorn", p=2, blur=0.01)(x, x_prev)

from utils import *

class proxpoint:
    def __init__(self, rng, proxdist, inneriter=100, gamma=1, dtype=torch.float32, device="cpu"):
        self.rng = rng
        self.proxdist = proxdist
        self.inneriter = inneriter
        self.gamma = gamma
        self.dtype = dtype
        self.device = device

    def obj(self, Wn):
        out = torch.relu(self.X @ Wn)@self.ly2
        yhat = torch.sum(out, axis=1)
        mseloss = torch.nn.MSELoss()(yhat.flatten(), self.Y.flatten())
        return mseloss 

    def step(self):
        x_prev = self.ly1.clone().detach()
        assert self.ly1.requires_grad
        itero = range(self.inneriter)
        if self.inneriter > 100:
            itero = tqdm(itero, desc="prox loop")
        for i in itero:
            self.optimizer.zero_grad()
            loss = self.obj(self.ly1) + 1/self.gamma*self.proxdist(self.ly1, x_prev, self.ly2)
            #assert self.ly1.requires_grad
            loss.backward()
            self.optimizer.step()
        #    obji, disti = self.obj(self.ly1).item(), 1/self.gamma*self.proxdist(self.ly1, x_prev).item()
        #print(f"{obji:.1E}, {disti:.1E} so {obji/disti*100:.1f}%")
        #if i == self.inneriter-1:
            #print("warning, used all", self.inneriter, "iterations in prox")
        with torch.no_grad():
            nrm = torch.norm(self.ly1.grad).item()
            print("norm", nrm)

    # sample code to do custom grad 
    def weirdstep(self):
        x_prev = self.ly1.clone().detach()
        for i in range(self.inneriter):
            loss = self.obj(self.ly1)
            loss.backward()
            with torch.no_grad():
                #self.ly1 = x_prev - self.ly1.grad
                self.ly1.zero_()
                self.ly1.add_(x_prev, alpha=1)
                self.ly1.add_(self.ly1.grad, alpha=-self.gamma)
            #self.optimizer.step()
            self.optimizer.zero_grad()

        #with torch.no_grad():
            #M = ot.emd(torch.Tensor([]), torch.Tensor([]), ot.dist(self.ly1.T, x_prev.T)).detach().numpy()
            #b = np.sum(np.abs(np.eye(self.m)/self.m - M))
            #print(f"sum(I/m - M)= {b:.3f}, wasser-L2Q = {a:.3E}")
        #print("")

    def load(self, X, Y, ly1, ly2, beta=0):
        torch.set_grad_enabled(True) # keyboard interrupt in the middle of a torch.no_grad...
        self.beta = beta
        self.d, self.m = ly1.shape
        self.ly1 = torch.tensor(ly1, dtype=self.dtype, device=self.device, requires_grad=True)
        self.ly2 = torch.tensor(ly2, dtype=self.dtype, device=self.device)
        self.X = torch.tensor(X, dtype=self.dtype, device=self.device)
        self.Y = torch.tensor(Y, dtype=self.dtype, device=self.device)

        from mechanic_pytorch import mechanize # lr magic (rollbacks)
        self.optimizer = mechanize(torch.optim.SGD)([self.ly1], lr=1)
        #self.optimizer = mechanize(torch.optim.AdamW)([self.ly1], lr=1, weight_decay=0)
        #self.optimizer = torch.optim.SGD([self.ly1], lr=self.gamma)

    def loss(self):
        with torch.no_grad():
            out = torch.relu(self.X @ self.ly1)@self.ly2
            yhat = torch.sum(out, axis=1)
            print(yhat.flatten().detach().cpu().numpy()[:10])
            print(self.Y.flatten().detach().cpu().numpy()[:10])
            return self.obj(self.ly1).item()

    def params(self):
        ly1 = self.ly1.cpu().detach().numpy()
        ly2 = self.ly2.cpu().detach().numpy()
        return np.array(ly1), np.array(ly2) # otherwise it's just a pointer...


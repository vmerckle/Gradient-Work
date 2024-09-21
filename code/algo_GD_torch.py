import numpy as np

#import functorch
import torch
#import torch.nn as nn
#import torch.nn.functional as F

from utils import *

def torch_2layers(X, ly1, ly2):
    return torch.relu(X@ly1)@ly2

def getOpti(optiname, params, optiD):
    if optiname == "prodigy":
        from prodigyopt import Prodigy
        return Prodigy(params,lr=optiD["lr"], weight_decay=optiD["weight_decay"])
    elif optiname == "mechanize":
        from mechanic_pytorch import mechanize # lr magic (rollbacks)
        return mechanize(torch.optim.SGD)(params, lr=optiD["lr"])
    elif optiname == "mechanizeadam":
        from mechanic_pytorch import mechanize # lr magic (rollbacks)
        return mechanize(torch.optim.AdamW)(params, lr=optiD["lr"])
    elif optiname == "SGD":
        return torch.optim.SGD(params, lr=optiD["lr"], weight_decay=optiD["weight_decay"])
    elif optiname == "AdamW":
        return torch.optim.AdamW(params, lr=optiD["lr"], weight_decay=optiD["weight_decay"])
    elif optiname == "Adadelta":
        return torch.optim.Adadelta(params, lr=optiD["lr"], weight_decay=optiD["weight_decay"])

class torch_descent:
    def __init__(self, D, dtype=torch.float32, device="cpu"):
        self.dtype = dtype
        self.device = device
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.SoftMarginLoss() # loss for 2-class classif
        self.pred_fn = torch_2layers
        self.opti = D["opti"]
        self.optiD = D["optiD"]
        self.optimizer = None
        self.onlyTrainFirstLayer = D["onlyTrainFirstLayer"]
        self.batched = D["batched"]
        self.batch_size = D["batch_size"]


    def load(self, X, Y, ly1, ly2, beta=0):
        self.n, self.d = X.shape
        self.ibatch = self.n # so that we reshuffle at first step
        self.m = ly1.shape[1]


        self.X = torch.tensor(X, dtype=self.dtype, device=self.device)
        self.Y = torch.tensor(Y, dtype=self.dtype, device=self.device)

        self.ly1 = torch.tensor(ly1, dtype=self.dtype, device=self.device, requires_grad=True)
        self.ly2 = torch.tensor(ly2, dtype=self.dtype, device=self.device, requires_grad=not self.onlyTrainFirstLayer)
        params = [self.ly1]
        if self.onlyTrainFirstLayer:
            params = [self.ly1, self.ly2]

        self.optimizer = getOpti(self.opti, params, self.optiD)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        

        with torch.no_grad():
            self.lastloss = self.loss_fn(self.pred_fn(self.X, self.ly1, self.ly2), self.Y).item()

    def step(self):
        if self.batched:
            if self.ibatch + self.batch_size > self.n:
                self.ibatch = 0
                rand_indx = torch.randperm(self.n)
                self.X = self.X[rand_indx]
                self.Y = self.Y[rand_indx]
            bY, bX = self.Y[self.ibatch:self.ibatch+self.batch_size], self.X[self.ibatch:self.ibatch+self.batch_size]
            loss = self.loss_fn(self.pred_fn(bX, self.ly1, self.ly2), bY)
            self.ibatch += self.batch_size
        else:
            loss = self.loss_fn(self.pred_fn(self.X, self.ly1, self.ly2), self.Y)
        self.lastloss = loss.item()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def params(self):
        ly1 = self.ly1.cpu().detach().numpy()
        ly2 = self.ly2.cpu().detach().numpy()
        return {"ly1": np.array(ly1),
                "ly2": np.array(ly2)} # otherwise it's just a pointer...

    def grads(self):
        ly1g = self.ly1.grad.cpu().detach().numpy()
        ly2g = self.ly2.grad.cpu().detach().numpy()
        return np.array(ly1g), np.array(ly2g) # no pointy things allowed 

    def pred(self, X):
        X = torch.tensor(X, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            return self.pred_fn(X, self.ly1, self.ly2).cpu().detach().numpy()

    def loss(self):
        return self.lastloss

            #ly1gg, ly2gg = dir_grad(X, Y, ly1, ly2)
            #ly1g = ly1gg.add(ly1, alpha=beta)
            #ly1.add_(ly1g, alpha=-lr)
            #ly2g = ly2gg.add(ly2, alpha=beta)
            #ly2.add_(ly2g, alpha=-lr)

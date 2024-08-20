import numpy as np

#import functorch
import torch
#import torch.nn as nn
#import torch.nn.functional as F

from utils import *

def torch_2layers(X, ly1, ly2):
    return torch.relu(X@ly1)@ly2

class torch_descent:
    def __init__(self, D, dtype=torch.float32, device="cpu"):
        self.opti = D["opti"]
        self.momentum = D["momentum"]
        self.dtype = dtype
        self.device = device
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.SoftMarginLoss() # loss for 2-class classif
        self.pred_fn = torch_2layers
        self.optimizer = None
        self.lr = D["lr"]
        self.beta = D["beta"]
        self.onlyTrainFirstLayer = D["onlyTrainFirstLayer"]


    def load(self, X, Y, ly1, ly2, beta=0):
        self.n, self.d = X.shape
        self.m = ly1.shape[1]

        self.ly1 = torch.tensor(ly1, dtype=self.dtype, device=self.device, requires_grad=True)
        self.X = torch.tensor(X, dtype=self.dtype, device=self.device)
        self.Y = torch.tensor(Y, dtype=self.dtype, device=self.device)

        if self.onlyTrainFirstLayer:
            params = [self.ly1]
            self.ly2 = torch.tensor(ly2, dtype=self.dtype, device=self.device, requires_grad=False)
        else:
            params = [self.ly1, self.ly2]
            self.ly2 = torch.tensor(ly2, dtype=self.dtype, device=self.device, requires_grad=True)

        if self.opti == "prodigy":
            from prodigyopt import Prodigy
            self.optimizer = Prodigy(params,lr=self.lr, weight_decay=self.beta)
        elif self.opti == "mechanize":
            from mechanic_pytorch import mechanize # lr magic (rollbacks)
            self.optimizer = mechanize(torch.optim.SGD)(params, lr=self.lr)
        elif self.opti == "mechanizeadam":
            from mechanic_pytorch import mechanize # lr magic (rollbacks)
            self.optimizer = mechanize(torch.optim.AdamW)(params, lr=self.lr)
        elif self.opti == "SGD":
            self.optimizer = torch.optim.SGD(params, lr=self.lr, weight_decay=self.beta)
        elif self.opti == "AdamW":
            self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.beta)
        elif self.opti == "Adadelta":
            self.optimizer = torch.optim.Adadelta(params, lr=self.lr, weight_decay=self.beta)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        with torch.no_grad():
            self.lastloss = self.loss_fn(self.pred_fn(self.X, self.ly1, self.ly2), self.Y).item()

    def step(self):
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

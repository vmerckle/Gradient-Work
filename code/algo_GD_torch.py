import numpy as np

#import functorch
import torch
#import torch.nn as nn
#import torch.nn.functional as F

from utils import *

def torch_2layers(X, ly1, ly2):
    return torch.relu(X@ly1)@ly2

class torch_descent:
    def __init__(self, algo, lr, momentum=0, dtype=torch.float32, device="cpu"):
        self.algo = algo
        self.momentum = momentum
        self.dtype = dtype  # float16 does a runtime error in pred
        self.device = device
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.SoftMarginLoss() # loss for 2-class classif
        self.pred_fn = torch_2layers
        self.optimizer = None
        self.lr = lr

    def load(self, X, Y, ly1, ly2, beta):
        self.beta = beta
        self.n, self.d = X.shape
        self.m = ly1.shape[1]
        self.X = torch.tensor(X, dtype=self.dtype, device=self.device)
        self.Y = torch.tensor(Y, dtype=self.dtype, device=self.device)
        self.ly1 = torch.tensor(ly1, dtype=self.dtype, device=self.device, requires_grad=True)
        self.ly2 = torch.tensor(ly2, dtype=self.dtype, device=self.device, requires_grad=False) # we only train First layer allo

        if self.algo == "gd":
            #self.optimizer = torch.optim.SGD([self.ly1], lr=self.lr, momentum=self.momentum, weight_decay=self.beta)
            # really slower(for large neurons, for some reason) or very far from GD
            #from mechanic_pytorch import mechanize # lr magic (rollbacks)
            #self.optimizer = mechanize(torch.optim.AdamW)([self.ly1], lr=1e-2)#, momentum=0, weight_decay=self.beta)

            from prodigyopt import Prodigy
            self.optimizer = Prodigy([self.ly1],lr=1e-2, weight_decay=0.)
            #self.optimizer = torch.optim.AdamW([self.ly1], lr=self.lr, weight_decay=0.)
            #self.optimizer = torch.optim.SGD([self.ly1], lr=self.lr, momentum=0.90)
        elif self.algo == "adam":
            self.optimizer = torch.optim.AdamW([self.ly1, self.ly2], lr=self.lr, weight_decay=self.beta)
        else:
            raise Exception

    def step(self):
        loss = self.loss_fn(self.pred_fn(self.X, self.ly1, self.ly2), self.Y)
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
        with torch.no_grad():
            loss = self.loss_fn(self.pred_fn(self.X, self.ly1, self.ly2), self.Y)
            return loss.item()

            #ly1gg, ly2gg = dir_grad(X, Y, ly1, ly2)
            #ly1g = ly1gg.add(ly1, alpha=beta)
            #ly1.add_(ly1g, alpha=-lr)
            #ly2g = ly2gg.add(ly2, alpha=beta)
            #ly2.add_(ly2g, alpha=-lr)

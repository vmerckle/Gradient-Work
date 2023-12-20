import numpy as np

#import functorch
import torch
#import torch.nn as nn
#import torch.nn.functional as F

from utils import *

class torch_descent:
    def __init__(self, algo="gd", momentum=0, dtype=torch.float32, device="cpu"):
        self.algo = algo
        self.momentum = momentum
        self.dtype = dtype  # float16 does a runtime error in pred
        self.device = device
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.SoftMarginLoss() # loss for 2-class classif
        self.pred_fn = torch_2layers
        self.optimizer = None

    def load(self, X, Y, ly1, ly2, lr, beta):
        self.lr, self.beta = lr, beta
        self.n, self.d = X.shape
        self.m = ly1.shape[1]
        self.X = torch.tensor(X, dtype=self.dtype, device=self.device)
        self.Y = torch.tensor(Y, dtype=self.dtype, device=self.device)
        self.ly1 = torch.tensor(ly1, dtype=self.dtype, device=self.device, requires_grad=True)
        self.ly2 = torch.tensor(ly2, dtype=self.dtype, device=self.device, requires_grad=True)
        if self.algo == "gd":
            self.optimizer = torch.optim.SGD([self.ly1, self.ly2], lr=self.lr, momentum=self.momentum, weight_decay=self.beta)
        elif self.algo == "adam":
            self.optimizer = torch.optim.AdamW([self.ly1, self.ly2], lr=self.lr, weight_decay=self.beta)
        else:
            raise Exception

    def step(self):
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.pred_fn(self.X, self.ly1, self.ly2), self.Y)
        loss.backward()
        self.optimizer.step()

    def params(self):
        ly1 = self.ly1.cpu().detach().numpy()
        ly2 = self.ly2.cpu().detach().numpy()
        return np.array(ly1), np.array(ly2) # otherwise it's just a pointer...

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

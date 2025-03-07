import numpy as np
from tqdm import tqdm
import ot
import torch
from torch.optim.lr_scheduler import _LRScheduler
#import geomloss as poto
    #    return poto.SamplesLoss("sinkhorn", p=2, blur=0.01)(x, x_prev)

from utils import *
from algos.algo_GD_torch import getOpti

class CustomScheduler(_LRScheduler):
    def __init__(self, optimizer, lr_decay=0.9993, last_epoch=-1):
        self.start_lr = [group['lr'] for group in optimizer.param_groups]
        self.last_loss = 100.0
        self.factor = 1.0
        self.lr_decay = lr_decay
        super(CustomScheduler, self).__init__(optimizer, last_epoch)

    def step(self, loss=None, epoch=None):
        if loss is not None:
            if loss >= self.last_loss:
                self.factor *= self.lr_decay
            self.last_loss = loss
        super().step(epoch)  # Proceed with the normal step functionality

    def get_lr(self):
        return [base_lr * self.factor for base_lr in self.start_lr]

class proxpoint:
    def __init__(self, D, dtype=torch.float32, device="cpu"):
        self.dtype = dtype
        self.device = device
        dist = D["dist"]
        if dist == "wasser":
            from algos.proxdistance import wasserstein
            self.proxdist = wasserstein
        elif dist == "frobenius":
            from algos.proxdistance import frobenius 
            self.proxdist = frobenius
        elif dist == "sliced":
            from algos.proxdistance import slicedwasserstein
            self.proxdist = slicedwasserstein
        else:
            raise Exception("config bad proxdist choice")

        self.opti = D["opti"]
        self.optiD = D["optiD"]
        self.LRdecay = D["LRdecay"]
        self.inneriter = D["inneriter"]
        self.gamma = D["gamma"]
        self.recordinner = D["recordinner"]
        self.recordinnerlayers = D["recordinnerlayers"]
        self.onlyTrainFirstLayer = D["onlyTrainFirstLayer"]
        if self.recordinner:
            self.innerD = {}

    def obj(self, Wn):
        out = torch.relu(self.X @ Wn)@self.ly2
        yhat = torch.sum(out, axis=1)
        mseloss = torch.nn.MSELoss()(yhat.flatten(), self.Y.flatten())
        return mseloss 

    def step_unused(self):
        x_prev = self.ly1.clone().detach()
        assert self.ly1.requires_grad
        itero = range(self.inneriter)
        firstloss = None
        if self.inneriter > 100:
            itero = tqdm(itero, desc="prox loop")
        for i in itero:
            self.optimizer.zero_grad()
            loss = self.obj(self.ly1) + 1/self.gamma*self.proxdist(self.ly1, x_prev, self.ly2)
            #assert self.ly1.requires_grad
            loss.backward()
            if firstloss is None:
                firstloss = loss.item()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        if loss.item() > firstloss:
            itero = tqdm(desc="over prox")
            while True:
                self.optimizer.zero_grad()
                loss = self.obj(self.ly1) + 1/self.gamma*self.proxdist(self.ly1, x_prev, self.ly2)
                #assert self.ly1.requires_grad
                loss.backward()
                self.optimizer.step()
                itero.update()
                if loss.item() < firstloss:
                    break
            itero.close()

    def updateInner(self, D=None, num=None):
        if not self.recordinner:
            return
        if D is None:
            self.innerD = {}
            return

        if self.recordinnerlayers:
            params = self.params()
            D["ly1"] = params["ly1"]
        self.innerD[num] = D

    def step(self):
        x_prev = self.ly1.clone().detach()
        assert self.ly1.requires_grad
        
        itero = range(self.inneriter)
        if self.inneriter > 100:
            itero = tqdm(itero, desc="prox loop")

        self.updateInner()
        for i in itero:
            self.optimizer.zero_grad()
            obj = self.obj(self.ly1)
            dis = 1/self.gamma*self.proxdist(self.ly1, x_prev, self.ly2)
            loss = obj + dis
            lossitem = loss.item()
            self.updateInner(num=i, D={"obj":obj.item(), "dist":dis.item(), "loss":lossitem, "lr":self.scheduler.get_last_lr()[0]})
            # todo include LR in datas
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step(loss=lossitem)
        self.lastloss = obj.item()

    def load(self, X, Y, ly1, ly2, beta=0):
        torch.set_grad_enabled(True) # keyboard interrupt in the middle of a torch.no_grad...
        self.beta = beta
        self.d, self.m = ly1.shape
        self.ly1 = torch.tensor(ly1, dtype=self.dtype, device=self.device, requires_grad=True)
        self.X = torch.tensor(X, dtype=self.dtype, device=self.device)
        self.Y = torch.tensor(Y, dtype=self.dtype, device=self.device)

        self.ly1 = torch.tensor(ly1, dtype=self.dtype, device=self.device, requires_grad=True)
        self.ly2 = torch.tensor(ly2, dtype=self.dtype, device=self.device, requires_grad=not self.onlyTrainFirstLayer)
        params = [self.ly1]
        if self.onlyTrainFirstLayer:
            params = [self.ly1, self.ly2]

        self.optimizer = getOpti(self.opti, params, self.optiD)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        #self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.LRdecay)
        self.scheduler = CustomScheduler(self.optimizer, lr_decay=self.LRdecay)


        with torch.no_grad():
            self.lastloss = self.obj(self.ly1).item()

    def loss(self):
        return self.lastloss

    def params(self):
        ly1 = self.ly1.cpu().detach().numpy()
        ly2 = self.ly2.cpu().detach().numpy()
        return {"ly1": np.array(ly1), # otherwise it's just a pointer...
                "ly2": np.array(ly2),
                "innerD": self.innerD,
                "loss": self.lastloss}


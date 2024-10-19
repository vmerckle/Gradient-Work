import numpy as np
import argparse
import sys
from scipy.optimize import minimize
from scipy.special import kl_div
from scipy.spatial import distance

from utils import *
import torch

import jko

"""
with proxf implemented
    - in pytorch gradient descent, multiplicate gradient update, and other tricks
    to keep away the NaN

results: GD is too hard to tune.
"""

class jko_torch(jko.JKO):
    def __init__(self, gamma=1, tau=1, interiter=10, verb=False, tol=1e-6, dtype=torch.float32, device="cpu", adamlr=1e-3):
        self.gamma = gamma
        self.tau = tau
        self.tol = tol
        self.verb = verb
        self.interiter = interiter
        self.dtype = dtype  # float16 does a runtime error in pred
        self.device = device
        self.adamlr = adamlr
        self.mynorm = lambda x: np.linalg.norm(x.flatten())
        self.jkotot = 0

        self.proxf = self.prox_f_pytorch_new

    def prox_f_pytorch(self, q_in):
        X = torch.tensor(self.X, dtype=self.dtype, device=self.device)
        Y = torch.tensor(self.Y, dtype=self.dtype, device=self.device)
        p = torch.tensor(q_in, dtype=self.dtype, device=self.device, requires_grad=True)
        q = torch.tensor(q_in, dtype=self.dtype, device=self.device)
        grid = torch.tensor(self.grid.T, dtype=self.dtype, device=self.device)
        activ = (X @ grid) > 0
        def obj(p, verb=False):
            effneurons = p * grid.T # (d,N) 
            out = activ * (X @ effneurons.T) # (n, d) * (d, N) = (n, N)
            yhat = torch.sum(out, axis=1)
            mseloss = torch.nn.MSELoss()(yhat.flatten(), Y.flatten())
            kldiv = p * torch.log(p/q) - p + q
            kldiv = torch.nan_to_num(kldiv, nan=0.0)
            #print("kldiv", kldiv)
            kldivs = torch.sum(kldiv)/len(p) # arguable
            #print(kldivs)
            if verb:
                print("p,q", p, q)
                print("kldiv", kldiv)
                sys.exit(0)
            return self.tau/self.gamma * mseloss +  kldivs

        #optimizer = torch.optim.AdamW([p], lr=self.adamlr, weight_decay=0)
        #optimizer = torch.optim.SGD([p], lr=self.adamlr, weight_decay=0)
        with torch.no_grad():
            fstloss = obj(p)
            #print("first", p)
        maxit = 100
        bestloss = 1e3
        for i in range(maxit):
            #optimizer.zero_grad()
            p.grad = None
            loss = obj(p)
            loss.backward()
            #optimizer.step()
            with torch.no_grad():
                lossdiff = bestloss - loss.item()
                if lossdiff < 1e-6:
                    #print("proxf exit on small loss at ite", i)
                    break
                #print(f"{loss.item():.4f} so.. {lossdiff:.8f}")
                if torch.norm(p.grad) < 0.1:
                    print("proxf exit on small grad at ite", i)
                    break
                #pg = torch.exp(-p.grad)/p.sum()
                #p = (p*torch.exp(-p.grad))/p.sum()
                #print(i, "GRADO", p.grad)
                pgrad = torch.nan_to_num(p.grad, nan=0.0)
                newp= torch.relu(p - self.adamlr * pgrad) # exact same as p.add_
                p.copy_(newp)

                #p.add_(p.grad, alpha=-self.adamlr) # p.add also works.. 
                # but p.add_ really is the same as p - self.lr * p.grad. p.add idk

                #p = p - self.adamlr * p.grad
                #p.add_(ly1g, alpha=1)
                if loss.item() < bestloss:
                    bestloss = loss.item()
                #print("OUR BELOVED P", p.flatten().detach())
            if np.isnan(loss.item()):
                print("nan loss at", i)
                obj(p, verb=True)
                break
            #print(f"{i} loss -> {loss.item()}")
        #print(f"improved loss by {fstloss} -> {loss.item()}, pfound={p.flatten()}")
        #print(f"proxf in:{q_in.flatten()} out:{p.flatten().cpu().detach().numpy()}")
        if i == maxit-1:
            print(f"did all {maxit} iterations, grad=", torch.norm(p.grad).item())
        return p.cpu().detach().numpy()

    def prox_f_pytorch(self, q_in):
        qsum = q_in.sum()
        #if qsum > 1.01:
        print("proxf in vector..sum=", qsum)
            #assert qsum < 1.2
        X = torch.tensor(self.X, dtype=self.dtype, device=self.device)
        Y = torch.tensor(self.Y, dtype=self.dtype, device=self.device)
        p = torch.tensor(q_in, dtype=self.dtype, device=self.device, requires_grad=True)
        q = torch.tensor(q_in, dtype=self.dtype, device=self.device)
        #print("so now ", q.flatten(), q.sum())
        #q = q/q.sum()
        grid = torch.tensor(self.grid.T, dtype=self.dtype, device=self.device)
        psigns= torch.tensor(self.psigns, dtype=self.dtype, device=self.device)
        activ = (X @ grid) > 0
        def obj(p, verb=False, onlyMSE=False):
            effneurons = p * grid.T * psigns # (d,N) 
            out = activ * (X @ effneurons.T) # (n, d) * (d, N) = (n, N)
            yhat = torch.sum(out, axis=1)
            mseloss = torch.nn.MSELoss()(yhat.flatten(), Y.flatten())
            kldiv = p * torch.log(p/q) - p + q
            kldivs = torch.sum(kldiv)/len(p)
            if verb:
                print("p,q", p, q)
                print("kldiv", kldiv)
                sys.exit(0)
            if onlyMSE:
                return mseloss.item(), kldivs.item()
            return self.tau/self.gamma * mseloss +  kldivs

        #optimizer = torch.optim.AdamW([p], lr=self.adamlr, weight_decay=0)
        #optimizer = torch.optim.SGD([p], lr=self.adamlr, weight_decay=0)
        with torch.no_grad():
            
            print("First loss(mseloss, kldivs) =",obj(p, onlyMSE=True))
        maxit = 10000
        bestloss = 1e3
        allps = []
        for i in range(maxit):
            #optimizer.zero_grad()
            p.grad = None
            loss = obj(p)
            loss.backward()
            #optimizer.step()
            with torch.no_grad():
                allps.append(p.cpu().detach().numpy())
                lossdiff = bestloss - loss.item()
                #print(f"{loss.item():.4f} so.. {lossdiff:.8f}")
                if torch.norm(p.grad) < 0.1 and False:
                    print("proxf exit on small grad at ite", i)
                    break
                #pg = torch.exp(-p.grad)/p.sum()
                #p = (p*torch.exp(-p.grad))/p.sum()

                #newp= p - self.adamlr * p.grad # exact same as p.add_
                newp = p*torch.exp(-self.adamlr*p.grad)
                newp = torch.clamp(newp, min=1e-7) # needed...
                #newp = newp/newp.sum()
                p.copy_(newp)

                #p.add_(p.grad, alpha=-self.adamlr) # p.add also works.. 
                # but p.add_ really is the same as p - self.lr * p.grad. p.add idk

                #p = p - self.adamlr * p.grad
                #p.add_(ly1g, alpha=1)
                if loss.item() < bestloss:
                    bestloss = loss.item()
                if i % 3000 == 0 or i == maxit-1:
                    print("ité MWA:mse,kldiv", i, obj(p, onlyMSE=True))
            if np.isnan(loss.item()):
                print("nan loss at", i)
                obj(p, verb=True)
                break
            #print(f"{i} loss -> {loss.item()}")
        #print(f"improved loss by {fstloss} -> {loss.item()}, pfound={p.flatten()}")
        #print(f"proxf in:{q_in.flatten()} out:{p.flatten().cpu().detach().numpy()}")
        nallps = len(allps)
        allps = np.array(allps)
        if i == maxit-1:
            pass
            #print(f"did all {maxit} iterations, grad=", torch.norm(p.grad).item())
        return allps.sum(axis=0)/nallps
        #return p.cpu().detach().numpy()

    def prox_f_pytorch_new(self, q_in):
        X = torch.tensor(self.X, dtype=self.dtype, device=self.device)
        Y = torch.tensor(self.Y, dtype=self.dtype, device=self.device)
        p = torch.tensor(q_in, dtype=self.dtype, device=self.device, requires_grad=True)
        q = torch.tensor(q_in, dtype=self.dtype, device=self.device)
        grid = torch.tensor(self.grid.T, dtype=self.dtype, device=self.device)
        psigns= torch.tensor(self.psigns, dtype=self.dtype, device=self.device)
        activ = (X @ grid) > 0
        def obj(p, verb=False, onlyMSE=False):
            effneurons = p * grid.T * psigns # (d,N) 
            out = activ * (X @ effneurons.T) # (n, d) * (d, N) = (n, N)
            yhat = torch.sum(out, axis=1)
            mseloss = torch.nn.MSELoss()(yhat.flatten(), Y.flatten())
            kldiv = p * torch.log(p/q) - p + q
            kldivs = torch.sum(kldiv)/len(p)
            if verb:
                print("p,q", p, q)
                print("kldiv", kldiv)
                sys.exit(0)
            if onlyMSE:
                return mseloss.item(), kldivs.item()
            return self.tau/self.gamma * mseloss +  kldivs

        optimizer = torch.optim.AdamW([p], lr=self.adamlr, weight_decay=0)
        #optimizer = torch.optim.SGD([p], lr=self.adamlr, weight_decay=0)
        maxit = 100
        for i in range(maxit):
            p.grad = None
            loss = obj(p)
            loss.backward()
            #print(loss.item())
            optimizer.step()
            p.data = torch.clamp(p.data, min=1e-16) # necessary
        return p.cpu().detach().numpy()

        
    def prox_f_pytorch_ok(self, q_in):
        q_in = np.log(q_in+1e-7)
        X = torch.tensor(self.X, dtype=self.dtype, device=self.device)
        Y = torch.tensor(self.Y, dtype=self.dtype, device=self.device)
        p = torch.tensor(q_in, dtype=self.dtype, device=self.device, requires_grad=True)
        q = torch.tensor(q_in, dtype=self.dtype, device=self.device)
        grid = torch.tensor(self.grid.T, dtype=self.dtype, device=self.device)
        activ = (X @ grid) > 0
        def obj(p, verb=False):
            effneurons = p * grid.T # (d,N) 
            out = activ * (X @ effneurons.T) # (n, d) * (d, N) = (n, N)
            yhat = torch.sum(out, axis=1)
            mseloss = torch.nn.MSELoss()(yhat.flatten(), Y.flatten())
            #kldiv = p * torch.log(p/q) - p + q
            #kldivs = torch.sum(kldiv)/len(p)
            kldivs = torch.nn.KLDivLoss(log_target=True, reduction="batchmean")(q, p)
            if verb:
                print("p,q", p, q)
                print("kldiv", kldivs)
                sys.exit(0)
            return self.tau/self.gamma * mseloss +  kldivs

        #optimizer = torch.optim.AdamW([p], lr=self.adamlr, weight_decay=0)
        #optimizer = torch.optim.SGD([p], lr=self.adamlr, weight_decay=0)
        with torch.no_grad():
            fstloss = obj(p)
            #print("first", p)
        maxit = 100
        bestloss = 1e3
        for i in range(maxit):
            #optimizer.zero_grad()
            p.grad = None
            loss = obj(p)
            loss.backward()
            #optimizer.step()
            with torch.no_grad():
                lossdiff = bestloss - loss.item()
                #if lossdiff < 1e-6:
                #    print("proxf exit on small loss at ite", i)
                #    break
                #print(f"{loss.item():.4f} so.. {lossdiff:.8f}")
                #print(p.grad)
                if torch.norm(p.grad) < 0.1:
                    print("proxf exit on small grad at ite", i)
                    break
                #pg = torch.exp(-p.grad)/p.sum()
                #p = (p*torch.exp(-p.grad))/p.sum()
                print(torch.exp(p))
                assert (torch.exp(p) > 0).all() # yeah 
                #newp= p - self.adamlr / torch.exp(p) * p.grad # assert exp > 0 fails
                #newp= p - self.adamlr * torch.exp(p) * loss increase a bit slower
                #newp= p - self.adamlr * p.grad # loss increase
                #newp = p*torch.exp(-self.adamlr*p.grad)
                newp = p*torch.exp(-self.adamlr*p.grad)
                #newp = p*p.grad
                #newp = newp/newp.sum()
                p.copy_(newp)

                #p.add_(p.grad, alpha=-self.adamlr) # p.add also works.. 
                # but p.add_ really is the same as p - self.lr * p.grad. p.add idk

                #p = p - self.adamlr * p.grad
                #p.add_(ly1g, alpha=1)
                if loss.item() < bestloss:
                    bestloss = loss.item()
            if np.isnan(loss.item()):
                print("nan loss at", i)
                obj(p, verb=True)
                assert False

            #print(f"{i} loss -> {loss.item()}")
        return np.exp(p.cpu().detach().numpy())

import numpy as np
import argparse
import sys
from scipy.optimize import minimize
from scipy.special import kl_div
from scipy.spatial import distance

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import *
import torch

class jko_descent:
    def __init__(self, gamma=1, tau=1, interiter=10, verb=False, tol=1e-6, dtype=torch.float32, device="cpu", adamlr=1e-3, proxf="scipy"):
        self.gamma = gamma
        self.tau = tau
        self.tol = tol
        self.verb = verb
        self.interiter = interiter
        self.dtype = dtype  # float16 does a runtime error in pred
        self.device = device
        self.adamlr = adamlr
        if proxf == "scipy":
            self.proxf = self.prox_f_scipy
        elif proxf == "torch":
            self.proxf = self.prox_f_pytorch
        else:
            raise Exception("You're drunk")

    def load(self, X, Y, ly1, ly2, lr=0, beta=0, grid=None, p=None):
        self.X = X
        self.Y = Y
        self.psigns = np.sign(ly2)
        ly2 = np.abs(ly2)
        if grid is None or p is None:
            norm = ly2.sum()
            #print("ly1", ly1.shape, "its norm", norm.shape)
            self.normori = norm
            self.grid = ly1.T*norm
            #print("grid", self.grid.shape)
            #print(self.grid)
            self.p = ly2/norm
            #print("hopefully..", self.p.sum())
            #print("p", self.p.shape,  self.p)
        else:
            self.normori = 1
            self.grid = grid
            self.p = p

        self.K = getKernel(self.grid, self.gamma)

    def step(self):
        q = self.p
        #print(f"qsum={q.sum():.2f}, q={q.flatten()}")
        #print([w1**2+w2**2 for w1, w2 in self.grid])
        # print("start step?", q.sum())
        proxf = self.proxf
        mynorm = lambda x: np.linalg.norm(x.flatten())

        a, b, u, u1, v, v1, pr = ( np.ones_like(self.p) for _ in range(7) )

        for i in range(self.interiter):
            if self.verb == 1:
                print(f".", end="", flush=True)

            # memory
            a1 = a.copy()
            b1 = b.copy()

            u2 = u1.copy()
            u1 = u.copy()

            v2 = v1.copy()
            v1 = v.copy()

            # odd update 
            a = a1 * u2
            b = q / self.K(a)
            #print(self.K(b).shape)

            u = u2*a1/a
            v = v2*b1/b

            # Violation 
            ConstrOdd = mynorm(a * self.K(b) - pr) / mynorm(q)
            

            # memory
            a1 = a.copy()
            b1 = b.copy()

            u2 = u1.copy()
            u1 = u.copy()

            v2 = v1.copy()
            v1 = v.copy()


            # even update
            b = b1 * v2
            pr = proxf(a1 * u2 * self.K(b))
            a = pr / self.K(b)

            u = u2*a1/a
            v = v2*b1/b


            # violation
            ConstrEven = mynorm(b * self.K(a) - q) / mynorm(q)
           

            if ConstrOdd < self.tol and ConstrEven < self.tol:
                if self.verb == 1:
                    print(f"     early exit after {(i+1)*2} iterations")
                self.p = pr
                return 
        if self.verb == 1:
            print(f"went over {self.interiter} iterations, exit")
        self.p = pr
        return 

    def step(self):
        print("running slimmed down")
        q = self.p
        #print("in", self.p.flatten())
        mynorm = lambda x: np.linalg.norm(x.flatten())

        a, b = ( np.ones_like(self.p) for _ in range(2) )

        for i in range(self.interiter):
            if self.verb == 1:
                print(f".", end="", flush=True)
            #print("before", f"a: {a.sum()}, b:{b.sum()}, p:{self.p.sum()}")
            #print("kb?", self.K(b))
            eps = 0
            b = q / (self.K(a)+eps)
            #print("before", f"a: {a.sum()}, b:{b.sum()}, p:{self.p.sum()}")
            ConstrOdd = mynorm(a * self.K(b) - self.p) / mynorm(q) < self.tol
            kb = self.K(b)
            #print("kb?", kb)
            #print(b.flatten())
            #print(f"in -> {self.K(b).sum():.3f}")
            self.p = self.proxf(a * self.K(b)) # as algo
            #print("before", f"a: {a.sum()}, b:{b.sum()}, p:{self.p.sum()}")
            #print(f"out[{i}]: {self.p.sum()}", "we got this out..")
            # self.p = self.proxf(kb/kb.sum()) # optional forced normalization..
            a = self.p / (self.K(b)+eps)
            ConstrEven = mynorm(b * self.K(a) - q) / mynorm(q) < self.tol
            #print("after", f"a: {a.sum()}, b:{b.sum()}, p:{self.p.sum()}")
           
            if ConstrOdd and ConstrEven:
                if self.verb == 1:
                    print(f"early exit after {i} iterations")
                return


        if self.verb == 1:
            print(f"went over {self.interiter} iterations, exit")
        print("psum before norma:" ,self.p.sum())
        self.p = self.p/self.p.sum()

    def params(self, gridout=False):
        if gridout:
            return self.grid, self.p, self.psigns
        else:
            #ly1 = self.grid * np.sqrt(self.p)
            #ly2 = np.sqrt(self.p)
            ly1 = self.grid.T/self.normori # some attempt at not modifying the scale
            ly2 = self.p*self.normori * self.psigns
            return ly1, ly2

    def loss(self):
        return self.f(self.p)

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

    def prox_f_scipy(self, q):
        step = self.tau/self.gamma
        #print(step)
        def objective(p):
            # annoying facts: p=(N,) q=(N, 1)
            return self.f(p)*step + kl_div(p,q.flatten()).sum()
        bounds = [(0, None) for _ in q] # positivity
        res = minimize(objective , q.flatten(), bounds=bounds)

        return res.x.reshape(-1,1)

    def f(self, p):
        p = p.flatten()
        activ = (self.X @ self.grid.T) > 0 # (n, N)
        effneurons = self.psigns.flatten() * p * self.grid.T # (d,N) 
        out = activ * (self.X @ effneurons) # (n, d) * (d, N) = (n, N)
        yhat = np.sum(out, axis=1)[:, None] # avoid broadcasting..

        return MSEloss(yhat, self.Y, coeff=len(yhat))

def getKernel(grid, gamma):
    dist = distance.pdist(grid, metric="sqeuclidean") #sq-uared
    d = distance.squareform(dist) # get NxN matrix of all distance diffs
    Gibbs = np.exp(-d/gamma)

    def aux(p):  # Gibbs Kernel applied to a vector p 
        return np.dot(Gibbs,p)
    return aux

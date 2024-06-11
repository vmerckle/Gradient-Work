import numpy as np
import datetime

def gethours(timestamp):
    delta = datetime.datetime.now() - datetime.datetime.fromtimestamp(float(timestamp))
    return delta.total_seconds() / 3600

def hoursago(timestamp):
    delta = datetime.datetime.now() - datetime.datetime.fromtimestamp(float(timestamp))
    s = delta.total_seconds()
    return deltatimestring(s)

def deltatimestring(s):
    d = int(s//(60*60*24)) # i know it's available in the timedelta object
    h = int(s//(60*60) - d*24) # shorter
    m = int(s//60 - h*60)
    s = int(s%60)
    ss = ""
    if d > 0:
        return f"{d}d {h}h"
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def gethours(timestamp):
    delta = datetime.datetime.now() - datetime.datetime.fromtimestamp(float(timestamp))
    return delta.total_seconds() / 3600

def hoursago(timestamp):
    delta = datetime.datetime.now() - datetime.datetime.fromtimestamp(float(timestamp))
    s = delta.total_seconds()
    return deltatimestring(s)

def deltatimestring(s):
    d = int(s//(60*60*24)) # i know it's available in the timedelta object
    h = int(s//(60*60) - d*24) # shorter
    m = int(s//60 - h*60)
    s = int(s%60)
    ss = ""
    if d > 0:
        return f"{d}d {h}h"
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def add_bias(x):
    """ add a column of 1 to data to allow for bias """
    if x.ndim == 1:
        x = x[:, None] # (n,) -> (n, 1) so we can stack
    
    return np.hstack((x, np.ones((x.shape[0], 1))))

# attempt to not load torch here
# def torch_2layers(X, ly1, ly2):
    # return torch.relu(X@ly1)@ly2

def np_2layers(X, ly1, ly2):
    #a = X@ly1
    #return ((a>0)*a).dot(ly2) # exactly the same
    return np.maximum(X@ly1, 0) @ ly2

def MSEloss(yhat, y, coeff=None):
    if coeff is None:
        return np.square(yhat-y).sum()/len(yhat)
    return np.square(yhat-y).sum()/coeff

# linear [a,b] ->  [left, right]
def normData(D, key, left, right):
    allki = np.array([di[key] for di in D])
    b, a = np.min(allki), np.max(allki)
    for di in D:
        if (a-b) == 0:
            di[key] = di[key]*0 + (right-left)/2
        else:
            di[key] = (di[key]-b)/(a-b)*(right-left)+left

def focuslim(axis, xl, yl):
    ax, bx = np.min(xl), np.max(xl)
    ay, by = np.min(yl), np.max(yl)
    eps = 1e-1
    axis.set_xlim(ax-eps, bx+eps)
    axis.set_ylim(ay-eps, by+eps)
    #axis.set_xlim(-eps, bx+eps)
    #axis.set_ylim(-eps, by+eps)

def getMotifNow(X, ly1):
    n, d = X.shape
    nl = []
    found = {}

    for w in ly1.T:
        res = (X.dot(w) >= 0).flatten()
        resb = res.tobytes()
        if resb not in found:
            found[resb] = True
            nl.append(res*1.0)
    return nl

def getMotif(X, rng):
    n, d = X.shape
    nl = []
    found = {}

    for i in range(2000):
        w = rng.standard_normal(d)
        res = (X.dot(w) >= 0).flatten()
        resb = res.tobytes()
        if resb not in found:
            found[resb] = True
            nl.append(res*1.0)
    return nl

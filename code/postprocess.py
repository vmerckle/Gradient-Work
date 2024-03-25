import numpy as np
import time
from utils import *

# compute plot information from data and layers
# used in live and post animation
def NNtoIter(Xt, Yt, allX, ly1, ly2, run=False):
    d, m = ly1.shape
    if d == 2:
        #lnorm = np.abs(ly1[1,:].flatten() * ly2.flatten()) # -> slope of the ReLU unit
        #lnorm =np.abs(ly2.flatten()) # -> similar alpha = similar speed. But the first layer's norm also matter...
        lspeed = 1/(1e-10+np.linalg.norm(ly1, ord=2, axis=0)) * np.abs(ly2.flatten())
        lslope = np.abs(ly1[1,:].flatten() * ly2.flatten())
        lnorm = np.linalg.norm(ly1, ord=2, axis=0) * ly2.flatten() # -> mult or addition, just an "idea" of how big the neuron is
        lsize = 1/(lspeed+1) + (40 if run else 0)# slower = bigger
        lsize = ly2.flatten() # size=slope
        lnorm = lslope
        lact = np.array([-w2/w1 if w1 != 0 else 1e20 for w1, w2 in ly1.T])
        Yout = np_2layers(allX, ly1, ly2)
        Yhat = np_2layers(Xt, ly1, ly2)
        signedE = Yhat-Yt
        loss = MSEloss(Yhat, Yt)
        pdirecs = []
        motifs = getMotifNow(Xt, ly1)
        for m in motifs:
            w1, w2 = np.sum(np.atleast_2d(m).T*Xt * signedE , axis=0) # n,d * 10, * 10, =sum> 2
            #print((np.sum(Xt * signedE * m, axis=0)).shape)
            #print(-w2/w1)
            if w1 != 0:
                pdirecs.append(-w2/w1)
                pdirecs.append(w2/w1)
        return {"ly1": ly1, "ly2": ly2, "lact": lact, "lnorm": lnorm, "loss": loss, "Yout": Yout, "lsize": lsize, "signedE":signedE, "pdirecs":np.array(pdirecs)}
    elif d == 1:
        lsize = ly2.flatten() # size=slope
        lnorm = ly1[0, :].flatten()
        lact = np.zeros_like(ly1[0])
        Yout = np_2layers(allX, ly1, ly2)
        Yhat = np_2layers(Xt, ly1, ly2)
        signedE = Yhat-Yt
        loss = MSEloss(Yhat, Yt)
        return {"ly1": ly1, "ly2": ly2, "lact": lact, "lnorm": lnorm, "loss": loss, "Yout": Yout, "lsize": lsize}
    else:
        raise Exception("d>3 has no animation yet")

def simplecalcs(X):
    X, Y, lly1, lly2, rng = [X[x] for x in ["X", "Y", "lly1", "lly2", "rng"]]
    d = X.shape[1]
    allXb = np.linspace(-4,4, 1000)
    if d == 2:
        allX = add_bias(allXb)
    elif d == 1:
        allX = allXb[:, None]
    iterdata = [NNtoIter(X, Y, allX, lly1[i], lly2[i]) for i in range(len(lly1))]
    normData(iterdata, "lnorm", 0, 1) 
    normData(iterdata, "lsize", 1, 100)
    return {"Xout": allXb, "iterdata": iterdata}

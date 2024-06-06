import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib.markers import MarkerStyle 
# https://stackoverflow.com/questions/58089375/question-about-changing-marker-of-scatter-during-animation
# the only god damn correct answer, all others will have bugs.

from utils import *


## beautiful lines don't touch
s = u'$\u2193$'
fun_marker = MarkerStyle(s).get_path().transformed(MarkerStyle(s).get_transform())
pos_marker = MarkerStyle('>').get_path().transformed(MarkerStyle('>').get_transform())
neg_marker = MarkerStyle('<').get_path().transformed(MarkerStyle('<').get_transform())

class LessNiceAnimNotUsed:
    def __init__(self, fig, data, runanim=False):
        # load and check data
        self.Xb, self.Y, self.X = data["Xb"], data["Y"][:, 0], data["X"]
        self.Xout = data["Xout"]
        self.n, self.d = self.X.shape
        self.m = data["ly1"].shape[1]
        assert self.Y.ndim == 1 and len(self.Y) == self.n
        if runanim:
            self.D = None
        else:
            self.D = data["iterdata"]
            self.Nframe = len(self.D)
            # sanity checks on data, looks useless but this is python
            assert np.all([di["ly1"].shape == (self.d, self.m) for di in self.D])
            assert np.all([di["ly2"].shape == (self.m, 1) for di in self.D])
        # plotting setup
        self.fig = fig
        self.ax = self.fig.add_subplot()
        self.ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        self.ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        self.ax.grid(True, alpha=0.2)
        self.ax.scatter(self.Xb, np.ones_like(self.Xb), marker="x", color="red", s=60, alpha=0.5)
        self.signedE = self.ax.scatter([], [], marker="*", color="blue", s=60, alpha=1)
        ## beautiful lines don't touch
        s = u'$\u2193$'
        # artist engaged in animation
        self.text = self.ax.text(0.05, 0.9, f"start frame", transform=self.ax.transAxes)
        self.outline, = self.ax.plot([], [])
        self.allscat = self.ax.scatter([], [], marker='o')

    def update_aux(self, di, frame):
        ly1, ly2 = di["ly1"], di["ly2"]
        lact, lnorm = di["lact"], di["lnorm"]
        loss, Yout = di["loss"], di["Yout"]
        lsize = di["lsize"]
        #ly1g, ly2g = di["ly1g"], di["ly2g"]

        xl, yl, mks = np.zeros(self.m), np.zeros(self.m), []
        sizes, colors = lsize*4, []
        for i in range(self.m):
            w1, w2 = ly1.T[i]
            alpha, = ly2[i]
            xl[i], yl[i] = w1*alpha, w2*alpha
            if alpha > 0:
                colors.append("green")
            else:
                colors.append("red")
            d = fun_marker.transformed(mpl.transforms.Affine2D().rotate_deg(np.arctan(w2/w1)*360/2/np.pi+90))
            mks.append(d)

        self.allscat.set_offsets(np.column_stack((xl, yl)))
        self.allscat.set_paths(mks)
        self.allscat.set_sizes(sizes)
        self.allscat.set_color(colors)
        #self.signedE.set_offsets(np.column_stack((self.Xb, di["signedE"])))
        #self.pdirecs.set_offsets(np.column_stack((di["pdirecs"], -0.1*np.ones(di["pdirecs"].shape))))
        #set_alpha : same way 0-1
        self.text.set_text(f"frame {frame}, loss {loss}")
        #self.outline.set_data(self.Xout, Yout)
        return self.text, self.allscat

    def update(self, frame):
        di = self.D[frame]
        return self.update_aux(di, frame)

    def getAnim(self, interval, blit=True):
        return animation.FuncAnimation(self.fig, self.update, frames=list(range(self.Nframe)), blit=blit, interval=interval)

class LessNiceAnim:
    def __init__(self, fig, data, runanim=False, frames=None):
        # load and check data
        self.Xb, self.Y, self.X = data["Xb"], data["Y"][:, 0], data["X"]
        self.Xout = data["Xout"]
        self.n, self.d = self.X.shape
        self.m = data["ly1"].shape[1]
        assert self.Y.ndim == 1 and len(self.Y) == self.n
        if runanim:
            self.D = None
        else:
            self.D = data["iterdata"]
            self.Nframe = len(self.D)
            # sanity checks on data, looks useless but this is python
            assert np.all([di["ly1"].shape == (self.d, self.m) for di in self.D])
            assert np.all([di["ly2"].shape == (self.m, 1) for di in self.D])
            if frames is None:
                self.frames = list(range(self.Nframe))
            else:
                self.frames = frames
        # plotting setup
        self.fig = fig
        self.ax = self.fig.add_subplot(frameon=False)
        self.ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        self.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        self.ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        self.ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        self.ax.grid(True, alpha=0.2)

        cvx_n1, cvx_n2, cvx_c = [], [], []
        if "ly1s" in data and data["ly1s"] is not None:
            for (w1, w2), alp in zip(data["ly1s"].T, data["ly2s"]):
                cvx_n1.append(w1*alp)
                cvx_n2.append(w2*alp)
                cvx_c.append("violet")

            self.ax.scatter(cvx_n1, cvx_n2, marker="*", color=cvx_c, s=60, alpha=0.9)

        self.ax.scatter(self.Xb, np.ones_like(self.Xb), marker="x", color="red", s=60, alpha=0.5)
        for x in self.Xb:
            self.ax.axline((0,0), xy2=(-1, float(x)), color="C1", alpha=0.3)
        self.signedE = self.ax.scatter([], [], marker="*", color="blue", s=60, alpha=1)
        ## beautiful lines don't touch
        # artist engaged in animation
        self.text = self.ax.text(0.05, 0.9, f"start frame", transform=self.ax.transAxes)
        self.outline, = self.ax.plot([], [])
        self.allscat = self.ax.scatter([], [], marker='o')

    def update_aux(self, di, frame):
        ly1, ly2 = di["ly1"], di["ly2"]
        lact, lnorm = di["lact"], di["lnorm"]
        loss, Yout = di["loss"], di["Yout"]
        lsize = di["lsize"]
        #ly1g, ly2g = di["ly1g"], di["ly2g"]

        xl, yl, mks = np.zeros(self.m), np.zeros(self.m), []
        sizes, colors = lsize*4, []
        for i in range(self.m):
            w1, w2 = ly1.T[i]
            alpha, = ly2[i]
            xl[i], yl[i] = w1*alpha, w2*alpha
            if alpha > 0:
                colors.append("green")
            else:
                #colors.append("green")
                colors.append("red")
            d = fun_marker.transformed(mpl.transforms.Affine2D().rotate_deg(np.arctan(w2/w1)*360/2/np.pi+90))
            mks.append(d)

        self.allscat.set_offsets(np.column_stack((xl, yl)))
        self.allscat.set_paths(mks)
        self.allscat.set_sizes(sizes)
        self.allscat.set_color(colors)
        #self.signedE.set_offsets(np.column_stack((self.Xb, di["signedE"])))
        #self.pdirecs.set_offsets(np.column_stack((di["pdirecs"], -0.1*np.ones(di["pdirecs"].shape))))
        #set_alpha : same way 0-1
        self.text.set_text(f"Step: {frame}, loss={loss:.4f}")
        #self.outline.set_data(self.Xout, Yout)
        return self.text, self.allscat

    def update(self, frame):
        di = self.D[frame]
        return self.update_aux(di, frame)

    def getAnim(self, interval, blit=True):
        return animation.FuncAnimation(self.fig, self.update, frames=self.frames, blit=blit, interval=interval)

class oldNiceAnim:
    def __init__(self, fig, data, runanim=False, frames=None):
        self.fig = fig
        # load data
        self.Xb, self.Y, self.X = data["Xb"], data["Y"][:, 0], data["X"]
        self.Xout = data["Xout"]
        self.n, self.d = self.X.shape
        self.m = data["ly1"].shape[1]
        assert self.Y.ndim == 1 and len(self.Y) == self.n
        if runanim:
            self.D = None
        else:
            self.D = data["iterdata"]
            self.Nframe = len(self.D)

            if frames is None:
                self.frames = list(range(self.Nframe))
            else:
                self.frames = frames
            # sanity checks on data, looks useless but this is python
            assert np.all([di["ly1"].shape == (self.d, self.m) for di in self.D])
            assert np.all([di["ly2"].shape == (self.m, 1) for di in self.D])
        # plotting setup
        self.ax = self.fig.add_subplot()
        self.ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        self.ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        self.ax.grid(True, alpha=0.2)
        self.ax.scatter(self.Xb, self.Y, marker="x", color="red", s=40, alpha=1)
        ## not very useful but works
        #self.signedE = self.ax.scatter([], [], marker="*", color="blue", s=60, alpha=1)
        #self.pdirecs= self.ax.scatter([], [], marker="^", color="blue", s=60, alpha=1)
        ## 
        # artist engaged in animation
        self.text = self.ax.text(0.05, 0.9, f"start frame", transform=self.ax.transAxes)
        self.outline, = self.ax.plot([], [])
        self.allscat = self.ax.scatter([], [], marker='o')

    def update_aux(self, di, frame):
        ly1, ly2 = di["ly1"], di["ly2"]
        lact, lnorm = di["lact"], di["lnorm"]
        loss, Yout = di["loss"], di["Yout"]
        lsize = di["lsize"]
        #ly1g, ly2g = di["ly1g"], di["ly2g"]

        xl, yl, mks = np.zeros(self.m), np.zeros(self.m), []
        sizes, colors = lsize, []
        for i in range(self.m):
            w1, w2 = ly1.T[i]
            alpha, = ly2[i]
            xl[i], yl[i] = lact[i], 0.5
            xl[i], yl[i] = lact[i], min(1, max(alpha*20, 0.1))
            xl[i], yl[i] = lact[i], lnorm[i]*10#*100
            if alpha > 0:
                colors.append("green")
            else:
                colors.append("red")
            if w1 > 0:
                mks.append(pos_marker)
            else:
                mks.append(neg_marker)

        self.allscat.set_offsets(np.column_stack((xl, yl)))
        self.allscat.set_paths(mks)
        self.allscat.set_sizes(sizes)
        self.allscat.set_color(colors)
        ## not very useful but works
        #self.signedE.set_offsets(np.column_stack((self.Xb, di["signedE"])))
        #self.pdirecs.set_offsets(np.column_stack((di["pdirecs"], -0.1*np.ones(di["pdirecs"].shape))))
        ## 
        #set_alpha : same way 0-1
        self.text.set_text(f"frame {frame}, loss {loss}")
        self.outline.set_data(self.Xout, Yout)
        return self.outline, self.text, self.allscat#, self.signedE, self.pdirecs

    def update(self, frame):
        di = self.D[frame]
        return self.update_aux(di, frame)

    def getAnim(self, interval, blit=True):
        return animation.FuncAnimation(self.fig, self.update, frames=self.frames, blit=blit, interval=interval)


class NiceAnim:
    def __init__(self, fig, data, runanim=False, frames=None):
        self.fig = fig
        # load data
        self.Xb, self.Y, self.X = data["Xb"], data["Y"][:, 0], data["X"]
        self.Xout = data["Xout"]
        self.n, self.d = self.X.shape
        self.m = data["ly1"].shape[1]
        assert self.Y.ndim == 1 and len(self.Y) == self.n
        if runanim:
            self.D = None
        else:
            self.D = data["iterdata"]
            self.Nframe = len(self.D)

            if frames is None:
                self.frames = list(range(self.Nframe))
            else:
                self.frames = frames
            # sanity checks on data, looks useless but this is python
            assert np.all([di["ly1"].shape == (self.d, self.m) for di in self.D])
            assert np.all([di["ly2"].shape == (self.m, 1) for di in self.D])
        # plotting setup
        self.ax = self.fig.add_subplot()
        #self.ax.xaxis.set_major_locator(plt.MaxNLocator(5)) # only 5 ticks
        #self.ax.yaxis.set_major_locator(plt.MaxNLocator(5)) # it's just cleaner
        self.ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        self.ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        self.ax.grid(True, alpha=0.2)
        self.ax.scatter(self.Xb, self.Y, marker="x", color="red", s=40, alpha=1)

        ## not very useful but works
        #self.signedE = self.ax.scatter([], [], marker="*", color="blue", s=60, alpha=1)
        #self.pdirecs= self.ax.scatter([], [], marker="^", color="blue", s=60, alpha=1)
        ## 
        # artist engaged in animation
        self.text = self.ax.text(0.05, 0.9, f"start frame", transform=self.ax.transAxes)
        self.outline, = self.ax.plot([], [])
        self.allscat = self.ax.scatter([], [], marker='o')

    def update_aux(self, di, frame):
        ly1, ly2 = di["ly1"], di["ly2"]
        d = ly1.shape[0]
        lact, lnorm = di["lact"], di["lnorm"]
        loss, Yout = di["loss"], di["Yout"]
        lsize = di["lsize"]
        #ly1g, ly2g = di["ly1g"], di["ly2g"]

        xl, yl, mks = np.zeros(self.m), np.zeros(self.m), []
        sizes, colors = lsize, []
        for i in range(self.m):
            if d == 2:
                w1, w2 = ly1.T[i]
            elif d == 1:
                w1 = ly1[0][i]
            alpha, = ly2[i]
            xl[i], yl[i] = lact[i], 0.5
            xl[i], yl[i] = lact[i], min(1, max(alpha*20, 0.1))
            xl[i], yl[i] = lact[i], lnorm[i]*10#*100
            xl[i], yl[i] = lact[i], ly1.flatten()[i]/np.max(ly1) # for jko
            if alpha >= 0:
                colors.append("green")
            else:
                colors.append("red")
            if w1 >= 0:
                mks.append(pos_marker)
            else:
                mks.append(neg_marker)

        self.allscat.set_offsets(np.column_stack((xl, yl)))
        self.allscat.set_paths(mks)
        self.allscat.set_sizes(sizes)
        self.allscat.set_color(colors)
        ## not very useful but works
        #self.signedE.set_offsets(np.column_stack((self.Xb, di["signedE"])))
        #self.pdirecs.set_offsets(np.column_stack((di["pdirecs"], -0.1*np.ones(di["pdirecs"].shape))))
        ## 
        #set_alpha : same way 0-1
        self.text.set_text(f"frame {frame}, loss {loss}")
        self.outline.set_data(self.Xout, Yout)
        return self.outline, self.text, self.allscat#, self.signedE, self.pdirecs

    def update(self, frame):
        di = self.D[frame]
        return self.update_aux(di, frame)

    def getAnim(self, interval, blit=True):
        return animation.FuncAnimation(self.fig, self.update, frames=self.frames, blit=blit, interval=interval)

class WasserNiceAnim:
    def __init__(self, fig, data, runanim=False, frames=None):
        self.fig = fig
        # load data
        self.Xb, self.Y, self.X = data["Xb"], data["Y"][:, 0], data["X"]
        self.Xout = data["Xout"]
        self.n, self.d = self.X.shape
        self.m = data["ly1"].shape[1]
        assert self.Y.ndim == 1 and len(self.Y) == self.n
        if runanim:
            self.D = None
        else:
            self.D = data["iterdata"]
            self.Nframe = len(self.D)

            if frames is None:
                self.frames = list(range(self.Nframe))
            else:
                self.frames = frames
            # sanity checks on data, looks useless but this is python
            assert np.all([di["ly1"].shape == (self.d, self.m) for di in self.D])
            assert np.all([di["ly2"].shape == (self.m, 1) for di in self.D])
        # plotting setup
        self.ax = self.fig.add_subplot(111)
        self.ax2 = self.ax.twinx().twiny()

        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.xaxis.set_major_locator(plt.MaxNLocator(3)) # set number of ticks
        self.ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        self.ax2.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        self.ax2.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        self.ax2.grid(True, alpha=0.2)


        # Initialization of animation artists
        self.text = self.ax.text(0.05, 0.9, f"start frame", transform=self.ax.transAxes)
        self.allscat = self.ax.scatter([], [], marker='o')
        self.ax2.scatter(self.Xb, self.Y, marker="x", color="red", s=40, alpha=1)
        self.outline, = self.ax2.plot([], [])
        #self.ax.set_yscale("symlog")
        #self.ax.set_xscale("symlog")
        a = 1.0
        self.ax.set_xlim(-a, a)
        self.ax.set_ylim(-a, a)
        a = 1.0
        self.ax2.set_xlim(-a*1, a*1)
        self.ax2.set_ylim(-a*2, a*5)

    # update only depends on the data of the i-th iteration
    # so that it can be used for live data
    def update_aux(self, di, frame):
        ly1, ly2 = di["ly1"], di["ly2"]
        d = ly1.shape[0]
        lact, lnorm = di["lact"], di["lnorm"]
        loss, Yout = di["loss"], di["Yout"]
        lsize = di["lsize"]

        xl, yl, mks = np.zeros(self.m), np.zeros(self.m), []
        colors = []
        for j in range(self.m):
            w1, w2 = ly1.T[j]
            alpha, = ly2[j]
            xl[j], yl[j] = lact[j], 0.5*ly2.flatten()[j]#/np.max(ly1) # for jko
            xl[j], yl[j] = w1, w2
            colors.append("green" if alpha >= 0 else "red")
            mks.append(pos_marker if w1 >= 0 else neg_marker)

        self.allscat.set_offsets(np.column_stack((xl, yl)))
        self.allscat.set_paths(mks)
        self.allscat.set_sizes([20]*self.m)
        self.allscat.set_color(colors)
        self.allscat.set_alpha(0.5)
        self.text.set_text(f"frame {frame}, loss {loss}")
        self.outline.set_data(self.Xout, Yout)
        #focuslim(self.ax, xl, yl)
        return self.outline, self.text, self.allscat

    def update(self, frame):
        di = self.D[frame]
        return self.update_aux(di, frame)

    def getAnim(self, interval, blit=True):
        return animation.FuncAnimation(self.fig, self.update, frames=self.frames, blit=blit, interval=interval)

def niceplots(wasserstats):
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(frameon=False)

    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_xlabel('gradient step', loc='center')
    ax.set_ylabel('values', loc='center')
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
    ax.grid(True, alpha=0.2)
    ax.set_yscale("log")
    A, B, C = wasserstats["obj"], wasserstats["wasser"], wasserstats["ldeuxdist"]
    D = np.abs(np.array(B)-np.array(C))
    #cc = list(range(len(tloss)))
    #print(len(tloss))
    ax.plot(A, label="Objective", color="C3")
    ax.plot(B, label="Wasser_2^2", color="C4")
    ax.plot(C, label="L_2^2", color="C5")
    #ax.plot(D, label="|Frob-Wasser|", color="C5")
    plt.legend()

    #plt.savefig(f"{codemov}_plot.png", dpi=400)
    plt.show()

import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib.markers import MarkerStyle 
# https://stackoverflow.com/questions/58089375/question-about-changing-marker-of-scatter-during-animation
# the only god damn correct answer, all others will have bugs.

from utils import *

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
        self.fun_marker = MarkerStyle(s).get_path().transformed(MarkerStyle(s).get_transform())
        self.pos_marker = MarkerStyle('>').get_path().transformed(MarkerStyle('>').get_transform())
        self.neg_marker = MarkerStyle('<').get_path().transformed(MarkerStyle('<').get_transform())
        # artist engaged in animation
        self.text = self.ax.text(0.05, 0.9, f"start frame", transform=self.ax.transAxes)
        self.outline, = self.ax.plot([], [])
        self.allscat = self.ax.scatter([], [], marker='o')
        # administrative stuff
        self.starttime = time.time()
        self.verbose = False

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
            d = self.fun_marker.transformed(mpl.transforms.Affine2D().rotate_deg(np.arctan(w2/w1)*360/2/np.pi+90))
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
        if data["ly1s"] is not None:
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
        s = u'$\u2193$'
        self.fun_marker = MarkerStyle(s).get_path().transformed(MarkerStyle(s).get_transform())
        self.pos_marker = MarkerStyle('>').get_path().transformed(MarkerStyle('>').get_transform())
        self.neg_marker = MarkerStyle('<').get_path().transformed(MarkerStyle('<').get_transform())
        # artist engaged in animation
        self.text = self.ax.text(0.05, 0.9, f"start frame", transform=self.ax.transAxes)
        self.outline, = self.ax.plot([], [])
        self.allscat = self.ax.scatter([], [], marker='o')
        # administrative stuff
        self.starttime = time.time()
        self.verbose = False

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
                colors.append("green")
                #colors.append("red")
            d = self.fun_marker.transformed(mpl.transforms.Affine2D().rotate_deg(np.arctan(w2/w1)*360/2/np.pi+90))
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

class NiceAnim:
    def __init__(self, fig, data, runanim=False):
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
        ## beautiful lines don't touch
        self.pos_marker = MarkerStyle('>').get_path().transformed(MarkerStyle('>').get_transform())
        self.neg_marker = MarkerStyle('<').get_path().transformed(MarkerStyle('<').get_transform())
        # artist engaged in animation
        self.text = self.ax.text(0.05, 0.9, f"start frame", transform=self.ax.transAxes)
        self.outline, = self.ax.plot([], [])
        self.allscat = self.ax.scatter([], [], marker='o')
        # administrative stuff
        self.starttime = time.time()
        self.verbose = False

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
            xl[i], yl[i] = lact[i], lnorm[i]*100
            if alpha > 0:
                colors.append("green")
            else:
                colors.append("red")
            if w1 > 0:
                mks.append(self.pos_marker)
            else:
                mks.append(self.neg_marker)

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
        return animation.FuncAnimation(self.fig, self.update, frames=list(range(self.Nframe)), blit=blit, interval=interval)

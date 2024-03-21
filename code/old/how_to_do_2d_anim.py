# for 2d networks
class NiceAnim:
    def __init__(self, fig, data):
        self.fig = fig
        self.Xb, self.Y, self.X = data["Xb"], data["Y"][:, 0], data["X"]
        self.Xout, self.Nout = data["Xout"], data["Nout"]
        self.outleft, self.outright = data["leftout"], data["rightout"]
        self.D = data["iterdata"]
        self.xboth = data["xboth"]
        self.Nframe = len(self.D)
        # plotting setup
        self.ax = self.fig.add_subplot()
        self.ax.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        self.ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=0.4)
        self.ax.grid(True, alpha=0.2)
        self.dat = self.ax.scatter(self.Xb[:, 0], self.Xb[:, 1], marker="x", color=["red" if y < 0 else "cyan" for y in self.Y], s=40, alpha=1)
        Yout = self.D[0]["Yout"]
        self.im = self.ax.imshow(Yout.reshape(self.Nout, self.Nout).T, extent=[self.outleft, self.outright, self.outleft, self.outright], alpha=1.0, cmap=mpl.cm.coolwarm)

        #self.im.set_clim(min(np.min(Yout),-1), max(1, np.max(Yout))) # blit must be disabled for colormap to auto update
        self.im.set_clim(-1.1, 1.1) # blit must be disabled for colormap to auto update
        # artist engaged in animation
        self.text = self.ax.text(0.05, 0.9, f"start frame", transform=self.ax.transAxes)
        # administrative stuff
        self.starttime = time.time()
        self.verbose = False
        self.lines = []
        nb_neurons = len(self.D[0]["slopeY"])
        # plot the lines
        for yy in self.D[0]["slopeY"]:
            x, = self.ax.plot(self.xboth, yy, color="green", alpha=2/np.sqrt(nb_neurons))
            self.lines.append(x)
        div = make_axes_locatable(self.ax)
        cax = div.append_axes('right', '5%', '5%')
        self.colorbar = fig.colorbar(self.im, cax=cax)
        #self.title = self.ax.set_title('Frame 0')
    def update(self, frame):
        #self.title.set_text(f"Frame {frame}") #blit=false
        di = self.D[frame]
        ly1, ly2 = di["ly1"], di["ly2"]
        loss, Yout = di["loss"], di["Yout"]
        self.text.set_text(f"Frame {frame}, loss {loss}")
        for i, l in enumerate(self.lines):
            l.set_data(di["slopeX"], di["slopeY"][i])
        self.im.set_data(Yout.reshape(self.Nout, self.Nout))#, extent=[-1, 1, -1, 1])
        # self.im.set_clim(np.min(Yout), np.max(Yout)) # blit must be disabled for colormap to auto update
        return self.text, self.im, self.dat, *self.lines

    def getAnim(self, interval, blit=True):
        # blit is important, see update()
        return animation.FuncAnimation(self.fig, self.update, frames=list(range(self.Nframe)), blit=blit, interval=interval)

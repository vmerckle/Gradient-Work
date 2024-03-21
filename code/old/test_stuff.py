def verifopti():
    seed = 2
    rng = np.random.default_rng(seed) # do not use np.random, see https://numpy.org/doc/stable/reference/random/generator.html#distributions
    torch.use_deterministic_algorithms(True)
    gpudevice = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if gpudevice != "cpu":
        print(f"Found a gpu: {gpudevice}..")
    device = "cpu"
    m,d,n,lr,beta = 10,2,5,1e-3,0
    Xb = np.linspace(-1, 1, n)[:, None]
    X = add_bias(Xb)
    Y = Xb*1.3
    scaling = np.sqrt(1/m)*100
    ly1 = rng.uniform(-scaling, scaling, size=(d, m))
    ly2 = rng.uniform(-scaling, scaling, size=(m, 1))
    print(f"X: {X.shape}, Xb: {Xb.shape}, Y: {Y.shape}")
    print(f"ly1: {ly1.shape}, ly2: {ly2.shape}")
    opti = torch_descent(device=device)
    opti.load(X, Y, ly1, ly2, lr, beta)
    print(f"Initial loss from opti {opti.loss()}, our MSE loss: {MSEloss(opti.pred(X), Y)}")
    opti.step()
    nly1, nly2 = opti.params()
    print(f"nly1: {nly1.shape}, nly2: {nly2.shape}")
    ly1g, ly2g = opti.grads()
    print(f"ly1g: {ly1g.shape}, ly2g: {ly2g.shape}")
    Yhat = opti.pred(X)
    print(f"Y: {Y.shape}, Yhat: {Yhat.shape}")
    print(f"Loss after one step from opti {opti.loss()}, our MSE loss: {MSEloss(Yhat, Y)}")

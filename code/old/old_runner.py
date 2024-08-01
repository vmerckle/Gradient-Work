# run without animation
def run(setupDict):
    setup = SimpleNamespace(**setupDict)
    lly1, lly2 = [setup.ly1], [setup.ly2]
    opti = setup.opti
    num = 0
    print("it=", num, "loss=", opti.loss())
    try:
        while True:
            if setup.steps != -1 and num >= setup.steps:
                break
            num += 1
            opti.step()
            nly1, nly2 = opti.params()
            lly1.append(nly1)
            lly2.append(nly2)
            l = opti.loss()
            print("it=", num, "loss=", l)
    except KeyboardInterrupt:
        print("Normal interrupt at num=", num)

    return {"lly1":lly1, "lly2":lly2}

# run without animation
def simpleRun(setupDict):
    setup = SimpleNamespace(**setupDict)
    lly1, lly2 = [setup.ly1], [setup.ly2]
    opti = setup.opti
    X, Y = setup.X, setup.Y
    d, m = setup.ly1.shape

    num = 0
    bestloss = opti.loss()
    print("it=", num, "loss=", opti.loss())
    #print("layer1", ",".join([f"{x:.2f}" for x in lly1[-1].flatten()]))
    #print("layer2", ",".join([f"{x:.2f}" for x in lly2[-1].flatten()]))
    try:
        while True:
            num += 1
            if setup.steps != -1 and num >= setup.steps:
                break
            opti.step()
            nly1, nly2 = opti.params()
            lly1.append(nly1)
            lly2.append(nly2)
            l = opti.loss()
            print("it=", num, "loss=", l)
            if m <= 20 and 1:
                #print("layer2", ",".join([f"{x:.2f}" for x in lly2[-1].flatten()]))
                # print("layer2", ",".join([f"{x:.2f}" for x in lly2[-1].flatten()]), f"loss: {l:.4f}, sum {np.sum(lly2[-1]):.4f}")
                pass
            else:
                # print("layer2", ",".join([f"{x:.2f}" for x in lly2[-1].flatten()]), f"loss: {l:.4f}, sum {np.sum(lly2[-1]):.4f}")
                #print(f"{num}: loss: {l:.4f}, sum {np.sum(lly2[-1]):.4f}")
                pass
            #print(f"loss: {l}")
            if l < bestloss:
                bestloss = l
            if l/bestloss > 10:
                pass
                #print(".. completely diverged.")
                #assert False
    except KeyboardInterrupt:
        print("Normal interrupt at num=", num)
    #except Exception as e:
    #   print("Something went really wrong at num=", num)
    #   print(e)

    return {"lly1":lly1, "lly2":lly2}

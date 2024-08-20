config
---
- NNseed: int
- dataseed: int
- typefloat: float32, float64
- threadcount: int
- device: cpu, cuda
- algo: GD, JKO, proxpoint
- algoD (all parameters for algo)
(all)
    - opti: "prodigy", "mechanize", "mechanizeadam", "SGD", "AdamW"
    - momentum: float
(GD)
    - lr: float
    - onlyTrainFirstLayer: bool
(JKO)
    - proxf: "cvxpy", "scipy", "pytorch"
    - tol: float
    - tau: float
    - gamma: float
(proxpoint)
    - dist: "frobenius", "wasser" ot.emd2, "sliced" ot.sliced...
    - gamma: float
    - innerlr: float
    - inneriter: int
    - recordinner: bool
    - recordinnerlayers: bool
- datatype: linear2d, rnglinear, sinus, random, mnist
- Xsampling: uniform, normal
- onlypositives: bool
- Ynoise: float
- beta: float
- scale: float
- m (nb of neurons): int
- d (dimension): int
- n (nb of datapoints): int

apply config
---
- X (n, d) (added bias column)
- Xb (n, d-1)
- Y (n, 1)

default logger
---
- iter DICT-LIST of {}, 0 is initialization, last item is numsteps 
    (opti.params() for every iteraitons)
    - ly1: (d, m)
    - ly2: (m, 1)
    - loss: float
    - innerD (for Prox, if recordinner): DICT-LIST of {}
        (empty for iter=0)
        - obj: float
        - dist (1/gamma dist): float
        - loss (obj+dis): float
        - ly1 (if recordinnerlayers): (d, m)

postprocess
---
- Xout: (1000,)
- iterdata LIST of {}
    - ly1: (d, m)
    - ly2: (m, 1)
    - lact
    - lnorm
    - loss
    - Yout
    - lsize
    - signedE
    - pdirecs

meta
---
- timetaken (seconds)
- numsteps
- timestamp (datetime.timestamp() x 1000)
- timestart (_time.time() )
- _lastprint
- _lastsave

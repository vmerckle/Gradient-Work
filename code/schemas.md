default config
---
- NNseed: int
- dataseed: int
- typefloat: float32, float64
- threadcount: int
- device: cpu, cuda
- algo: GD, JKO, proxpoint
- proxf (for JKO): cvxpy, scipy, pytorch
- proxD (for proxpoint)
    - dist: "frobenius", "wasser" ot.emd2, "sliced" ot.sliced...
    - opti: "prodigy", "mechanize", "mechanizeadam", "SGD", "AdamW"
    - innerlr
    - inneriter
    - gamma
- datatype: linear2d, rnglinear, sinus, random
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
- iter LIST of {}, 0 is initialization, last item is numsteps 
    - ly1
    - ly2

postprocess
---
- Xout: (1000,)
- iterdata LIST of {}
    - ly1
    - ly2
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

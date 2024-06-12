setup:
- seed
- typefloat
- device
- algo
- proxdist
- gamma
- inneriter
- lr
- beta
- scale
- m
- d
- n

descent
- lly1 list of (d, m)
- lly2 list of (m, 1)

postprocess
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
- config (just the name)
- steps
- timetaken (seconds)

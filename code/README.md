- *better_runner.py*: 1d animation runner

Example run: 

start by a real time animation

    python better_runner.py --algo jko --jkosteps 100 --jkogamma 0.001 --jkotau 0.00001 --runanim --noanim

all data has been saved, now run a smoother animation on the same data:

    python better_runner.py -kr

encode animation in "bestrun_movie.mp4"

    python better_runner.py -kr --movie --ouput bestrun

all options:

      -h, --help            show this help message and exit
      -k, --keepfirst       keep first step
      -r, --keepsecond      keep second step
      --run                 ignore steps number and run until interrupted
      --steps STEPS         how many iterations to optimize the network
      --noanim              do not show the end animation
      --runanim             show a real time animation, enables option 'run' as well
      -m, --movie           save movie
      -o OUTPUT, --output OUTPUT
                            output name
      --verbose
      --algo {torch,jko}
      -lr LR                learning rate for algo='torch'
      --jkosteps JKOSTEPS   algo=jko, number of internal iterations
      --jkogamma JKOGAMMA   algo=jko, float
      --jkotau JKOTAU       algo=jko, float
      --proxf {scipy,torch}
                            algo=jko, how to compute the prox
      --adamlr ADAMLR       algo=jko, proxf=torch, learning rate for gradient descent


- *torch_descent.py*: pytorch implem of 2 layer ReLU
- *jko_descent.py*: mean-field discretization using JKO
- *anim_2d_classif.py*: algo visualization  with 2d data 
- *utils.py*: helper functions and such

### obsolete files
- *old_jko/*: (OLD) mean-field discretization using JKO -> all code in jko_descent.py
- *anim_1d_allneurons.py*: algo visualization  with 1d data  -> recoded in better_runner.py

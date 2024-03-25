# user interface: main.py

options:

    -h, --help            show this help message and exit
    --verbose
    --seed SEED           seed
    --config {config2d_new,config2d_new_grid,config2d_new_grid_wasser,config1d_new}
                          config name
    -o OUTPUT, --output OUTPUT
                          output name
    -k, --keepfirst       reload descent
    -r, --keepsecond      reload descent & postprocess
    --run                 ignore steps number and run until interrupted
    --steps STEPS         how many iterations to optimize the network
    --noanim              do not show the end animation
    --runanim             show a real time animation, enables option 'run' as well
    --anim {output+neurons,dataspace,dataspaceb}
                          what animation
    --movie               save movie
    --movieout MOVIEOUT   output movie name
    --fps FPS             movie fps
    --skiptoseconds SKIPTOSECONDS
                        maximum time in seconds, will skip frame to match

## --config?

- *config.py*: configuration are python files (contain algo choice, data setup, hyperparameters...)

## sliced wasserstein

- *algo_wasserstein.py*: 1D, projections.. c/c as of now

## "library" files

- *runner.py*: different loops (animation, loss display...)
- *postprocess.py*: compute indicators and data for display purposes
- *animations.py*: 2d, 1d animations
- *utils.py*: helper functions and such
- *algo_GD_torch.py*: pytorch implem of 2 layer ReLU gradient descent
- *algo_convex_cvxpy.py*: 2 layer ReLU gradient descent convex solver

## jko solvers
- *algo_jko.py*: mean-field discretization using JKO, replace Wasserstein proxf by kl_div proxf.
- *jko_proxf_scipy.py*: proxf scipy solver
- *jko_proxf_cvxpy.py*: proxf cvxpy solver
- *jko_proxf_pytorch.py*: proxf pytorch gradient descent solver

## obsolete files
- *old/\**: some are useful code snippets

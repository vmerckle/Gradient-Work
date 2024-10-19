# Gradient Work - Neural Network Analysis Framework for Research

![](/imgs/extra3.png)

This repository contains a standalone research framework for short to long (+20 hours) experiments, targeted at iterative algorithms like gradient descent.

As well as three different algorithms for ReLU Shallow Networks training.

- Convex Reformulation, as explained in my [blog post @ ICLR2024](https://iclr-blogposts.github.io/2024/blog/hidden-convex-relu/), following the work [Neural Network are Convex regularizers](https://arxiv.org/abs/2002.10553) (Pilanci, M. and Ergen, T.,).
- Wasserstein Gradient Flow Discretization
    - As a proximal point algorithm, on shallow networks described in [Chizat, L. and Bach, F., NIPS2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/a1afc58c6ca9540d057299ec3016d726-Abstract.html)
    - As a JKO-step on a fixed grid, a variant of the sinkhorn algorithm [Peyré, G., SIAM2015](https://epubs.siam.org/doi/10.1137/15M1010087).

To do so, set what should be saved during the experiments and how often, when to stop the experiment and what to log in real time. Each experiment will be stored along with all the parameters in a file for future analysis and exploitation.

## Example

![](/imgs/extra1.png)

Once it's finished, we can check the file that has been created.

![](/imgs/extra2.png)

Run *plot.py* without arguments to create plots for the latest experiment.

![](/imgs/extra4.png)

## Experiments and helper files

- *config.py*: configuration are python files (contain algo choice, data setup, hyperparameters...)
- *runner.py*: different loops (animation, loss display...)
- *postprocess.py*: compute indicators
- *utils.py*: helper functions and such

This project uses but does not depend on pytorch, cvxpy. However it depends on NumPy.

# Implemented Algorithms

## Gradient Descent

- *algo_GD_torch.py*: pytorch implem of 2 layer ReLU gradient descent

## Convex Reformulation

- *algo_convex_cvxpy.py*: 2 layer ReLU gradient descent convex solver

## Wasserstein Gradient Flow Simulation

### Proximal and Wasserstein Descent

- *algo_prox.py*: Proximal Point
- *proxdistance.py* : implements Frobenius, Wasserstein, Sliced Wasserstein distances

### JKO-step and Proximal Solvers

- *algo_jko.py*: Mean-field discretization using JKO, replace Wasserstein proxf by kl_div proxf.
- *jko_proxf_scipy.py*: Proximal Scipy solver
- *jko_proxf_cvxpy.py*: Proximal Cvxpy solver
- *jko_proxf_pytorch.py*: Proximal solver using pytorch gradient descent

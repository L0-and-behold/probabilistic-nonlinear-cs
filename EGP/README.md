# EGP - Exact Gradient Pruning

This repository contains an implementation of Exact Gradient Pruning (EGP), introduced in our article *Probabilistic and Nonlinear Compressive Sensing*, as well as code for benchmarking EGP against other compressive sensing techniques. 

It efficiently finds good local optima of the following non-convex objective,

$$ 
L_\lambda(\theta) = ||y-F\theta||_2^2 + \lambda_0 \ell_0(\theta)+ \lambda_1 \ell_1(\theta),
$$

by employing a probabilistic reformulation that allows gradient descent based optimization that is drastically faster than Monte Carlo based methods, while outperforming conventional compressive sensing algorithms in terms of result quality on a wide range of settings and signal-to-noise-ratios.

EGP is implemented in the high-performance programming language [Julia](https://julialang.org/), using the [Lux.jl](https://lux.csail.mit.edu/stable/) deep learning framework.

## Installation

Clone the repository, move to the EGP folder, open the terminal there and run:
```bash
$ julia --project=. -e "using Pkg; Pkg.instantiate()"
```
This creates a new julia environment called `EGP` and installs all necessary packages for the julia simulations in this folder. If you only want to use EGP, then this is all you need.

If you also want to run the benchmarks that compare EGP to other methods, then some `R` libraries have to be installed as well. If conda is used as package manager, then those can for example be installed as follows:
```bash
$ conda create --name R r-base=4.4.2
$ conda activate R

$ conda install -c conda-forge r-devtools r-glmnet r-ggplot2

# Install bestsubset from GitHub
$ R -e "library(devtools); install_github(repo='ryantibs/best-subset', subdir='bestsubset')"
```
Thereafter move to the `EGP/benchmarks` folder, open `Relaxed-Lasso-FS-IHT.jl` and check that the correct path is set for the R libaries in line 8.

Finally, the Monte Carlo method comparison also requires Python to be installed and the libraries used inside the corresponding Jupyter notebook in `EGP/benchmarks/comparison-with-MonteCarlo-methods/`.

## Usage

Please take a look at the minimal example at `EGP/examples/minimal_example.jl`.


## Benchmarking

The code implements 3 different benchmarks.

1. Comparison of EGP with Monte Carlo based probabilistic pruning methods, see `EGP/benchmarks/comparison-with-MonteCarlo-methods/`
2. Comparison of EGP with other Compressive Sensing algorithms, namely _Lasso_, _Relaxed Lasso_, _Forward Stepwise_ and _IHT_, see `EGP/benchmarks/systematic-compressive-sensing-experiments/`
3. Runtime benchmarks of _EGP_, _Lasso_, _Relaxed Lasso_, _Forward Stepwise_ and _IHT_, see `EGP/benchmarks/runtime-benchmarks/`

To run any of the benchmarks, go to the corresponding folder and execute the corresponding julia file. For the Monte Carlo benchmarks, thereafter run the code in the corresponding notebook that loads the julia generated files.

A batch job submission example file is also provided for the longer simulations (2. and 3.), though amendments are perhaps necessary, depending on your batch job submission system.

To visualize the results, `plot_results.jl` files are provided for simulations 2. and 3., while the jupyter notebook handles visualizations for the Monte Carlo comparisons.


Benchmarking setup and simulation details are described in our publication *Probabilistic and Nonlinear Compressive Sensing*.

### Benchmarks summary

1. EGP performs drastically faster than Monte Carlo based methods.

2. EGP has lower test error and lower active set reconstruction error than Lasso, Relaxed Lasso, IHT and Forward Stepwise over a wide range of settings and signal to noise ratios.

3. EGP computation time scales linearly with sample size and number of variables and is at most a constant factor slower than other compressive sensing methods. It is therefore an excellent choice for large datasets.
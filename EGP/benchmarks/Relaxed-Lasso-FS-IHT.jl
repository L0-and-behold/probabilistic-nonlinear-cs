using RCall

# In order for the following to work, you must have run `install_R_libraries.sh` which does the following, which you can also do manually: It uses miniforge3 conda (or another conda version in which case you have to change the "miniforge3" path below) to create a conda environment called R, and in that environment installs the r-base package and the r-devtools package. Then it opens R in that environment and installs the bestsubset package from this repo https://github.com/ryantibs/best-subset, using the following commands (in R console)
#  library(devtools)
#  install_github(repo="ryantibs/best-subset", subdir="bestsubset")

# Make sure that library pathes are equivalent.
ENV["R_LIBS_USER"] = joinpath(homedir(), "miniconda3", "envs", "R", "lib", "R", "library")
R"""
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
"""

## load the R file
current_dir = @__DIR__
relative_path = "load_bestsubset_library.R"
absolute_path = joinpath(current_dir, relative_path)

R"""
source($absolute_path)
"""

## Define Lasso and Relaxed Lasso and Forward Stepwise using the R library

function Lasso(X, y, Xval, yval; nlambda=50, nrelax=10, seed=42)
    @rput X
    @rput y
    @rput nrelax
    @rput nlambda
    @rput seed

    t0 = time()
    R"""
    set.seed(seed)
    r_lasso <- lasso(X,y,intercept=FALSE, nlambda=nlambda, nrelax=nrelax)
    beta_lasso = as.matrix(coef(r_lasso))
    y_lasso = as.matrix(predict(r_lasso,X))
    """;

    j_beta_lasso = @rget beta_lasso
    j_y_lasso = @rget y_lasso

    val_losses_depending_on_beta_lasso = sum((yval .- Xval * j_beta_lasso).^2, dims=1) ./ length(yval)
    ind_lasso = argmin(val_losses_depending_on_beta_lasso)[2]
    final_beta_lasso = j_beta_lasso[:,ind_lasso]

    t1 = time()
    total_time = t1-t0

    return j_beta_lasso, j_y_lasso, final_beta_lasso, val_losses_depending_on_beta_lasso, ind_lasso, total_time
end
function Relaxed_Lasso(X, y, Xval, yval)
    return Lasso(X, y, Xval, yval; nlambda=50, nrelax=10)
end
function Plain_Lasso(X, y, Xval, yval)
    return Lasso(X, y, Xval, yval; nlambda=500, nrelax=1)
end

function Forward_Stepwise(X, y, Xval, yval)
    @rput X
    @rput y

    t0 = time()
    R"""
    r_stepwise <- fs(X, y, intercept = FALSE, verbose = FALSE)
    beta_stepwise = as.matrix(coef(r_stepwise))
    y_stepwise = as.matrix(predict(r_stepwise,X))
    """;

    j_beta_stepwise = @rget beta_stepwise
    j_y_stepwise = @rget y_stepwise

    val_losses_depending_on_beta_stepwise = sum((yval .- Xval * j_beta_stepwise).^2, dims=1) ./ length(yval)
    ind_stepwise = argmin(val_losses_depending_on_beta_stepwise)[2]
    final_beta_stepwise = j_beta_stepwise[:,ind_stepwise]

    t1 = time()
    total_time = t1-t0

    return j_beta_stepwise, j_y_stepwise, final_beta_stepwise, val_losses_depending_on_beta_stepwise, ind_stepwise, total_time
end

## Next use the MendelIHT library to load the IHT method

using MendelIHT, Random
using Base.Threads
# println("Threads: ", Threads.nthreads())
# println("Total available threads: ", Sys.CPU_THREADS)
function IHT(X, y, Xval, yval; max_k=150)
    
    t0 = time()

    p = size(X)[2]
    max_k = min(max_k,p)

    betas = zeros(p,max_k)
    Threads.@threads for k in 1:max_k
        betas[:,k] = fit_iht(y, X, k=k; verbose=false).beta
    end

    val_losses_depending_on_beta = sum((yval .- Xval * betas).^2, dims=1) ./ length(yval)
    
    ind_best = argmin(val_losses_depending_on_beta)[2]
    final_beta = betas[:,ind_best]
    
    y_predicted = Xval * final_beta

    t1 = time()
    total_time = t1-t0

    return betas, y_predicted, final_beta, val_losses_depending_on_beta, ind_best, total_time
end

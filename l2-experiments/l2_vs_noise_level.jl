"""
Experiment: ℓ₂-Distance vs Noise Level

This script measures the ℓ₂-distance between a trained student network
and its teacher as a function of dataset noise level σ² 
(given a fixed number of training steps).

For each noise level:
1. A fixed teacher network generates noisy training data.
2. The student network is initialized as a (possibly perturbed) copy of the teacher.
3. The student is trained for a fixed number of epochs.
4. ℓ₂-distance after training is recorded.

The experiment explores how the final mismatch depends on noise level,
averaged over multiple random seeds.
"""

"""
Package Loading and Environment Setup

Activates the project environment and loads required dependencies for
neural network training, dataset generation, and plotting.
"""

using Pkg
Pkg.activate(".")

using Flux, ProgressMeter, Plots, Statistics, Random, Revise
using Flux: params, mse, Descent, cpu
using Random: MersenneTwister

include("src/Teacher_nn/Teacher_nn.jl")
include("src/Metrics/Metrics.jl")
include("src/TrainFunctions/TrainFunctions.jl")
include("src/NonDegenerateProjection.jl")
include("src/TrainArgs.jl")

import .Teacher_nn: get_random_teacher, get_dataset
import .Metrics: l2_distance
import .TrainFunctions: vanilla_train!, l1_train!, DRR_train!
import .NonDegenerateProjection: project_onto_F
import .TrainingArguments: TrainArgs

gr() # change plot backend if desired

"""
Experiment Configuration Parameters

This section defines all hyperparameters and σ² levels to be explored.
Adjust these values to reproduce different conditions.
"""

#######
# Experiment Settings
#######
args = TrainArgs()

# Noise levels (logarithmic sweep)
σ_levels = [1e-5 * 1.615^i for i in 0:24]

# Training hyperparameters
learning_rate = 1e-3
initial_permut = 0.0      # Initial perturbation magnitude
seed = 42
epochs = 100
num_runs = 10             # Independent runs per noise level
dataset_size = 10000
projection_frequency = 0 # epochs (0 means no projection)
teacher_dimensions = [2, 25, 25, 1]

opt = Descent(learning_rate) # Optimizer

train_function! = vanilla_train! # vanilla_train!, l1_train!, or DRR_train!
args.α = 0.0001 # ℓ₀- or ℓ₁-regularization strength

"""
Pre-allocation and Teacher Initialization

We create a single teacher network to be used across all experiments
for consistent comparison.
"""
    
title = "ℓ₂_σ² - lr=$(learning_rate), initial_permut=$(initial_permut), epochs=$(epochs), N=$(num_runs)"

l2_results = zeros(Float32, length(σ_levels), num_runs)
rng = MersenneTwister(seed)

teacher = get_random_teacher(teacher_dimensions...; device=cpu, rng=rng)
if projection_frequency != 0
    teacher = project_onto_F(teacher; device=cpu)
end

"""
Main Training Loop

Iterates over noise levels and independent runs. For each combination:
1. Generate noisy dataset from the fixed teacher.
2. Initialize student as perturbed copy of teacher.
3. Train student network for fixed epochs.
4. Record ℓ₂-distance after training.
"""

@showprogress Threads.@threads for (i, j) in collect(Iterators.product(1:length(σ_levels), 1:num_runs))
    noise_σ = σ_levels[i]

    # Generate noisy dataset
    train_set = get_dataset(teacher, dataset_size, dataset_size; noise_σ=noise_σ, device=cpu, rng=rng)

    # Initialize student network
    student = deepcopy(teacher)
    for p in params(student)
        p .+= initial_permut * randn(rng, size(p))
    end

    # Loss function
    loss(x, y) = mse(student(x), y)

    # Training
    for epoch in 1:epochs
        if train_function! == vanilla_train!
            vanilla_train!(loss, student, train_set, opt)
        else
            train_function!(loss, student, train_set, opt, args)
        end

        if projection_frequency != 0 && projection_frequency%epoch == 0
            student = project_onto_F(student; device=cpu)
        end
    end

    # Record ℓ₂-distance
    l2_results[i, j] = l2_distance(student, teacher)
end

"""
Data Analysis and Visualization

Computes mean ℓ₂-distance across runs for each σ² level,
and produces a log-log plot.
"""

l2_avg = mean(l2_results, dims=2)

# Custom color palette

p = plot(
    σ_levels, l2_avg,
    title = title,
    xlabel = "σ²",
    ylabel = "ℓ₂ after training",
    xscale = :log10,
    yscale = :log10,
    legend = false,
    color = :orange,
    linewidth = 2,
    size = (500, 500),
    titlefontsize = 9,
    frame = :box
)

# Tick formatting
xticks!(p, [10.0^-i for i in 0:2:6])
yticks!(p, [10.0^-i for i in -5:2:30])

# Save figure
savefig(title*".svg")

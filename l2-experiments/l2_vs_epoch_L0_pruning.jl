"""
This script investigates the ℓ₂-distance evolution between student and teacher networks
during training as a function of training steps (epochs). The experiment explores scaling
behavior under two conditions:
1. Setting initial_permut to zero (student exactly matches teacher) and training with noisy data
2. Setting initial_permut to positive values and training with zero noise
3. Combinations of both approaches
"""


"""
Package Loading and Environment Setup

Activates the current project environment and loads all necessary dependencies
for neural network training, optimization, and visualization.
"""

using Pkg
Pkg.activate(".")

using ProgressMeter, Plots, Statistics
using Flux: params, mse, Descent, cpu , gpu, Adam
using Random: MersenneTwister

include("src/Teacher_nn/Teacher_nn.jl")
include("src/Metrics/Metrics.jl")
include("src/TrainFunctions/TrainFunctions.jl")
include("src/NonDegenerateProjection.jl")
include("src/TrainArgs.jl")

import .Teacher_nn: get_random_teacher, get_random_student, get_dataset, pad_small_into_big, plot_nn
import .Metrics: l2_distance
import .TrainFunctions: vanilla_train!, DRR_train!, l1_train!
import .NonDegenerateProjection: project_onto_F
import .TrainingArguments: TrainArgs

include("src/TrainFunctions/FineTuning.jl")
import .FineTuning: initialize_mask, update_mask!, apply_mask!

plotlyjs() # change plot backend if desired

using Revise: includet
includet("src/ExperimentHelpers.jl")
using .ExperimentHelpers: normalize_and_average, do_plot

"""
Experiment Configuration Parameters

This section defines all hyperparameters and settings for the student-teacher training experiment.
Modify these values to explore different experimental conditions.

See README.md for a description how to use each parameter.
"""

#######
# Experiment Settings
#######
args = TrainArgs()

learning_rate = 1e-2
noise_σ = 1e-3
initial_permut = 0.0
seed = 42
epochs = 5000
num_runs = 1
dataset_size = 10000
test_set_size = 1000
projection_frequency = 0 # epochs (0 means no projection)
teacher_dimensions = [2, 10, 5, 1]
student_dimensions = [2, 25, 25, 1]

# opt = Descent(learning_rate) #Optimizer
opt = Adam(learning_rate) #Optimizer

train_function! = DRR_train! # vanilla_train!, l1_train!, or DRR_train!
args.α = 0.1 # ℓ₀- or ℓ₁-regularization strength

"""
Pre-allocation and Setup

Initialize data structures and objects that remain constant across all experimental runs.
This includes the optimizer configuration and result storage arrays.
"""

title = "ℓ₂_epoch - lr="*string(learning_rate)*", σ="*string(noise_σ)*", initial_permut="*string(initial_permut)*", epochs="*string(epochs)*", num_runs="*string(num_runs)*", seed="*string(seed)

l2 = zeros(Float32, epochs, num_runs)
train_loss = zeros(Float32, epochs, num_runs)
test_loss = zeros(Float32, epochs, num_runs)

"""
Main Training Loop

Executes the core experimental procedure across multiple independent runs.
Each run involves:
1. Generating a random teacher network and corresponding datasets
2. Initializing a student network (copy of teacher + optional perturbation)
3. Training the student network while tracking metrics
4. Optional projection onto constraint manifolds

The loop is parallelized across runs for computational efficiency.
"""

# @showprogress Threads.@threads for i in 1:num_runs
    i = 1
    current_seed = seed + i
    rng = MersenneTwister(current_seed)

    original_teacher = get_random_teacher(teacher_dimensions..., device=cpu, rng=rng)
    if projection_frequency != 0
        original_teacher = project_onto_F(original_teacher; device=cpu)
    end
    teacher = pad_small_into_big(original_teacher, student_dimensions[2], student_dimensions[3]; device=cpu)

    train_set = get_dataset(teacher, dataset_size, dataset_size; noise_σ=noise_σ, device=cpu, rng=rng)
    test_set = get_dataset(teacher, test_set_size, test_set_size; noise_σ=noise_σ, device=cpu, rng=rng)

    # student = deepcopy(teacher)    
    # for p in params(student)
    #     p .+= initial_permut * randn(rng, size(p))
    # end
    student = get_random_student(student_dimensions..., device=cpu, rng=rng)

    l2_distance(student, teacher)

    plot_nn(original_teacher)
    plot_nn(teacher)
    plot_nn(student)
    
    loss(x, y) = mse(student(x), y)

    for epoch in 1:epochs

        println(epoch, " / ", string(epochs))
        if train_function! == vanilla_train!
            vanilla_train!(loss, student, train_set, opt)
        else
            train_function!(loss, student, train_set, opt, args)
        end

        if projection_frequency != 0 && projection_frequency%epoch == 0
            student = project_onto_F(student; device=cpu)
        end

        l2[epoch, i] = l2_distance(student, teacher)
        train_loss[epoch, i] = mean(loss(x, y) for (x, y) in train_set)
        test_loss[epoch, i] = mean(loss(x, y) for (x, y) in test_set)
    end
# end
# plot_nn(student)
mask = initialize_mask(student; device=cpu)
threshold = 1f-3
update_mask!(threshold, student, mask)
pruned_student = deepcopy(student)
apply_mask!(pruned_student, mask)
plot_nn(pruned_student)

"""
Data Analysis and Visualization

Process the collected data across all runs to compute statistical averages
and generate comprehensive plots showing the evolution of key metrics.
"""

l2_avg = normalize_and_average(l2)
train_loss_avg = normalize_and_average(train_loss)
test_loss_avg = normalize_and_average(test_loss)

plotlyjs()
do_plot(l2_avg, train_loss_avg, test_loss_avg, title)

"""
Results

Save the generated plots to file with descriptive filename containing
all relevant experimental parameters for easy identification and comparison.
"""

savefig(title*".svg")
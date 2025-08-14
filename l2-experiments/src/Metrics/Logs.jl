module Logs

using Flux: params
using Dates
using JSON
using DataStructures

include("../../TrainingTools/TrainingTools.jl")
using .TrainingTools: accuracy

export number_of_small_weights, 
    accuracy_zeroing_small_weights, 
    set_small_weights_zero!

"""
Count the number of parameters with absolute values below specified thresholds.

# Arguments
- `model`: Neural network model
- `ϵ::Vector{Float64}`: Vector of threshold values

# Returns
- `Vector{Int}`: Number of small weights for each threshold
"""
function number_of_small_weights(model, ϵ::Vector{Float64})
    return map(x -> number_of_small_weights(model, x), ϵ)
end

"""
Count the number of parameters with absolute values below a threshold.

# Arguments
- `model`: Neural network model  
- `ϵ::Float64`: Threshold value

# Returns
- `Int`: Number of parameters with |parameter| < ϵ
"""
function number_of_small_weights(model, ϵ::Float64)
    small_weights = 0
    for p in params(model)
        small_weights += sum(abs.(p) .< ϵ)
    end
    return small_weights
end

"""
Evaluate model accuracy after zeroing small weights for multiple thresholds.

# Arguments
- `trained_model`: Trained neural network model
- `ϵ::Vector{Float64}`: Vector of threshold values
- `test_set`: Test dataset

# Returns
- `Vector{Float64}`: Test accuracies for each threshold
"""
function accuracy_zeroing_small_weights(trained_model, ϵ::Vector{Float64}, test_set)
    test_accuracy = []
    for x in ϵ
        push!(test_accuracy, accuracy_zeroing_small_weights(trained_model, x, test_set))
    end
    return test_accuracy
end

"""
Evaluate model accuracy after zeroing small weights below a threshold.

# Arguments
- `trained_model`: Trained neural network model
- `ϵ::Number`: Threshold value
- `test_set`: Test dataset

# Returns
- `Float64`: Test accuracy after pruning small weights
"""
function accuracy_zeroing_small_weights(trained_model, ϵ::Number, test_set)
    model = deepcopy(trained_model)
    set_small_weights_zero!(model, ϵ)
    return accuracy(test_set[1][1], test_set[1][2], model)
end

"""
Set parameters with absolute values below threshold to zero (in-place).

# Arguments
- `model`: Neural network model to modify
- `ϵ::Number`: Threshold value

# Returns
- `Nothing`: Modifies model parameters in-place
"""
function set_small_weights_zero!(model, ϵ::Number)
    for p in params(model)
        p[abs.(p) .< ϵ] .= 0
    end
end

end
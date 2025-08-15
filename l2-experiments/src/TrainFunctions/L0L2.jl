"""
L0L2 submodule for training neural networks with L0 and L0L2 regularization.

Implements regularized training methods as described in https://arxiv.org/abs/2109.05075v3.
All training functions follow the Flux.train! interface style, modifying gradients
according to the regularization terms described in the paper.

# Function Hierarchy
All training functions are special cases of `l0l2_train_FT!`:
- `l0l2_train_FT!`: Full L0L2 regularization with fine-tuning mask
- `l0_train_FT!`: L0 regularization only (ρ=0) with fine-tuning mask  
- `l0l2_train!`: L0L2 regularization without masking
- `l0_train!`: L0 regularization only (ρ=0) without masking

# Exports
## Training Functions
- `l0_train_FT!`: L0 regularized training with fine-tuning mask
- `l0l2_train_FT!`: L0L2 regularized training with fine-tuning mask
- `l0l2_train!`: L0L2 regularized training without masking
- `l0_train!`: L0 regularized training without masking

## Regularization Terms
- `l0l2_regularizer_term`: Combined L0 + L2 regularization gradient term
- `l0_regularizer_term`: L0 regularization gradient term

# Usage
Load functions via `using .TrainFunctions`
"""
module L0L2

import Flux
using Flux.Optimise: Optimiser, update!
using Flux: params, gradient, Chain
using CUDA

using ..TrainingArguments: TrainArgs
using ..FineTuning: initialize_mask, update_mask!, apply_mask!

export l0_train_FT!, l0l2_train_FT!, 
    l0l2_train!, l0_train!,
    l0l2_regularizer_term, l0_regularizer_term

####
# training rules
####

"""
Train a neural network with L0L2 regularization using Flux.train!-like interface.

This is the master function - all other training functions are special cases of this one.

# Arguments
- `loss::Function`: Loss function taking (input, target) and returning scalar loss
- `model::Union{Chain, Flux.Chain}`: Neural network model to train
- `mask::Vector`: Binary mask indicating which weights to freeze at 0.0 during fine-tuning
- `data::Vector{<:Tuple}`: Training data as vector of (input, target) tuples
- `opt`: Flux optimizer (see Flux.Optimise)
- `args`: TrainArgs instance with regularization parameters (α, β, ρ, ϵ, lr, etc.)
"""
function l0l2_train_FT!(loss::Function, model::Union{Chain, Flux.Chain}, mask::Vector, data::Vector{<:Tuple}, opt, args)
    layers = 1:length(params(model))

    α, β, ρ = args.α, args.β, args.ρ
    f(w) = l0l2_regularizer_term(w, α, β, ρ)

    for d in data
        x, y = d
        grads = gradient(() -> loss(x, y), params(model))

        # it is NOT possible to touch all parameters fully vectorized such as grads[params[:]]
        for i in layers
            grads[params(model)[i]] .+= f.(params(model)[i])
        end

        update!(opt, params(model), grads)
        apply_mask!(model, mask)
    end
end

"""
Train a neural network with L0L2 regularization using default training arguments.

# Arguments
- `loss::Function`: Loss function taking (input, target) and returning scalar loss
- `model::Union{Chain, Flux.Chain}`: Neural network model to train
- `mask::Vector`: Binary mask for fine-tuning
- `data::Vector{<:Tuple}`: Training data as vector of (input, target) tuples
- `opt`: Flux optimizer
"""
function l0l2_train_FT!(loss::Function, model::Union{Chain, Flux.Chain}, mask::Vector, data::Vector{<:Tuple}, opt)
    args = TrainArgs()
    l0l2_train_FT!(loss, model, mask, data, opt, args)
end


"""
Train a neural network with L0 regularization only (ρ=0) using default arguments.

# Arguments
- `loss::Function`: Loss function taking (input, target) and returning scalar loss
- `model::Union{Chain, Flux.Chain}`: Neural network model to train
- `mask::Vector`: Binary mask for fine-tuning
- `data::Vector{<:Tuple}`: Training data as vector of (input, target) tuples
- `opt`: Flux optimizer
"""
function l0_train_FT!(loss::Function, model::Union{Chain, Flux.Chain}, mask::Vector, data::Vector{<:Tuple}, opt)
    args = TrainArgs()
    l0_train_FT!(loss, model, mask, data, opt, args)
end 

"""
Train a neural network with L0 regularization only (ρ=0) with custom arguments.

# Arguments
- `loss::Function`: Loss function taking (input, target) and returning scalar loss
- `model::Union{Chain, Flux.Chain}`: Neural network model to train
- `mask::Vector`: Binary mask for fine-tuning
- `data::Vector{<:Tuple}`: Training data as vector of (input, target) tuples
- `opt`: Flux optimizer
- `args`: TrainArgs instance (ρ will be set to 0.0)
"""
function l0_train_FT!(loss::Function, model::Union{Chain, Flux.Chain}, mask::Vector, data::Vector{<:Tuple}, opt, args)
    local_args = deepcopy(args)
    local_args.ρ = 0.0
    l0l2_train_FT!(loss, model, mask, data, opt, local_args)
end

"""
Train a neural network with L0L2 regularization without masking (no fine-tuning).

# Arguments
- `loss::Function`: Loss function taking (input, target) and returning scalar loss
- `model::Union{Chain, Flux.Chain}`: Neural network model to train
- `data::Vector{<:Tuple}`: Training data as vector of (input, target) tuples
- `opt`: Flux optimizer
"""
function l0l2_train!(loss::Function, model::Union{Chain, Flux.Chain}, data::Vector{<:Tuple}, opt)
    args = TrainArgs()
    l0l2_train!(loss, model, data, opt, args)
end 

"""
Train a neural network with L0L2 regularization without masking with custom arguments.

# Arguments
- `loss::Function`: Loss function taking (input, target) and returning scalar loss
- `model::Union{Chain, Flux.Chain}`: Neural network model to train
- `data::Vector{<:Tuple}`: Training data as vector of (input, target) tuples
- `opt`: Flux optimizer
- `args`: TrainArgs instance with regularization parameters
"""
function l0l2_train!(loss::Function, model::Union{Chain, Flux.Chain}, data::Vector{<:Tuple}, opt, args)
    local_args = deepcopy(args)
    mask = initialize_mask(model)
    l0l2_train_FT!(loss, model, mask, data, opt, local_args)
end

"""
Train a neural network with L0 regularization only without masking.

# Arguments
- `loss::Function`: Loss function taking (input, target) and returning scalar loss
- `model::Union{Chain, Flux.Chain}`: Neural network model to train
- `data::Vector{<:Tuple}`: Training data as vector of (input, target) tuples
- `opt`: Flux optimizer
"""
function l0_train!(loss::Function, model::Union{Chain, Flux.Chain}, data::Vector{<:Tuple}, opt)
    args = TrainArgs()
    l0_train!(loss, model, data, opt, args)
end 

"""
Train a neural network with L0 regularization only without masking with custom arguments.

# Arguments
- `loss::Function`: Loss function taking (input, target) and returning scalar loss
- `model::Union{Chain, Flux.Chain}`: Neural network model to train
- `data::Vector{<:Tuple}`: Training data as vector of (input, target) tuples
- `opt`: Flux optimizer
- `args`: TrainArgs instance (ρ will be set to 0.0)
"""
function l0_train!(loss::Function, model::Union{Chain, Flux.Chain}, data::Vector{<:Tuple}, opt, args)
    local_args = deepcopy(args)
    local_args.ρ = 0.0
    l0l2_train!(loss, model, data, opt, local_args)
end

####
# regularization terms
####

"""
Compute L0 regularization gradient term.

# Arguments
- `weight`: Parameter value
- `α`: L0 regularization strength
- `β`: L0 regularization sharpness parameter

# Returns
- Gradient contribution from L0 regularization: α*β*sign(weight)*exp(-β*|weight|)
"""
function l0_regularizer_term(weight, α, β)
    return α*β * sign(weight) * exp(-β * abs(weight))
end

"""
Compute combined L0L2 regularization gradient term.

# Arguments
- `weight`: Parameter value
- `α`: L0 regularization strength  
- `β`: L0 regularization sharpness parameter
- `ρ`: L2 regularization strength

# Returns
- Gradient contribution from L0L2 regularization: 2*ρ*weight + L0_term
"""
function l0l2_regularizer_term(weight, α, β, ρ)
    return 2 * ρ * weight + l0_regularizer_term(weight, α, β)
end


end # module
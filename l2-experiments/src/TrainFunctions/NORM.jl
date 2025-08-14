"""
NORM submodule for layer-wise normalized L0L2 regularized training.

Implements custom training rules for deep neural networks with layer-wise 
normalized regularization as described in https://arxiv.org/abs/2109.05075v3.

# Key Features
- Layer-wise normalization of regularization parameters α and ρ (β remains global)
- Flux.train!-style interface with gradient modification
- Support for fine-tuning with binary masks
- GPU-compatible operations avoiding scalar indexing

# Function Hierarchy
- `NORM_train_FT!`: Full training with fine-tuning mask (master function)
- `NORM_train!`: Training without masking (special case of NORM_train_FT!)

# Exports
## Training Functions
- `NORM_train_FT!`: Layer-normalized L0L2 training with fine-tuning mask
- `NORM_train!`: Layer-normalized L0L2 training without masking
- `NORM_regularizing_terms`: Compute layer-normalized regularization terms

## Helper Functions
- `scalarwise_product`: Element-wise operations on tensors/vectors
- `scalarwise_addition`: Element-wise addition operations
- `vector_of_tensors`: Create tensors from layer index functions
- `scaled_parameters`: Layer-wise parameter scaling
- `biases_follow_weights`: Validate parameter structure
- `weights_follow_biases`: Validate parameter structure

# Usage
Load functions via `using .TrainFunctions`
"""
module NORM

import Flux
using Flux.Optimise: Optimiser, update!
using Flux: params, gradient, Chain, gpu
using CUDA

using ..TrainingArguments: TrainArgs
using ..FineTuning: initialize_mask, update_mask!, apply_mask!

include("norm_helpers.jl")

export NORM_train_FT!, 
    NORM_train!,
    NORM_regularizing_terms, 
    scalarwise_product, 
    scalarwise_addition, 
    vector_of_tensors, 
    scaled_parameters, 
    biases_follow_weights, 
    weights_follow_biases


####
# training rules
####

"""
Train neural network with layer-wise normalized L0L2 regularization and fine-tuning mask.

This is the master training function implementing layer-wise normalized regularization.
Parameters α and ρ are scaled according to layer sizes, while β remains global.

# Arguments
- `loss::Function`: Loss function taking (input, target) and returning scalar loss
- `model::Union{Chain, Flux.Chain}`: Neural network model to train
- `mask::Vector`: Binary mask indicating which weights to freeze during fine-tuning
- `data::Vector{<:Tuple}`: Training data as vector of (input, target) tuples
- `opt`: Flux optimizer
- `args`: TrainArgs instance with regularization parameters (α, β, ρ) and device settings

# Notes
- Pre-computes layer-wise scaling constants for efficiency
- Avoids GPU scalar indexing by working with parameter vectors
- Applies mask after each parameter update to maintain sparsity
"""
function NORM_train_FT!(loss::Function, model::Union{Chain, Flux.Chain}, mask::Vector, data::Vector{<:Tuple}, opt, args)
    layers = 1:length(params(model))

    device, dtype = args.dev, args.dtype
    
    all_parameters = [device(params(model)[i]) for i in 1:length(params(model))] |> device
    Alpha = scaled_parameters(args.α, all_parameters, device=device)
    Rho = scaled_parameters(args.ρ, all_parameters, device=device)
    Two = vector_of_tensors(all_parameters, i -> dtype(2.0), device=device)

    for d in data
        x, y = d
        grads = gradient(() -> loss(x, y), params(model))

        all_parameters = [device(params(model)[i]) for i in 1:length(params(model))] |> device
        
        regularizing_terms = NORM_regularizing_terms(all_parameters, args, (Alpha, Rho, Two))

        for i in layers 
            grads[params(model)[i]] .+= regularizing_terms[i]
        end

        update!(opt, params(model), grads)
        apply_mask!(model, mask)
    end
end

"""
Train neural network with layer-wise normalized L0L2 regularization using default arguments.

# Arguments
- `loss::Function`: Loss function taking (input, target) and returning scalar loss
- `model::Union{Chain, Flux.Chain}`: Neural network model to train
- `mask::Vector`: Binary mask for fine-tuning
- `data::Vector{<:Tuple}`: Training data as vector of (input, target) tuples
- `opt`: Flux optimizer
"""
function NORM_train_FT!(loss::Function, model::Union{Chain, Flux.Chain}, mask::Vector, data::Vector{<:Tuple}, opt)
    args = TrainArgs()
    NORM_train_FT!(loss, model, mask, data, opt, args)
end

"""
Train neural network with layer-wise normalized L0L2 regularization without masking.

# Arguments
- `loss::Function`: Loss function taking (input, target) and returning scalar loss
- `model::Union{Chain, Flux.Chain}`: Neural network model to train
- `data::Vector{<:Tuple}`: Training data as vector of (input, target) tuples
- `opt`: Flux optimizer
- `device`: Target device (default: gpu)
"""
function NORM_train!(loss::Function, model::Union{Chain, Flux.Chain}, data::Vector{<:Tuple}, opt; device=gpu)
    args = TrainArgs()
    args.dev = device
    NORM_train!(loss, model, data, opt, args)
end

"""
Train neural network with layer-wise normalized L0L2 regularization without masking.

# Arguments
- `loss::Function`: Loss function taking (input, target) and returning scalar loss
- `model::Union{Chain, Flux.Chain}`: Neural network model to train
- `data::Vector{<:Tuple}`: Training data as vector of (input, target) tuples
- `opt`: Flux optimizer
- `args`: TrainArgs instance with regularization parameters
"""
function NORM_train!(loss::Function, model::Union{Chain, Flux.Chain}, data::Vector{<:Tuple}, opt, args)
    device = args.dev
    mask = initialize_mask(model, device=device)
    NORM_train_FT!(loss, model, mask, data, opt, args)
end

####
# regularization terms
####

"""
Compute layer-wise normalized L0L2 regularization terms.

# Arguments
- `all_parameters`: Vector of model parameter tensors
- `args`: TrainArgs instance with regularization parameters

# Returns
- `Vector`: Regularization gradient terms for each parameter tensor

# Formula
2ρw + αβ·sign(w)·exp(-β|w|) with layer-wise normalized α and ρ
"""
function NORM_regularizing_terms(all_parameters, args)
    α, β, ρ, dtype = args.α, args.β, args.ρ, args.dtype

    Alpha = scaled_parameters(α, all_parameters)
    Rho = scaled_parameters(ρ, all_parameters)
    Two = vector_of_tensors(all_parameters, i -> dtype(2.0))
    
    return NORM_regularizing_terms(all_parameters, args, (Alpha, Rho, Two))
end

"""
Compute layer-wise normalized L0L2 regularization terms using pre-computed constants.

More efficient version that reuses pre-computed scaling constants across training steps.

# Arguments
- `all_parameters`: Vector of model parameter tensors
- `args`: TrainArgs instance with regularization parameters
- `constants::Tuple`: Pre-computed (Alpha, Rho, Two) scaling tensors

# Returns
- `Vector`: Regularization gradient terms for each parameter tensor

# Formula
2ρw + αβ·sign(w)·exp(-β|w|) with layer-wise normalized α and ρ
"""
function NORM_regularizing_terms(all_parameters, args, constants::Tuple)
    
    Alpha = constants[1]
    Rho = constants[2]
    Two = constants[3]
    β = args.β

    Helper = deepcopy(all_parameters)

    for layer in Helper
        layer .= β .* sign.(layer) .* exp.(-β .* abs.(layer))
    end

    regularizing_terms = scalarwise_addition(
        scalarwise_product( Two, scalarwise_product(Rho, all_parameters) ),
        scalarwise_product(Alpha, Helper)
    )

    return regularizing_terms
end

end#module


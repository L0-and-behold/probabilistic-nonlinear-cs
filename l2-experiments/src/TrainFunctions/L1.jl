"""
L1 submodule for training neural networks with L1 regularization.

Implements L1 (Lasso) regularization for neural network training, following the 
Flux.train! interface style. Provides both fine-tuning (with masking) and 
standard training variants.

# Exports
## Training Functions
- `l1_train_FT!`: L1 regularized training with fine-tuning mask
- `l1_train!`: L1 regularized training without masking

## Regularization Terms
- `l1_regularizer_term`: L1 regularization gradient term

# Usage
Load functions via `using .TrainFunctions`
"""
module L1

import Flux
using Flux.Optimise: Optimiser, update!
using Flux: params, gradient, Chain
using CUDA

using ..TrainingArguments: TrainArgs
using ..FineTuning: initialize_mask, update_mask!, apply_mask!

export l1_train_FT!, l1_regularizer_term, l1_train!

####
# training rules
####

"""
Train a neural network with L1 regularization using Flux.train!-like interface with masking.

# Arguments
- `loss::Function`: Loss function taking (input, target) and returning scalar loss
- `model::Union{Chain, Flux.Chain}`: Neural network model to train
- `mask::Vector`: Binary mask indicating which weights to freeze at 0.0 during fine-tuning
- `data::Vector{<:Tuple}`: Training data as vector of (input, target) tuples
- `opt`: Flux optimizer (e.g., Flux.Adam(0.01))
- `args`: TrainArgs instance with L1 regularization parameter α
"""
function l1_train_FT!(loss::Function, model::Union{Chain, Flux.Chain}, mask::Vector, data::Vector{<:Tuple}, opt, args)
    layers = 1:length(params(model))

    α= args.α
    f(w) = l1_regularizer_term(w, α)

    for d in data
        x, y = d
        grads = gradient(() -> loss(x, y), params(model))

        for i in layers
            grads[params(model)[i]] .+= f.(params(model)[i])
        end

        update!(opt, params(model), grads)
        apply_mask!(model, mask)
    end
end

"""
Train a neural network with L1 regularization without masking.

# Arguments
- `loss::Function`: Loss function taking (input, target) and returning scalar loss
- `model::Union{Chain, Flux.Chain}`: Neural network model to train
- `data::Vector{<:Tuple}`: Training data as vector of (input, target) tuples
- `opt`: Flux optimizer (e.g., Flux.Adam(0.01))
- `args`: TrainArgs instance with L1 regularization parameter α
"""
function l1_train!(loss::Function, model::Union{Chain, Flux.Chain}, data::Vector{<:Tuple}, opt, args)
    layers = 1:length(params(model))

    α= args.α
    f(w) = l1_regularizer_term(w, α)

    for d in data
        x, y = d
        grads = gradient(() -> loss(x, y), params(model))

        for i in layers
            grads[params(model)[i]] .+= f.(params(model)[i])
        end

        update!(opt, params(model), grads)
    end
end

####
# regularization terms
####

"""
Compute L1 regularization gradient term.

# Arguments
- `weight`: Parameter value
- `α`: L1 regularization strength

# Returns
- Gradient contribution from L1 regularization: α*sign(weight)
"""
function l1_regularizer_term(weight, α)
    return α * sign(weight)
end


end # module
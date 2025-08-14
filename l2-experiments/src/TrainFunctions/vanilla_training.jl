"""
Train a neural network without regularization using a Flux.train!-like interface.

Provides standard gradient descent training similar to Flux.train!, but with
a different parsing approach for the loss function and model parameters.

# Arguments
- `loss::Function`: Loss function taking (input, target) and returning scalar loss
- `model::Union{Chain, Flux.Chain}`: Neural network model to train
- `data::Vector{<:Tuple}`: Training data as vector of (input, target) tuples
- `opt`: Flux optimizer

# Notes
- No regularization applied to gradients
- Compatible with all Flux optimizers
- Different parameter parsing than standard Flux.train!
"""
function vanilla_train!(loss::Function, model::Union{Chain, Flux.Chain}, data::Vector{<:Tuple}, opt)
    for d in data
        x, y = d
        grads = gradient(() -> loss(x, y), params(model))
        update!(opt, params(model), grads)
    end
end

"""
Train a neural network without regularization with fine-tuning mask support.

Provides standard gradient descent training with the ability to freeze certain
parameters at zero using a binary mask.

# Arguments
- `loss::Function`: Loss function taking (input, target) and returning scalar loss
- `model::Union{Chain, Flux.Chain}`: Neural network model to train
- `mask::Vector`: Binary mask indicating which weights to freeze at zero
- `data::Vector{<:Tuple}`: Training data as vector of (input, target) tuples
- `opt`: Flux optimizer
- `args`: TrainArgs instance (not used but maintained for interface consistency)

# Notes
- Applies mask after each parameter update to maintain sparsity
- Useful for fine-tuning pre-pruned networks
- No regularization applied to gradients
"""
function vanilla_train_FT!(loss::Function, model::Union{Chain, Flux.Chain}, mask::Vector, data::Vector{<:Tuple}, opt, args)
    layers = 1:length(params(model))

    for d in data
        x, y = d
        grads = gradient(() -> loss(x, y), params(model))
        update!(opt, params(model), grads)
        apply_mask!(model, mask)
    end
end
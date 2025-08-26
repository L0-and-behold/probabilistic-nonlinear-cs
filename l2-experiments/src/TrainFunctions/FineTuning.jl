"""
FineTuning submodule for global magnitude pruning and fine-tuning operations.

This module provides utilities for implementing structured pruning workflows through
binary masking operations on neural network parameters.

# Workflow
1. Initialize a binary mask with `initialize_mask`
2. Perform global magnitude pruning with `update_mask!`
3. Apply mask during training with `apply_mask!` to freeze pruned parameters at zero
4. Use with `TrainFunctions.l0l2_train_FT!` or `TrainFunctions.DRR_train_FT!`

# Exports
- `initialize_mask`: Create initial binary mask for model parameters
- `update_mask!`: Update mask based on magnitude threshold (pruning step)
- `apply_mask!`: Apply mask to zero out pruned parameters

# Usage
Load functions via `using .TrainFunctions`
"""
module FineTuning

import Flux
using Flux: params, gradient, Chain, gpu
using CUDA

export initialize_mask,
    update_mask!, 
    apply_mask!

"""
Initialize a binary mask for all model parameters.

# Arguments
- `model::Union{Chain, Flux.Chain}`: Neural network model
- `device`: Target device (default: gpu)

# Returns
- `Vector`: Binary mask with same structure as model parameters, initialized to `true`
"""
function initialize_mask(model::Union{Chain, Flux.Chain}; device=gpu)
    mask = Any[]
    for (i, p) in enumerate(params(model))
        mask_layer = fill(true, size(Array(p)))
        mask_layer = mask_layer |> device
        push!(mask, mask_layer)
    end
    return mask |> device
end

"""
Update binary mask based on magnitude threshold for global pruning.

# Arguments
- `threshold::AbstractFloat`: Magnitude threshold for pruning
- `model::Union{Chain, Flux.Chain}`: Neural network model
- `mask::Vector`: Existing binary mask to update

# Returns
- `Vector`: Updated mask where `true` indicates parameters to keep (|p| >= threshold)
"""
function update_mask!(threshold::AbstractFloat, model::Union{Chain, Flux.Chain}, mask::Vector)
    for (i, p) in enumerate(params(model))
        mask[i] .= abs.(p) .>= threshold
    end
    return mask
end

"""
Apply binary mask to model parameters, setting masked parameters to zero.

# Arguments
- `model::Union{Chain, Flux.Chain}`: Neural network model to modify
- `mask::Vector`: Binary mask indicating which parameters to keep

# Returns
- `Chain`: Modified model with masked parameters set to zero
"""
function apply_mask!(model::Union{Chain, Flux.Chain}, mask::Vector)
    for (i, p) in enumerate(params(model))
        p .= p .* mask[i]
    end
    return model
end 

end#module
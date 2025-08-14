"""
TrainFunctions module for custom neural network training with regularization.

Provides custom training rules that extend the Flux.train! interface to implement
various regularization techniques, including layer-wise normalized L0L2 regularization
as described in https://arxiv.org/abs/2109.05075v3.
"""
module TrainFunctions

using Flux: gradient, params, update!, Chain

include("../TrainArgs.jl")
using .TrainingArguments

include("FineTuning.jl")
using .FineTuning
export initialize_mask, 
    update_mask!, 
    apply_mask!

include("L0L2.jl")
using .L0L2
export l0_train_FT!, 
    l0l2_train_FT!, 
    l0l2_train!, 
    l0_train!,
    l0l2_regularizer_term, 
    l0_regularizer_term

include("L1.jl")
using .L1
export l1_train_FT!,
    l1_train!,
    l1_regularizer_term

include("NORM.jl")
using .NORM
export NORM_train_FT!, 
    NORM_train!,
    NORM_regularizing_terms, 
    scalarwise_product, 
    scalarwise_addition, 
    vector_of_tensors, 
    scaled_parameters, 
    biases_follow_weights, 
    weights_follow_biases

include("vanilla_training.jl")
export vanilla_train!, vanilla_train_FT!


end#module
"""
Metrics module for neural network analysis and evaluation.

This module provides comprehensive tools for measuring neural network properties including:
- Description length and compression metrics
- Weight analysis and pruning utilities  
- Distance measures between networks
- Model size and sparsity evaluation

# Exports
## Description Length
- `description_length`: Compute minimum description length (MDL) principle

## Compression Metrics  
- `true_byte_size`: Calculate actual memory footprint of model parameters
- `byte_size_compression`: Measure compression ratios

## Weight Analysis
- `number_of_small_weights`: Count parameters below threshold values
- `accuracy_zeroing_small_weights`: Evaluate accuracy after weight pruning
- `set_small_weights_zero!`: Zero out small weights in-place

## Distance Measures
- `l2_distance`: L2 (Euclidean) distance between network parameters
- `l1_distance`: L1 (Manhattan) distance between network parameters  
- `l0_distance`: Sparsity difference between networks
"""
module Metrics

using Flux
using Flux: Chain, params

include("DescriptionLength.jl")
using .DescriptionLength: description_length
export description_length

include("ByteSizeCompression/ByteSizeCompression.jl")
using .ByteSizeCompression: true_byte_size, byte_size_compression
export true_byte_size, byte_size_compression

include("Logs.jl")
using .Logs: number_of_small_weights, accuracy_zeroing_small_weights, set_small_weights_zero!
export number_of_small_weights, accuracy_zeroing_small_weights, set_small_weights_zero!

include("distances.jl")
export l2_distance, l1_distance, l0_distance

end # module
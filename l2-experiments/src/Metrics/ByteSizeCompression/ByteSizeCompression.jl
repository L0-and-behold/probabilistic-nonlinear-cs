"""
ByteSizeCompression module for neural network parameter compression and analysis.

This module provides tools for analyzing and compressing neural network parameters,
including sparse matrix representations and memory footprint calculations.
"""
module ByteSizeCompression 

using Flux: params, cpu, gpu
using Flux
using CUDA
using SparseArrays

include("../Logs.jl")
using .Logs: set_small_weights_zero!

include("metrics.jl")
export byte_size_compression, 
        true_byte_size

include("sparse_matrices.jl")
export CSC, COO

include("logical_helpers.jl")
export biases_follow_weights,
    is_matrix,
    is_vector,
    check_for_size!,
    last_matching_colptr_index

include("structural_helpers.jl")
export fourTensor_to_matrix,
    matrices,
    parameter

end
"""
TrainingTools module for neural network training utilities.

This module provides essential tools for training and evaluating neural networks.

# Exports
## Evaluation Metrics
- `accuracy`: Compute classification accuracy for images/labels or dataset tuples
"""
module TrainingTools

import Flux
using Flux: gpu, Chain, onehotbatch, onecold
using CUDA
using Statistics: mean, std

include("accuracy.jl")
export accuracy

end
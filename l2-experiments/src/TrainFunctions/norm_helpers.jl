using Flux, CUDA
using Flux: gpu
### parallel operations on model-sized tensors

"""
Compute element-wise product of two scalars.

# Arguments
- `scalar_1::Float32`: First scalar value
- `scalar_2::Float32`: Second scalar value

# Returns
- `Float32`: Product of the two scalars
"""
function scalarwise_product(scalar_1::Float32, scalar_2::Float32)
    return scalar_1 * scalar_2
end

"""
Compute element-wise product of corresponding tensors in two vectors.

# Arguments
- `vector_of_tensor1::Vector`: First vector of tensors
- `vector_of_tensor2::Vector`: Second vector of tensors

# Returns
- `Vector`: Vector of element-wise products of corresponding tensor pairs
"""
function scalarwise_product(vector_of_tensor1::Vector, vector_of_tensor2::Vector)
    # println("vectorwise")
    return scalarwise_product.(vector_of_tensor1, vector_of_tensor2)
end

"""
Compute element-wise product of two tensors (CPU or GPU arrays).

# Arguments
- `tensor1::Union{Array, CuArray}`: First tensor
- `tensor2::Union{Array, CuArray}`: Second tensor

# Returns
- `Union{Array, CuArray}`: Element-wise product of the tensors
"""
function scalarwise_product(tensor1::Union{Array, CuArray}, tensor2::Union{Array, CuArray})
    return tensor1 .* tensor2
end

"""
Compute element-wise sum of two scalars.

# Arguments
- `scalar_1::Float32`: First scalar value
- `scalar_2::Float32`: Second scalar value

# Returns
- `Float32`: Sum of the two scalars
"""
function scalarwise_addition(scalar_1::Float32, scalar_2::Float32)
    return scalar_1 + scalar_2
end

"""
Compute element-wise sum of corresponding tensors in two vectors.

# Arguments
- `vector_of_tensor1::Vector`: First vector of tensors
- `vector_of_tensor2::Vector`: Second vector of tensors

# Returns
- `Vector`: Vector of element-wise sums of corresponding tensor pairs
"""
function scalarwise_addition(vector_of_tensor1::Vector, vector_of_tensor2::Vector)
    return scalarwise_addition.(vector_of_tensor1, vector_of_tensor2)
end

"""
Compute element-wise sum of two tensors (CPU or GPU arrays).

# Arguments
- `tensor1::Union{Array, CuArray}`: First tensor
- `tensor2::Union{Array, CuArray}`: Second tensor

# Returns
- `Union{Array, CuArray}`: Element-wise sum of the tensors
"""
function scalarwise_addition(tensor1::Union{Array, CuArray}, tensor2::Union{Array, CuArray})
    return tensor1 .+ tensor2
end


"""
Create a vector of tensors with values determined by a layer index function.

# Arguments
- `all_parameters::Vector`: Vector of model parameters
- `function_of_layerindex::Function`: Function mapping layer index to tensor values
- `device`: Target device (default: gpu)

# Returns
- `Vector`: Vector of tensors with same shapes as input parameters

# Notes
- Uses deepcopy to avoid side effects on input parameters
- Each tensor is filled with values from function_of_layerindex(layer_index)
"""
function vector_of_tensors(all_parameters::Vector, function_of_layerindex::Function; device=gpu)
    # without the deepcopy, the function has sideeffects on all_parameters
    vector_of_tensors = deepcopy(all_parameters)
    for (i, p) in enumerate(params(all_parameters))
        tensor = fill(function_of_layerindex(i), size(Array(p)))
        tensor = tensor |> device
        vector_of_tensors[i] = tensor
    end
    return vector_of_tensors
end

### rescaling parameters layerwise

"""
Scale a parameter value layer-wise based on total parameters and layer sizes.

Creates tensors with scaled parameter values for each layer, accounting for 
different layer sizes to achieve balanced regularization across the network.

# Arguments
- `parameter::Number`: Base parameter value to scale
- `all_parameters::Vector`: Vector of all model parameters
- `device`: Target device (default: gpu)

# Returns
- `Vector`: Vector of tensors with layer-wise scaled parameter values

# Formula
scaled_value = parameter × (total_parameters / number_layers) × (1 / layer_size)

# Notes
- Assumes parameters are ordered as weight-tensor followed by bias-vector pairs
- Combines weights and biases when calculating layer sizes
- Number of layers = length(all_parameters) / 2 (accounting for weight-bias pairs)
- Raises error if bias arrangement doesn't follow expected pattern
"""
function scaled_parameters(parameter::Number, all_parameters::Vector; device=gpu)
    
    parameter = Float32(parameter)

    f(x) = Float32(1.0)
    output = vector_of_tensors(all_parameters, f; device=device)

    total_parameters = Float32(sum(length.(all_parameters)))
    number_layers = Float32(length(all_parameters)/2) # since we have weights and biases
    layer_sizes = length.(all_parameters)

    if !biases_follow_weights(all_parameters)
        error("This function can only handle where parameters are lists ordered as weight-tensor followed by bias-vector, but found an index that should be a bias-vector, but is not.")
    end
    if !weights_follow_biases(all_parameters)
        @warn("This function can only handle where parameters are lists ordered as weight-tensor followed by bias-vector, but found an index that should be a weight-tensor, but is a vector.")
    end

    # add the weights and biases together and set them to the same value to realize layerwise scaling
    even_indices = 2:2:length(layer_sizes)
    odd_indices = even_indices .- 1
    layer_sizes[even_indices] .+= layer_sizes[odd_indices]
    layer_sizes[odd_indices] = layer_sizes[even_indices]

    reziproke_layer_sizes = Float32.(ones(length(all_parameters)) ./ layer_sizes)

    scaled_parameters = parameter * total_parameters/number_layers .* reziproke_layer_sizes

    output .*= scaled_parameters

    return  output
end

### logical helpers

"""
Check if bias parameters follow weight parameters in the expected pattern.

Verifies that even-indexed parameters are 1-dimensional (biases) in a 
weight-bias alternating structure.

# Arguments
- `all_parameters::Vector`: Vector of model parameters

# Returns
- `Bool`: True if all even-indexed parameters are 1-dimensional (biases)
"""
function biases_follow_weights(all_parameters::Vector)
    # odd indices are weights and even indices are biases
    probably_biases = length.(size.(all_parameters)) .== 1
    even_indices = 2:2:length(probably_biases)
    return sum(probably_biases[even_indices]) == length(probably_biases[even_indices])
end

"""
Check if weight parameters follow bias parameters in the expected pattern.

Verifies that odd-indexed parameters are multi-dimensional (weights) in a 
weight-bias alternating structure.

# Arguments
- `all_parameters::Vector`: Vector of model parameters

# Returns
- `Bool`: True if all odd-indexed parameters are multi-dimensional (weights)
"""
function weights_follow_biases(all_parameters::Vector)
    # odd indices are weights and even indices are biases
    probably_biases = length.(size.(all_parameters)) .== 1
    even_indices = 2:2:length(probably_biases)
    odd_indices = even_indices .- 1
    return sum(probably_biases[odd_indices]) == 0
end
"""
    matrices(model)

Extract weight matrices from a neural network model and concatenate them with their corresponding bias vectors.

# Arguments
- `model`: Neural network model containing parameters

# Returns
- `Vector{Matrix{Float32}}`: List of augmented weight matrices with biases concatenated as additional columns

Throws `ArgumentError` if parameters are not ordered as weight-matrix followed by bias-vector.
"""
function matrices(model)
    p = parameter(model)
    
    matrices_list = Vector{Matrix{Float32}}()

    if !biases_follow_weights(p)
        throw(ArgumentError("Parameters must be lists ordered as weight-matrix followed by bias-vector."))
    end
    
    for (i, array) in enumerate(p)
        if !is_matrix(array)
           continue
        end
        push!(matrices_list, hcat(array, p[i+1]))
    end

    return matrices_list
end

"""
    parameter(model)

Extract and preprocess parameters from a neural network model.

# Arguments
- `model`: Neural network model

# Returns
- `Vector`: Processed parameters moved to CPU with 4D tensors converted to 2D matrices
"""
function parameter(model)
    parameter = [] |> cpu
    for l in params(model)
        l = cpu(l)
        l = fourTensor_to_matrix(l)
        push!(parameter, l)
    end
    check_for_size!(parameter)
    return parameter
end

"""
    fourTensor_to_matrix(tensor::Union{Array, CuArray})

Convert a 4-dimensional tensor to a 2-dimensional matrix by flattening feature maps.

# Arguments
- `tensor::Union{Array, CuArray}`: Input tensor of any dimensionality

# Returns
- `Union{Array, CuArray}`: Original tensor if not 4D, otherwise a 2D matrix where each row represents a flattened feature map
"""
function fourTensor_to_matrix(tensor::Union{Array, CuArray})

    if length(size(tensor)) != 4
        return tensor
    end

    a, b, c, d = size(tensor)

    # For each feature map (4th dimension), flatten all parameters to a vector and append vertically.
    matrix = vcat([reshape(tensor[:, :, :, i], 1, a*b*c) for i in 1:d]...)
    return matrix
end
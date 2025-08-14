"""
    biases_follow_weights(parameter::Array) -> Bool

Check if bias vectors properly alternate with weight matrices in a parameter array.

# Arguments
- `parameter::Array`: Array containing alternating weight matrices and bias vectors

# Returns
- `Bool`: `true` if biases follow the expected alternating pattern with weights, `false` otherwise
"""
function biases_follow_weights(parameter::Array)
    matrix_logic = map(x -> is_matrix(x), parameter)
    vector_logic = map(x -> is_vector(x), parameter)
    
    a = Array(1:length(matrix_logic)) .% 2 # i.e. [1, 0, 1, 0, ...]

    b = (Array(1:length(vector_logic)) .+ 1) .% 2 # i.e. [0, 1, 0, ...]

    return a == matrix_logic && b == vector_logic
end

"""
    is_matrix(array::Array) -> Bool

Check if an array is a 2-dimensional matrix.

# Arguments
- `array::Array`: Input array to check

# Returns
- `Bool`: `true` if array has exactly 2 dimensions, `false` otherwise
"""
function is_matrix(array::Array)
    return ndims(array) == 2
end

"""
    is_vector(array::Array) -> Bool

Check if an array is a 1-dimensional vector.

# Arguments
- `array::Array`: Input array to check

# Returns
- `Bool`: `true` if array has exactly 1 dimension, `false` otherwise
"""
function is_vector(array::Array)
    return ndims(array) == 1
end

"""
    check_for_size!(parameter::Array)

Validate that bias vectors and weight matrices have compatible dimensions.

Checks that vectors (biases) properly follow matrices or 4-tensors (weights) and that
bias vector sizes match the first dimension of their corresponding weight arrays.

# Arguments
- `parameter::Array`: Array containing weight matrices/tensors and bias vectors

# Throws
- `ArgumentError`: If biases don't follow weights in the expected pattern
- `DimensionMismatch`: If bias vector size doesn't match corresponding weight matrix size
"""
function check_for_size!(parameter::Array)
    matrix_logic = map(x -> is_matrix(x), parameter)
    vector_logic = map(x -> is_vector(x), parameter)
    
    if matrix_logic != map(x -> !x, vector_logic) && tensor_logic != map(x -> !x, vector_logic)
        throw(ArgumentError("Vectors, representing biases, should follow matrices or 4-tensors, representing weights."))
    end
    
    for (i, array) in enumerate(parameter)
        if matrix_logic[i]
            continue
        elseif size(array)[1] != size(parameter[i-1])[1]
            throw(DimensionMismatch("The size of the bias vector does not match the corresponding weight matrix."))
        end
    end
end

"""
    last_matching_colptr_index(i::Int64, colptr::Vector{Int64}) -> Int64

Find the index of the last element in `colptr` that is less than or equal to `i`.

# Arguments
- `i::Int64`: Target value to compare against
- `colptr::Vector{Int64}`: Sorted vector of column pointer values

# Returns
- `Int64`: Index of the last element in `colptr` where `colptr[index] <= i`
"""
function last_matching_colptr_index(i::Int64, colptr::Vector{Int64})
    logic = colptr .<= i
    helper = 1:length(colptr)
    helper = helper[logic]
    return helper[end]
end
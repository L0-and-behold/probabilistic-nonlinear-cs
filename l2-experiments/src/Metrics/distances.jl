using Flux: params, Chain

"""
Compute the L2 (Euclidean) distance between parameters of two neural networks.

# Arguments
- `nn1::Chain`: First neural network
- `nn2::Chain`: Second neural network

# Returns
- `Number`: L2 distance between the flattened parameter vectors
"""
function l2_distance(nn1::Chain, nn2::Chain)::Number
    total_distance = 0.0
    for (i, p) in enumerate(params(nn1))
        total_distance += sum((p - params(nn2)[i]).^2)
    end
    return total_distance
end

"""
Compute the L1 (Manhattan) distance between parameters of two neural networks.

# Arguments
- `nn1::Chain`: First neural network
- `nn2::Chain`: Second neural network

# Returns
- `Number`: L1 distance between the flattened parameter vectors
"""
function l1_distance(nn1::Chain, nn2::Chain)::Number
    total_distance = 0.0
    for (i, p) in enumerate(params(nn1))
        total_distance += sum(abs.(p - params(nn2)[i]))
    end
    return total_distance
end

"""
Compute the difference in sparsity (number of zero parameters) between two neural networks.

# Arguments
- `nn1::Chain`: First neural network
- `nn2::Chain`: Second neural network

# Returns
- `Number`: Absolute difference in the count of zero-valued parameters
"""
function l0_distance(nn1::Chain, nn2::Chain)::Number
    non_zero_counter_1 = 0
    non_zero_counter_2 = 0
    for (i, p) in enumerate(params(nn1))
        non_zero_counter_1 += sum(p .== 0.0)
        non_zero_counter_2 += sum(params(nn2)[i] .== 0.0)
    end
    return abs(non_zero_counter_1 - non_zero_counter_2)
end

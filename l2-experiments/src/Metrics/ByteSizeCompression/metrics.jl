"""
    true_byte_size(model::Chain)

Calculate the minimum memory footprint of a neural network model across different sparse matrix formats.

# Arguments
- `model::Chain`: Neural network model to analyze

# Returns
- `Int`: Minimum byte size among dense, CSC, and COO representations
"""
function true_byte_size(model::Chain)

    a = Base.summarysize(matrices(model))
    b = Base.summarysize(CSC(matrices(model)))
    c = Base.summarysize(COO(matrices(model)))
    
    return minimum([a, b, c]) #bytes
end

"""
    byte_size_compression(model_post::Chain, ϵ::Union{Array, Vector})

Calculate compression ratios for multiple sparsity thresholds.

# Arguments
- `model_post::Chain`: Neural network model to compress
- `ϵ::Union{Array, Vector}`: Array of sparsity thresholds to test

# Returns
- `Vector`: Compression ratios corresponding to each threshold in `ϵ`
"""
function byte_size_compression(model_post::Chain{}, ϵ::Union{Array, Vector})
    CR = map(e -> byte_size_compression(model_post, e), ϵ)
    return CR
end

"""
    byte_size_compression(model_post::Chain, ϵ::Number)

Calculate the byte size compression ratio after applying weight pruning with threshold ϵ.

# Arguments
- `model_post::Chain`: Neural network model to compress
- `ϵ::Number`: Sparsity threshold for weight pruning

# Returns
- `Float64`: Compression ratio (original size / compressed size)
"""
function byte_size_compression(model_post::Chain{}, ϵ::Number)

    pre = Base.summarysize(matrices(model_post))
    
    model = deepcopy(model_post)
    set_small_weights_zero!(model, ϵ)

    post_1 = Base.summarysize(matrices(model))
    post_2 = Base.summarysize(CSC(matrices(model)))
    post_3 = Base.summarysize(COO(matrices(model)))
    post = minimum([post_1, post_2, post_3])

    return pre / post
end

"""
    byte_size_compression(model_pre::Chain, model_post::Chain)

Calculate the compression ratio between two models using optimal sparse matrix representations.

# Arguments
- `model_pre::Chain`: Original (uncompressed) neural network model
- `model_post::Chain`: Compressed neural network model

# Returns
- `Float64`: Compression ratio (pre-compression size / post-compression size)
"""
function byte_size_compression(model_pre::Chain{}, model_post::Chain{})

    pre_1 = Base.summarysize(matrices(model_pre))
    pre_2 = Base.summarysize(CSC(matrices(model_pre)))
    pre_3 = Base.summarysize(COO(matrices(model_pre)))
    pre = minimum([pre_1, pre_2, pre_3]) #the CSC format will be bigger for untrained models

    post_2 = Base.summarysize(matrices(model_post)) 
    post_1 = Base.summarysize(CSC(matrices(model_post)))
    post_3 = Base.summarysize(COO(matrices(model_post)))
    post = minimum([post_1, post_2, post_3])#the CSC format will be bigger for untrained models

    return pre / post
end
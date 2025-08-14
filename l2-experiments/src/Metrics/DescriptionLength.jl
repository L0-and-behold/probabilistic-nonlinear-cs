module DescriptionLength

using CUDA
using Flux: gpu, flatten, Chain, relu, gpu, Dense, onehot, onecold, softmax, OneHotMatrix, params

include("ByteSizeCompression/ByteSizeCompression.jl")
using .ByteSizeCompression: true_byte_size
using .ByteSizeCompression

export codinglength, 
    uncoded_codelength, 
    description_length

"""
Compute the minimum description length (MDL) principle: -∑log(p(x)) + l(p).

# Arguments
- `model::Chain`: Neural network model
- `train_set::Vector{<:Tuple}`: Training dataset
- `test_set::Vector{<:Tuple}`: Test dataset

# Returns
- `Float64`: Total description length in bytes (rounded to nearest integer)
"""
function description_length(model::Chain, train_set::Vector{<:Tuple}, test_set::Vector{<:Tuple})
    L = codinglength(model, train_set) + codinglength(model, test_set) + true_byte_size(model)
    return round(L, digits = 0) #bytes
end

"""
Compute the minimum description length (MDL) principle: -∑log(p(x)) + l(p).

# Arguments
- `model::Chain`: Neural network model
- `dataset::Vector{<:Tuple}`: Dataset

# Returns
- `Float64`: Total description length in bytes (rounded to nearest integer)
"""
function description_length(model::Chain, dataset::Vector{<:Tuple})
    L = codinglength(model, dataset) + true_byte_size(model)
    return round(L, digits = 0) #bytes
end

"""
Compute the minimum description length (MDL) principle: -∑log(p(x)) + l(p).

# Arguments
- `model::Chain`: Neural network model
- `train_set::Vector{<:Tuple}`: Training dataset
- `val_set::Vector{<:Tuple}`: Validation dataset
- `test_set::Vector{<:Tuple}`: Test dataset

# Returns
- `Float64`: Total description length in bytes (rounded to nearest integer)
"""
function description_length(model::Chain, train_set::Vector{<:Tuple}, val_set::Vector{<:Tuple}, test_set::Vector{<:Tuple})
    L = codinglength(model, train_set) + codinglength(model, val_set) + codinglength(model, test_set) + true_byte_size(model)
    return round(L, digits = 0) #bytes
end

"""
Compute the negative log-likelihood coding length: -∑log(p(x)).

# Arguments
- `model::Chain`: Neural network model
- `img_batch::CuArray`: Batch of input images
- `label_batch::OneHotMatrix`: Batch of one-hot encoded labels

# Returns
- `Float64`: Coding length in bytes
"""
function codinglength(model::Chain, img_batch::CuArray, label_batch::OneHotMatrix)
    a = softmax(model(img_batch)) .* label_batch
    a = reduce(max, a, dims = 1)

    if sum(a.==0.0) > 0
        @warn "Some probabilities are zero. Setting them to eps(Float32)."
        a[a.== 0.0] .= eps(Float32)
    end

    a  = - sum( lb.(a) ) #bits
    return a / 8 #bytes
end

"""
Compute the negative log-likelihood coding length: -∑log(p(x)).

# Arguments
- `model::Chain`: Neural network model
- `imglable::Tuple{CuArray, OneHotMatrix}`: Tuple of images and labels

# Returns
- `Float64`: Coding length in bytes
"""
function codinglength(model::Chain, imglable::Tuple{CuArray, OneHotMatrix}) 
    return codinglength(model, imglable...)
end

"""
Compute the negative log-likelihood coding length: -∑log(p(x)).

# Arguments
- `model::Chain`: Neural network model
- `imglable::Tuple{Array, OneHotMatrix}`: Tuple of images and labels

# Returns
- `Float64`: Coding length in bytes
"""
function codinglenght(model::Chain, imglable::Tuple{Array, OneHotMatrix}) 
    return codinglength(model, imglable...)
end

"""
Compute the total negative log-likelihood coding length for a dataset.

# Arguments
- `model::Chain`: Neural network model
- `set::Vector{<:Tuple}`: Dataset as vector of (image, label) tuples

# Returns
- `Float64`: Total coding length in bytes
"""
function codinglength(model::Chain, set::Vector{<:Tuple}) 
    return sum(map(x -> codinglength(model, x), set))
end

"""
Compute logarithm base 2.

# Arguments
- `x`: Input value

# Returns
- Logarithm base 2 of x
"""
lb(x) = log(x) / log(2)

end#module
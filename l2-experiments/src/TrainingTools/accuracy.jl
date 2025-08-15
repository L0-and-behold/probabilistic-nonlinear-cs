"""
Compute classification accuracy for images and labels.

# Arguments
- `images::Union{CuArray, Array}`: 4D array of input images (H×W×C×N)
- `labels`: One-hot encoded labels
- `model`: Neural network model
- `device`: Target device (default: gpu)

# Returns
- `Float64`: Classification accuracy as a fraction between 0 and 1
"""
function accuracy(images::Union{CuArray, Array}, labels, model; device=gpu)
    if !(typeof(Array(images)) <: Array{Float32, 4})
        throw(DomainError("images should be a 4D array"))
    end
    
    if !(isa(labels, Flux.OneHotMatrix))
        throw(DomainError("labels should be a OneHotMatrix"))
    end

    images = images |> device
    labels = labels |> device
    model = model |> device

    result = model(images) |> device
    result = onecold(result) |> device

    labels = onecold(labels) |> device

    acc = mean(result .== labels)
    return acc
end

"""
Compute classification accuracy for a dataset of image-label tuples.

# Arguments
- `data::Vector{<:Tuple}`: Vector of (image, label) tuples
- `model::Flux.Chain{<:Any}`: Neural network model
- `device`: Target device (default: gpu)

# Returns
- `Float64`: Classification accuracy as a fraction between 0 and 1
"""
function accuracy(data::Vector{<:Tuple}, model::Flux.Chain{<:Any}; device=gpu)
    if typeof(Array(data[1][1])) != Array{Float32, 4}
        throw(ArgumentError("data should be a vector of tuples where the images are 4D arrays"))
    end
    if !isa(data[1][2], Flux.OneHotMatrix)
        throw(ArgumentError("data should be a vector of tuples where the labels are OneHotMatrix"))
    end

    images = cat([x[1] for x in data]..., dims=4) |> device
    labels = cat([x[2] for x in data]..., dims=2) |> device

    # Calculate accuracy for all images and labels at once
    acc = accuracy(images, labels, model, device=device)

    return acc
end
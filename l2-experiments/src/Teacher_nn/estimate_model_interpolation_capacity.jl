using Flux
using Statistics

"""
    pertub_model(model::Union{Chain, Flux.Chain})::Chain

Create a slightly perturbed copy of the input neural network model.

# Arguments
- `model`: The original model to be perturbed.

# Returns
A perturbed copy of the input model.
"""
function pertub_model(model::Union{Chain, Flux.Chain}; device=gpu)::Chain
    nn = deepcopy(model) |> cpu
    for p in params(nn)
        p .+= p .* eps(Float32) .* randn(Float32, size(p))
    end
    return nn |> device
end

"""
    estimate_model_interpolation_capacity(
        teacher::Union{Chain, Flux.Chain}, 
        loss_fn::Function = Flux.mse, 
        optimizer = Flux.Adam(0.0001),
        batch_size::Int = 1000, 
        num_trials::Int = 10
            )::Tuple{Float64, Float64} 

Estimate the lowest achievable loss for a given model and training setup.

# Arguments
- `teacher`: The original model used to generate training data.
- `loss_fn`: Loss function (default: `Flux.mse`).
- `optimizer`: Optimization algorithm (default: `Flux.Adam(0.0001)`).
- `batch_size`: Number of samples per batch (default: 1000).
- `num_trials`: Number of trials to run (default: 10).

# Returns
A tuple containing the mean and standard deviation of the final losses.
"""
function estimate_model_interpolation_capacity(
    teacher::Union{Chain, Flux.Chain}, 
    loss_fn::Function = Flux.mse, 
    optimizer = Flux.Adam(0.0001),
    batch_size::Int = 1000, 
    num_trials::Int = 10;
    device=gpu
        )::Tuple{Float64, Float64}        

    losses = []

    for _ in 1:num_trials

        train_set = get_dataset(teacher, 20*batch_size, batch_size, device=device)
        test_set = get_dataset(teacher, 5*batch_size, 5*batch_size, device=device)
        
        opt = deepcopy(optimizer) |> device # copy optimizer to avoid side effects
        nn = pertub_model(teacher, device=device)
        loss(x, y) = loss_fn(nn(x), y)
        
        TrainFunctions.vanilla_train!(loss, nn, train_set, opt)
        push!(losses, loss(test_set[1][1], test_set[1][2]))
    end

    return mean(losses), std(losses)
end
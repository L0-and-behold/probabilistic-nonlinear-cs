using Flux
using Flux: Chain
import Random

"""
Generate a dataset from a teacher neural network with optional noise.

Creates a vector of input-output batch pairs by sampling from a "noisy teacher" 
model. The teacher network provides deterministic outputs on random inputs, which 
serve as the mean of a normal distribution from which the final outputs are sampled.

# Arguments
- `nn::Union{Chain, Flux.Chain}`: Teacher network used to generate target outputs
- `set_size::Int`: Total number of samples in the dataset
- `batch_size::Int`: Number of samples per batch (must divide set_size evenly)
- `noise_σ::Number`: Standard deviation of Gaussian noise added to teacher outputs (default: 0.0)
- `device::Function`: Target device function for tensor allocation (default: gpu)
- `rng`: Random number generator (default: Random.default_rng())

# Returns
- `Vector{Tuple{Matrix{Float32}, Matrix{Float32}}}`: Dataset as vector of (input, output) batch pairs

# Dataset Structure
- Total batches: set_size ÷ batch_size
- Each batch: (input_matrix, output_matrix)
  - input_matrix: (input_dim × batch_size) uniformly distributed in [-1, 1]
  - output_matrix: (output_dim × batch_size) teacher predictions + Gaussian noise

# Generation Process
1. Generate random input batches uniformly distributed in [-1, 1]
2. Compute deterministic teacher outputs: signal = nn(input)
3. Add Gaussian noise: output = signal + N(0, noise_σ²)
4. Return dataset as vector of (input, output) batch tuples

# Notes
- Input dimension determined from first layer weight matrix: size(W)[2]
- Noise is sampled independently for each output element
- All tensors allocated on specified device
- Useful for student-teacher training scenarios and model distillation

# Constraints
- set_size > 0
- batch_size > 0  
- set_size must be divisible by batch_size
- noise_σ ≥ 0
"""
function get_dataset(
    nn::Union{Chain, Flux.Chain}, 
    set_size::Int, 
    batch_size::Int;
    noise_σ::Number=0f0,
    device::Function=gpu,
    rng=Random.default_rng()
    )::Array{Tuple{Matrix{Float32}, Matrix{Float32}}, 1}
    
    @assert set_size > 0
    @assert batch_size > 0
    @assert set_size % batch_size == 0 "set_size must be divisible by batch_size"
    @assert noise_σ >= 0 "noise_σ must be greater or equal to 0"

    input_dim = size(params(nn.layers[1])[1])[2]

    dataset = Tuple{Matrix{Float32}, Matrix{Float32}}[] |> device
    number_batches = Int(set_size / batch_size)
    for i in 1:number_batches
        x = 2 * rand(rng, Float32, input_dim, batch_size) .- 1 |> device

        # we draw from a normal distribution with mean determined by deterministic teacher and standard deviation noise_σ
        signal = nn(x) |> device
        noise = Float32(noise_σ) * randn(rng, Float32, 1, batch_size) |> device
        y = signal + noise

        push!(dataset, (x, y))
    end
    return dataset |> device
end
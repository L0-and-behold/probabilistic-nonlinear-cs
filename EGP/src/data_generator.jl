
using LinearAlgebra, Random, Distributions
using CUDA
using Lux, LuxCUDA 
using Lux: cpu_device, gpu_device

function gpu_MvNormal(Σ, p, n)
    L = CUDA.cholesky(Σ).L
    Z = CUDA.randn(p, n)
    X = L * Z
    return X'
end


function cuda_driver_available(;verbose=false)
    try
        if CUDA.functional()
            if verbose
                println("CUDA is available and functional")
            end
            return true
        else
            if verbose
                println("CUDA is installed but not functional, using CPU")
            end
            return false
        end
    catch e
        if verbose
            println("CUDA not available, using CPU: ", e)
        end
        return false
    end
end

function generate_data(n, p, s1, rho, nu; s2 = p+1, dtype=Float32::DataType, seed=42, dev=cpu_device())
    if dev == gpu_device()
        if !cuda_driver_available()
            dev == cpu_device()
        end
    end
    # Step 0: Set seed
    rng = MersenneTwister()
    Random.seed!(rng, seed)
    if dev == cpu_device()

        # Step 1: Generate vector β
        β = [dtype(i <= s1 ? 1.0 : i % s2 == 0 ? 4.0 : 0) for i in 1:p]

        # Step 2: Create matrix Σ
        Σ = [dtype(rho^abs(i-j)) for i in 1:p, j in 1:p]

        # Step 3: Create matrix X
        X = dtype.(rand(rng, MvNormal(zeros(p), Σ), n)')
        # X2 = dtype.(rand(rng, RandnNormalMatrix(Σ, n))')

        # Step 4: Compute σ²
        σ² = dtype(dot(β, Σ * β) / nu)

        # Step 5: Generate vector y
        I_s = dtype.(Diagonal(fill(σ², n)))
        y = dtype.(rand(rng,MvNormal(X * β, I_s)))

        return X, y, β, Σ, σ²
    else # if GPU is available, then this is considerably faster for bigger n, p
        CUDA.seed!(seed)

        i = accumulate(+, CUDA.ones(1,p))
        Σ = rho .^ abs.(i .- i')

        β = CUDA.zeros(p)
        β[1:s1] .= 1f0
        β[vec(i .% s2 .== 0)] .= 4f0

        X = gpu_MvNormal(Σ, p, n) |> cpu_device()

        σ² = β' * Σ * β / nu

        β = β |> cpu_device()
        Σ = Σ |> cpu_device()

        I_s = dtype.(Diagonal(fill(σ², n)))
        y = dtype.(rand(rng,MvNormal(X * β, I_s)))

        return X, y, β, Σ, σ²
    end
end

function generate_fast_data(n, p; dtype=Float32::DataType, seed=42, s1=10, s2=n+1)
    # Step 0: Set seed
    rng = MersenneTwister()
    Random.seed!(rng, seed)

    # Step 1: Generate vector β
    β = [dtype(i <= s1 ? 1.0 : i % s2 == 0 ? 4.0 : 0) for i in 1:p]
    
    # Step 2: Create random matrix X
    X = rand32(rng, n, p)

    # Step 3: Generate vector y
    y = X * β

    return X, y, β
end
## Example usage

# n = 10
# p = 20
# s = 5
# rho = 0.35
# nu = 0.1

# X, y, β, Σ, σ², rng = generate_data(n, p, s, rho, nu)


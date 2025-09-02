using Lux, Random, WeightInitializers # Importing `Lux` also gives you access to `LuxCore`

struct ProlinearMatrixGauss{F1,F2,F3} <: LuxCore.AbstractLuxLayer
    n::Int
    p::Int
    initial_w_params::F1
    initial_p_params::F2
    initial_sigma::F3
end
 
function ProlinearMatrixGauss(n::Int, p::Int; initial_w_params=glorot_uniform, initial_p_params=(rng, n, p)->Float32.(rand(rng, Bool, n, p)), initial_sigma=(rng, n, p)->ones32(1,p))
    return ProlinearMatrixGauss{typeof(initial_w_params), typeof(initial_p_params), typeof(initial_sigma)}(n, p, initial_w_params, initial_p_params, initial_sigma)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::ProlinearMatrixGauss)
    return (w=l.initial_w_params(rng, l.n, l.p), p=l.initial_p_params(rng, l.n, l.p), sigma=l.initial_sigma(rng, l.n, l.p))
end

LuxCore.initialstates(::AbstractRNG, ::ProlinearMatrixGauss) = NamedTuple()
LuxCore.parameterlength(l::ProlinearMatrixGauss) = 2*l.n*l.p + l.n
LuxCore.statelength(::ProlinearMatrixGauss) = 0

function (l::ProlinearMatrixGauss)(x::AbstractMatrix, ps, st::NamedTuple)
    y = x * (ps.w .* ps.p)
    return y, st
end


## run it
# l = ProlinearMatrixGauss(2,3)

# rng = Random.default_rng()
# Random.seed!(rng, 0)

# ps, st = LuxCore.setup(rng, l)

# println("Parameter Length: ", LuxCore.parameterlength(l), "; State Length: ", LuxCore.statelength(l))

# x = randn(rng, Float32, 5, 2)

# mu = LuxCore.apply(l, x, ps, st)[1] # or `l(x, ps, st)`

# size(mu)

# ps.sigma .* mu
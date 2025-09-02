using Lux, Random, WeightInitializers # Importing `Lux` also gives you access to `LuxCore`

struct ProlinearMatrix{F1,F2} <: LuxCore.AbstractLuxLayer
    n::Int
    p::Int
    initial_w_params::F1
    initial_p_params::F2
end
 
function ProlinearMatrix(n::Int, p::Int; initial_w_params=glorot_uniform, initial_p_params=(rng, n, p)->Float32.(rand(rng, Bool, n, p)))
    return ProlinearMatrix{typeof(initial_w_params), typeof(initial_p_params)}(n, p, initial_w_params, initial_p_params)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::ProlinearMatrix)
    return (w=l.initial_w_params(rng, l.n, l.p), p=l.initial_p_params(rng, l.n, l.p))
end

LuxCore.initialstates(::AbstractRNG, ::ProlinearMatrix) = NamedTuple()
LuxCore.parameterlength(l::ProlinearMatrix) = 2*l.n*l.p
LuxCore.statelength(::ProlinearMatrix) = 0

function (l::ProlinearMatrix)(x::AbstractMatrix, ps, st::NamedTuple)
    y = x * (ps.w .* ps.p)
    return y, st
end


## run it

# l = ProlinearMatrix(2)

# rng = Random.default_rng()
# Random.seed!(rng, 0)

# ps, st = LuxCore.setup(rng, l)

# println("Parameter Length: ", LuxCore.parameterlength(l), "; State Length: ",
#     LuxCore.statelength(l))

# x = randn(rng, Float32, 5, 2)

# LuxCore.apply(l, x, ps, st) # or `l(x, ps, st)`

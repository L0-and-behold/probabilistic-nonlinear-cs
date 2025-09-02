using Lux, Random, WeightInitializers # Importing `Lux` also gives you access to `LuxCore`

struct Linear{F1} <: LuxCore.AbstractLuxLayer
    dims::Int
    dims2::Int
    initial_params::F1
end

function Linear(dims::Int; dims2::Int=1, initial_params=glorot_uniform)
    return Linear{typeof(initial_params)}(dims, dims2, initial_params)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::Linear)
    return (v_params=l.initial_params(rng, l.dims, l.dims2),)
end

LuxCore.initialstates(::AbstractRNG, ::Linear) = NamedTuple()
LuxCore.parameterlength(l::Linear) = l.dims * l.dims2
LuxCore.statelength(::Linear) = 0

function (l::Linear)(x::AbstractMatrix, ps, st::NamedTuple)
    y = x * ps.v_params
    return y, st
end


## run it

# l = Linear(2)
# display(l)

# rng = Random.default_rng()
# Random.seed!(rng, 0)

# randn(rng,(2,3))
# glorot_uniform(rng,Float64,2,3)
# glorot_uniform(Float64)(rng,2,3)
# ps, st = LuxCore.setup(rng, l)

# println("Parameter Length: ", LuxCore.parameterlength(l), "; State Length: ",
#     LuxCore.statelength(l))

# # x = randn(rng, Float32, 5, 2)

# # LuxCore.apply(l, x, ps, st) # or `l(x, ps, st)`

using Lux, Random, WeightInitializers # Importing `Lux` also gives you access to `LuxCore`

struct LinearGauss{F1, F2} <: LuxCore.AbstractLuxLayer
    dims::Int
    dims2::Int
    initial_params::F1
    initial_sigma::F2
end

function LinearGauss(dims::Int; dims2::Int=1, initial_params=glorot_uniform, initial_sigma=(rng, dims, dims2)->ones32(1, dims2))
    return LinearGauss{typeof(initial_params), typeof(initial_sigma)}(dims, dims2, initial_params, initial_sigma)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::LinearGauss)
    return (v_params=l.initial_params(rng, l.dims, l.dims2), sigma=l.initial_sigma(rng, l.dims, l.dims2))
end

LuxCore.initialstates(::AbstractRNG, ::LinearGauss) = NamedTuple()
LuxCore.parameterlength(l::LinearGauss) = l.dims * l.dims2 * 2
LuxCore.statelength(::LinearGauss) = 0

function (l::LinearGauss)(x::AbstractMatrix, ps, st::NamedTuple)
    y = x * ps.v_params
    return y, st
end

## run it

# l = LinearGauss(2)
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


using Optimisers

# Optimisers.@def 
struct ArrayAdam <: AbstractRule
    eta::AbstractArray
    beta::Tuple{Float32, Float32}
    epsilon::Float32
    function ArrayAdam(eta; beta = (0.9, 0.999), epsilon = 1e-8)
        new(eta, beta, epsilon)
    end
end

Optimisers.init(o::ArrayAdam, x::AbstractArray{T}) where T = (zero(x), zero(x), T.(o.beta))

# o = ArrayAdam([1f0; 1f0;;]); state = tstate.optimizer_state.v_params.state; x = tstate.parameters.v_params; dx = grads.v_params;
function Optimisers.apply!(o::ArrayAdam, state, x::AbstractArray{T}, dx) where T
    η, β, ϵ = T.(o.eta), T.(o.beta), Optimisers._eps(T, o.epsilon)
    mt, vt, βt = state
  
    Optimisers.@.. mt = β[1] * mt + (1 - β[1]) * dx
    Optimisers.@.. vt = β[2] * vt + (1 - β[2]) * abs2(dx)
    dxn = Optimisers.@lazy mt / (1 - βt[1]) / (sqrt(vt / (1 - βt[2])) + ϵ) * η # @lazy already contains broadcasting
  
    return (mt, vt, βt .* β), dxn
end

# mt .= β[1] .* mt .+ (1 - β[1]) .* dx
# vt .= β[2] .* vt .+ (1 - β[2]) .* abs2.(dx)
# dxn = Optimisers.@lazy mt ./ (1 - βt[1]) ./ (sqrt.(vt ./ (1 - βt[2])) .+ ϵ) .* η
# a
## Tests for ArrayAdam
# using Zygote
# include("custom_linear_model.jl")

# seed = 42
# opt = ArrayAdam([1f-1 2f-1 3f-1])
# Random.seed!(seed)
# rng = Random.default_rng(seed)
# model = Linear(2; dims2=3)
# ps, st = Lux.setup(rng, model)
# ps.v_params .= 1f0

# tstate = Training.TrainState(model, ps, st, opt)
# # grads = (v_params=[1f0; 1f0;;],)
# # grads.v_params
# # ostate = tstate.optimizer_state.v_params.state

# # Optimisers.apply!(opt, ostate, tstate.parameters.v_params, grads.v_params)


# function myloss(model, ps::NamedTuple, st::NamedTuple, (X,y)::Tuple)
#     loss = 0.5 * sum((ps.v_params .- y).^2)
#     stats = nothing
#     return loss, st, stats
# end
# batch = ([[1f0 2f0]; [3f0 4f0]], [0f0; 0f0;;])
# myloss(tstate.model, tstate.parameters, tstate.states, batch)

# vjp = AutoZygote()
# grads, loss, _, tstate = Training.compute_gradients(vjp, myloss, batch, tstate);

# grads.v_params

# tstate = Training.apply_gradients!(tstate, grads);

# tstate.parameters.v_params

# tstate.optimizer_state.v_params.state[1]
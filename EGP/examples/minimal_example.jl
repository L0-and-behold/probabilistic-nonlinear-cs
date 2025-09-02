include("../src/Exact_Gradient_Pruning.jl")

using Plots
plotlyjs()

n, p, s, noise_factor = (500, 1000, 20, 0.1f0)
val_set_size = 500

# BEGIN data generation
rng = MersenneTwister(42)
X = randn(rng,Float32,n,p)
Xval = randn(rng,Float32,val_set_size,p)
beta_initial = randn(rng,Float32,s)
noise_vector = randn(rng,Float32,n)
noise_vector_val = randn(rng, Float32, val_set_size)
y = X[:,1:s] * beta_initial  .+ noise_factor .* noise_vector
yval = Xval[:,1:s] * beta_initial  .+ noise_factor .* noise_vector_val
actual_beta = zeros(Float32,p)
actual_beta[1:s] .= beta_initial
# END data generation

_, logs, _, _, _, final_beta, final_y_pred, _, _, _, _, _, total_time, _, _ = Exact_Gradient_Pruning(X, y; Xval = Xval, yval = yval)

plot(logs["train_loss"], label="L0 Loss", linewidth=2, legend=:topright)

println(findall(x->x!=0, final_beta), "found active set")
println(findall(x->x!=0, true_active_set), "true active set")
println(beta_initial, " - true beta")
println(final_beta[1:s], " - found beta")

loss = sum((final_y_pred-y).^2) / length(y)
reconstruction_error = sum(abs.(final_beta-actual_beta))

active_set = zeros(Float32, size(final_beta))
active_set[final_beta.!=0] .= 1f0
ASRE = sum(abs.(active_set.-true_active_set))
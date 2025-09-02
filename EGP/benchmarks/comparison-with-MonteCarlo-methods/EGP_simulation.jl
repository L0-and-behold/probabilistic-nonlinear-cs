include("../../src/Exact_Gradient_Pruning.jl")

using Plots
plotlyjs()

using CSV, DataFrames, Statistics

function nt_to_string(nt)
    parts = ["__$(k)_$(v)" for (k, v) in pairs(nt)]
    return join(parts, "") * "__"
end

n, p, p_sub, noise_factor, lambda_value = (60, 200, 3, 0.5f0, 2f0)

batch_size = n
seed = 1
optimizerNames = ["Descent", "Adam"]

dir_path = @__DIR__
results_path = dir_path * "/results"
if !isdir(results_path) # Create the folder for storing the results if it doesn't exist
    mkpath(results_path)
end
date_string = "2025-08-20"

for optimizerName in optimizerNames
    # optimizerName = "Descent"
    rng = MersenneTwister(seed)
    X = randn(rng,Float32,n,p)
    beta_initial = randn(rng,Float32,p_sub)
    noise_vector = randn(rng,Float32,n)
    y = X[:,1:p_sub] * beta_initial  .+ noise_factor .* noise_vector
    actual_beta = zeros(Float32,p)
    actual_beta[1:p_sub] .= beta_initial
    lr = 1f-1
    initial_p_value=0.5f0
    if optimizerName == "Descent"
        optimizer = Descent
    else
        optimizer = ArrayAdam
    end
    setting = (n = n, p = p, s = p_sub, noise = noise_factor, lambda = lambda_value, lr = lr, seed = seed, initp=initial_p_value, optimizer=optimizerName)
    epochs = 1500

    l_e = []
    rec_errs = []
    final_L0_losses = []
    runtimes = []
    ASREs = []
    for i in 1:10
        _, logs, _, _, _, final_beta, final_y_pred, _, _, _, _, _, total_time, _, _ = Exact_Gradient_Pruning(X, y;  optimizer=optimizer, log_L0_loss=true, verbose=false, initial_p_value=initial_p_value, lr=lr, l1_regularization = false, finetuning = false, alpha_range = (lambda_value,lambda_value), l1_block_size = 1, l0_block_size = 1, lr_block_size = 1, min_epochs = epochs, max_epochs=epochs);

        activeElements = final_beta[final_beta .!= 0]
        active_set = zeros(Float32, size(final_beta))
        active_set[final_beta.!=0] .= 1f0
        true_active_set = zeros(Float32, size(actual_beta))
        true_active_set[actual_beta.!=0] .= 1f0
        println()
        println(findall(x->x!=0, final_beta), "found active set")
        println(findall(x->x!=0, true_active_set), "true active set")
        println(activeElements)
        println()
        println(beta_initial, "true beta")
        println(final_beta[1:p_sub], "found beta")
        println()

        push!(l_e, logs["L0_loss"])

        loss = sum((final_y_pred-y).^2) / length(y)
        l0_loss = loss + length(activeElements)
        push!(final_L0_losses, l0_loss)
        println("final L0 loss: ", l0_loss)

        reconstruction_error = sum(abs.(final_beta-actual_beta))

        intersection = intersect(true_active_set[true_active_set.!=0],active_set[active_set.!=0])
        println("intersection: ", intersection)
        ASRE = sum(abs.(active_set.-true_active_set))
        push!(ASREs, ASRE)
        println("ASRE: ",ASRE)
        println()

        push!(rec_errs, reconstruction_error)
        println("reconstruction error: ",reconstruction_error)
        println()
        push!(runtimes, total_time)
    end
    # plot(l_e[end], label="L0 Loss", linewidth=2, legend=:topright) #, markershape=:circle)

    average_rec_err = sum(rec_errs) / length(rec_errs)
    average_final_L0_loss = sum(final_L0_losses) / length(final_L0_losses)
    average_runtime = sum(runtimes) / length(runtimes)
    average_ASRE = sum(ASREs) / length(ASREs)

    std_rec_err = std(rec_errs)
    std_final_L0_loss = std(final_L0_losses)
    std_runtime = std(runtimes)
    std_ASRE = std(ASREs)

    metrics = (average_rec_err = average_rec_err, average_final_L0_loss = average_final_L0_loss, average_runtime = average_runtime, average_ASRE=average_ASRE)
    metrics_std = (std_rec_err = std_rec_err, std_final_L0_loss = std_final_L0_loss, std_runtime = std_runtime, std_ASRE=std_ASRE)

    # Convert the vector of vectors into a matrix
    data_matrix = reduce(hcat, l_e)'
    means = mapslices(mean, data_matrix, dims=1)[:]
    std_devs = mapslices(std, data_matrix, dims=1)[:]

    # plot(means, label="L0 Loss", linewidth=2, legend=:topright) #, markershape=:circle)
    # plot!(means, ribbon=std_devs, fillalpha=0.3, label="", color=:blue)

    df_X = DataFrame(X, :auto)
    df_beta_initial = DataFrame(hcat(beta_initial), :auto) # hcat(y) makes vector y into a length(y) x 1 column matrix, alternatives: hcat(y) == [[y;;]] == reshape(y,:,1) == reshape(y,(length(y),1))
    df_actual_beta = DataFrame(hcat(actual_beta), :auto)
    df_noise_vector = DataFrame(hcat(noise_vector), :auto)
    df_y = DataFrame(hcat(y), :auto)
    df_setting = DataFrame([setting,])
    df_EGP_results = DataFrame(data_matrix, :auto)
    df_metrics = DataFrame([metrics,])
    df_metrics_std = DataFrame([metrics_std,])

    names_df = [("X",df_X), 
                ("beta_initial",df_beta_initial), 
                ("actual_beta",df_actual_beta), 
                ("noise_vector",df_noise_vector), 
                ("y",df_y), 
                ("setting",df_setting), 
                ("metrics",df_metrics), 
                ("metrics_std",df_metrics_std), 
                ("EGP_results",df_EGP_results)
                ]
    
    for (name, df_name) in names_df
        save_path = results_path * "/EGP_" * date_string *  nt_to_string(setting) * name * ".csv"
        CSV.write(save_path, df_name)
    end
end
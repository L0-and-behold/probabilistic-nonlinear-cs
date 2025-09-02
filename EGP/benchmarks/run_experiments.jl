using Pkg

# For the following to work, all necessary package must have been installed in a julia environment called `EGP`
# homedir_path = homedir()
# env_path = homedir_path * "/.julia/environments/EGP/Project.toml"
environment_path = dirname(@__DIR__)
env_path = environment_path * "/Project.toml"
println("Start")
Pkg.activate(env_path)

println(Pkg.project().path)

using CUDA
using Lux, LuxCUDA
using JLD2 # library for storing files in julia
using Dates

include("Relaxed-Lasso-FS-IHT.jl")
include("../src/data_generator.jl")
includet("../src/Exact_Gradient_Pruning.jl")


function logspace(start_value, end_value, num_points; digits=2)
    log_start = log10(start_value)
    log_end = log10(end_value)
    log_range = range(log_start, stop=log_end, length=num_points)
    values = Float32.(round.(10 .^ log_range, digits=digits))
    return values
end

example_settings = [
    (ids = 1, n = 100, p = 10, s1 = 5, s2 = 1000000), # low
    (ids = 2, n = 50, p = 1000, s1 = 5, s2 = 1000000), # high-5
    (ids = 3, n = 100, p = 1000, s1 = 10, s2 = 1000000), # high-10
    (ids = 4, n = 100, p = 10000, s1 = 10, s2 = 1000000), # very high-10
]


function run_experiments(settings; SNRs = logspace(0.05,60000,20), rhos = Float32.([0, 0.35, 0.7, 0.9]), val_set_size = 500, test_set_size = 1000, test_methods = [("EGP",Exact_Gradient_Pruning), ("Lasso",Plain_Lasso), ("Relaxed Lasso", Relaxed_Lasso), ("Forward Stepwise", Forward_Stepwise), ("IHT",IHT)],     repetitions = 10, seed = 42, date_string = nothing, path_addendum = "", add_n_p_to_path = false, save_models = true)

    ## Let methods run once to get rid of compile time
    ids, n, p, s1, s2 = settings[1]
    rho = 0f0; nu = 100000f0;
    Xd, yd, β, Σ, sigmaSquare = generate_data(n+val_set_size, p, s1, rho, nu; s2 = s2, dtype=Float32, seed=44, dev=gpu_device())
    X, y = Xd[1:end-val_set_size,:], yd[1:end-val_set_size]
    X_val, y_val = Xd[end-val_set_size+1:end,:], yd[end-val_set_size+1:end]
    for (tm_name, tm) in test_methods
        if tm == Exact_Gradient_Pruning
            compile_beta, compile_logs, compile_pp, compile_mask, compile_final_beta_before_finetuning, compile_final_beta, compile_final_y_pred, compile_logs_finetuning, compile_alphas, compile_val_losses_depending_on_beta, compile_trank, compile_tstate, compile_total_time, final_sigma, sigma_before_finetuning = Exact_Gradient_Pruning(X, y; tstate=nothing, Xval=X_val, yval=y_val, alpha_range=(0f0, Float32(180)), lr=1f-1,  lr_finetuning_range=(1f-2,1f-5), optimizer=ArrayAdam, dev=gpu_device(), seed=42, batch_size=100, max_epochs=1200, min_epochs=100, min_epochs_finetuning=500, max_epochs_finetuning = 15000, convergence_check_interval=20, convergence_check_interval_finetuning=5, verbose=true, smoothing_window=10, min_possible_deviation=1f-4, min_possible_deviation_finetuning=0f0, weight_factor=15f0, weight_factor_trank = 0.0025f0, finetuning=true, initial_p_value = 1.0f0, sieve_factor=0.9f0, active_set_criterion_number=0, gpu_training_starting_from_size=5000, convergence_max_count=1, lr_variation = false, alpha_block_size = 5, block_number = 33, l1_regularization=true, l1_range = (0.0f0,100f0), l0_regularization=true, interpolation_form=:logarithmic, log_val_loss_every_epoch=true, log_selection_loss_every_epoch=true, ft_noise_range=(0f0, 0.7f0), ft_beta_number=200, equalize_train_val_loss = false, a_interpolation_factor=0.1f0, mask_stability_factor=0.01f0, lr_number=3);
        else
            compile_beta, compile_y_pred, compile_final_beta, compile_val_losses_depending_on_beta, compile_val_ind, compile_total_time = tm(X, y, X_val, y_val)
        end
    end



    tls = length(settings) * repetitions * length(SNRs) * length(rhos) * length(test_methods) # total number of simulations

    Random.seed!(seed)
    idd = 0

    dir_path = @__DIR__
    results_path = dir_path * "/results"
    if !isdir(results_path) # Create the folder for storing the results if it doesn't exist
        mkpath(results_path)
    end

    if isnothing(date_string)
        date_string = string(today())
    end
    
    for (ids, n, p, s1, s2) in settings
        for (method_name, tmethod) in test_methods
            results_path_and_title = results_path * "/results_" * date_string * path_addendum * "_setting_" * string(ids) * "_method_" * method_name
            if add_n_p_to_path
                results_path_and_title = results_path_and_title * "_n_" * string(n) * "_p_" * string(p) * ".jld2"
            else
                results_path_and_title = results_path_and_title * ".jld2"
            end
            jldopen(results_path_and_title, "w") do file
                for repetition_number in 1:repetitions
                    for nu in SNRs
                        for rho in rhos
                            idd += 1
                            println("")
                            println("id: ",idd," / ", tls)
                            println("n=",n," p=",p," s1=",s1," s2=",s2," nu=",nu," rho=",rho)
                            println(method_name)
                            println("rep-number: ", repetition_number)
                            flush(stdout) # this is necessary when submitting this as a batch job so that output is immediately written to the corresponding .out file, cf. https://discourse.julialang.org/t/running-julia-with-slurm-output-not-being-printed-until-after-completion/102728

                            Xd, yd, β, Σ, sigmaSquare = generate_data(n+val_set_size, p, s1, rho, nu; s2 = s2, dtype=Float32, seed=42+repetition_number, dev=gpu_device())
                                
                            X, y = Xd[1:end-val_set_size,:], yd[1:end-val_set_size]
                            
                            X_val, y_val = Xd[end-val_set_size+1:end,:], yd[end-val_set_size+1:end]

                            model_seed = 42+repetitions+repetition_number
            
                            if tmethod == Exact_Gradient_Pruning
                                beta, logs, pp, mask, final_beta_before_finetuning, final_beta, final_y_pred, logs_finetuning, alphas, val_losses_depending_on_beta, trank, tstate, total_time, final_sigma, sigma_before_finetuning = Exact_Gradient_Pruning(X, y; tstate=nothing, Xval=X_val, yval=y_val, alpha_range=(0f0, Float32(180)), lr=1f-1, lr_finetuning_range=(1f-2,1f-5), optimizer=ArrayAdam, dev=gpu_device(), seed=model_seed, batch_size=100, max_epochs=1200, min_epochs=100, min_epochs_finetuning=500, max_epochs_finetuning = 15000, convergence_check_interval=20, convergence_check_interval_finetuning=5, verbose=false, smoothing_window=10, min_possible_deviation=1f-4, min_possible_deviation_finetuning=0f0, weight_factor=15f0, weight_factor_trank = 0.0025f0, finetuning=true, initial_p_value = 1.0f0, sieve_factor=0.9f0, active_set_criterion_number=0,  gpu_training_starting_from_size=5000, convergence_max_count=1, lr_variation = false,  alpha_block_size = 5, block_number = 33, l1_regularization=true, l1_range = (0.0f0,100f0), l0_regularization=true, interpolation_form=:logarithmic, log_val_loss_every_epoch=true, log_selection_loss_every_epoch=true, ft_noise_range=(0f0, 0.7f0), ft_beta_number=200, equalize_train_val_loss = false, a_interpolation_factor=0.1f0, mask_stability_factor=0.01f0, lr_number=3);
                            else
                                beta, _, final_beta, val_losses_depending_on_beta, val_ind, total_time = tmethod(X, y, X_val, y_val)
                            end

                            # compute metrics
                            ## active set reconstruction error
                            a1 = zeros32(length(final_beta))
                            a1[final_beta .!= 0] .= 1
                            a2 = zeros32(length(final_beta))
                            a2[β .!= 0] .= 1
                            active_set_reconstruction_error = sum(abs.(a1.-a2))
                            ## Relative Risk
                            beta_difference = final_beta .- β
                            RR_num = beta_difference' * Σ * beta_difference
                            RR_denom = β' * Σ * β
                            RR = RR_num / RR_denom
                            ## Relative Test Error
                            RTE = (RR_num + sigmaSquare) / sigmaSquare # (perfect score is 1 and null score is `nu + 1`)
                            ## Proportion of variance explained
                            PVE = 1 - (RR_num + sigmaSquare) / (RR_denom + sigmaSquare)
                            ## number of non-zeros
                            nnz = sum(a1)
                            ## Test Error
                            Xt, yt, βt, Σt, sigmaSquaret = generate_data(test_set_size, p, s1, rho, nu; s2 = s2, dtype=Float32, seed=43, dev=gpu_device())
                            
                            TE = sum((yt .- Xt * final_beta).^2) / length(yt)
                            ## Expected Test Loss
                            ETL = RR_num

                            if save_models
                                if tmethod == Exact_Gradient_Pruning
                                    result = (method_name = method_name, setting=(ids=ids,n=n,p=p,s1=s1,s2=s2,nu=nu,rho=rho,seed=seed), logs=logs, beta = beta, pp=pp, mask=mask, final_beta_before_finetuning=final_beta_before_finetuning, final_beta=final_beta, logs_finetuning=logs_finetuning, alphas=alphas, val_losses_depending_on_beta=val_losses_depending_on_beta, trank=trank, total_time=total_time, ASRE = active_set_reconstruction_error, RR=RR, RTE = RTE, PVE = PVE, nnz=nnz, TE=TE, ETL=ETL)
                                else
                                    result = (method_name = method_name, setting=(ids=ids,n=n,p=p,s1=s1,s2=s2,nu=nu,rho=rho,seed=seed), beta=beta, final_beta=final_beta, val_losses_depending_on_beta=val_losses_depending_on_beta, val_ind=val_ind, total_time=total_time, ASRE = active_set_reconstruction_error, RR=RR, RTE = RTE, PVE = PVE, nnz=nnz, TE=TE, ETL=ETL)
                                end
                            else
                                if tmethod == Exact_Gradient_Pruning
                                    result = (method_name = method_name, setting=(ids=ids,n=n,p=p,s1=s1,s2=s2,nu=nu,rho=rho,seed=seed), total_time=total_time, ASRE = active_set_reconstruction_error, RR=RR, RTE = RTE, PVE = PVE, nnz=nnz, TE=TE, ETL=ETL)
                                else
                                    result = (method_name = method_name, setting=(ids=ids,n=n,p=p,s1=s1,s2=s2,nu=nu,rho=rho,seed=seed), total_time=total_time, ASRE = active_set_reconstruction_error, RR=RR, RTE = RTE, PVE = PVE, nnz=nnz, TE=TE, ETL=ETL)
                                end
                            end
                            println(total_time, " seconds")

                            file["result_$idd"] = result
                        end
                    end
                end
            end
        end
    end
    println("Done.")
end
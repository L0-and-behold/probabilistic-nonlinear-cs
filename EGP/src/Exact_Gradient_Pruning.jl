
using CUDA
using Lux, LuxCUDA, Optimisers, Printf, Random, Zygote
using Octavian, LoopVectorization
using Accessors

include("ArrayAdam.jl")
include("custom_linear_probabilistic_model_matrix.jl")
include("custom_linear_probabilistic_model_matrix_gauss.jl")
include("convergence.jl")
include("custom_linear_model.jl")
include("custom_linear_model_Gauss.jl")

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

function Exact_Gradient_Pruning(X::Matrix{Float32}, y::Vector{Float32}; tstate=nothing, Xval = nothing, yval = nothing, alpha_range::Tuple{Float32, Float32} = (0f0, Float32(180)), lr::Float32 = 1f-1, lr_finetuning_range=(1f-2,1f-5), optimizer=ArrayAdam, dev = gpu_device(), seed = 42, batch_size = 100, max_epochs=1200, max_epochs_finetuning=15000, min_epochs=100, min_epochs_finetuning=500, convergence_check_interval = 20, convergence_check_interval_finetuning = 5, verbose=true, smoothing_window = 10, smoothing_window_finetuning=smoothing_window, min_possible_deviation=1f-4, min_possible_deviation_finetuning = 0f0, finetuning = true, weight_factor = 15f0, initial_p_value = 1.0f0, sieve_factor = 0.9f0, active_set_criterion_number=0, mask_stability_factor=0.01f0, log_L0_loss=false, convergence_max_count=1, gpu_training_starting_from_size=5000, lr_variation = false, weight_factor_trank = 0.0025f0, l0_regularization=true, l1_regularization=true, l1_range::Tuple{Float32, Float32} = (0f0,100f0), interpolation_form=:logarithmic, l1_regularization_finetuning=false, log_val_loss_every_epoch=false, log_selection_loss_every_epoch=false, ft_beta_number = 200, ft_noise_range=(0f0, 0.7f0), equalize_train_val_loss = false, a_interpolation_factor=0.1f0, gauss_loss=false, initial_sigma_value=1f0, l1_block_size = 5, l0_block_size = 33, lr_block_size=3)
    t0 = time()
    if dev == gpu_device()
        if !cuda_driver_available()
            dev == cpu_device()
        end
    end
    # if normalize_data
    #     X .-= minimum(X)
    #     X ./= maximum(X)
    #     y .-= minimum(y)
    #     y ./= maximum(y)
    #     if !isnothing(Xval)
    #         Xval .-= minimum(Xval)
    #         Xval ./= maximum(Xval)
    #     end
    #     if !isnothing(yval)
    #         yval .-= minimum(yval)
    #         yval ./= maximum(yval)
    #     end
    # end

    # if !lr_variation
    #     lr_block_size = 1
    # end
    if !gauss_loss
        log_gauss_loss=false
    end
    initial_seed = seed
    n, p = size(X)
    beta_number = l1_block_size * l0_block_size * lr_block_size
    if p * beta_number <= gpu_training_starting_from_size # if p * beta_number is small, then cpu training can be much faster
        dev = cpu_device()
    end
    batch_size = min(batch_size, n)

    function log2space(num_points, start_value, end_value; factor=0.5)
        if num_points == 1
            return Float32.([start_value])
        else
            eps = 1e-20
            log_start = log(start_value+eps)
            log_end = log(end_value+eps)
            log_range = range(log_start, stop=log_end, length=num_points)
            values = Float32.(exp.(factor .* log_range))
            values .-= minimum(values) 
            values ./= maximum(values)
            values .= start_value .+ (end_value - start_value) .* values
            return values
        end
    end
    function interpolation_block(p, a_range; factor=x->2^(0.5*x))
        a1, a2 = a_range
        if p > 1 
            block = collect(1e0:Float64(p))
            block .= factor.(block) .- factor(1e0)
            block .*= (a2-a1) / Float64(factor(p)-factor(1))
        else
            block = zeros64(p)
        end
        block .+= a1
        return Float32.(block)
    end
    function interpolate(step, total_steps)
        return total_steps == 1 ? 0f0 : Float32((step - 1) / (total_steps-1))
    end
    function interpolation_chain(l0_block_size, l1_block, lr_range, ar_range; interpolation_form=:logarithmic, a_interpolation_factor=0.1f0, lr_block_size=2)
        l1_length = length(l1_block)
        lr_block_length = l1_length * l0_block_size
        beta_number = lr_block_length * lr_block_size
        lr_vec = zeros32(beta_number) |> dev
        ar_vec = zeros32(beta_number) |> dev
        l1_vec = zeros32(beta_number) |> dev
        if interpolation_form == :logarithmic
            a_seq = log2space(l0_block_size, ar_range[1],ar_range[2]; factor=a_interpolation_factor)
        else
            a_seq = interpolation_block(l0_block_size, ar_range; factor=x->x)
        end
        for ln in 1:lr_block_size
            for bn in 1:l0_block_size
                ar_vec[1+(ln-1)*lr_block_length+l1_length*(bn-1):(ln-1)*lr_block_length+l1_length*bn] .= a_seq[bn]
                l1_vec[1+(ln-1)*lr_block_length+l1_length*(bn-1):(ln-1)*lr_block_length+l1_length*bn] .= l1_block
            end
            interpolation_factor = interpolate(ln, lr_block_size)
            lr_vec[1+(ln-1)*lr_block_length:ln*lr_block_length] .= lr_range[1] .+ ((lr_range[2]-lr_range[1]) * interpolation_factor)
        end
        lr_matrix = reshape(lr_vec, 1, beta_number)
        a_matrix = reshape(ar_vec, 1, beta_number)
        l1_matrix = reshape(l1_vec, 1, beta_number)
        return lr_matrix, a_matrix, l1_matrix
    end
    
    function create_batches(X, y, batch_size, beta_number)
        n, _ = size(X)
        num_batches = div(n, batch_size) + (n % batch_size != 0 ? 1 : 0)
        data = Vector{Tuple{Matrix{Float32}, Vector{Float32}}}(undef, num_batches) 
        for i in 1:num_batches
            start_index = (i - 1) * batch_size + 1
            end_index = min(i * batch_size, n)
            X_batch = X[start_index:end_index, :]
            y_batch = y[start_index:end_index]
            data[i] = (X_batch, y_batch)
        end
        return data, num_batches
    end

    train_data, num_batches = create_batches(X, y, batch_size, beta_number)

    bb = Float32(1 / batch_size / beta_number) # / 5000 ) # / beta_number)
    bba = Float32(1 / batch_size / num_batches / beta_number)
    ar_range = (alpha_range[1] * bba, alpha_range[2] * bba)
    l1_range_bba = (l1_range[1] * bba, l1_range[2] * bba)
    l1_block = interpolation_block(l1_block_size, l1_range_bba; factor=x->x) |> dev

    lr_range = (1f0, Float32(1/sqrt(6e0/p)) ); # sqrt(6/p) is the max value of glorot_uniform initialization
    init_matrix, a_matrix, l1_matrix = interpolation_chain(l0_block_size, l1_block, lr_range, ar_range; interpolation_form=interpolation_form, a_interpolation_factor=a_interpolation_factor, lr_block_size=lr_block_size)

    lr_matrix = similar(init_matrix)
    lr_matrix .= lr

    loss_fun = Lux.MSELoss()
    

    if isnothing(tstate)
        @label before_tstate_initialization
        # Random.seed!(seed)
        # rng = Random.default_rng(seed)
        rng = MersenneTwister(seed)

        if gauss_loss
            model = ProlinearMatrixGauss(size(X)[2], beta_number)
        else
            model = ProlinearMatrix(size(X)[2], beta_number)
        end
        ps, st = Lux.setup(rng, model) |> dev
        ps.p .= initial_p_value
        if gauss_loss
            ps.sigma .= initial_sigma_value
        end
        # ps.w .-= minimum(ps.w)
        # ps.w ./= maximum(ps.w)

        ps.w .*= init_matrix
        if optimizer == ArrayAdam
            opt = optimizer(lr_matrix)
        else
            opt = optimizer(lr)
        end

        
        tstate = Training.TrainState(model, ps, st, opt)
    end
    initial_w = copy(ps.w)
    vjp = AutoZygote()

    function update!(grads, tstate, (Xb,yb))
        if gauss_loss
            gauss_factor = 1f0 ./ (2f0 .* tstate.parameters.sigma.^2 .* log(2f0))
        else
            gauss_factor = 1f0
        end
        wp = tstate.parameters.w .* tstate.parameters.p
        lo = (yb .- Xb * wp)
        diff_v = -2 .* bb .* (Xb' * lo) .* gauss_factor
        grads.w .= diff_v .* tstate.parameters.p # MSE gradients
        grads.p .= diff_v .* tstate.parameters.w # MSE gradients

        Xbs = sum(Xb.^2, dims=1)' # Xb.^2 is typically the most compute-intense operation
        gp = (tstate.parameters.w.^2 .* (1 .- 2 .* tstate.parameters.p)) .* Xbs
        grads.p .+= gp .* bb .* gauss_factor # .+ a_matrix
        if l0_regularization
            grads.p .+= a_matrix
        end
        gw = (2 .* wp .* (1 .- tstate.parameters.p)) .* Xbs
        grads.w .+= gw .* bb .* gauss_factor

        loss = sum(lo.^2) * bb
        if gauss_loss
            plain_loss = sum(lo.^2, dims=1) # / Float32(batch_size)
            grads.sigma .= 1f0 ./ (log(2f0) .* tstate.parameters.sigma) .* Float32(1 / beta_number) .- plain_loss ./ (log(2f0) .* tstate.parameters.sigma.^3) .* bb
        end
        
        if l1_regularization
            grads.w .+= l1_matrix .* tstate.parameters.p .* sign.(tstate.parameters.w)
            grads.p .+= l1_matrix .* abs.(tstate.parameters.w)
        end

        if lr_variation
            grads.p .*= lr_matrix
            grads.w .*= lr_matrix
        end

        if log_L0_loss
            loss_L0 = loss + sum(Xb.^2 * (tstate.parameters.w .* wp .* (1 .- tstate.parameters.p))) * bb
        else
            loss_L0 = 0f0
        end
        if log_gauss_loss
            loss_gauss = tstate.parameters.sigma # bb .* (plain_loss .* gauss_factor .+ 0.5f0 .* log.(tstate.parameters.sigma.^2))
        else
            loss_gauss = 0f0
        end

        return grads, loss, loss_L0, loss_gauss
    end

    if !isnothing(Xval) && !isnothing(yval)
        val_data, _ = create_batches(Xval, yval, batch_size, beta_number)
    else
        val_data = nothing
    end

    function compute_val_loss(val_data, logs, val_loss_fun, betas, sigma=initial_sigma_value)
        if !isnothing(val_data)
            val_loss = 0f0
            for (Xv,yv) in val_data
                Xv, yv = (Xv,yv) .|> dev
                vloss = (yv .- Xv * betas).^2
                if gauss_loss
                    len_vloss = Float32(size(vloss, 1))
                    vloss = sum(vloss,dims=1) ./ (2f0 .* sigma.^2 .* log(2f0)) / len_vloss .+ 0.5f0 .* log2.(sigma.^2)
                end
                vloss = sum(vloss) / length(vloss)
                val_loss += vloss
            end
            val_loss /= length(val_data)
            if !isnothing(logs)
                push!(logs["val_loss"], val_loss)
            end
            return val_loss
        else
            return nothing
        end
    end
    function compute_loss_ft(val_data, logs, betas, sigma=initial_sigma_value; mode=:min, mask=nothing)
        if !isnothing(val_data)
            val_loss = zeros(Float32, 1, size(betas,2)) |> dev
            for (Xv,yv) in val_data
                Xv, yv = (Xv,yv) .|> dev
                if !isnothing(mask)
                    vloss = (yv .- Xv[:,mask] * betas).^2
                else
                    vloss = (yv .- Xv * betas).^2
                end
                vloss = sum(vloss, dims=1) / length(yv)
                if gauss_loss
                    vloss = vloss ./ (2f0 .* sigma.^2 .* log(2f0)) .+ 0.5f0 .* log2.(sigma.^2)
                end
                val_loss .+= vloss
            end
            val_loss ./= length(val_data)
            if mode == :min || mode == :avg
                if mode == :min
                    argm = argmin(val_loss)[2]
                    val_loss = val_loss[argm]
                else
                    argm = nothing
                    val_loss = sum(val_loss) / length(val_loss)
                end
                if !isnothing(logs)
                    push!(logs["val_loss"], val_loss)
                end
            else
                argm = nothing
            end
            return val_loss, argm
        else
            return nothing, nothing
        end
    end

    function some_column_exhibits_active_set_convergence(p)
        columns_only_containing_0_or_1 = all(x -> x == 0 || x == 1, p, dims=1)
        valid_columns_indices = findall(columns_only_containing_0_or_1)
        if length(valid_columns_indices) >= active_set_criterion_number
            valid_columns_indices = map(x->x[2],valid_columns_indices) |> dev
            # Check if those columns have at least one non-zero entry
            valid_columns = p[:, valid_columns_indices]
            contains_non_zero = any(x -> x != 0, valid_columns, dims=1)
            result = sum(contains_non_zero) >= active_set_criterion_number
        else
            result = false
        end
        return result
    end

    function select_beta(pw, pp, sdata, sieve_factor, sigmas=nothing; add_zero=false, epoch_n=0)
        beta_s = pw .* pp
        if add_zero
            beta_s = hcat(beta_s, zeros32(size(beta_s,1))) # make sure the 0-column is always included as candidate for scenarios in which noise is particularly high.
            # y_pred = X * beta_s
        end
    
        losses_depending_on_beta = zeros32(1, size(beta_s,2)) |> dev
        for (Xd,yd) in sdata
            Xd, yd = (Xd, yd) .|> dev
            # generate mask using weighted average, where the weight is e^{-weight_factor * val_loss}
            loss_summand_depending_on_beta = sum((yd .- Xd * beta_s).^2, dims=1) ./ length(yd)
            if gauss_loss && !isnothing(sigmas)
                if add_zero
                    loss_summand_depending_on_beta[:,1:end-1] .= loss_summand_depending_on_beta[:,1:end-1] ./ (2f0 .* sigmas.^2 .* log(2f0)) .+ 0.5f0 .* log2.(sigmas.^2)
                else
                    loss_summand_depending_on_beta .= loss_summand_depending_on_beta ./ (2f0 .* sigmas.^2 .* log(2f0)) .+ 0.5f0 .* log2.(sigmas.^2)
                end
            end
            losses_depending_on_beta .+= loss_summand_depending_on_beta
        end
        losses_depending_on_beta ./= length(sdata)
        if gauss_loss
            losses_depending_on_beta .-= minimum(losses_depending_on_beta) # to avoid negative losses
        end

        # estimate final_beta by assuming that the true active set is exponentially smaller than p
        active_set_sizes = sum(pp, dims=1)
        if add_zero
            active_set_sizes = hcat(active_set_sizes, 0f0)
        end

        trank = weight_factor_trank .* active_set_sizes .+ log.( 1f-15 .+ losses_depending_on_beta) # trank expresses that active_set_sizes are exponentially more important for selection than losses. 1f-15 is inserted only for numerical stability. weight_factor_trank regulates how important sparsity is for selection.

        # normalize trank. important to achieve more dataset independence
        trank .-= minimum(trank)
        if maximum(trank) > 0
            trank ./= maximum(trank)
        else
            trank .= 1
        end

        argmin_trank = argmin(trank)[2]
        argmin_losses = argmin(losses_depending_on_beta)[2]
        coefs = exp.(-weight_factor .* trank) # use an exponential family for the weighted average
        coefs[coefs .< minimum(coefs) + (maximum(coefs) - minimum(coefs)) * sieve_factor] .= 0 # keep only the top sieve_factor*100 percent of candidates
        coefs .= coefs ./ sum(coefs)
        if dev == gpu_device() && cuda_driver_available()
            selected_beta = CUDA.zeros(size(X)[2]) 
        else
            selected_beta = zeros32(size(X)[2]) 
        end
        if add_zero
            weighted_avg = coefs .* hcat(pp, zeros32(size(pp,1)))
        else
            weighted_avg = coefs .* pp
        end
        mask = Bool.(vec(round.(sum(weighted_avg, dims=2))))
        
        selected_beta[mask] = sum(coefs .* beta_s[mask,:], dims=2)
        if gauss_loss && !isnothing(sigmas)
            if add_zero
                selected_sigma = sum(coefs[:,1:end-1] .* sigmas)
            else
                selected_sigma = sum(coefs .* sigmas)
            end
        else
            selected_sigma = nothing
        end

        loss_selected_beta = 0f0
        for (Xd,yd) in sdata
            Xd, yd = (Xd, yd) .|> dev
            # generate mask using weighted average, where the weight is e^{-weight_factor * val_loss}
            loss_selected_beta_summand = sum((yd .- Xd[:,mask] * selected_beta[mask]).^2) / length(yd)
            if gauss_loss
                loss_selected_beta_summand = loss_selected_beta_summand / (2f0 * selected_sigma^2 * log(2f0)) + 0.5f0 * log2(selected_sigma^2)
            end
            loss_selected_beta += loss_selected_beta_summand
        end
        loss_selected_beta /= length(sdata)

        return selected_beta, mask, losses_depending_on_beta, coefs, loss_selected_beta, argmin_trank, argmin_losses, selected_sigma
    end

    function train(tstate::Training.TrainState, vjp, loss_fun, train_data, val_data, max_epochs, min_epochs, convergence_check_interval, smoothing_window, min_possible_deviation)
        logs = Dict{String, Any}(
            "epoch" => Int[],
            "train_loss" => Float32[],
            "val_loss" => Float32[],
            "L0_loss" => Float32[],
            "gauss_loss" => Matrix{Float32}[],
            "selection_loss" => Float32[],
        )
        val_loss_fun = Lux.MSELoss()
        if gauss_loss
            prev_val_loss = compute_val_loss(val_data, nothing, val_loss_fun, tstate.parameters.w .* tstate.parameters.p, tstate.parameters.sigma)
        else
            prev_val_loss = compute_val_loss(val_data, nothing, val_loss_fun, tstate.parameters.w .* tstate.parameters.p)
        end
        if !isnothing(val_data)
            c_data = val_data
        else
            c_data = train_data
        end
        if gauss_loss
            selected_beta, mask, _, _, prev_loss_c, _, argmin_losses, selected_sigma = select_beta(tstate.parameters.w, tstate.parameters.p, c_data, sieve_factor, tstate.parameters.sigma)
        else
            selected_beta, mask, _, _, prev_loss_c, _, argmin_losses, selected_sigma = select_beta(tstate.parameters.w, tstate.parameters.p, c_data, sieve_factor)
        end
        best_betas = zeros32(size(tstate.parameters.w, 1)) |> dev # selected_beta
        best_masks = mask
        best_sigmas = Float32[initial_sigma_value]
        convergence_counter = 0
        prev_p = copy(tstate.parameters.p[:,argmin_losses])
        if gauss_loss
            grads = (w = copy(tstate.parameters.w), p = copy(tstate.parameters.p), sigma = copy(tstate.parameters.sigma))
        else
            grads = (w = copy(tstate.parameters.w), p = copy(tstate.parameters.p))
        end
        for epoch in 1:max_epochs
            avg_loss = 0f0
            if log_L0_loss 
                avg_L0_loss = 0f0
                L0_loss_columns = a_matrix .* sum(tstate.parameters.p, dims=1)
                L0_loss_avg = sum(L0_loss_columns) / length(L0_loss_columns)
            end
            if log_gauss_loss
                avg_gauss_loss = zeros32(size(tstate.parameters.sigma)[1],size(tstate.parameters.sigma)[2])
            end
            for batch in train_data
                batch = batch .|> dev
                grads, loss, loss_L0, loss_gauss = update!(grads, tstate, batch)
                tstate = Training.apply_gradients!(tstate, grads)
                clamp!(tstate.parameters.p, 0, 1)
                avg_loss += loss
                if log_L0_loss 
                    avg_L0_loss += loss_L0
                end
                if log_gauss_loss 
                    avg_gauss_loss .+= loss_gauss
                end
            end
            avg_loss /= length(train_data)
            push!(logs["epoch"], epoch)
            push!(logs["train_loss"], avg_loss)
            if log_L0_loss
                avg_L0_loss /= length(train_data)
                push!(logs["L0_loss"], avg_L0_loss + L0_loss_avg / bb)
            end
            if log_gauss_loss
                avg_gauss_loss ./= length(train_data)
                push!(logs["gauss_loss"], avg_gauss_loss)
            end

            val_loss = nothing
            if log_val_loss_every_epoch
                val_loss = compute_val_loss(val_data, logs, val_loss_fun, tstate.parameters.w .* tstate.parameters.p)
                verbose && !isnothing(val_loss) && @printf "\r Epoch: %3d \t Loss: %.5g \t  Val-Loss: %.5g" epoch avg_loss val_loss
            else
                verbose && @printf "\r Epoch: %3d \t Loss: %.5g" epoch avg_loss
            end
            if log_selection_loss_every_epoch
                if gauss_loss
                    _, _, _, _, loss_c, _, _, _ = select_beta(tstate.parameters.w, tstate.parameters.p, c_data, sieve_factor, tstate.parameters.sigma; epoch_n=epoch)
                else
                    _, _, _, _, loss_c, _, _, _ = select_beta(tstate.parameters.w, tstate.parameters.p, c_data, sieve_factor; epoch_n=epoch)
                end
                push!(logs["selection_loss"], loss_c)
            end

            # BEGIN check convergence
            if epoch % convergence_check_interval == 0
                if gauss_loss
                    selected_beta, mask, _, _, loss_c, argmin_trank, argmin_losses, selected_sigma = select_beta(tstate.parameters.w, tstate.parameters.p, c_data, sieve_factor, tstate.parameters.sigma; add_zero=true, epoch_n=epoch)
                else
                    selected_beta, mask, losses_depending_on_beta, coefs, loss_c, argmin_trank, argmin_losses, selected_sigma = select_beta(tstate.parameters.w, tstate.parameters.p, c_data, sieve_factor; add_zero=true, epoch_n=epoch)
                end                            
                if !log_selection_loss_every_epoch
                    push!(logs["selection_loss"], loss_c)
                end
                if loss_c >= prev_loss_c - min_possible_deviation
                    best_betas = hcat(best_betas, selected_beta)
                    best_masks = hcat(best_masks, mask)
                    if gauss_loss
                        best_sigmas = push!(best_sigmas, selected_sigma)
                    end
                    convergence_counter += 1
                    if epoch >= min_epochs && convergence_counter >= convergence_max_count
                        if argmin_losses == size(tstate.parameters.p, 2) + 1
                            cf_vec = zeros32(size(tstate.parameters.p, 1)) |> dev
                        else
                            cf_vec = tstate.parameters.p[:,argmin_losses]
                        end
                        sub_dev = abs.(prev_p .- cf_vec)
                        active_set_conv = all(sub_dev .<= mask_stability_factor)
                        if active_set_conv
                            verbose && println("\nConvergence at epoch ", epoch)
                            break
                        end
                    end
                else
                    convergence_counter = 0
                end
                prev_loss_c = loss_c
                if argmin_losses == size(tstate.parameters.p, 2) + 1
                    prev_p .= zeros32(size(tstate.parameters.p, 1)) |> dev
                else
                    prev_p .= tstate.parameters.p[:,argmin_losses]
                end
            end
            # END check convergence
        end
        best_sigmas = reshape(best_sigmas, 1, length(best_sigmas))
        return tstate, logs, best_betas, best_masks, best_sigmas
    end

    function gauss_loss_fun(model, ps, st, (X,y))
        unaugmented_loss = sum((y .- X * ps.v_params).^2, dims=1) ./ Float32(size(X, 1))
        loss = sum(unaugmented_loss ./ (log(2f0) .* ps.sigma.^2 .* 2f0) .+ 0.5f0 .* log2.(ps.sigma.^2)) / length(unaugmented_loss)
        stats = []
        return loss, st, stats
    end
    
    function finetune(tstate::Training.TrainState, vjp, train_data, val_data, max_epochs, min_epochs, convergence_check_interval, smoothing_window, min_possible_deviation, ft_l1_param, initial_best_sigma_value=initial_sigma_value)
        ft_logs = Dict{String, Any}(
            "epoch" => Int[],
            "train_loss" => Float32[],
            "val_loss" => Float32[],
        )
        if gauss_loss
            loss_fun = gauss_loss_fun
            best_sigmas = [initial_best_sigma_value]
        else
            loss_fun = Lux.MSELoss()
            best_sigmas = nothing
        end
        best_betas = tstate.parameters.v_params[:,1]
        if gauss_loss
            prev_val_loss, _ = compute_loss_ft(val_data, nothing, tstate.parameters.v_params, tstate.parameters.sigma)
        else
            prev_val_loss, _ = compute_loss_ft(val_data, nothing, tstate.parameters.v_params)
        end
        if equalize_train_val_loss
            if gauss_loss
                prev_train_loss, _ = compute_loss_ft(train_data, nothing, tstate.parameters.v_params, tstate.parameters.sigma)
            else
                prev_train_loss, _ = compute_loss_ft(train_data, nothing, tstate.parameters.v_params)
            end
            dlo = prev_train_loss - prev_val_loss 
            cross_direction = sign(dlo)
            if cross_direction < 0
                tmp = copy(val_data)
                val_data = copy(train_data)
                train_data = tmp
            elseif cross_direction == 0
                return tstate, ft_logs
            end
        end
        prev_params = copy(tstate.parameters.v_params)
        if gauss_loss
            prev_sigmas = copy(tstate.parameters.sigma)
        end
        for epoch in 1:max_epochs
            avg_loss = 0f0
            for batch in train_data
                batch = batch .|> dev
                if gauss_loss
                    grads, loss, _, tstate = Training.compute_gradients(vjp, loss_fun, (batch[1], batch[2]), tstate)
                else
                    grads, loss, _, tstate = Training.compute_gradients(vjp, loss_fun, (batch[1], repeat(batch[2],1,size(tstate.parameters.v_params,2))), tstate)
                end
                if l1_regularization_finetuning
                    grads.v_params .+= ft_l1_param .* sign.(tstate.parameters.v_params)
                end
                tstate = Training.apply_gradients!(tstate, grads)
                avg_loss += loss
            end
            avg_loss /= length(train_data)
            push!(ft_logs["epoch"], epoch)
            push!(ft_logs["train_loss"], avg_loss)
            
            if log_val_loss_every_epoch
                if gauss_loss
                    val_loss, argm = compute_loss_ft(val_data, ft_logs, tstate.parameters.v_params, tstate.parameters.sigma)
                else
                    val_loss, argm = compute_loss_ft(val_data, ft_logs, tstate.parameters.v_params)
                end
            end
            # BEGIN check convergence
            if epoch % convergence_check_interval == 0
                if !log_val_loss_every_epoch
                    if gauss_loss
                        val_loss, argm = compute_loss_ft(val_data, ft_logs, tstate.parameters.v_params, tstate.parameters.sigma)
                    else
                        val_loss, argm = compute_loss_ft(val_data, ft_logs, tstate.parameters.v_params)
                    end
                end
                if !isnothing(val_loss)
                    verbose && @printf "\r FT-Epoch: %3d \t Loss: %.5g \t  Val-Loss: %.5g" epoch avg_loss val_loss
                    if equalize_train_val_loss
                        if (avg_loss - val_loss) <= min_possible_deviation 
                            verbose && println("\nValidation loss converged at epoch ", epoch)
                            tstate.parameters.v_params .= prev_params
                            if gauss_loss
                                tstate.parameters.sigma .= prev_sigmas
                            end
                            break
                        end
                    else
                        if (prev_val_loss - val_loss) <= min_possible_deviation
                            best_betas = hcat(best_betas, tstate.parameters.v_params[:,argm])
                            if gauss_loss
                                best_sigmas = push!(best_sigmas, sum(tstate.parameters.sigma[:,argm]))
                            end
                            if epoch >= min_epochs
                                verbose && println("\nValidation loss converged at epoch ", epoch)
                                break
                            end
                        end
                    end
                    prev_val_loss = val_loss
                    prev_params .= tstate.parameters.v_params
                    if gauss_loss
                        prev_sigmas .= tstate.parameters.sigma
                    end
                else
                    verbose && @printf "\r FT-Epoch: %3d \t Loss: %.5g \t" epoch avg_loss
                    if epoch >= min_epochs && is_saturated(ft_logs["train_loss"], smoothing_window; min_possible_deviation=min_possible_deviation) && is_saturated(ft_logs["val_loss"], round(Int, smoothing_window/5); min_possible_deviation=min_possible_deviation) 
                        verbose && println("\nTrain loss converged at epoch ", epoch)
                        break
                    end
                end
            end
            # END check convergence
        end
        if gauss_loss
            best_sigmas = reshape(best_sigmas, 1, length(best_sigmas))
        end
        return tstate, ft_logs, best_betas, best_sigmas
    end

    if verbose
        println("Training. Min-epochs: ", min_epochs)
    end

    @label before_training
    tstate, logs, best_betas, best_masks, best_sigmas = train(tstate, vjp, loss_fun, train_data, val_data, max_epochs, min_epochs, convergence_check_interval, smoothing_window, min_possible_deviation)
    
    beta = tstate.parameters.w .* tstate.parameters.p
    beta = hcat(beta, zeros32(size(beta,1))) # make sure the 0-column is always included as candidate for scenarios in which noise is particularly high.
   
    if !isnothing(val_data)
        c_data = val_data
        final_beta_before_finetuning, mask, losses_depending_on_beta, coefs, loss_selected_beta_before_finetuning, argmin_trank, argmin_losses, selected_sigma = select_beta(best_betas, best_masks, c_data, sieve_factor, best_sigmas; add_zero=true) # add_zero column to ensure that 0-solution is always among the candidates.
        @show argmin_trank
        @show argmin_losses
        @show losses_depending_on_beta
        @show size(losses_depending_on_beta)
    else
        c_data = train_data
        println("Note: final_beta is selected according to train_loss because no validation data was provided. We recommmend providing validation data to improve the final_beta estimate.")

        final_beta_before_finetuning, mask, losses_depending_on_beta, coefs, loss_selected_beta_before_finetuning, argmin_trank, argmin_losses, selected_sigma = select_beta(tstate.parameters.w, tstate.parameters.p, c_data, sieve_factor, best_sigmas; add_zero=true) # add_zero column to ensure that 0-solution is always among the candidates.

        @show argmin_trank
        @show argmin_losses
        @show losses_depending_on_beta


    end
    if gauss_loss
        sigma_before_finetuning = copy(selected_sigma)
    else
        sigma_before_finetuning = nothing
    end
    if verbose
        println("")
    end
    
    if finetuning
        if verbose
            println("Finetuning")
        end
        mask_cpu = mask |> cpu_device()
        X_reduced = X[:,mask_cpu]
        if size(X_reduced)[2] <= gpu_training_starting_from_size # if active_set_size * beta_number is small, then cpu training can be much faster
            dev = cpu_device()
            mask = mask |> dev
            final_beta_before_finetuning = final_beta_before_finetuning |> dev
            selected_sigma = selected_sigma |> dev
        end
        train_data_ft, _ = create_batches(X_reduced, y, batch_size, 1)
        if !isnothing(Xval) && !isnothing(yval)
            Xval_reduced = Xval[:,mask_cpu]
            val_data_ft, _ = create_batches(Xval_reduced, yval, batch_size, 1)
        else
            val_data_ft = nothing
        end

        if l1_regularization_finetuning
            weighted_l1_parameter = coefs .* hcat(l1_matrix, 1f0)
            ft_l1_param = sum(weighted_l1_parameter) / length(weighted_l1_parameter)
        else
            ft_l1_param = 0f0
        end

        rng = MersenneTwister(seed)
        deviation = reshape(interpolation_block(ft_beta_number, ft_noise_range; factor=x->x), 1, ft_beta_number)
        deviation = deviation .* randn(rng, Float32, size(X_reduced,2), ft_beta_number) |> dev
        sample_betas = repeat(final_beta_before_finetuning[mask], 1, ft_beta_number) .+ deviation
        if gauss_loss
            sample_sigmas = repeat([selected_sigma], 1, ft_beta_number)
            model_ft = LinearGauss(size(X_reduced,2); dims2 = ft_beta_number)
        else
            model_ft = Linear(size(X_reduced,2); dims2 = ft_beta_number)
        end
        ps_ft, st_ft = Lux.setup(rng, model_ft) |> dev
        ps_ft.v_params .= sample_betas
        if gauss_loss
            ps_ft.sigma .= sample_sigmas |> dev
        end
        lr_finetuning_matrix = interpolation_block(ft_beta_number, lr_finetuning_range; factor=x->x)
        lr_finetuning_matrix = reshape(lr_finetuning_matrix, 1, length(lr_finetuning_matrix))
        if optimizer == ArrayAdam
            tstate_ft = Training.TrainState(model_ft, ps_ft, st_ft, optimizer(lr_finetuning_matrix))
        else
            ft_lr = lr_finetuning_matrix[1, ceil(Int, size(lr_finetuning_matrix)[2]/2)]
            tstate_ft = Training.TrainState(model_ft, ps_ft, st_ft, optimizer(ft_lr))
        end

        tstate_ft, logs_finetuning, best_betas, best_sigmas = finetune(tstate_ft, vjp, train_data_ft, val_data_ft, max_epochs_finetuning, min_epochs_finetuning, convergence_check_interval_finetuning, smoothing_window_finetuning, min_possible_deviation_finetuning, ft_l1_param, selected_sigma)
        
        if !isnothing(Xval)
            if gauss_loss
                best_best_beta_loss, best_beta_argm = compute_loss_ft(val_data, nothing, best_betas, best_sigmas; mode=:min, mask=mask)
            else
                best_best_beta_loss, best_beta_argm = compute_loss_ft(val_data, nothing, best_betas; mode=:min, mask=mask)
            end
        else
            if gauss_loss
                best_best_beta_loss, best_beta_argm  = compute_loss_ft(train_data, nothing, best_betas, best_sigmas; mode=:min, mask=mask)
            else
                best_best_beta_loss, best_beta_argm  = compute_loss_ft(train_data, nothing, best_betas; mode=:min, mask=mask)
            end
        end
        
        final_beta = zeros32(size(X)[2])
        final_beta[mask_cpu] .= best_betas[:,best_beta_argm] |> cpu_device()
        if gauss_loss
            final_sigma = best_sigmas[:,best_beta_argm] |> cpu_device()
        else
            final_sigma = nothing
        end
        

        final_y_pred = X_reduced * final_beta[mask]
        if verbose
            println("")
        end
    else
        final_beta = final_beta_before_finetuning
        if gauss_loss
            final_sigma = selected_sigma
        else
            final_sigma = nothing
        end
        mask_cpu = mask |> cpu_device()
        final_beta_before_finetuning_cpu = final_beta_before_finetuning |> cpu_device()
        final_y_pred = X[:,mask_cpu] * final_beta_before_finetuning_cpu[mask_cpu] 
        logs_finetuning = Dict{String, Any}(
            "epoch" => Int[],
            "train_loss" => Float32[],
            "val_loss" => Float32[],
        )
    end

    t1 = time()
    total_time = t1-t0

    alphas = a_matrix ./ bba
    verbose && println("")
    beta, logs, pp, mask, final_beta_before_finetuning, final_beta, final_y_pred, logs_finetuning, alphas, losses_depending_on_beta, argmin_trank, final_sigma, sigma_before_finetuning = (beta, logs, tstate.parameters.p, mask, final_beta_before_finetuning, final_beta, final_y_pred, logs_finetuning, alphas, losses_depending_on_beta, argmin_trank, final_sigma, sigma_before_finetuning) |> cpu_device()
    return beta, logs, pp, mask, final_beta_before_finetuning, final_beta, final_y_pred, logs_finetuning, alphas, losses_depending_on_beta, argmin_trank, tstate, total_time, final_sigma, sigma_before_finetuning
end

using Flux: Chain

### Training

"""
Initialize empty logging dictionary with predefined keys for training metrics.
"""
function initialize_logs()
    logs = Dict{String, Any}(
        "epoch" => [],
        "train_loss" => [],
        "l0_loss" => [],
        "teacher_student_size_divergence" => [],
        "α_history" => [],
        "description_length" => [],
    )
    return logs
end

"""
Update training logs with current model metrics on test dataset.

# Arguments
- `model::Chain`: Student model to evaluate
- `teacher::Chain`: Teacher model for comparison
- `logs::Dict{String, Any}`: Existing logs dictionary to update
- `loss::Function`: Loss function to compute training loss
- `args`: Training arguments containing device and α parameter

# Returns
Updated logs dictionary with new metric values appended.

# Note
Assumes logging occurs every 10 epochs.
"""
function update_logs(
    model::Chain,
    teacher::Chain, 
    logs::Dict{String, Any}, 
    loss::Function,
    args
    )::Dict{String, Any}

    # println("Updating logs...")
    
    device = args.dev

    test_set = get_dataset(teacher, 1000, 1000, device=device)

    l0_loss(model) = args.α*true_byte_size(model)
    
    append!(logs["epoch"], length(logs["train_loss"])*10)
    append!(logs["train_loss"], loss(test_set[1][1], test_set[1][2]))
    append!(logs["teacher_student_size_divergence"], byte_size_compression(normal_form(model, device=device), normal_form(teacher, device=device)))
    append!(logs["α_history"], args.α)
    append!(logs["description_length"], description_length(teacher, model, 10000, device=device)[1])
    append!(logs["l0_loss"], l0_loss(model))

    return logs
end

"""
Save plot to artifact folder with error handling.

# Arguments
- `artifact_folder::String`: Directory path to save plot
- `plt`: Plot object to save
- `title::String`: Filename prefix for saved plot
"""
function save_plot(artifact_folder::String, plt, title::String)
    plotfilename = joinpath(artifact_folder, title*".png")
    try
        savefig(plt, plotfilename)
    catch e # the plot library is a bit buggy and sometimes throws errors
        @warn "Chaught error: $e. Not saving plot $plotfilename"
    end
end

"""
Save X-Y data pairs to CSV file.

# Arguments
- `artifact_folder`: Directory path to save CSV file
- `X_data::Vector`: X-axis data values
- `Y_data::Vector`: Y-axis data values  
- `title::String`: Filename prefix for saved CSV
"""
function save_CSV(artifact_folder, X_data::Vector, Y_data::Vector, title::String)
    filename = joinpath(artifact_folder, title*".csv")
    CSV.write(filename, DataFrame(X=X_data, Y=Y_data))
end

### Plots

"""
Create dual-panel plot of combined loss (training + L0 regularization).

# Arguments
- `logs::Dict{String, Any}`: Training logs containing epoch and loss data

# Returns
Plot object with global view and zoomed-in view of combined loss.
"""
function combined_loss_plot(logs::Dict{String, Any})
    X = logs["epoch"]
    Y = logs["l0_loss"] .+ logs["train_loss"]

    # combined loss - global
    p1 = plot(X, Y, title="combined loss", legend = :false, yaxis = :log)

    # combined loss - zoomed in
    if length(Y) > 10
        last_few_percent = Int(ceil(0.25*length(Y)))
        offset = length(Y) - last_few_percent
        X = [10*(x-1) for x in offset:(offset+last_few_percent)]
        l = Y[end-last_few_percent:end]
        p2 = plot(X, l, label = "l + αֽ||model||", legend = :outertop)
    else
        p2 = plot(X, Y, label = "l + αֽ||model||", legend = :outertop)
    end

    p = plot(p1, p2, layout = (1, 2))
    return p
end

"""
Create dual-panel plot of description length over training epochs.

# Arguments
- `logs::Dict{String, Any}`: Training logs containing epoch and description length data

# Returns
Plot object with global view and zoomed-in view of description length.
"""
function description_length_plot(logs::Dict{String, Any})
    X = logs["epoch"]
    Y = logs["description_length"]

    # description length - global
    p1 = plot(X, Y, title="Description length", legend = :false)
    
    # description length - zoomed in
    if length(Y) > 10
        last_few_percent = Int(ceil(0.25*length(Y)))
        offset = length(Y) - last_few_percent
        X = [10*(x-1) for x in offset:(offset+last_few_percent)]
        l = Y[end-last_few_percent:end]
        p2 = plot(X, l, label = "L [bytes]", legend = :outertop)
    else
        p2 = plot(X, Y, label = "L [bytes]", legend = :outertop)
    end

    p = plot(p1, p2, layout = (1, 2))

    return p
end

"""
Create dual-panel plot of training loss with baseline comparison.

# Arguments
- `logs::Dict{String, Any}`: Training logs containing epoch and loss data
- `teacher::Chain`: Teacher model for baseline estimation
- `args`: Training arguments containing device and prelude epochs

# Returns
Plot object with global view and zoomed-in view of training loss vs baseline.
"""
function loss_plot(logs::Dict{String, Any}, teacher::Chain, args)

    device = args.dev

    baseline_mean, baseline_std = estimate_model_interpolation_capacity(teacher, device=device)

    # loss - global
    X = logs["epoch"]
    p1 = plot(X, logs["train_loss"], title = "train_loss", yaxis = :log, label = "train_loss", legend = :false)
    yticks!(p1, [10^-i for i in 0.0:1:10.0])

    plot!(p1, X, [baseline_mean for _ in X], ribbon = [baseline_std for _ in X], label="best possible interpolator", color=:red, linestyle=:dash)

    # loss - zoomed in
    p(X, l) = plot(X, l, label = "train_loss", legend = :outertop)
    if args.prelude_epochs < 10 && length(logs["train_loss"]) > 10
        last_few_percent = Int(ceil(0.25*length(logs["train_loss"])))
        offset = length(logs["train_loss"]) - last_few_percent
        X = [10*x for x in offset:(offset+last_few_percent)]
        l = logs["train_loss"][end-last_few_percent:end]
        p2 = p(X, l)
    elseif args.prelude_epochs < 10
        p2 = p(X, logs["train_loss"])
    elseif 10*length(logs["train_loss"]) > args.prelude_epochs
        last_part = Int((10*length(logs["train_loss"]) - args.prelude_epochs)/10)
        X = [10*(x-1) for x in (length(logs["train_loss"])-last_part):length(logs["train_loss"])]
        l = logs["train_loss"][end-last_part:end]
        p2 = p(X, l)
    elseif length(logs["train_loss"]) > 10
        last_few_percent = Int(ceil(0.25*length(logs["train_loss"])))
        offset = length(logs["train_loss"]) - last_few_percent
        X = [10*x for x in offset:(offset+last_few_percent)]
        l = logs["train_loss"][end-last_few_percent:end]
        p2 = p(X, l)
    else
        p2 = plot(X, logs["train_loss"], title = "train_loss", label = "train_loss", legend = :best)
    end
    plot!(p2, X, [baseline_mean for _ in X], ribbon = [baseline_std for _ in X], color=:red, linestyle=:dash, label="best possible interpolator")

    plt = plot(p1, p2, layout = (1, 2))

    return plt
end

"""
Create dual-panel plot of teacher-student model size ratio over training.

# Arguments
- `logs::Dict{String, Any}`: Training logs containing epoch and size divergence data

# Returns
Plot object with global view and zoomed-in view of size ratio.
"""
function teacher_student_size_divergence_plot(logs::Dict{String, Any})

    X = logs["epoch"]
    Y = logs["teacher_student_size_divergence"]

    # teacher-student size divergence - global
    p1 = plot(X,
        Y, title="||Student|| / ||Teacher||", titlefontsize=10, legend = :false)
    
    # teacher-student size divergence - zoomed in
    if length(Y) > 10
        last_few_percent = Int(ceil(0.25*length(Y)))
        offset = length(Y) - last_few_percent
        X = [10*(x-1) for x in offset:(offset+last_few_percent)]
        l = Y[end-last_few_percent:end]
        p2 = plot(X, l, legend = :false)
    else
        p2 = plot(X,
        Y, legend = :false)
    end

    p = plot(p1, p2, layout = (1, 2))
    return p
end
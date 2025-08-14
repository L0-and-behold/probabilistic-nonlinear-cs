"""
Execute main teacher-student training loop with regularization and optional masking.

# Arguments
- `teacher::Chain`: Teacher model to generate training data
- `student::Chain`: Student model to train
- `logs::Dict{String, Any}`: Training logs dictionary
- `loss::Function`: Loss function for training
- `opt`: Optimizer for gradient updates
- `update_rule!::Function`: Function to perform parameter updates
- `args`: Training arguments containing hyperparameters and configuration
- `mask`: Optional parameter mask for selective training (default: missing)
- `α_state`: State for dynamic α scheduling (default: missing)  
- `finetuning::Bool`: Whether this is a finetuning phase (default: false)

# Returns
Tuple containing updated student model, optimizer, logs, and α_state.

# Requirements
- Prelude epochs must be multiple of 10 and ≥10 if non-zero
- Dataset must be Tuple{String, Number} with first element "TeacherStudent"
"""
function teacher_student_training_loop(
    teacher::Chain,
    student::Chain,
    logs::Dict{String, Any},
    loss::Function,
    opt,
    update_rule!::Function,
    args;
    mask=missing,
    α_state=missing,
    finetuning=false
    )::Tuple{Chain, Any, Dict{String, Any}, Any}

    if args.prelude_epochs < 10 || args.prelude_epochs % 10 != 0
        error("Prelude epochs must be a multiple of 10 and at least 10 if non-zero.")
    end

    if !isa(args.dataset, Tuple{String, Number}) || args.dataset[1] != "TeacherStudent"
        error("Got bad dataset argument. Expected Tuple{String, Number} with first element 'TeacherStudent'.")
    end

    start_time = time()

    println("Beginning teacher-student training loop...")

    rng = Random.MersenneTwister(args.seed)

    if finetuning
        epochs = args.finetuning_epochs
    else
        epochs = args.epochs
    end

    for epoch in 1:epochs

        if args.dynamic_α[1] == "s_shape_scaling"
            AlphaScaling.update_alpha_state!(α_state, epoch, args)
        end

        train_set = get_dataset(teacher, args.dataset[2], args.batch_size, device=args.dev, rng=rng)

        if ismissing(mask)
            update_rule!(loss, student, train_set, opt, args)
        else
            update_rule!(loss, student, mask, train_set, opt, args)
        end

        if epoch % 10 == 0
            logs = update_logs(student, teacher, logs, loss, args)
        end

        if args.dynamic_α == "linear_scaling"
            AlphaScaling.update_alpha_state!(α_state, train_loss[end], args)
        end
        
        if (epoch % ceil(epoch/10) == 0 && 101 > epoch > 10) || (epoch > 100 && epoch % 50 == 0)
            runtime = round((time() - start_time) / epoch, digits=5)
            println("Epoch: ", epoch, " - runtime: ", runtime, "s/epoch")
        end

        if TrainingTools.break_loop_here(args, logs["train_loss"], logs["train_loss"])
            break
        end
    end

    if !ismissing(α_state)
        return student, opt, logs, α_state
    end
    return student, opt, logs, missing
end

"""
Execute prelude training phase without regularization (α=0).

# Arguments
- `teacher::Chain`: Teacher model to generate training data
- `student::Chain`: Student model to train
- `logs::Dict{String, Any}`: Training logs dictionary
- `loss::Function`: Loss function for training
- `opt`: Optimizer for gradient updates
- `args`: Training arguments containing hyperparameters and configuration

# Returns
Tuple containing updated student model, optimizer, and logs.

# Requirements
- Prelude epochs must be multiple of 10 and ≥10 if non-zero
- Dataset must be Tuple{String, Number} with first element "TeacherStudent"

# Note
Temporarily sets α=0 during training, restores original value afterward.
"""
function prelude_training_loop(
    teacher::Chain, 
    student::Chain, 
    logs::Dict{String, Any},
    loss::Function,
    opt,
    args
    )::Tuple{Chain, Any, Dict{String, Any}}

    # Set args.α to 0 for prelude training for logging purposes. Does not affect training. However, the training does correspond to α = 0 here.
    α = args.α
    args.α = 0.0

    rng = Random.MersenneTwister(args.seed)

    if args.prelude_epochs == 0
        println("No prelude training.")
        return student, opt, logs
    elseif args.prelude_epochs < 10 || args.prelude_epochs % 10 != 0
        error("Prelude epochs must be a multiple of 10 and at least 10 if non-zero.")
    end

    if !isa(args.dataset, Tuple{String, Number}) || args.dataset[1] != "TeacherStudent"
        error("Got bad dataset argument. Expected Tuple{String, Number} with first element 'TeacherStudent'.")
    end

    println("Beginning prelude training loop...")

    start_time = time()

    for epoch in 1:args.prelude_epochs
        train_set = get_dataset(teacher, args.dataset[2], args.batch_size, device=args.dev, rng=rng)

        TrainFunctions.vanilla_train!(loss, student, train_set, opt)

        if epoch % 10 == 0
            logs = update_logs(student, teacher, logs, loss, args)
        end
        if (epoch % ceil(epoch/10) == 0 && 101 > epoch > 10) || (epoch > 100 && epoch % 50 == 0)
            runtime = round((time() - start_time) / epoch, digits=5)
            println("Epoch: ", epoch, " - runtime: ", runtime, "s/epoch")
        end

    end

    args.α = α

    return student, opt, logs

end
"""
Module for generating teacher and student neural networks, plotting them, generating datasets, and comparing them by defining a normal form/symbolic_representation.
Refer to `demonstation.jl` for documnentation and examples.
"""
module Teacher_nn
    using Plots, ColorSchemes
    using Random: randperm
    using Flux: Chain, params, Dense, relu, gpu, cpu, sigmoid_fast, tanh_fast
    using Revise: includet

    include("../Metrics/ByteSizeCompression/ByteSizeCompression.jl")
    include("../TrainingTools/TrainingTools.jl")
    include("../TrainFunctions/TrainFunctions.jl")
    include("../NonDegenerateProjection.jl")

    using .ByteSizeCompression: byte_size_compression, true_byte_size
    using .TrainingTools
    using .AlphaScaling
    import .TrainFunctions
    import .NonDegenerateProjection: project_onto_F

    
    include("plot_nn.jl")
    export plot_nn

    include("random_networks.jl")
    export get_random_teacher,
        get_random_student, 
        pad_small_into_big, 
        shuffle, 
        adjust_params

    include("symbolic_representation.jl")
    export sort_by_bias, 
        sort_nn_along_weights, 
        delete_zero_neurons, 
        normal_form, 
        is_same, 
        print_normal_form

    include("estimate_model_interpolation_capacity.jl")
    export estimate_model_interpolation_capacity,
        pertub_model

    include("get_dataset.jl")
    export get_dataset

    include("training_helpers.jl")
    export initialize_logs,
        update_logs,
        save_plot,
        save_CSV,
        combined_loss_plot,
        description_length_plot,
        loss_plot,
        student_size_plot,
        teacher_student_size_divergence_plot
        
    include("training_loop.jl")
    export teacher_student_training_loop,
        prelude_training_loop

end# module

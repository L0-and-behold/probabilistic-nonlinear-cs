"""
TrainingArguments module for neural network training configuration.

Defines the `TrainArgs` type and default values for training arguments used throughout
the codebase as a configuration object for various training procedures.

# Usage

```julia
args = TrainArgs()  # defaults to TrainArgs{Float32}()
# or
args = TrainArgs{Float64}()
```

# Constructor
- `TrainArgs(; T=Float32)`: Create TrainArgs with specified floating-point precision

# Fields

## SGD Parameters
- `dtype::DataType`: Floating-point precision type (Float32 or Float64)
- `lr::T`: Learning rate for optimizer (default: 1e-5)
- `batch_size::Int`: Number of samples per gradient update (default: 1000)
- `epochs::Int`: Target number of training epochs (default: 5000)
- `min_epochs::Int`: Minimum training epochs (default: 500)
- `max_epochs::Int`: Maximum training epochs (default: 10000)
- `prelude_epochs::Int`: Epochs before applying regularization (default: 0)
- `finetuning_epochs::Int`: Epochs for fine-tuning phase (default: 1000)

## Regularization Parameters
- `α::T`: L0 regularization coefficient (α_l0 in arXiv:2109.05075, default: 1.0)
- `β::T`: L0 regularization sharpness parameter (default: 10.0)
- `ρ::T`: L2 regularization coefficient (α_l2 in arXiv:2109.05075, default: 0.01)
- `ϵ::T`: Global magnitude pruning threshold (default: 0.02)
  - Range: 0.0 (no pruning) to Inf (full pruning)
- `N::Int`: Number of fine-tuning iterations (default: 2)
- `smoothing_window::Int`: Window size for smoothing operations (default: 100)

## Dataset and Architecture
- `dataset`: Dataset specification (default: "MNIST")
  - `("Teacher_NN", train_set_size)`
- `architecture`: Model architecture (default: "LeNet300100")
  - `"LeNet300100"`, `"CNNmodel"`, `"ciresan2010deep"`

## Training Configuration
- `training_rule`: Training function for main training phase
- `training_rule_FT`: Training function for fine-tuning with masks
- `optimizer::String`: Optimizer type (default: "Descent")
  - `"Descent"`, `"Adam"`
- `dev::Function`: Device function (default: cpu)
- `break_loop_after`: Early stopping condition (default: "train_acc_saturation")
  - `false`: No early stopping

## Advanced Options
- `use_pretrained_model_if_possible::Bool`: Use pretrained models when available (default: false)
- `noise::T`: Noise level for training (default: 0.0)
- `seed::Int`: Random seed for reproducibility (default: 42)
- `verbose::Bool`: Enable verbose logging (default: true)

## Logging Flags
These fields are primarily for logging and do not affect training functionality:
- `used_pretrained_model::Bool`: Indicates if pretrained model was used (default: false)
- `train_set_size`: Training set size for logging (default: "see dataset")
- `optimization_procedure::String`: Optimization method description for comparison

# Notes
- All parameters are mutable and can be modified after construction
- Type parameter T determines precision for floating-point fields
- See referenced modules for available options
"""


module TrainingArguments

export TrainArgs

abstract type AbstractTrainArgs end

using Flux: gpu, cpu

Base.@kwdef mutable struct TrainArgs{T<:Union{Float32,Float64}} <: AbstractTrainArgs
    #SGD parameters
    dtype::DataType = T
    lr::T = 1e-5
    epochs::Int = 5000
    min_epochs::Int = 500
    max_epochs::Int = 10000
    prelude_epochs::Int = 0 # Do not set high in combination with dynamic α (otherwise α will stay low).
    finetuning_epochs::Int = 1000
    batch_size::Int = 1000
    #l0l2, FT parameters
    α::T = 1.0 
    β::T = 10.0
    ρ::T = 0.01
    ϵ::T = 0.02
    N::Int = 2
    smoothing_window::Int = 100
    #Switches
    dataset::Union{Function, String, Tuple{String, Number}} = "MNIST"
    architecture::Union{String, Function} = "LeNet300100" # LeNet300100, CNNmodel, ciresan2010deep
    training_rule::Union{Function, String} = ""
    training_rule_FT::Union{Function, String} = ""
    break_loop_after::Union{String, Bool} = "train_acc_saturation" # 
    use_pretrained_model_if_possible::Bool = false
    optimizer::String = "Descent" # Descent, Adam
    dev::Function = cpu
    #Flags
    used_pretrained_model::Bool = false
    train_set_size::Union{String, Int} = "see dataset" # does not change the training process, only for logging purposes. Change dataset instead.
    optimization_procedure::String = ""
    verbose::Bool = true
    noise::T = 0.0
    seed::Int = 42
end

function TrainArgs(; T=Float32)
    return TrainArgs{T}()
end

end
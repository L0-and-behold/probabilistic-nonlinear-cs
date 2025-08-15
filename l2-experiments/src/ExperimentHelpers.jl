module ExperimentHelpers

using Statistics, Plots

export normalize, normalize_and_average, normalize_and_std, do_plot

"""
    normalize(vec::AbstractVector) -> Vector

Apply min-max normalization to scale vector values to the range [0, 1].

This function transforms the input vector such that the minimum value becomes 0
and the maximum value becomes 1, with all other values scaled proportionally.

## Arguments
- `vec::AbstractVector`: Input vector to be normalized

## Returns
- `Vector`: Normalized vector with values in [0, 1] range

## Notes
- Preserves relative ordering of elements
- Zero vector output indicates no variation in input
"""
function normalize(vec)
    min_val = minimum(vec)
    max_val = maximum(vec)
    
    # Handle degenerate case where all values are identical
    if min_val == max_val
        return zeros(eltype(vec), length(vec))
    end
    
    # Apply min-max normalization: (x - min) / (max - min)
    normalized = (vec .- min_val) ./ (max_val - min_val)
    
    return normalized
end

"""
    normalize_and_average(mat::AbstractMatrix) -> Vector

Normalize each column of a matrix independently, then compute row-wise averages.

This function is designed for analyzing results from multiple experimental runs
where each column represents one run and each row represents a time point (epoch).
The normalization ensures that each run contributes equally to the average,
regardless of the absolute scale of values in that run.

## Arguments
- `mat::AbstractMatrix`: Input matrix where columns are runs and rows are time points

## Returns
- `Vector`: Row-wise averages of the normalized matrix (time-series average)

## Notes
- Normalization removes scale differences between runs
- Preserves temporal patterns within each run
- Robust to outlier runs with extreme values
"""
function normalize_and_average(mat)
    # Normalize each column (run) independently
    mat_normalized = mapslices(normalize, mat, dims=1)
    
    # Compute row-wise means (average across runs for each time point)
    return mean(mat_normalized, dims=2)
end

"""
    normalize_and_std(mat::AbstractMatrix) -> Vector

Normalize each column of a matrix independently, then compute row-wise standard deviations.

Companion function to `normalize_and_average` that provides uncertainty estimates
for the averaged trajectories. The normalization ensures that variability estimates
reflect relative differences rather than absolute scale differences.

## Arguments
- `mat::AbstractMatrix`: Input matrix where columns are runs and rows are time points

## Returns
- `Vector`: Row-wise standard deviations of the normalized matrix
"""
function normalize_and_std(mat)
    # Normalize each column (run) independently
    mat_normalized = mapslices(normalize, mat, dims=1)
    
    # Compute row-wise standard deviations (variability across runs for each time point)
    return std(mat_normalized, dims=2)
end

"""
    do_plot(l2, train_loss, test_loss) -> Nothing

Generate comprehensive visualization of neural network training metrics.

## Arguments
- `l2::Union{Vector, Matrix}`: ℓ₂-distance between student and teacher networks
- `train_loss::Union{Vector, Matrix}`: Training loss evolution
- `test_loss::Union{Vector, Matrix}`: Test/validation loss evolution
- `title::String` : Descriptive title of the plot

## Plot Structure
**Upper Panel**: ℓ₂-distance evolution
- Shows convergence/divergence of student network towards/from teacher
- Y-axis normalized to [0, 1] range
- Orange line with appropriate legend

**Lower Panel**: Loss evolution comparison  
- Training loss (red line)
- Test loss (blue line)
- Y-axis normalized to [0, 1] range
- Shared X-axis with upper panel

## Customization
The function uses a global `title` variable for the plot title, which should
contain experimental parameters for proper documentation.

## Output
- Displays the plot in the current Julia environment
- Plot can be saved using `savefig()` after calling this function
- Uses SVG format recommended for publication-quality figures
"""
function do_plot(l2, train_loss, test_loss, title)
    # Determine data type for appropriate labeling
    if isa(l2, Vector)
        suffix = " [raw - i.e. N=1]"
    else
        suffix = " [normalized & averaged]"
    end

    # Create two-panel figure with shared x-axis
    p = plot(layout=(2,1), 
            size = (500, 500),
            link=:x,
            margin=1Plots.mm,
            frame=:box,
            )
    
    # Upper panel: ℓ₂-distance evolution
    plot!(
        p[1], 1:length(l2), l2,
        label="ℓ₂(teacher - student)"*suffix,
        ylabel="ℓ₂",
        color = :orange,
        linewidth=2,
        titlefontsize=7,
        title=title,
        framestyle=:box,
        grid=false,
        ylims=(-0.05, 1.05),
        yticks=0:0.25:1
    )
    
    # Lower panel: Training and test loss comparison
    plot!(p[2], 1:length(train_loss), train_loss, 
        label="Train loss"*suffix,
        ylabel="Loss",
        linewidth=2,
        color = :red,
        framestyle=:box,
        grid=false,
        ylims=(-0.05, 1.05),
        yticks=0:0.25:1)
        
    # Add test loss to lower panel    
    plot!(p[2], 1:length(test_loss), test_loss, 
        color = :blue,              # Standard color for validation metrics
        linewidth=2,
        label="Test loss"*suffix)
        
    # Add x-axis label only to bottom panel
    xlabel!(p[2], "Epoch")
    
    # Optimize legend positioning for typical data patterns
    plot!(p[1], legend=:bottomright, legendfontsize=8)  # ℓ₂ typically increases
    plot!(p[2], legend=:topright, legendfontsize=8)     # Loss typically decreases
    
    # Display the completed plot
    display(p)
end

end #module
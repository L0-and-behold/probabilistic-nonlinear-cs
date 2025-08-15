"""
Visualize a dense neural network with exactly 3 layers (2 hidden layers).

Creates a network diagram showing nodes (neurons) colored by bias values and 
connections (weights) colored and sized according to their magnitudes and signs.

# Visual Encoding
## Connections (Weights)
- **Color**: Blue for positive weights, red for negative weights, white for zero
- **Width**: Proportional to absolute weight value (minimum 0.4 for visibility)
- **Transparency**: Full opacity for all non-zero weights

## Nodes (Neurons)  
- **Input layer**: Gray nodes (no bias)
- **Hidden/Output layers**: Color-mapped by bias value using RdYlGn colorscheme
  - Red tones: Negative bias
  - Yellow tones: Near-zero bias  
  - Green tones: Positive bias
- **Size**: Scaled based on maximum layer size for optimal visibility

# Arguments
- `nn::Chain`: Neural network with exactly 3 Dense layers
- `legend`: Legend position (default: :topright)
- `title::String`: Plot title (default: empty)
- `dev`: Device function for network (default: gpu, temporarily moves to CPU for plotting)

# Returns
- Plot object showing the network visualization

# Example Usage
```julia
using Flux: Chain, Dense, params

nn = Chain(
    Dense(2, 4),    # Input layer: 2 → 4
    Dense(4, 4),    # Hidden layer: 4 → 4  
    Dense(4, 2)     # Output layer: 4 → 2
)

# Set some random biases for demonstration
params(nn.layers[1])[2] .= randn(4)

plt = plot_nn(nn, title="My Neural Network")
```

# Layout
- **Horizontal**: Layers arranged left to right (input → hidden1 → hidden2 → output)
- **Vertical**: Nodes centered within each layer column
- **Scaling**: Automatic sizing based on largest layer for optimal readability

# Requirements
- Network must have exactly 3 layers (input → hidden → hidden → output)
- All layers must be Dense layers with weight matrices and bias vectors
- Assumes bias values approximately in [-1, 1] range for optimal color mapping

# Notes
- Zero weights are skipped during rendering for performance
- Network temporarily moved to CPU for parameter access
- Bias color mapping assumes normalized range; extreme values may appear saturated
- Plot dimensions automatically adjusted based on network architecture
"""
function plot_nn(nn::Chain; legend = :topright, title = "", dev=gpu)

    nn = nn |> cpu

    if length(nn.layers) != 3
        error("The network must have exactly 3 layers (2 hidden layers)")
    end
    
    input_nodes = size(params(nn.layers[1])[1])[2]
    hidden_nodes_1 = length(params(nn.layers[1])[2])
    hidden_nodes_2 = length(params(nn.layers[2])[2])
    output_nodes = size(params(nn.layers[3])[1])[1]

    # Create weight matrices and bias vectors
    weights = [params(layer)[1] for layer in nn.layers]
    biases = [params(layer)[2] for layer in nn.layers]

    layer_sizes = [input_nodes, hidden_nodes_1, hidden_nodes_2, output_nodes]
    max_layer_size = maximum(layer_sizes)
    positions = [[(i-1, j) for j in 1:size] for (i, size) in enumerate(layer_sizes)]


    # Function to map weight to color
    function weight_to_color(weight)
        if weight == 0.0
            return RGB(1, 1, 1)  # white
        elseif weight > 0
            return RGB(0, 0, 1)  # Blue for positive weights
        else
            return RGB(1, 0, 0)  # Red for negative weights
        end
    end

    # Function to map bias to color
    function bias_to_color(bias)
        colorscheme = ColorSchemes.RdYlGn
        normalized_bias = (bias + 1) / 2  # Assuming bias is in [-1, 1] range
        return get(colorscheme, normalized_bias)
    end

    # Plot
    p = plot(size=(1000, 600), legend=legend, title=title)

    # Draw connections with weights

    scaling_factor = max(20/max_layer_size, 1)
    linewdth(w) = max(abs(w) * scaling_factor, 0.4)

    for layer in 1:3
        for i in 1:size(weights[layer], 2)
            for j in 1:size(weights[layer], 1)
                weight = weights[layer][j, i]
                if weight == 0.0 # for performance reasons, we explicitly skip zero weights
                    continue
                end
                start, stop = positions[layer][i], positions[layer+1][j]
                plot!(p, [start[1], stop[1]], [start[2], stop[2]], 
                    color=weight_to_color(weight), 
                    alpha=1.0, 
                    linewidth=linewdth(weight),
                    label="")
            end
        end
    end

    # Draw nodes with colors based on bias
    for (i, pos) in enumerate(positions)
        color = i == 1 ? fill(:gray, length(pos)) : [bias_to_color(b) for b in biases[i-1]]
        scatter!(p, first.(pos), last.(pos), 
                    markersize=4*scaling_factor, 
                    color=color, 
                    label=i == 1 ? "No bias (input)" : false)
    end

    # Add legend items for weight colors
    plot!(p, [], [], color=:blue, linewidth=2, label="Positive weight")
    plot!(p, [], [], color=:red, linewidth=2, label="Negative weight")

    # Add legend items for bias colors
    scatter!(p, [], [], color=bias_to_color(-0.8), markersize=6, label="Negative bias")
    scatter!(p, [], [], color=bias_to_color(0), markersize=6, label="Zero bias")
    scatter!(p, [], [], color=bias_to_color(0.8), markersize=6, label="Positive bias")

    xlims!(p, -0.5, 3.5)
    ylims!(p, 0.5, max_layer_size + 0.5)
    
    # Remove axis and ticks
    plot!(p, xticks=false, yticks=false, xaxis=false, yaxis=false)

    nn = nn |> dev

    return p
end
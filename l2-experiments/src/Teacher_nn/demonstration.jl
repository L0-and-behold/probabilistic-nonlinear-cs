"""
# Demonstation of Teacher_nn module
"""

"""
Loading `Teacher_nn` module and other dependencies used in this script.
"""

using Flux: params
using Plots

include("Teacher_nn.jl")
using .Teacher_nn: get_random_teacher, 
    get_random_student,
    pad_small_into_big, 
    shuffle, 
    sort_by_bias, 
    sort_nn_along_weights, 
    delete_zero_neurons, 
    normal_form, 
    is_same, 
    print_normal_form, 
    plot_nn,
    get_dataset

"""
We define a small neural network with 2 input and 2 output neurons and 2 hidden layers with 2 neurons each and plot it.
"""

input_dim = 2
output_dim = 2
hidden_small_1 = 2
hidden_small_2 = 2
small = get_random_teacher(input_dim, hidden_small_1, hidden_small_2, output_dim)
p1 = plot_nn(small)

"""
In the small network, we set some biases to zero. The motivation will get clear later.
Plotting the networks together reveals the changes.
"""

params(small.layers[1])[2] .= 0.0
p2 = plot_nn(small)
plot(p1, p2, layout=(1, 2))

"""
Now we embed the small network into a bigger network with 2 hidden layers and 25 neurons each.
"""

hidden_big_1 = 25
hidden_big_2 = 25

big = pad_small_into_big(small, hidden_big_1, hidden_big_2)
p3 = plot_nn(big)

big = shuffle(big)
p4 = plot_nn(big)

"""
We can plot the networks together to see the changes. All three networks compute the same function. The only difference is that we add trivial neurons to the small network and shuffle their order.
"""

plot(p2, p3, p4, layout=(3, 1))

"""
Now that our teacher network is set up, we can use it to train a student network.
To do so, we need to create a dataset. We will create a dataset with 1000 samples and a batch size of 100.
"""

train_set = get_dataset(big, 1000, 100)
test_set = get_dataset(big, 1000, 1000)

"""
We initialize a random student network with the same architecture as the teacher network.
The student is a big, dense network.
"""

student = get_random_student(input_dim, hidden_big_1, hidden_big_2, output_dim)
plot_nn(student)

"""
At this point we can train the student network using the dataset we created. We can test sparsity-inducing techniques on the student network and compare the results with the teacher network.
"""

"""
To compare the student and teacher networks, it is helpful to bring the networks into a normal form.
The function computed by the network stays the same under permutation of neurons in the hidden layers and under deletion of trivial neurons.
We can bring the network to normal form by:
1. sort neurons by bias
2. sort neurons by weights
3. delete trivial neurons
"""


"""
1. Sorting by bias.
In the plot below, we see that the neurons in the second hidden layer are sorted by bias. 
In the first hidden layer, however, the neurons are not sorted by bias since we set some biases to zero in the very beginning. It is in general possible, that multiple biases have identical values. In that case sorting neurons along the bias vector is not deterministic and ambiguous.
This motivates us to sort neurons also by weights in the next step. 
"""

big_sorted_by_bias = sort_by_bias(big) ## I get here: "ERROR: Scalar indexing is disallowed.
# Invocation of getindex resulted in scalar indexing of a GPU array.
# This is typically caused by calling an iterating implementation of a method.
# Such implementations *do not* execute on the GPU, but very slowly on the CPU,
# and therefore should be avoided. " 
# Perhaps somewhere in the code, we need to change the sorting procedure to act on arrays instead of indices, or we need to use @allowscalar, if we want to keep the code as is.
p5 = plot_nn(big_sorted_by_bias)
plot(p4, p5, layout=(2, 1))

"""
2. Sorting by weights.
Those neurons that can not be sorted by bias are identified and sorted by weights in the next step. Those neurons that can be sorted by bias are not affected by this step.
For each hidden layer, we consider the neurons in the previous layer. Starting with the first neuron in the previous layer, we sort the neurons by the magnitude of the weight connecting them to the neuron in the current layer.
If this sorting is still ambiguous, we consider the next neuron in the previous layer and so on.
"""

big_sorted_by_weights = sort_nn_along_weights(big_sorted_by_bias)
p6 = plot_nn(big_sorted_by_weights)
plot(p4, p5, p6, layout=(3, 1))

"""
3. Deleting trivial neurons.
We delte all neurons with zero weights and biases.
"""

sorted_and_truncated = delete_zero_neurons(big_sorted_by_weights)
p7 = plot_nn(sorted_and_truncated)
plot(p4, p7, layout=(2, 1))

"""
This procedure is summarized in the `normal_form` function.
"""

big_normal_form = normal_form(big)
p8 = plot_nn(big_normal_form)
plot(p7, p8, layout=(2, 1))

"""
We see that i.g. the original small teacher network is different from the normal form of the big teacher network as demonstrated in the plot below.
To quickly verify that the two networks actually can be transformed into each other, we can use the `is_same` function.
"""

plot(p2, p8)
println("The networks are the same: ", is_same(big, small))

"""
The last helpful function is `print_normal_form` which prints the weights and biases of a neural network in normal form.
"""

print_normal_form(big)

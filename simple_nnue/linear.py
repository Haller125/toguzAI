class LinearLayer:
    def __init__(self, weight, bias, num_inputs, num_outputs):
        self.weight = weight
        self.bias = bias
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

# float* linear(
#     const LinearLayer& layer,  // the layer to use. We have two: L_1, L_2
#     float*             output, // the already allocated storage for the result
#     const float*       input   // the input, which is the output of the previous ClippedReLU layer
# ) {
#     // First copy the biases to the output. We will be adding columns on top of it.
#     for (int i = 0; i < layer.num_outputs; ++i) {
#         output[i] = layer.bias[i];
#     }
#
#     // Remember that rainbowy diagram long time ago? This is it.
#     // We're adding columns one by one, scaled by the input values.
#     for (int i = 0; i < layer.num_inputs; ++i) {
#         for (int j = 0; j < layer.num_outputs; ++j) {
#             output[j] += input[i] * layer.weight[i][j];
#         }
#     }
#
#     // Let the caller know where the used buffer ends.
#     return output + layer.num_outputs;
# }

def linear(layer, output, input):
    # Copy the biases to the output
    for i in range(layer.num_outputs):
        output[i] = layer.bias[i]

    # Add the weighted input values to the output
    for i in range(layer.num_inputs):
        for j in range(layer.num_outputs):
            output[j] += input[i] * layer.weight[i][j]

def test():
    # Initialize the layer
    layer = LinearLayer([[1, 2], [3, 4]], [5, 6], 2, 2)

    # Initialize the input and output buffers
    input = [7, 8]
    output = [0, 0]

    # Call the linear function
    linear(layer, output, input)

    # Print the result
    print(output)

test()

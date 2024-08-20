# float* crelu(,
#     int          size,   // no need to have any layer structure, we just need the number of elements
#     float*       output, // the already allocated storage for the result
#     const float* input   // the input, which is the output of the previous linear layer
# ) {
#     for (int i = 0; i < size; ++i) {
#         output[i] = min(max(input[i], 0), 1);
#     }
#
#     return output + size;
# }

def crelu(size, output, input):
    for i in range(size):
        output[i] = min(max(input[i], 0), 1)

    return output

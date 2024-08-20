# def make_feature_transformer_slice_backward_kernel(max_active_features, output_size):
#     ''''
#         @param: max_active_features
#             The maximum number of features that are active
#             (non-zero) for a single position. This value determines
#             the shape of the inputs.
#             This value is of type uint32_t.
#
#         @param: output_size
#             The number of outputs. Must match the shape of weights
#             and biases.
#             This value is of type uint32.
#     '''
#     num_threads = _get_num_threads_for_backward(output_size)
#     output_thread_slice_size = output_size // num_threads
#     key = (max_active_features, output_size, num_threads)
#     if key not in _feature_transformer_slice_backward_kernel_cache:
#         kernel = cp.RawKernel(r'''
#
# typedef unsigned int uint32_t;
# typedef int int32_t;
#
# extern "C" __global__
# /*
#     @assumptions:
#         The blocks must have dimensionality (BATCH_SIZE,)
#         The threads must have dimensionality (N,), where
#         N * output_thread_slice_size == output_size.
#
#     @param: feature_indices
#         A matrix of shape (BATCH_SIZE, max_active_features)
#         containing indices of active features for each position
#         in a batch. Feature index of -1 means that the slot is empty
#         and the weights will not be accumulated for it. Moreover
#         no further indices from this block will be considered.
#         The indices form an implicit matrix of shape
#         (BATCH_SIZE, NUM_INPUTS), where the first dimension index is
#         inferred from the memory location (BATCH_SIZE), and the
#         second dimension index is stored in the feature_indices matrix.
#         The type for feature indices is int32_t.
#
#     @param: feature_values
#         A matrix of shape (BATCH_SIZE, max_active_features)
#         containing the values (arity) of the corresponding
#         feature index in feature_indices.
#         The type for the feature value (arity) is float32.
#
#     @param: weight_grad
#         The weight gradient matrix of shape (NUM_INPUTS, output_size).
#         The gradient is accumulated, i.e. it must be zero initialized
#         on the first call.
#         Weights must be of type float32.
#
#     @param: bias_grad
#         The bias gradient vector of shape (output_size,).
#         The gradient is accumulated, i.e. it must be zero initialized
#         on the first call.
#         Bias values must be of type float32.
#
#     @param: output_grad
#         An output gradient matrix of shape (BATCH_SIZE, output_size).
#         Output values must have type float32.
# */
# void feature_transformer_slice_backward(
#     const int32_t* const feature_indices,
#     const float*   const feature_values,
#           float*   const weight_grad,
#           float*   const bias_grad,
#     const float*   const output_grad
# ) {{
#     __shared__
#           float          shared_output_grad[{output_size}];
#
#     const uint32_t       block_idx                = blockIdx.x;
#     const uint32_t       slice_offset             = threadIdx.x * {output_thread_slice_size};
#
#     const float*   const output_grad_slice        = output_grad + block_idx * {output_size} + slice_offset;
#           float*   const bias_grad_slice          = bias_grad                               + slice_offset;
#           float*         shared_output_grad_slice = shared_output_grad                      + slice_offset;
#
#     const int32_t* const feature_index_row        = feature_indices + block_idx * {max_active_features};
#     const float*   const feature_value_row        = feature_values  + block_idx * {max_active_features};
#
#     #pragma unroll
#     for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
#     {{
#         shared_output_grad_slice[s] = output_grad_slice[s];
#     }}
#
#     #pragma unroll
#     for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
#     {{
#         const float sog = shared_output_grad_slice[s];
#         if (sog != 0.0f)
#         {{
#             atomicAdd(&bias_grad_slice[s], sog);
#         }}
#     }}
#
#     for (uint32_t k = 0; k < {max_active_features}; ++k)
#     {{
#         const int32_t feature_index = feature_index_row[k];
#         const float   feature_value = feature_value_row[k];
#         if (feature_index != -1)
#         {{
#             float* const weight_grad_slice = weight_grad + feature_index * {output_size} + slice_offset;
#             #pragma unroll
#             for (int s = 0; s < {output_thread_slice_size}; ++s)
#             {{
#                 const float sog = shared_output_grad_slice[s];
#                 if (sog != 0.0f)
#                 {{
#                     atomicAdd(&weight_grad_slice[s], sog * feature_value);
#                 }}
#             }}
#         }} else break;
#     }}
# }}
#
# '''.format(
#                 max_active_features=max_active_features,
#                 output_thread_slice_size=output_thread_slice_size,
#                 output_size=output_size),
#             'feature_transformer_slice_backward')
#         kernel.compile()
#         _feature_transformer_slice_backward_kernel_cache[key] = _kernel_with_threads(kernel, (num_threads,))
#     return _feature_transformer_slice_backward_kernel_cache[key]

import numpy as np

def make_feature_transformer_slice_backward_kernel(max_active_features, output_size):
    '''
        @param: max_active_features
            The maximum number of features that are active
            (non-zero) for a single position. This value determines
            the shape of the inputs.
            This value is of type uint32_t.

        @param: output_size
            The number of outputs. Must match the shape of weights
            and biases.
            This value is of type uint32.
    '''

    def kernel(feature_indices, feature_values, weight_grad, bias_grad, output_grad):
        batch_size = feature_indices.shape[0]
        output_thread_slice_size = output_size

        for block_idx in range(batch_size):
            output_grad_slice = output_grad[block_idx, :]
            bias_grad_slice = bias_grad[:]

            for s in range(output_thread_slice_size):
                sog = output_grad_slice[s]
                if sog != 0.0:
                    bias_grad_slice[s] += sog

            for k in range(max_active_features):
                feature_index = feature_indices[block_idx, k]
                feature_value = feature_values[block_idx, k]
                if feature_index != -1:
                    weight_grad_slice = weight_grad[feature_index, :]
                    for s in range(output_thread_slice_size):
                        sog = output_grad_slice[s]
                        if sog != 0.0:
                            weight_grad_slice[s] += sog * feature_value

    return kernel

def test():

    # feature_indices = np.random.randint(-1, 10, size=(32, 64)).astype(np.int32)
    # feature_values = np.random.rand(32, 64).astype(np.float32)
    # weight_grad = np.zeros((10, 512), dtype=np.float32)
    # bias_grad = np.zeros(512, dtype=np.float32)
    # output_grad = np.random.rand(32, 512).astype(np.float32)

    feature_indices = np.array([[0, 1]])
    feature_values = np.array([[0.5, 0.8]])
    weight_grad = np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32)
    bias_grad = np.array([0, 0, 0, 0], dtype=np.float32)
    output_grad = np.array([[0.1, 0.2, 0.3, 0.4]])

    print(weight_grad)
    print(bias_grad)

    kernel = make_feature_transformer_slice_backward_kernel(2, 4)
    kernel(feature_indices, feature_values, weight_grad, bias_grad, output_grad)

    print(weight_grad)
    print(bias_grad)

test()

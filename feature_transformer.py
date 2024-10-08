import torch
from torch import nn
from torch import autograd
import math
import numpy as np
from feature_transformer_slice_backward import make_feature_transformer_slice_backward_kernel
from feature_transformer_slice_forward import make_feature_transformer_slice_forward_kernel

def _find_nearest_divisor(value, target):
    divisors = []
    for i in range(1, value+1):
        if value % i == 0:
            divisors.append((i, abs(target-i)))
    divisors.sort(key=lambda x:x[1])
    return divisors[0][0]

_num_threads_forward_cache = dict()
def _get_num_threads_for_forward(output_size):
    optimal_num_threads = 512
    if output_size not in _num_threads_forward_cache:
        _num_threads_forward_cache[output_size] = _find_nearest_divisor(output_size, optimal_num_threads)

    return _num_threads_forward_cache[output_size]

_num_threads_backward_cache = dict()
def _get_num_threads_for_backward(output_size):
    optimal_num_threads = 512
    if output_size not in _num_threads_backward_cache:
        _num_threads_backward_cache[output_size] = _find_nearest_divisor(output_size, optimal_num_threads)

    return _num_threads_backward_cache[output_size]

def _kernel_with_threads(kernel, threads):
    def f(grid, args):
        kernel(grid=grid, block=threads, args=args)
    return f

_feature_transformer_slice_forward_kernel_cache = dict()

class FeatureTransformerSliceFunction(autograd.Function):

    @staticmethod
    def forward(ctx, feature_indices, feature_values, weight, bias):
        ctx.save_for_backward(feature_indices, feature_values, weight, bias)

        assert len(feature_indices.shape) == 2
        assert len(feature_values.shape) == 2
        assert feature_indices.shape[0] == feature_values.shape[0]
        assert feature_indices.shape[1] == feature_values.shape[1]
        assert feature_indices.dtype == torch.int32
        assert feature_values.dtype == torch.float32

        assert len(weight.shape) == 2
        assert weight.dtype == torch.float32

        assert len(bias.shape) == 1
        assert bias.dtype == torch.float32

        assert feature_indices.is_cuda
        assert feature_values.is_cuda
        assert weight.is_cuda
        assert bias.is_cuda

        assert feature_values.device == feature_indices.device
        assert weight.device == feature_indices.device
        assert bias.device == feature_indices.device

        assert feature_indices.is_contiguous()
        assert feature_values.is_contiguous()
        assert weight.is_contiguous()
        assert bias.is_contiguous()

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]
        output_size = weight.shape[1]

        output = torch.empty(batch_size, output_size, dtype=torch.float32, device=device, requires_grad=True)

        kernel = make_feature_transformer_slice_forward_kernel(max_active_features, output_size)
        kernel(
            grid=(batch_size,),
            args=(
                feature_indices.data_ptr(),
                feature_values.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                output.data_ptr()
            )
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert not ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]

        grad_output = grad_output.contiguous()

        feature_indices, feature_values, weight, bias = ctx.saved_tensors

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]
        output_size = weight.shape[1]

        weight_grad = torch.zeros(weight.shape[0], weight.shape[1], dtype=torch.float32, device=device)
        bias_grad = torch.zeros(output_size, dtype=torch.float32, device=device)

        kernel = make_feature_transformer_slice_backward_kernel(max_active_features, output_size)
        kernel(
            grid=(batch_size,),
            args=(
                feature_indices.data_ptr(),
                feature_values.data_ptr(),
                weight_grad.data_ptr(),
                bias_grad.data_ptr(),
                grad_output.data_ptr()
            )
        )

        return None, None, weight_grad, bias_grad

class DoubleFeatureTransformerSliceFunction(autograd.Function):

    @staticmethod
    def forward(ctx, feature_indices_0, feature_values_0, feature_indices_1, feature_values_1, weight, bias):
        ctx.save_for_backward(feature_indices_0, feature_values_0, feature_indices_1, feature_values_1, weight, bias)

        assert len(feature_indices_0.shape) == 2
        assert len(feature_values_0.shape) == 2
        assert feature_indices_0.shape[0] == feature_values_0.shape[0]
        assert feature_indices_0.shape[1] == feature_values_0.shape[1]
        assert feature_indices_0.dtype == torch.int32
        assert feature_values_0.dtype == torch.float32

        assert len(feature_indices_1.shape) == 2
        assert len(feature_values_1.shape) == 2
        assert feature_indices_1.shape[0] == feature_values_1.shape[0]
        assert feature_indices_1.shape[1] == feature_values_1.shape[1]
        assert feature_indices_1.dtype == torch.int32
        assert feature_values_1.dtype == torch.float32

        assert len(weight.shape) == 2
        assert weight.dtype == torch.float32

        assert len(bias.shape) == 1
        assert bias.dtype == torch.float32

        assert feature_indices_0.is_cuda
        assert feature_values_0.is_cuda
        assert feature_indices_1.is_cuda
        assert feature_values_1.is_cuda
        assert weight.is_cuda
        assert bias.is_cuda

        assert feature_values_0.device == feature_indices_0.device
        assert feature_values_1.device == feature_indices_1.device
        assert feature_indices_0.device == feature_indices_1.device
        assert weight.device == feature_indices_0.device
        assert bias.device == feature_indices_0.device

        assert feature_indices_0.is_contiguous()
        assert feature_values_0.is_contiguous()
        assert feature_indices_1.is_contiguous()
        assert feature_values_1.is_contiguous()
        assert weight.is_contiguous()
        assert bias.is_contiguous()

        device = feature_indices_0.device
        batch_size = feature_indices_0.shape[0]
        max_active_features = feature_indices_0.shape[1]
        output_size = weight.shape[1]

        output0 = torch.empty(batch_size, output_size, dtype=torch.float32, device=device, requires_grad=True)
        output1 = torch.empty(batch_size, output_size, dtype=torch.float32, device=device, requires_grad=True)

        kernel = make_feature_transformer_slice_forward_kernel(max_active_features, output_size)
        kernel(
            grid=(batch_size,),
            args=(
                feature_indices_0.data_ptr(),
                feature_values_0.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                output0.data_ptr()
            )
        )

        kernel(
            grid=(batch_size,),
            args=(
                feature_indices_1.data_ptr(),
                feature_values_1.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                output1.data_ptr()
            )
        )

        return output0, output1

    @staticmethod
    def backward(ctx, grad_output_0, grad_output_1):
        assert not ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]

        grad_output_0 = grad_output_0.contiguous()
        grad_output_1 = grad_output_1.contiguous()

        feature_indices_0, feature_values_0, feature_indices_1, feature_values_1, weight, bias = ctx.saved_tensors

        device = feature_indices_0.device
        batch_size = feature_indices_0.shape[0]
        max_active_features = feature_indices_0.shape[1]
        output_size = weight.shape[1]

        weight_grad = torch.zeros(weight.shape[0], weight.shape[1], dtype=torch.float32, device=device)
        bias_grad = torch.zeros(output_size, dtype=torch.float32, device=device)

        kernel = make_feature_transformer_slice_backward_kernel(max_active_features, output_size)
        kernel(
            grid=(batch_size,),
            args=(
                feature_indices_0.data_ptr(),
                feature_values_0.data_ptr(),
                weight_grad.data_ptr(),
                bias_grad.data_ptr(),
                grad_output_0.data_ptr()
            )
        )

        kernel(
            grid=(batch_size,),
            args=(
                feature_indices_1.data_ptr(),
                feature_values_1.data_ptr(),
                weight_grad.data_ptr(),
                bias_grad.data_ptr(),
                grad_output_1.data_ptr()
            )
        )

        return None, None, None, None, weight_grad, bias_grad

class FeatureTransformerSlice(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(FeatureTransformerSlice, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        sigma = math.sqrt(1/num_inputs)
        self.weight = nn.Parameter(torch.rand(num_inputs, num_outputs, dtype=torch.float32) * (2 * sigma) - sigma)
        self.bias = nn.Parameter(torch.rand(num_outputs, dtype=torch.float32) * (2 * sigma) - sigma)

    def forward(self, feature_indices, feature_values):
        return FeatureTransformerSliceFunction.apply(feature_indices, feature_values, self.weight, self.bias)

class DoubleFeatureTransformerSlice(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DoubleFeatureTransformerSlice, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        sigma = math.sqrt(1/num_inputs)
        self.weight = nn.Parameter(torch.rand(num_inputs, num_outputs, dtype=torch.float32) * (2 * sigma) - sigma)
        self.bias = nn.Parameter(torch.rand(num_outputs, dtype=torch.float32) * (2 * sigma) - sigma)

    def forward(self, feature_indices_0, feature_values_0, feature_indices_1, feature_values_1):
        return DoubleFeatureTransformerSliceFunction.apply(feature_indices_0, feature_values_0, feature_indices_1, feature_values_1, self.weight, self.bias)

if __name__ == '__main__':
    import time
    import sys
    import os

    def FeatureTransformerSliceFunctionEmulate(feature_indices, feature_values, weight, bias):
        batch_size = feature_indices.shape[0]
        num_inputs = weight.shape[0]
        max_active_features = feature_indices.shape[1]
        inputs = torch.zeros(batch_size, num_inputs, dtype=torch.float32, device=weight.device)
        for i in range(batch_size):
            for j in range(max_active_features):
                feature = feature_indices[i, j]
                value = feature_values[i, j]
                inputs[i, feature] += value

        return torch.mm(inputs, weight) + bias

    def test():
        BATCH_SIZE = 16
        INPUT_SIZE = 10
        MAX_ACTIVE_FEATURES = 32
        STRIDE = 128
        MAX_ERROR = 1e-4

        torch.manual_seed(0)
        weight0 = torch.rand(INPUT_SIZE, STRIDE, dtype=torch.float32, requires_grad=True)
        bias0 = torch.rand(STRIDE, dtype=torch.float32, requires_grad=True)
        torch.manual_seed(0)
        weight1 = torch.rand(INPUT_SIZE, STRIDE, dtype=torch.float32, requires_grad=True)
        bias1 = torch.rand(STRIDE, dtype=torch.float32, requires_grad=True)
        indices0 = (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES) * INPUT_SIZE).to(dtype=torch.int32)
        indices1 = (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES) * INPUT_SIZE).to(dtype=torch.int32)
        values0 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32)
        values1 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32)

        output00 = FeatureTransformerSliceFunctionEmulate(indices0.clone(), values0.clone(), weight0, bias0)
        output01 = FeatureTransformerSliceFunctionEmulate(indices1.clone(), values1.clone(), weight0, bias0)
        #output10 = FeatureTransformerSliceFunction.apply(indices0.clone().cuda(), values0.clone().cuda(), weight1.cuda(), bias1.cuda())
        #output11 = FeatureTransformerSliceFunction.apply(indices1.clone().cuda(), values1.clone().cuda(), weight1.cuda(), bias1.cuda())
        output10, output11 = DoubleFeatureTransformerSliceFunction.apply(indices0.clone().cuda(), values0.clone().cuda(), indices1.clone().cuda(), values1.clone().cuda(), weight1.cuda(), bias1.cuda())

        assert torch.max(output00.cpu() - output10.cpu()) < MAX_ERROR
        assert torch.max(output01.cpu() - output11.cpu()) < MAX_ERROR
        (output00 - output01).sum().backward()
        (output10 - output11).sum().backward()
        assert torch.max(weight0.grad.cpu() - weight1.grad.cpu()) < MAX_ERROR
        assert torch.max(bias0.grad.cpu() - bias1.grad.cpu()) < MAX_ERROR
        print('Tests passed.')

    def bench():
        INPUT_SIZE = 40960
        BATCH_SIZE = 8192
        ITERS = 64
        STRIDE = 264
        MAX_ACTIVE_FEATURES = 64

        layer = DoubleFeatureTransformerSlice(INPUT_SIZE, STRIDE).cuda()
        indices0 = torch.cat([torch.sort((torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES * 3 // 4) * INPUT_SIZE), dim=1)[0].to(dtype=torch.int32), torch.full((BATCH_SIZE, MAX_ACTIVE_FEATURES // 4), -1, dtype=torch.int32)], dim=1).cuda()
        values0 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32).cuda()
        indices1 = torch.cat([torch.sort((torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES * 3 // 4)) * INPUT_SIZE, dim=1)[0].to(dtype=torch.int32), torch.full((BATCH_SIZE, MAX_ACTIVE_FEATURES // 4), -1, dtype=torch.int32)], dim=1).cuda()
        values1 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32).cuda()

        output0, output1 = layer(indices0, values0, indices1, values1)

        device = indices0.device

        start = time.time()

        for i in range(ITERS):
            output0, output1 = layer(indices0, values0, indices1, values1)
            output0 = torch.clamp(output0, 0.0, 1.0)
            output1 = torch.clamp(output1, 0.0, 1.0)

            g = ((output0 - output1)**2).mean()
            g.backward()

            torch.cuda.synchronize()

        end = time.time()

        #for param in layer.parameters():
        #    print(param.grad)

        print('{} pos/s'.format((ITERS * BATCH_SIZE) / (end - start)))

    test()
    bench()
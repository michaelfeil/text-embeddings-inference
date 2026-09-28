// SPDX-License-Identifier: MIT
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

// Reduction order matches Candle fast_sum (58c5155, candle-kernels/src/reduce.cu).
// The reduction is adapted under Candle's MIT license (LICENSE.candle-MIT).
extern "C" __global__ void gemma4_norm_fused_bf16(
    const __nv_bfloat16 *input, const __nv_bfloat16 *weight,
    __nv_bfloat16 *output, uint32_t width, float scale, float epsilon, int weighted) {
    __shared__ float partial[1024];
    const uint32_t tid = threadIdx.x;
    const uint64_t base = uint64_t(blockIdx.x) * width;
    float sum = 0.f;
    // Match the per-thread serial sum and the subsequent halving tree exactly.
    for (uint32_t col = tid; col < width; col += blockDim.x) {
        const float value = __bfloat162float(input[base + col]);
        sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    partial[tid] = sum;
    for (uint32_t step = blockDim.x / 2; step >= 32; step >>= 1) {
        __syncthreads();
        if (tid < step) partial[tid] = __fadd_rn(partial[tid], partial[tid + step]);
    }
    // The final warp follows the same halving tree using register shuffles.
    __syncthreads();
    if (tid < 32) {
        float reduced = partial[tid];
        for (int offset = 16; offset > 0; offset >>= 1) {
            reduced = __fadd_rn(reduced, __shfl_down_sync(0xffffffff, reduced, offset));
        }
        if (tid == 0) {
            const float variance = __fmul_rn(reduced, scale);
            partial[0] = sqrtf(__fadd_rn(variance, epsilon));
        }
    }
    __syncthreads();
    const float denominator = partial[0];
    for (uint32_t col = tid; col < width; col += blockDim.x) {
        float value = __fdiv_rn(__bfloat162float(input[base + col]), denominator);
        if (weighted) value = __fmul_rn(value, __bfloat162float(weight[col]));
        output[base + col] = __float2bfloat16_rn(value);
    }
}

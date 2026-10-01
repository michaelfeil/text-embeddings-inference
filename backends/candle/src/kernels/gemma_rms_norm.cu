// SPDX-License-Identifier: MIT
#include <cuda_bf16.h>

// Apply the supplied scale and normalization in FP32, rounding only
// the final result to BF16. No full-sized FP32 activation buffers are needed.
extern "C" __global__ void gemma_rms_norm_bf16(
    const __nv_bfloat16* x, const float* scale, __nv_bfloat16* out,
    unsigned int width, unsigned int heads, unsigned long long token_stride, float eps) {
    const size_t base = static_cast<size_t>(blockIdx.x / heads) * token_stride
        + static_cast<size_t>(blockIdx.x % heads) * width;
    const size_t out_base = static_cast<size_t>(blockIdx.x) * width;
    float sum = 0.f;
    for (unsigned int col = threadIdx.x; col < width; col += blockDim.x) {
        float v = __bfloat162float(x[base + col]);
        sum += v * v;
    }
    for (int offset = 16; offset; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float partial[8];
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < blockDim.x / 32 ? partial[threadIdx.x] : 0.f;
        for (int offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (threadIdx.x == 0) partial[0] = rsqrtf(sum / width + eps);
    }
    __syncthreads();
    float inv = partial[0];
    for (unsigned int col = threadIdx.x; col < width; col += blockDim.x) {
        float normalized = __bfloat162float(x[base + col]) * inv;
        out[out_base + col] = __float2bfloat16_rn(normalized * scale[col]);
    }
}

// Preserve the reduction tree and FP32 operations of Candle's unfused path.
extern "C" __global__ void gemma_rms_norm_reference_bf16(
    const __nv_bfloat16* x, const float* scale, __nv_bfloat16* out,
    unsigned int width, unsigned int heads, unsigned long long token_stride, float eps) {
    const size_t base = static_cast<size_t>(blockIdx.x / heads) * token_stride
        + static_cast<size_t>(blockIdx.x % heads) * width;
    const size_t out_base = static_cast<size_t>(blockIdx.x) * width;
    __shared__ float sums[1024];
    float sum = 0.f;
    for (unsigned int col = threadIdx.x; col < width; col += blockDim.x) {
        float v = __bfloat162float(x[base + col]);
        sum += v * v;
    }
    sums[threadIdx.x] = sum;
    for (unsigned int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        __syncthreads();
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    }
    // Finish the same reduction tree within the first warp.
    if (threadIdx.x < 32) {
        sum = sums[threadIdx.x];
        const unsigned int mask = __activemask();
        for (unsigned int stride = blockDim.x < 32 ? blockDim.x / 2 : 16;
             stride; stride >>= 1) {
            float other = __shfl_down_sync(mask, sum, stride);
            if (threadIdx.x < stride) sum += other;
        }
        if (threadIdx.x == 0) {
            float mean = sum * static_cast<float>(1.0 / static_cast<double>(width));
            sums[0] = sqrtf(mean + eps);
        }
    }
    __syncthreads();
    for (unsigned int col = threadIdx.x; col < width; col += blockDim.x) {
        float normalized = __bfloat162float(x[base + col]) / sums[0];
        out[out_base + col] = __float2bfloat16_rn(normalized * scale[col]);
    }
}

// NeoX rotary embedding with BF16 rounding after each product, as in Candle's
// composed expression. Write packed [tokens, heads, width] directly.
extern "C" __global__ void gemma_rope_reference_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* cos, const __nv_bfloat16* sin,
    __nv_bfloat16* out, unsigned int count, unsigned int heads,
    unsigned int width, unsigned long long head_stride,
    unsigned long long token_stride, unsigned long long col_stride) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned int col = i % width;
    unsigned int head = (i / width) % heads;
    unsigned int token = i / (width * heads);
    size_t base = static_cast<size_t>(head) * head_stride
        + static_cast<size_t>(token) * token_stride;
    unsigned int other_col = col < width / 2 ? col + width / 2 : col - width / 2;
    float a = __bfloat162float(x[base + static_cast<size_t>(col) * col_stride]);
    float b = __bfloat162float(x[base + static_cast<size_t>(other_col) * col_stride]);
    if (col < width / 2) b = -b;
    size_t freq = static_cast<size_t>(token) * width + col;
    float first = __bfloat162float(__float2bfloat16_rn(__fmul_rn(a, __bfloat162float(cos[freq]))));
    float second = __bfloat162float(__float2bfloat16_rn(__fmul_rn(b, __bfloat162float(sin[freq]))));
    out[i] = __float2bfloat16_rn(__fadd_rn(first, second));
}

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

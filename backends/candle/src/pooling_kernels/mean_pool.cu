#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Match Candle fast_sum's dtype, strided partial sums and halving tree exactly.
// Eight adjacent features share a block for coalesced input reads. Scaling keeps
// the rounded sum and dtype-rounded reciprocal used by Candle's affine kernel.
template <typename T>
__device__ void packed_mean_pool(const T* input, const uint32_t* spans,
                                T* output, uint32_t width) {
    __shared__ T partial[1024 * 8];
    const uint32_t start = spans[2 * blockIdx.y];
    const uint32_t length = spans[2 * blockIdx.y + 1];
    uint32_t threads = 1;
    while (threads < length && threads < 1024) threads *= 2;
    const uint32_t base = blockIdx.x * 8;
    for (uint32_t i = threadIdx.x; i < threads * 8; i += blockDim.x) {
        const uint32_t feature = base + i % 8;
        T sum = T(0.f);
        if (feature < width) {
            for (uint64_t row = i / 8; row < length; row += threads)
                sum += input[(uint64_t(start) + row) * width + feature];
        }
        partial[i] = sum;
    }
    for (uint32_t step = threads / 2; step; step /= 2) {
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < step * 8; i += blockDim.x)
            partial[i] += partial[i + step * 8];
    }
    __syncthreads();
    if (threadIdx.x < 8 && base + threadIdx.x < width) {
        const T reciprocal = T(1.0 / double(length));
        output[uint64_t(blockIdx.y) * width + base + threadIdx.x] =
            partial[threadIdx.x] * reciprocal + T(0.f);
    }
}
extern "C" __global__ void packed_mean_pool_f16(const __half* x, const uint32_t* s,
                                               __half* y, uint32_t width) {
    packed_mean_pool(x, s, y, width);
}
#if __CUDA_ARCH__ >= 800
extern "C" __global__ void packed_mean_pool_bf16(const __nv_bfloat16* x, const uint32_t* s,
                                                __nv_bfloat16* y, uint32_t width) {
    packed_mean_pool(x, s, y, width);
}
#endif

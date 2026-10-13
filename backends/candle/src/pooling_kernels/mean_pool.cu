#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

template <typename T>
__device__ T packed_mean_finite_fallback(const T*, uint32_t, uint32_t,
                                        uint32_t, uint32_t, T original) {
    return original;
}

// Keep every finite Half result's historical rounding. Only a reduction that
// overflows on finite input is reevaluated in F32 before the final Half cast.
template <>
__device__ __half packed_mean_finite_fallback(const __half* input, uint32_t start,
    uint32_t length, uint32_t width, uint32_t feature, __half original) {
    if (isfinite(__half2float(original))) return original;
    float sum = 0.f, correction = 0.f;
    for (uint64_t row = 0; row < length; ++row) {
        const float value = __half2float(input[(uint64_t(start) + row) * width + feature]);
        // Genuine nonfinite hidden states keep their original propagation.
        if (!isfinite(value)) return original;
        // Compensation also keeps long spans near Half's largest finite
        // value from drifting above it before the final cast.
        const float adjusted = __fsub_rn(value, correction);
        const float next = __fadd_rn(sum, adjusted);
        correction = __fsub_rn(__fsub_rn(next, sum), adjusted);
        sum = next;
    }
    return __float2half_rn(__fdiv_rn(sum, float(length)));
}

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
        const T original = partial[threadIdx.x] * reciprocal + T(0.f);
        output[uint64_t(blockIdx.y) * width + base + threadIdx.x] =
            packed_mean_finite_fallback(input, start, length, width,
                                       base + threadIdx.x, original);
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

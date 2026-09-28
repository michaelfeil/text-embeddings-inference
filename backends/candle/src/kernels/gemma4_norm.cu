// Keep the reduction unchanged and fuse only Gemma4's pointwise normalization.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
extern "C" __global__ void gemma4_norm_finish_bf16(
    const __nv_bfloat16 *input, const float *variance,
    const __nv_bfloat16 *weight, __nv_bfloat16 *output,
    uint64_t count, uint32_t width, float epsilon, int weighted) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        // Explicitly retain division (not reciprocal multiplication) and the
        // FP32 rounding boundaries of the existing Candle operations.
        const float denominator = sqrtf(__fadd_rn(variance[i / width], epsilon));
        float value = __fdiv_rn(__bfloat162float(input[i]), denominator);
        if (weighted) value = __fmul_rn(value, __bfloat162float(weight[i % width]));
        output[i] = __float2bfloat16_rn(value);
    }
}

// Casting BF16 to FP32 is exact. Fuse the cast with the original FP32 square,
// leaving the subsequent Candle reduction and its summation order unchanged.
extern "C" __global__ void gemma4_square_bf16_f32(
    const __nv_bfloat16 *input, float *output, uint64_t count) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float value = __bfloat162float(input[i]);
        output[i] = __fmul_rn(value, value);
    }
}

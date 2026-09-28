#include <cuda_bf16.h>
#include <stdint.h>

// Input/output: [1, tokens, heads, dim]. Cos/sin: [1, 1, tokens, dim].
// Preserve Candle's two BF16 product roundings before the final BF16 addition;
// computing the whole expression in FP32 can change downstream MoE routing.
extern "C" __global__ void gemma4_rope_bf16(
    const __nv_bfloat16 *input, const __nv_bfloat16 *cos,
    const __nv_bfloat16 *sin, __nv_bfloat16 *output, uint64_t count,
    uint64_t heads, uint64_t dim) {
  const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;

  const uint64_t col = i % dim;
  const uint64_t pair = col < dim / 2 ? i + dim / 2 : i - dim / 2;
  const uint64_t cache = (i / (heads * dim)) * dim + col;
  const float rotated = col < dim / 2 ? -__bfloat162float(input[pair])
                                     : __bfloat162float(input[pair]);
  const float a = __bfloat162float(__float2bfloat16_rn(
      __fmul_rn(__bfloat162float(input[i]), __bfloat162float(cos[cache]))));
  const float b = __bfloat162float(__float2bfloat16_rn(
      __fmul_rn(rotated, __bfloat162float(sin[cache]))));
  output[i] = __float2bfloat16_rn(__fadd_rn(a, b));
}

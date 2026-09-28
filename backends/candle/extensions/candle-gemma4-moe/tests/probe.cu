// SPDX-License-Identifier: MIT
// Test-only entry points; the production library does not expose these.
#include "../kernels/grouped_gemm.cu"
extern "C" int gemma4_test_gelu(
    const __nv_bfloat16 *input, __nv_bfloat16 *output,
    int tokens, int width, cudaStream_t stream) {
    if (tokens <= 0 || width <= 0) return -1;
    gemma4_moe_gelu_mul<<<(uint64_t(tokens)*width+255)/256,256,0,stream>>>(
        input,output,tokens,width);
    return cudaGetLastError();
}

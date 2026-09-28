// SPDX-License-Identifier: MIT
// Gemma4-26B routing: softmax over all 128 experts, select eight,
// renormalize selected probabilities, then apply learned expert scales.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

extern "C" __global__ void gemma4_moe_route_128_8_f32(
    const float *logits, const float *expert_scale, uint32_t *ids, float *weights) {
    const int lane = threadIdx.x;
    const int token = blockIdx.x;
    __shared__ float remaining[128];
    __shared__ float reduction[128];
    __shared__ uint32_t indices[128];
    __shared__ float chosen[8];
    float score = logits[token * 128 + lane];
    remaining[lane] = score;
    reduction[lane] = score;
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
        if (lane < stride) reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
        __syncthreads();
    }
    float probability = expf(score - reduction[0]);
    __syncthreads();
    reduction[lane] = probability;
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    probability /= reduction[0];
    __syncthreads();
    for (int rank = 0; rank < 8; ++rank) {
        reduction[lane] = remaining[lane];
        indices[lane] = lane;
        __syncthreads();
        for (int stride = 64; stride; stride >>= 1) {
            if (lane < stride) {
                const float other = reduction[lane + stride];
                const uint32_t other_id = indices[lane + stride];
                const bool equal = other == reduction[lane];
                // vLLM's sortable float key orders +0 before -0, then expert ID.
                const uint32_t other_bits = __float_as_uint(other);
                const uint32_t current_bits = __float_as_uint(reduction[lane]);
                if (other > reduction[lane] || (equal &&
                    (other_bits < current_bits ||
                     (other_bits == current_bits && other_id < indices[lane])))) {
                    reduction[lane] = other;
                    indices[lane] = other_id;
                }
            }
            __syncthreads();
        }
        const uint32_t winner = indices[0];
        if (lane == winner) {
            ids[token * 8 + rank] = winner;
            chosen[rank] = probability;
            remaining[lane] = -INFINITY;
        }
        __syncthreads();
    }
    if (lane < 8) {
        float sum = 0.f;
        for (int i = 0; i < 8; ++i) sum += chosen[i];
        weights[token * 8 + lane] = (chosen[lane] / sum) * expert_scale[ids[token * 8 + lane]];
    }
}

// Input layout is [tokens, 8, hidden]. Combine in FP32 and round once.
extern "C" __global__ void gemma4_moe_combine_8_bf16(
    const __nv_bfloat16 *expert_outputs, const float *weights,
    __nv_bfloat16 *output, uint64_t tokens, uint32_t hidden) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= tokens * hidden) return;
    const uint64_t token = index / hidden;
    const uint32_t column = index % hidden;
    float sum = 0.f;
    for (int rank = 0; rank < 8; ++rank) {
        sum += __bfloat162float(expert_outputs[(token * 8 + rank) * hidden + column])
            * weights[token * 8 + rank];
    }
    output[index] = __float2bfloat16_rn(sum);
}

extern "C" __global__ void gemma4_moe_count(const uint32_t *ids, int *counts, int slots) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < slots) atomicAdd(counts + ids[i], 1);
}
extern "C" __global__ void gemma4_moe_offsets(const int *counts, int *offsets) {
    const int expert = threadIdx.x;
    int offset = 0;
    for (int i = 0; i < expert; ++i) offset += counts[i];
    offsets[expert] = offset;
    if (expert == 127) offsets[128] = offset + counts[127];
}
extern "C" __global__ void gemma4_moe_assign(
    const uint32_t *ids, const int *offsets, int *cursors, uint32_t *mapping, int slots) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < slots) {
        const int expert = ids[i];
        mapping[i] = offsets[expert] + atomicAdd(cursors + expert, 1);
    }
}
extern "C" __global__ void gemma4_moe_pack(
    const __nv_bfloat16 *input, const uint32_t *mapping,
    __nv_bfloat16 *packed, uint64_t slots, uint32_t hidden) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < slots * hidden) {
        const uint64_t slot = i / hidden;
        const uint32_t column = i % hidden;
        packed[uint64_t(mapping[slot]) * hidden + column] = input[(slot / 8) * hidden + column];
    }
}
extern "C" __global__ void gemma4_moe_gelu_mul(
    const __nv_bfloat16 *gate_up, __nv_bfloat16 *output, uint64_t slots, uint32_t width) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < slots * width) {
        const uint64_t row = i / width;
        const uint32_t column = i % width;
        const float gate = __bfloat162float(gate_up[row * (2 * width) + column]);
        const float up = __bfloat162float(gate_up[row * (2 * width) + width + column]);
        const float gelu = 0.5f * gate * (1.f + tanhf(0.7978845608028654f *
            (gate + 0.044715f * gate * gate * gate)));
        output[i] = __float2bfloat16_rn(gelu * up);
    }
}
extern "C" __global__ void gemma4_moe_unpermute_combine(
    const __nv_bfloat16 *expert_outputs, const uint32_t *mapping, const float *weights,
    __nv_bfloat16 *output, uint64_t tokens, uint32_t hidden) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < tokens * hidden) {
        const uint64_t token = i / hidden;
        const uint32_t column = i % hidden;
        float sum = 0.f;
        for (int k = 0; k < 8; ++k) {
            const uint64_t slot = token * 8 + k;
            sum += __bfloat162float(expert_outputs[uint64_t(mapping[slot]) * hidden + column]) * weights[slot];
        }
        output[i] = __float2bfloat16_rn(sum);
    }
}

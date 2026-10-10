#pragma once
#include <ATen/ATen.h>
#include <utility>
namespace tei {
at::Tensor fused_add_layer_norm(const at::Tensor& x, const at::Tensor& residual,
 const at::Tensor& weight, const at::Tensor& bias, double epsilon,
 const at::Tensor& second_residual = at::Tensor());
std::pair<at::Tensor, at::Tensor> fused_add_rms_norm(const at::Tensor& x, const at::Tensor& residual,
 const at::Tensor& weight, double epsilon);
// Activation codes: 0 SiLU, 1 GELU exact, 2 GELU tanh, 3 ReLU,
// 4 Candle GELU tanh, 5 Candle SiLU, with model-dtype intermediate rounding.
at::Tensor fused_gated_activation(const at::Tensor& gate_up, int32_t activation, bool gate_first = true);
std::pair<at::Tensor, at::Tensor> fused_rotary(const at::Tensor& q, const at::Tensor& k,
 const at::Tensor& cosine, const at::Tensor& sine, bool contract_first_product = false);
// Inference-only: Q/K must be disjoint views from a fresh QKV allocation.
// Writes each rotary pair in place and preserves the untouched V storage.
void encoder_rotary_inplace(const at::Tensor& q, const at::Tensor& k,
 const at::Tensor& cosine, const at::Tensor& sine);
at::Tensor fused_gelu(const at::Tensor& x, bool approximate_tanh);
at::Tensor fused_bias_gelu(const at::Tensor& x, const at::Tensor& bias, bool approximate_tanh);
std::pair<at::Tensor, at::Tensor> fused_qk_norm_rope(const at::Tensor& q,
 const at::Tensor& k, const at::Tensor& q_weight, const at::Tensor& k_weight,
 const at::Tensor& cosine, const at::Tensor& sine, double epsilon);
}

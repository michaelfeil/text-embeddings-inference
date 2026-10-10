#pragma once
#include "model.h"
namespace tei {
at::Tensor gemma_norm_cuda(const at::Tensor&,const at::Tensor&,double,bool);
at::Tensor gemma_gated_cuda(const at::Tensor&,bool);
at::Tensor gemma_activation_cuda(const at::Tensor&,bool);
at::Tensor qwen35_delta_cuda(const at::Tensor& qkv, const at::Tensor& z,
  const at::Tensor& ab, const at::Tensor& conv, const at::Tensor& a_log,
  const at::Tensor& dt_bias, const at::Tensor& norm, const PackedInput& input,
  int64_t key_heads, int64_t value_heads, double epsilon);
}

#pragma once
#include <ATen/ATen.h>
#include <utility>
namespace tei {
// Reuses Candle's exact CUDA normalization kernel up to width 1536 and rounded
// residual semantics. Wider custom configurations fall back to ATen LayerNorm.
std::pair<at::Tensor, at::Tensor> encoder_layer_norm(const at::Tensor& input,
  const at::Tensor& weight, double epsilon, const at::Tensor& residual = at::Tensor());
}

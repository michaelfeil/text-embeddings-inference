#pragma once
#include <ATen/ATen.h>
#include <utility>

namespace tei {
// Real rows only: the first dimension is never aligned or padded.
std::pair<at::Tensor, at::Tensor> fp8_quantize_rows(const at::Tensor& input);

// Checkpoint weights are quantized once, activations once per invocation.
// Attention, normalization, bias and output heads retain their model dtype.
class Fp8Linear {
  at::Tensor matrix_, scales_, unit_;
  at::ScalarType dtype_;
  bool rowwise_;
public:
  explicit Fp8Linear(const at::Tensor& weight);
  at::Tensor forward(const at::Tensor& input, const at::Tensor& bias = {}) const;
  int64_t input_width() const { return matrix_.size(1); }
  int64_t output_width() const { return matrix_.size(0); }
  const at::Tensor& matrix() const { return matrix_; }
  const at::Tensor& scales() const { return scales_; }
};
}

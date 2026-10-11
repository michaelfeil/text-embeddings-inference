#pragma once
#include <ATen/ATen.h>
#include <utility>
#include <memory>

namespace tei {
// Real rows only: the first dimension is never aligned or padded.
std::pair<at::Tensor, at::Tensor> fp8_quantize_rows(const at::Tensor& input);
std::pair<at::Tensor, at::Tensor> fp8_quantize_gated_rows(const at::Tensor& gate_up);

// Undefined output means the unchanged ATen FP8 path must be used.
// This C++ helper also accepts captured quantized operands for standalone proof.
class Fp8RowwiseExecutor {
  struct Impl;
  std::shared_ptr<Impl> impl_;
public:
  Fp8RowwiseExecutor();
  at::Tensor forward(const at::Tensor& activation, const at::Tensor& weight,
    const at::Tensor& activation_scales, const at::Tensor& weight_scales,
    at::ScalarType output_dtype = at::kHalf) const;
};

// Undefined output preserves the original ordinary ATen linear path.
// Bounded model-owned ordinary plans; graph capture never admits a new plan.
class OrdinaryLinearExecutor {
  struct Impl;
  std::shared_ptr<Impl> impl_;
public:
  class Session {
    struct State;
    std::shared_ptr<State> state_;
    friend class OrdinaryLinearExecutor;
  public:
    Session() = default;
  };
  OrdinaryLinearExecutor();
  // Request-local snapshot: retains its executor and captures policy, context,
  // stream and handle once. It stores no thread-local guard. Callers must keep
  // the same device, stream and policy while dispatching with it.
  Session prepare(const at::Tensor& input) const;
  at::Tensor forward(const at::Tensor& input, const at::Tensor& weight,
    const Session& session) const;
  at::Tensor forward(const at::Tensor& input, const at::Tensor& weight) const;
};

// Checkpoint weights are quantized once, activations once per invocation.
// Attention, normalization, bias and output heads retain their model dtype.
class Fp8Linear {
  at::Tensor matrix_, scales_, unit_;
  at::ScalarType dtype_;
  bool rowwise_;
  bool gated_quantization_;
  at::Tensor apply_quantized(const at::Tensor& input,const at::Tensor& scales,const at::Tensor& bias) const;
  std::shared_ptr<Fp8RowwiseExecutor> cached_executor_;
public:
  explicit Fp8Linear(const at::Tensor& weight, bool cached_rowwise = false,
    std::shared_ptr<Fp8RowwiseExecutor> executor = {}, bool gated_quantization = false);
  at::Tensor forward(const at::Tensor& input, const at::Tensor& bias = {}) const;
  at::Tensor forward_quantized(const at::Tensor& input,const at::Tensor& scales,
    const at::Tensor& bias = {}) const;
  bool can_fuse_gated_quantization() const {
    return gated_quantization_ && rowwise_ &&
      (dtype_==at::kHalf || dtype_==at::kBFloat16) &&
      input_width()==1152 && output_width()==768;
  }
  int64_t input_width() const { return matrix_.size(1); }
  int64_t output_width() const { return matrix_.size(0); }
  const at::Tensor& matrix() const { return matrix_; }
  const at::Tensor& scales() const { return scales_; }
};
}

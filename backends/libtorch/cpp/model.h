#pragma once
#include <ATen/ATen.h>
#include <ATen/ops/_flash_attention_forward.h>
#include <ATen/ops/_efficient_attention_forward.h>
#include <memory>
#include <array>
#include <string>
#include <unordered_map>

namespace tei {
using Weights = std::unordered_map<std::string, at::Tensor>;
struct Options {
  std::unordered_map<std::string, std::string> values;
  std::string string(const std::string& key, const std::string& fallback = "") const {
    auto it = values.find(key); return it == values.end() ? fallback : it->second;
  }
  int64_t integer(const std::string& key, int64_t fallback = 0) const {
    auto it = values.find(key); return it == values.end() ? fallback : std::stoll(it->second);
  }
  double real(const std::string& key, double fallback = 0) const {
    auto it = values.find(key); return it == values.end() ? fallback : std::stod(it->second);
  }
  bool boolean(const std::string& key, bool fallback = false) const {
    auto it = values.find(key); return it == values.end() ? fallback : it->second == "true";
  }
};
struct ImageInput {
  at::Tensor pixels;
  std::array<int64_t,3> grid_thw;
  int64_t merge_size=1, token_start=0, token_count=0, sequence_start=0;
};
struct AudioInput {
  at::Tensor features, validity;
  int64_t token_start=0, token_count=0;
  at::Tensor selected_frames{};
};
struct PackedInput {
  at::Tensor ids, types, positions, cumulative;
  const int32_t* offsets;
  int64_t batch, max_sequence;
  std::vector<ImageInput> images{};
  std::vector<AudioInput> audios{};
  at::Tensor multimodal_positions{};
};
class Model {
public:
  virtual ~Model() = default;
  virtual bool supports_images() const { return false; }
  virtual bool supports_audio() const { return false; }
  virtual void ready() = 0;
  virtual int64_t output_width() const { return 0; }
  virtual int64_t classification_width() const { return 0; }
  // Resolve data-dependent media indices before CUDA graph capture.
  virtual void prepare_input(PackedInput&) const {}
  virtual at::Tensor forward(const PackedInput&) const = 0;
  // Optional pooled-output transform; token-level projections belong in forward().
  virtual at::Tensor project_pooled(const at::Tensor& value) const { return value; }
  virtual at::Tensor token_scores(const at::Tensor&) const {
    TORCH_CHECK(false, "This model does not implement SPLADE");
  }
  virtual at::Tensor predict(const PackedInput&, bool) const {
    TORCH_CHECK(false, "This model does not implement classification");
  }
};
inline const at::Tensor& weight(const Weights& weights, const std::string& name) {
  auto it = weights.find(name);
  TORCH_CHECK(it != weights.end(), "Missing model weight: ", name);
  return it->second;
}
// Q/K/V are packed [total_tokens, heads, head_dim]; cumulative offsets delimit sequences.
inline at::Tensor packed_attention(const at::Tensor& q, const at::Tensor& k,
                                   const at::Tensor& v, const PackedInput& input,
                                   double scale, bool causal = false,
                                   int64_t window_left = -1, int64_t window_right = -1,
                                   const std::optional<at::Tensor>& alibi = std::nullopt) {
  if (q.is_cuda()) {
    if (alibi) {
      // PyTorch's prebuilt FlashAttention disables ALiBi despite exposing it in
      // the operator schema. Use its memory-efficient varlen operator for each
      // exact-length sequence; do not create a quadratic total_tokens bias or
      // padded Q/K/V. Bias rows have aligned storage strides, not padded tokens.
      std::vector<at::Tensor> results;
      for (int64_t row = 0; row < input.batch; ++row) {
        const int64_t start = input.offsets[row], length = input.offsets[row + 1] - start;
        auto idx = at::arange(length, q.options().dtype(at::kLong));
        auto delta = idx.unsqueeze(0) - idx.unsqueeze(1);
        const int64_t stride = ((length + 7) / 8) * 8;
        auto bias = at::empty_strided({1,q.size(1),length,length},
          {q.size(1)*length*stride,length*stride,stride,1}, q.options());
        bias.copy_((-alibi->view({1,q.size(1),1,1}) * delta.abs()).to(q.scalar_type()));
        if (window_left >= 0) bias.masked_fill_(delta < -window_left, -INFINITY);
        if (window_right >= 0) bias.masked_fill_(delta > window_right, -INFINITY);
        auto cumulative = input.cumulative.narrow(0,row,2) - start;
        auto slice = [&](const at::Tensor& t) { return t.narrow(0,start,length).unsqueeze(0); };
        results.push_back(std::get<0>(at::_efficient_attention_forward(
          slice(q),slice(k),slice(v),bias,cumulative,cumulative,length,length,
          0.0,causal ? 1 : 0,false,scale)).squeeze(0));
      }
      return at::cat(results,0);
    }
    return std::get<0>(at::_flash_attention_forward(q, k, v, input.cumulative,
      input.cumulative, input.max_sequence, input.max_sequence, 0.0, causal,
      false, scale, window_left, window_right, std::nullopt, alibi));
  }
  std::vector<at::Tensor> sequences;
  for (int64_t row = 0; row < input.batch; ++row) {
    const int64_t start = input.offsets[row], length = input.offsets[row+1] - start;
    auto slice = [&](const at::Tensor& t) {
      return t.narrow(0, start, length).transpose(0,1).unsqueeze(0);
    };
    std::optional<at::Tensor> mask;
    if (alibi || window_left >= 0 || window_right >= 0) {
      auto idx = at::arange(length, q.options().dtype(at::kLong));
      auto delta = idx.unsqueeze(0) - idx.unsqueeze(1);
      auto allowed = at::ones({length,length}, q.options().dtype(at::kBool));
      if (causal) allowed.logical_and_(delta <= 0);
      if (window_left >= 0) allowed.logical_and_(delta >= -window_left);
      if (window_right >= 0) allowed.logical_and_(delta <= window_right);
      auto bias = at::zeros({q.size(1),length,length}, q.options());
      if (alibi) bias -= alibi->view({q.size(1),1,1}) * delta.abs();
      mask = bias.masked_fill(allowed.logical_not(), -INFINITY).unsqueeze(0);
    }
    sequences.push_back(at::scaled_dot_product_attention(slice(q),slice(k),slice(v),mask,
      0.0, causal && !mask, scale, q.size(1) != k.size(1)).squeeze(0).transpose(0,1));
  }
  return at::cat(sequences,0);
}
}

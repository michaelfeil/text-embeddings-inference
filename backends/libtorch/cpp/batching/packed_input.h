#pragma once
#include <ATen/ATen.h>
#include <array>
#include <vector>
#include <string_view>

namespace tei {
// Gemma normalizes F32 pixels first; D3 must never round through FP16.
inline at::Tensor upload_image_pixels(const at::Tensor& pixels, const c10::Device& device,
 at::ScalarType dtype, std::string_view family, bool d3) {
 if(family=="gemma4"||family=="gemma4_text"||family=="embedding_gemma2")return pixels;
 const auto image_dtype=d3?at::kBFloat16:family.starts_with("qwen3_5")?at::kHalf:dtype;
 return pixels.to(device).to(image_dtype);
}

struct ImageInput {
  at::Tensor pixels;
  std::array<int64_t,3> grid_thw;
  int64_t merge_size=1, token_start=0, token_count=0, sequence_start=0;
  std::array<uint8_t,32> feature_key{};
  bool has_feature_key=false;
  at::Tensor prepared_features{};
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
  // Logical sequences remain full-length for attention and output metadata.
  // Radix maps share only actual token rows with identical causal prefixes.
  at::Tensor compact_ids{}, compact_positions{}, compact_types{};
  at::Tensor scatter_unfold{}, fold_gather{};
};
}

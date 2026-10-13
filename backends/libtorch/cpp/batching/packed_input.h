#pragma once
#include <ATen/ATen.h>
#include <array>
#include <vector>

namespace tei {
struct PrefixContext;
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
  // Execution-only state; original logical request metadata lives in the context.
  PrefixContext* prefix=nullptr;
  // Borrowed full-input host views, valid for this native submission only.
  // Prefix identity must not copy token metadata back from the device.
  at::Tensor host_ids{}, host_types{}, host_positions{}, host_multimodal_positions{}, host_scatter_unfold{};
};
}

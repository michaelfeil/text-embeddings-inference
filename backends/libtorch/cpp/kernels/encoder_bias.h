// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
namespace tei {
// Fill only real sequence-local bias tiles. Metadata is derived from validated
// packed lengths/bucket construction; no token rows are padded or transferred.
void packed_bias_fill(const at::Tensor& out, const at::Tensor& content_key,
                      const at::Tensor& position_content, const at::Tensor& indices,
                      const at::Tensor& index_offsets, const at::Tensor& cumulative,
                      int64_t max_sequence);
}

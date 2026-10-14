// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
#include <vector>
namespace tei {
// Batched, bit-preserving copies from row-strided state into contiguous owners.
// Returns false without submitting work if any layout is unsupported.
bool prefix_state_copies(const std::vector<at::Tensor>& targets,const std::vector<at::Tensor>& sources);
}

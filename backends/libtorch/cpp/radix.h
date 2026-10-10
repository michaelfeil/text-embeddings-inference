#pragma once
// SPDX-License-Identifier: MIT
// Compact/unfold contract adapted from Candle's RadixMLP by Michael Feil.
// Copyright (c) 2025 michaelfeil
#include "model.h"
namespace tei {
bool radix_enabled(const PackedInput& input);
const at::Tensor& radix_ids(const PackedInput& input);
const at::Tensor& radix_positions(const PackedInput& input);
const at::Tensor& radix_types(const PackedInput& input);
at::Tensor radix_unfold(const at::Tensor& rows,const PackedInput& input);
at::Tensor radix_fold(const at::Tensor& rows,const PackedInput& input);
// Validate host metadata before transfer/capture. No GPU reads or synchronization.
void validate_radix(const PackedInput& input,bool causal);
}

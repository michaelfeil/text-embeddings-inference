// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
namespace tei {
// CUDA pooling over selected exact sequence spans, preserving Candle's
// model-dtype reduction tree and reciprocal rounding.
at::Tensor packed_mean_pool(const at::Tensor& hidden, const at::Tensor& spans);
}

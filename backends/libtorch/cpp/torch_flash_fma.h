#pragma once
#include <ATen/ATen.h>

namespace tei {
// Exact-token native Torch Flash attention with fused softmax arithmetic.
// The supported head dimensions require no token or head-dimension padding.
bool torch_flash_fma_supported(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v);
at::Tensor torch_flash_fma(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
                          const at::Tensor& cumulative,int64_t max_sequence,
                          double scale,bool causal=false,int64_t window_left=-1,
                          int64_t window_right=-1);
}

#pragma once
#include <ATen/ATen.h>
#include <utility>
namespace tei {
// Exact Candle Welford reduction for the validated BF16, width2048 and3072
// decoder geometries (each uses its registered warp layout). Statistics use the unrounded FP32 residual sum, while
// the returned residual rounds to BF16. No input rows are padded.
std::pair<at::Tensor,at::Tensor> decoder_exact_rms_norm(const at::Tensor& input,
 const at::Tensor& residual,const at::Tensor& weight,double epsilon);
}

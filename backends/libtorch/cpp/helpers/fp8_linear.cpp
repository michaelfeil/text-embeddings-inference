#include "helpers/fp8_linear.h"
#include <ATen/ops/_scaled_mm.h>
#include <string>
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

namespace tei {
#ifndef TEI_TORCH_CUDA_KERNELS
std::pair<at::Tensor, at::Tensor> fp8_quantize_rows(const at::Tensor&) {
  TORCH_CHECK(false, "Dynamic FP8 requires a CUDA LibTorch build");
}
#endif
Fp8Linear::Fp8Linear(const at::Tensor& weight)
  : dtype_(weight.scalar_type()), rowwise_(false) {
  TORCH_CHECK(weight.is_cuda() && weight.dim()==2 && weight.size(0)>0 && weight.size(1)>0,
    "Dynamic FP8 requires nonempty CUDA matrix weights");
  TORCH_CHECK(dtype_==at::kHalf || dtype_==at::kBFloat16,
    "Dynamic FP8 requires an unquantized float16/bfloat16 checkpoint");
  TORCH_CHECK(weight.size(0)%16==0 && weight.size(1)%16==0,
    "Dynamic FP8 projection widths must be divisible by16; tokens are never padded");
#ifdef TEI_TORCH_CUDA_KERNELS
  c10::cuda::CUDAGuard guard(weight.device());
  const auto* props=at::cuda::getDeviceProperties(weight.device().index());
  const int capability=props->major*10+props->minor;
  TORCH_CHECK(capability==89 || capability==90 || capability==100 || capability==120,
    "Dynamic FP8 requires SM89/90/100/120");
  rowwise_=capability==90;
#else
  TORCH_CHECK(false, "Dynamic FP8 requires a CUDA LibTorch build");
#endif
  TORCH_CHECK(at::isfinite(weight).all().item<bool>(), "Dynamic FP8 checkpoint weights must be finite");
  auto quantized=fp8_quantize_rows(weight.contiguous());
  matrix_=std::move(quantized.first);scales_=std::move(quantized.second);
  unit_=at::ones({},weight.options().dtype(at::kFloat));
}
at::Tensor Fp8Linear::forward(const at::Tensor& input,const at::Tensor& bias) const {
  TORCH_CHECK(input.dim()==2 && input.device()==matrix_.device()
    && input.scalar_type()==dtype_ && input.size(1)==input_width(),
    "Invalid dynamic FP8 packed activations");
  TORCH_CHECK(!bias.defined() || (bias.dim()==1 && bias.size(0)==output_width()
    && bias.device()==input.device() && bias.scalar_type()==dtype_), "Invalid dynamic FP8 bias");
  if(input.size(0)==0)return at::empty({0,output_width()},input.options());
#ifdef TEI_TORCH_CUDA_KERNELS
  c10::cuda::CUDAGuard guard(input.device());
#endif
  auto quantized=fp8_quantize_rows(input.contiguous());
  at::Tensor output;
  if(rowwise_) {
    try {
      // Candle's Hopper recipe prefers fast accumulation. Some CUDA versions
      // lack fast tactics for short, wide-K shapes; preserve its full fallback.
      output=at::_scaled_mm(quantized.first,matrix_.t(),quantized.second,
        scales_.t(),{}, {},dtype_,true);
    } catch(const c10::Error& error) {
      if(std::string(error.what_without_backtrace()).find("CUBLAS_STATUS_NOT_SUPPORTED")==std::string::npos)throw;
      output=at::_scaled_mm(quantized.first,matrix_.t(),quantized.second,
        scales_.t(),{}, {},dtype_,false);
    }
  } else {
    // Match the portable recipe: scale full FP32 accumulation before the
    // single model-dtype rounding. No tensorwide activation scale is inferred.
    output=(at::_scaled_mm(quantized.first,matrix_.t(),unit_,unit_,{}, {},at::kFloat,false)
      *quantized.second*scales_.t()).to(dtype_);
  }
  // Candle's quantized MLP applies bias after GEMM model-dtype rounding.
  return bias.defined()?output+bias:output;
}
}

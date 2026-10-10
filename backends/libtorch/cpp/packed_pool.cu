// SPDX-License-Identifier: Apache-2.0
// Reuses the original packed reduction implementation from this TEI fork.
#include "packed_pool.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>
extern "C" __global__ void packed_mean_pool_bf16(const __nv_bfloat16*,const uint32_t*,__nv_bfloat16*,uint32_t);
#include "../../candle/src/pooling_kernels/mean_pool.cu"
namespace tei {
at::Tensor packed_mean_pool(const at::Tensor& hidden,const at::Tensor& spans) {
  TORCH_CHECK(hidden.is_cuda()&&hidden.dim()==2&&hidden.is_contiguous(),"Packed mean pooling requires contiguous CUDA hidden states");
  TORCH_CHECK(spans.device()==hidden.device()&&spans.scalar_type()==at::kInt&&spans.is_contiguous()&&spans.dim()==2&&spans.size(1)==2,"Invalid packed mean pooling spans");
  TORCH_CHECK(hidden.size(1)>0&&hidden.size(1)<=UINT32_MAX&&spans.size(0)<=65535,"Packed mean pooling dimensions exceed launch bounds");
  c10::cuda::CUDAGuard guard(hidden.device());
  auto output=at::empty({spans.size(0),hidden.size(1)},hidden.options());
  if(!spans.size(0))return output;
  auto stream=at::cuda::getCurrentCUDAStream();
  auto boundaries=reinterpret_cast<const uint32_t*>(spans.const_data_ptr<int32_t>());
  dim3 grid((hidden.size(1)+7)/8,spans.size(0));
  if(hidden.scalar_type()==at::kHalf) {
    ::packed_mean_pool_f16<<<grid,256,0,stream>>>(reinterpret_cast<const __half*>(hidden.const_data_ptr()),boundaries,reinterpret_cast<__half*>(output.data_ptr()),hidden.size(1));
  } else if(hidden.scalar_type()==at::kBFloat16) {
    // The shared implementation guards the exported BF16 entry point for PTX
    // compilation. Instantiate its device routine here for host CUDA builds.
    TORCH_CHECK(at::cuda::getDeviceProperties(hidden.device().index())->major>=8,"BF16 pooling requires SM80 or newer");
    ::packed_mean_pool_bf16<<<grid,256,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(hidden.const_data_ptr()),boundaries,reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),hidden.size(1));
  } else {TORCH_CHECK(false,"Packed mean pooling supports float16 and bfloat16");}
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
}

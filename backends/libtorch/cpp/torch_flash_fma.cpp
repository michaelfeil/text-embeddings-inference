// Adapter for the pinned native Torch Flash source in vendor/torch_flash_attention.
// Upstream kernel source and its full BSD-3-Clause notices remain unmodified.
#include "torch_flash_fma.h"
#include <cuda_runtime_api.h>
// Preserve the fork's fused softmax arithmetic without altering upstream headers.
#ifdef UNFUSE_FMA
#undef UNFUSE_FMA
#endif
#define FLASH_NAMESPACE tei_torch_flash
#include "vendor/torch_flash_attention/flash.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <limits>

namespace tei_torch_flash {
void launch_64(Flash_fwd_params&,cudaStream_t);
void launch_128(Flash_fwd_params&,cudaStream_t);
void launch_256(Flash_fwd_params&,cudaStream_t);
}
namespace tei {
bool torch_flash_fma_supported(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v) {
  return q.is_cuda() && q.dim()==3 && k.dim()==3 && v.dim()==3
    && (q.scalar_type()==at::kHalf || q.scalar_type()==at::kBFloat16)
    && (q.size(2)==64 || q.size(2)==128 || q.size(2)==256)
    && k.size(2)==q.size(2) && v.size(2)==q.size(2)
    && q.stride(2)==1 && k.stride(2)==1 && v.stride(2)==1;
}
at::Tensor torch_flash_fma(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
                          const at::Tensor& cumulative,int64_t max_sequence,
                          double scale,bool causal,int64_t window_left,int64_t window_right) {
  TORCH_CHECK(torch_flash_fma_supported(q,k,v),"Unsupported packed Torch FMA attention shape or dtype");
  TORCH_CHECK(k.device()==q.device() && v.device()==q.device()
    && k.scalar_type()==q.scalar_type() && v.scalar_type()==q.scalar_type()
    && k.size(0)==q.size(0) && v.sizes()==k.sizes()
    && k.size(1)>0 && q.size(1)>0 && q.size(1)%k.size(1)==0,"Packed Torch FMA Q/K/V mismatch");
  TORCH_CHECK(cumulative.device()==q.device() && cumulative.scalar_type()==at::kInt
    && cumulative.dim()==1 && cumulative.is_contiguous() && cumulative.numel()>1,
    "Packed Torch FMA requires contiguous int32 cumulative offsets");
  TORCH_CHECK(q.size(0)>0 && max_sequence>0 && max_sequence<=q.size(0)
    && q.size(0)<=std::numeric_limits<int>::max() && max_sequence<=std::numeric_limits<int>::max()-127
    && cumulative.numel()-1<=std::numeric_limits<int>::max() && q.size(1)<=std::numeric_limits<int>::max()
    && window_left>=-1 && window_right>=-1 && window_left<=std::numeric_limits<int>::max()
    && window_right<=std::numeric_limits<int>::max() && std::isfinite(static_cast<float>(scale)),
    "Invalid packed Torch FMA dimensions or attention scale");
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major>=8,"Packed Torch FMA requires CUDA compute capability >=8.0");
  auto out=at::empty(q.sizes(),q.options());
  auto lse=at::empty({q.size(1),q.size(0)},q.options().dtype(at::kFloat));
  tei_torch_flash::Flash_fwd_params p{};
  p.q_ptr=q.data_ptr();p.k_ptr=k.data_ptr();p.v_ptr=v.data_ptr();p.o_ptr=out.data_ptr();
  p.q_row_stride=q.stride(0);p.k_row_stride=k.stride(0);p.v_row_stride=v.stride(0);
  p.q_head_stride=q.stride(1);p.k_head_stride=k.stride(1);p.v_head_stride=v.stride(1);
  p.o_row_stride=out.stride(0);p.o_head_stride=out.stride(1);
  p.b=cumulative.numel()-1;p.h=q.size(1);p.h_k=k.size(1);p.h_h_k_ratio=p.h/p.h_k;
  p.seqlen_q=max_sequence;p.seqlen_k=max_sequence;
  p.seqlen_q_rounded=((max_sequence+127)/128)*128;p.seqlen_k_rounded=p.seqlen_q_rounded;
  p.d=q.size(2);p.d_rounded=p.d;p.total_q=q.size(0);
  p.cu_seqlens_q=cumulative.data_ptr<int>();p.cu_seqlens_k=p.cu_seqlens_q;
  p.softmax_lse_ptr=lse.data_ptr();p.unpadded_lse=true;p.is_seqlens_k_cumulative=true;
  p.scale_softmax=static_cast<float>(scale);p.scale_softmax_log2=p.scale_softmax*M_LOG2E;
  p.p_dropout=1.f;p.p_dropout_in_uint8_t=255;p.rp_dropout=1.f;p.scale_softmax_rp_dropout=p.scale_softmax;
  if(causal)window_right=0;
  p.is_causal=window_left<0 && window_right==0;
  if(window_left<0 && window_right>=0)window_left=max_sequence;
  if(window_left>=0 && window_right<0)window_right=max_sequence;
  p.window_size_left=window_left;p.window_size_right=window_right;
  p.is_bf16=q.scalar_type()==at::kBFloat16;p.num_splits=1;
  const auto stream=at::cuda::getCurrentCUDAStream();
  if(p.d==64)tei_torch_flash::launch_64(p,stream);
  else if(p.d==128)tei_torch_flash::launch_128(p,stream);
  else tei_torch_flash::launch_256(p,stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
}

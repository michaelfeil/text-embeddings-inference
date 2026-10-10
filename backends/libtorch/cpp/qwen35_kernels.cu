// SPDX-License-Identifier: MIT
// The included Candle kernels retain their MIT notices. This wrapper dispatches
// those native packed kernels on LibTorch's current stream, without Python.
#include "qwen35_kernels.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "../../candle/src/kernels/qwen35_gdn.cu"
#include "../../candle/src/kernels/gemma_rms_norm.cu"

namespace tei {
// Adapted from the MIT precise_activation in Candle's gated_activation.cu.
// The original source and its complete notices remain in the repository.
template<bool Gelu> __device__ __forceinline__ __nv_bfloat16 gemma_precise_activation(__nv_bfloat16 x) {
  using T=__nv_bfloat16;
  if constexpr(Gelu) {
    T square=x*x,cube=square*x;
    T alpha=x+T(.044715)*cube;
    T tanh_input=T(M_2_SQRTPI*M_SQRT1_2)*alpha;
    T tanh_value=T(tanhf(float(tanh_input)));
    return T(.5)*x*(T(1.)+tanh_value);
  } else {
    T exponential=hexp(-x),denominator=T(1.)+exponential;
    return x/denominator;
  }
}
template<bool Gelu> __global__ void gemma_precise_gate(const __nv_bfloat16* x,__nv_bfloat16* out,int64_t count,int64_t width) {
  auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=count)return;
  auto row=i/width,column=i%width;
  out[i]=gemma_precise_activation<Gelu>(x[row*2*width+column])*x[row*2*width+width+column];
}
template<bool Gelu> __global__ void gemma_precise_unary(const __nv_bfloat16* x,__nv_bfloat16* out,int64_t count) {
  auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<count)out[i]=gemma_precise_activation<Gelu>(x[i]);
}
at::Tensor gemma_activation_cuda(const at::Tensor& value,bool gelu) {
  c10::cuda::CUDAGuard guard(value.device());
  TORCH_CHECK(value.scalar_type()==at::kBFloat16,"Gemma precise activation requires BF16");
  auto x=value.contiguous(),out=at::empty_like(x);
  auto stream=at::cuda::getCurrentCUDAStream();
  auto input=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto output=reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  if(gelu)gemma_precise_unary<true><<<(out.numel()+255)/256,256,0,stream>>>(input,output,out.numel());
  else gemma_precise_unary<false><<<(out.numel()+255)/256,256,0,stream>>>(input,output,out.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
at::Tensor gemma_gated_cuda(const at::Tensor& x,bool gelu) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.scalar_type()==at::kBFloat16 && x.is_contiguous() && x.dim()==2 && x.size(1)%2==0,"Gemma precise GLU requires contiguous packed BF16 projection");
  auto width=x.size(1)/2;
  auto out=at::empty({x.size(0),width},x.options());
  auto stream=at::cuda::getCurrentCUDAStream(x.device().index());
  auto input=reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr());
  auto output=reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr());
  if(gelu)gemma_precise_gate<true><<<(out.numel()+255)/256,256,0,stream>>>(input,output,out.numel(),width);
  else gemma_precise_gate<false><<<(out.numel()+255)/256,256,0,stream>>>(input,output,out.numel(),width);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
at::Tensor gemma_norm_cuda(const at::Tensor& x,const at::Tensor& scale,double epsilon,bool reference) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.scalar_type()==at::kBFloat16 && scale.scalar_type()==at::kFloat &&
    x.dim()>=2 && x.dim()<=3 && x.stride(-1)==1,"Gemma CUDA norm requires packed BF16 rows and FP32 scales");
  auto width=x.size(-1),heads=x.dim()==3?x.size(1):1,rows=x.numel()/width;
  TORCH_CHECK(width>0 && width<=8192 && scale.numel()==width,"Invalid Gemma CUDA norm width");
  auto output=at::empty(x.sizes(),x.options());
  int threads=1;while(threads<width && threads<(reference?1024:256))threads*=2;
  // The optimized legacy reduction uses a full-warp shuffle mask.
  // Widths below32 still need32 participating lanes (extra lanes sum zero).
  if(!reference && threads<32)threads=32;
  auto stream=at::cuda::getCurrentCUDAStream(x.device().index());
  auto source=reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr());
  auto destination=reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr());
  if(reference)gemma_rms_norm_reference_bf16<<<rows,threads,0,stream>>>(source,scale.const_data_ptr<float>(),destination,width,heads,x.stride(0),epsilon);
  else gemma_rms_norm_bf16<<<rows,threads,0,stream>>>(source,scale.const_data_ptr<float>(),destination,width,heads,x.stride(0),epsilon);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
at::Tensor qwen35_delta_cuda(const at::Tensor& qkv, const at::Tensor& z,
  const at::Tensor& ab, const at::Tensor& conv, const at::Tensor& a_log,
  const at::Tensor& dt_bias, const at::Tensor& norm, const PackedInput& input,
  int64_t key_heads, int64_t value_heads, double epsilon) {
  c10::cuda::CUDAGuard guard(qkv.device());
  TORCH_CHECK(qkv.scalar_type()==at::kBFloat16 && z.scalar_type()==at::kBFloat16 &&
    ab.scalar_type()==at::kBFloat16 && conv.scalar_type()==at::kBFloat16 &&
    norm.scalar_type()==at::kBFloat16, "Packed DeltaNet CUDA kernels require BF16");
  TORCH_CHECK(key_heads>0 && value_heads>0 && value_heads<=128 && value_heads%key_heads==0,
    "Unsupported packed DeltaNet head geometry");
  auto channels=(2*key_heads+value_heads)*128, tokens=qkv.size(0);
  TORCH_CHECK(qkv.sizes()==at::IntArrayRef({tokens,channels}) &&
    z.sizes()==at::IntArrayRef({tokens,value_heads*128}) &&
    ab.sizes()==at::IntArrayRef({tokens,2*value_heads}) &&
    conv.size(0)==channels && conv.size(-1)>0 && conv.size(-1)<=16,
    "Invalid packed DeltaNet tensors");
  TORCH_CHECK(tokens<=INT32_MAX && tokens*channels<=UINT32_MAX, "DeltaNet token buffer exceeds kernel range");
  auto stream=at::cuda::getCurrentCUDAStream(qkv.device().index());
  auto convolution=at::empty_like(qkv), qk=at::empty({tokens*key_heads,256},qkv.options().dtype(at::kFloat));
  auto decay=at::empty({tokens,2*value_heads},qkv.options().dtype(at::kFloat));
  auto recurrence=at::empty({tokens,value_heads,128},qkv.options()), output=at::empty_like(recurrence);
  auto bf=[](const at::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.const_data_ptr()); };
  auto mut=[](at::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.mutable_data_ptr()); };
  auto cu=reinterpret_cast<const uint32_t*>(input.cumulative.const_data_ptr<int32_t>());
  qwen35_conv_bf16<<<(tokens*channels+255)/256,256,0,stream>>>(bf(qkv),bf(conv),cu,mut(convolution),tokens,channels,conv.size(-1),input.batch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  qwen35_gdn_prepare<<<tokens*key_heads,128,0,stream>>>(bf(convolution),bf(ab),a_log.const_data_ptr<float>(),dt_bias.const_data_ptr<float>(),qk.mutable_data_ptr<float>(),decay.mutable_data_ptr<float>(),tokens,key_heads,value_heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  qwen35_gdn_recurrent<<<dim3(input.batch,value_heads,16),256,0,stream>>>(bf(convolution),qk.const_data_ptr<float>(),decay.const_data_ptr<float>(),cu,mut(recurrence),key_heads,value_heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  qwen35_gdn_norm<<<tokens*value_heads,128,0,stream>>>(bf(recurrence),bf(z),bf(norm),mut(output),epsilon);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output.view({tokens,value_heads*128});
}
}

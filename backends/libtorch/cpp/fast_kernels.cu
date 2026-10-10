// SPDX-License-Identifier: Apache-2.0
// The included Q/K normalization kernel retains its original MIT notice.
#include "fast_kernels.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/_fused_rms_norm.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
// Candle emits PTX directly and guards its BF16 kernel by __CUDA_ARCH__.
// LibTorch builds a host CUDA translation unit, which also needs the declaration.
extern "C" __global__ void qk_norm_rope_bf16(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, int, float);
#include "../../candle/src/kernels/qk_norm_rope.cu"

namespace tei {
namespace {
__global__ void qk_bf16_bridge(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* qw,const __nv_bfloat16* kw,const __nv_bfloat16* cosine,const __nv_bfloat16* sine,__nv_bfloat16* out,int tokens,int qheads,int kheads,int qstride,int kstride,float eps) {
  ::fused_qk_norm_rope(q,k,qw,kw,cosine,sine,out,tokens,qheads,kheads,qstride,kstride,eps);
}
__device__ float block_sum(float value) {
  __shared__ float warps[8];
  const auto lane=threadIdx.x%32, warp=threadIdx.x/32;
  for(int d=16;d;d/=2)value+=__shfl_down_sync(0xffffffff,value,d);
  if(!lane)warps[warp]=value;
  __syncthreads();
  value=threadIdx.x<8?warps[lane]:0.f;
  if(warp==0)for(int d=16;d;d/=2)value+=__shfl_down_sync(0xffffffff,value,d);
  if(!threadIdx.x)warps[0]=value;
  __syncthreads();
  float total=warps[0];
  __syncthreads();
  return total;
}
template<class T> __global__ void add_norm_kernel(const T* x,const T* residual,const T* second,
 const T* weight,const T* bias,T* out,int64_t width,float epsilon) {
  const int64_t row=blockIdx.x;
  float sum=0;
  for(int64_t col=threadIdx.x;col<width;col+=blockDim.x){
    // Match eager tensor additions: round each intermediate to the model dtype.
    T value=T(float(x[row*width+col])+float(residual[row*width+col]));
    if(second)value=T(float(value)+float(second[row*width+col]));
    sum+=float(value);
  }
  const float mean=block_sum(sum)/width;
  float variance=0;
  for(int64_t col=threadIdx.x;col<width;col+=blockDim.x){
    T value=T(float(x[row*width+col])+float(residual[row*width+col]));
    if(second)value=T(float(value)+float(second[row*width+col]));
    float delta=float(value)-mean;variance+=delta*delta;
  }
  const float inverse=rsqrtf(block_sum(variance)/width+epsilon);
  for(int64_t col=threadIdx.x;col<width;col+=blockDim.x){
    T value=T(float(x[row*width+col])+float(residual[row*width+col]));
    if(second)value=T(float(value)+float(second[row*width+col]));
    float normalized=(float(value)-mean)*inverse;
    out[row*width+col]=T(normalized*(weight?float(weight[col]):1.f)+(bias?float(bias[col]):0.f));
  }
}
template<class T,int Columns> __global__ __launch_bounds__(256) void add_norm_register_kernel(const T* x,const T* residual,const T* second,
 const T* weight,const T* bias,T* out,int64_t width,float epsilon) {
  const int64_t row=blockIdx.x;
  float values[Columns],sum=0;
#pragma unroll
  for(int j=0;j<Columns;++j) {
    auto col=threadIdx.x+j*256;
    values[j]=0;
    if(col<width) {
      T value=T(float(x[row*width+col])+float(residual[row*width+col]));
      if(second)value=T(float(value)+float(second[row*width+col]));
      values[j]=float(value);sum+=values[j];
    }
  }
  const float mean=block_sum(sum)/width;float variance=0;
#pragma unroll
  for(int j=0;j<Columns;++j)if(threadIdx.x+j*256<width){float delta=values[j]-mean;variance+=delta*delta;}
  const float inverse=rsqrtf(block_sum(variance)/width+epsilon);
#pragma unroll
  for(int j=0;j<Columns;++j) {
    auto col=threadIdx.x+j*256;
    if(col<width) out[row*width+col]=T(((values[j]-mean)*inverse)*(weight?float(weight[col]):1.f)+(bias?float(bias[col]):0.f));
  }
}
template<class T> __global__ void add_rms_kernel(const T* x,const T* residual,const T* weight,T* out,T* summed,int64_t width,float epsilon) {
  const int64_t row=blockIdx.x;float variance=0;
  for(int64_t col=threadIdx.x;col<width;col+=blockDim.x){
    T value=residual?T(float(x[row*width+col])+float(residual[row*width+col])):x[row*width+col];
    summed[row*width+col]=value;variance+=float(value)*float(value);
  }
  float inverse=rsqrtf(block_sum(variance)/width+epsilon);
  for(int64_t col=threadIdx.x;col<width;col+=blockDim.x)
    out[row*width+col]=T((float(summed[row*width+col])*inverse)*float(weight[col]));
}
template<class T> __global__ void gelu_kernel(const T* x,const T* bias,T* out,int64_t count,int64_t width,bool approximate) {
  int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(index>=count)return;
  T rounded=x[index];if(bias)rounded=T(float(rounded)+float(bias[index%width]));
  float value=float(rounded);
  float y=approximate?0.5f*value*(1.f+tanhf(0.7978845608028654f*(value+0.044715f*value*value*value)))
    :0.5f*value*(1.f+erff(value*0.7071067811865475f));
  out[index]=T(y);
}
template<class T> __device__ __forceinline__ T candle_model_exp(T value){return T(expf(float(value)));}
template<> __device__ __forceinline__ __half candle_model_exp(__half value){return hexp(value);}
template<> __device__ __forceinline__ __nv_bfloat16 candle_model_exp(__nv_bfloat16 value){return hexp(value);}
template<class T> __device__ __forceinline__ T gated_value(float gate,float up,int activation) {
  float activated;
  if(activation==0)activated=gate/(1.f+expf(-gate));
  else if(activation==5) {
    T value=T(gate),exponential=candle_model_exp(T(-value));
    T denominator=T(T(1)+exponential);
    activated=float(T(value/denominator));
  }
  else if(activation==1)activated=.5f*gate*(1.f+erff(gate*.7071067811865475f));
  else if(activation==2)activated=.5f*gate*(1.f+tanhf(.7978845608028654f*(gate+.044715f*gate*gate*gate)));
  else if(activation==4) {
    // Match MIT-licensed Candle kernels/gated_activation.cu: each GELU
    // intermediate rounds to model dtype, including the tanh input/value.
    T x=T(gate),square=T(x*x),cube=T(square*x);
    T alpha=T(x+T(.044715)*cube);
    T argument=T(T(.7978845608028654)*alpha);
    T tangent=T(tanhf(float(argument)));
    activated=float(T(T(T(.5)*x)*T(T(1)+tangent)));
  }
  else activated=fmaxf(gate,0.f);
  return T(float(T(activated))*up);
}
template<class T> __global__ void gated_kernel(const T* x,T* out,int64_t count,int64_t width,int activation,bool gate_first) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
  auto row=i/width,col=i%width;
  out[i]=gated_value<T>(float(x[row*2*width+col+(gate_first?0:width)]),float(x[row*2*width+col+(gate_first?width:0)]),activation);
}
template<class T> struct alignas(sizeof(T)*4) GatedVector4 {T value[4];};
template<class T> __global__ void gated_vector4_kernel(const T* x,T* out,int64_t count,int64_t width,int activation,bool gate_first) {
  int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(index>=count/4)return;
  auto vectors_per_row=width/4,row=index/vectors_per_row,col=index%vectors_per_row;
  const auto* input=reinterpret_cast<const GatedVector4<T>*>(x);
  const auto gate=input[row*2*vectors_per_row+col+(gate_first?0:vectors_per_row)];
  const auto up=input[row*2*vectors_per_row+col+(gate_first?vectors_per_row:0)];
  GatedVector4<T> result;
#pragma unroll
  for(int lane=0;lane<4;++lane)result.value[lane]=gated_value<T>(float(gate.value[lane]),float(up.value[lane]),activation);
  reinterpret_cast<GatedVector4<T>*>(out)[index]=result;
}
template<class T> __device__ __forceinline__ T encoder_rotary_fma(T a,T b,T c) {
  return T(fmaf(float(a),float(b),float(c)));
}
template<> __device__ __forceinline__ __half encoder_rotary_fma(__half a,__half b,__half c) {
  return __hfma(a,b,c);
}
template<> __device__ __forceinline__ __nv_bfloat16 encoder_rotary_fma(__nv_bfloat16 a,__nv_bfloat16 b,__nv_bfloat16 c) {
  return __hfma(a,b,c);
}
template<class T> __global__ void rotary_kernel(const T* q,const T* k,const T* cosine,const T* sine,T* output,
 int64_t qrows,int64_t krows,int64_t qheads,int64_t kheads,int64_t qstride,int64_t kstride,int64_t dim,int64_t half,bool contract_first_product) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=(qrows+krows)*dim)return;
  auto row=i/dim,col=i%dim;bool isq=row<qrows;auto r=isq?row:row-qrows,heads=isq?qheads:kheads;
  auto token=r/heads,head=r%heads;const T* x=isq?q:k;auto offset=token*(isq?qstride:kstride)+head*dim;
  if(col>=2*half){output[i]=x[offset+col];return;}
  bool upper=col>=half;auto lane=upper?col-half:col;
  float a=float(x[offset+lane]),b=float(x[offset+lane+half]),c=float(cosine[token*half+lane]),s=float(sine[token*half+lane]);
  // Candle encoder rotary contracts the first half product into HFMA.
  // Decoder callers preserve the separately rounded products by default.
  if(contract_first_product)
    output[i]=upper?encoder_rotary_fma(T(b),T(c),T(a*s)):encoder_rotary_fma(T(a),T(c),T(-float(T(b*s))));
  else
    output[i]=upper?T(float(T(b*c))+float(T(a*s))):T(float(T(a*c))-float(T(b*s)));
}
template<class T> __global__ void encoder_rotary_inplace_kernel(T* q,T* k,const T* cosine,const T* sine,
 int64_t qheads,int64_t kheads,int64_t qstride,int64_t kstride,int64_t tokens,int64_t dim,int64_t half) {
  const int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(index>=tokens*(qheads+kheads)*half)return;
  const auto row=index/half,lane=index%half;
  const bool isq=row<tokens*qheads;
  const auto r=isq?row:row-tokens*qheads,heads=isq?qheads:kheads;
  const auto token=r/heads,head=r%heads;
  T* x=isq?q:k;
  const auto offset=token*(isq?qstride:kstride)+head*dim+lane;
  const T a=x[offset],b=x[offset+half],c=cosine[token*half+lane],s=sine[token*half+lane];
  // Load both original lanes before either write; every pair has one owner.
  x[offset]=encoder_rotary_fma(a,c,T(-float(T(b*s))));
  x[offset+half]=encoder_rotary_fma(b,c,T(a*s));
}
void validate(const at::Tensor& x,const at::Tensor& other,const char* name,bool full) {
  if(!other.defined())return;
  TORCH_CHECK(other.device()==x.device()&&other.scalar_type()==x.scalar_type()&&other.is_contiguous(),name," must be contiguous and match input device/dtype");
  TORCH_CHECK(full?other.sizes()==x.sizes():other.numel()==x.size(-1),"Invalid ",name," dimensions");
}
template<class T> void run_norm(const at::Tensor& x,const at::Tensor& residual,const at::Tensor& second,
 const at::Tensor& weight,const at::Tensor& bias,at::Tensor& output,double epsilon,cudaStream_t stream) {
  auto p=[](const at::Tensor& t){return t.defined()?reinterpret_cast<const T*>(t.const_data_ptr()):nullptr;};
  auto width=x.size(-1),rows=x.numel()/width;
#define TEI_REGISTER_LN(N) add_norm_register_kernel<T,N><<<rows,256,0,stream>>>(p(x),p(residual),p(second),p(weight),p(bias),reinterpret_cast<T*>(output.mutable_data_ptr()),width,epsilon)
  if(width<=256){TEI_REGISTER_LN(1);}else if(width<=512){TEI_REGISTER_LN(2);}else if(width<=768){TEI_REGISTER_LN(3);}else if(width<=1024){TEI_REGISTER_LN(4);}else if(width<=1536){TEI_REGISTER_LN(6);}else if(width<=2048){TEI_REGISTER_LN(8);}else if(width<=3072){TEI_REGISTER_LN(12);}else if(width<=4096){TEI_REGISTER_LN(16);}
  else add_norm_kernel<T><<<rows,256,0,stream>>>(p(x),p(residual),p(second),p(weight),p(bias),reinterpret_cast<T*>(output.mutable_data_ptr()),width,epsilon);
#undef TEI_REGISTER_LN
}
template<class T> void run_gelu(const at::Tensor& x,const at::Tensor& bias,at::Tensor& output,bool approximate,cudaStream_t stream) {
  gelu_kernel<T><<<(x.numel()+255)/256,256,0,stream>>>(reinterpret_cast<const T*>(x.const_data_ptr()),bias.defined()?reinterpret_cast<const T*>(bias.const_data_ptr()):nullptr,reinterpret_cast<T*>(output.mutable_data_ptr()),x.numel(),x.size(-1),approximate);
}
}
at::Tensor fused_add_layer_norm(const at::Tensor& x,const at::Tensor& residual,const at::Tensor& weight,const at::Tensor& bias,double epsilon,const at::Tensor& second) {
  TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.dim()>=2&&x.size(-1)>0,"Fused residual layernorm requires contiguous CUDA input");
  TORCH_CHECK(residual.defined(),"Residual must be defined");
  validate(x,residual,"residual",true);validate(x,second,"second residual",true);validate(x,weight,"weight",false);validate(x,bias,"bias",false);
  c10::cuda::CUDAGuard guard(x.device());
  auto output=at::empty_like(x);if(x.numel()==0)return output;
  auto stream=at::cuda::getCurrentCUDAStream(x.device().index());
  if(x.scalar_type()==at::kHalf)run_norm<__half>(x,residual,second,weight,bias,output,epsilon,stream);
  else if(x.scalar_type()==at::kBFloat16)run_norm<__nv_bfloat16>(x,residual,second,weight,bias,output,epsilon,stream);
  else if(x.scalar_type()==at::kFloat)run_norm<float>(x,residual,second,weight,bias,output,epsilon,stream);
  else TORCH_CHECK(false,"Unsupported fused layernorm dtype");
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
std::pair<at::Tensor,at::Tensor> fused_add_rms_norm(const at::Tensor& x,const at::Tensor& residual,const at::Tensor& weight,double epsilon) {
  TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.dim()>=2&&x.size(-1)>0,"Fused residual RMSNorm requires contiguous CUDA input");
  validate(x,residual,"residual",true);validate(x,weight,"weight",false);TORCH_CHECK(weight.defined(),"RMSNorm weight must be defined");
  c10::cuda::CUDAGuard guard(x.device());auto output=at::empty_like(x),sum=at::empty_like(x);if(x.numel()==0)return {output,sum};
  auto stream=at::cuda::getCurrentCUDAStream(x.device().index());
#define TEI_RMS_DISPATCH(T) add_rms_kernel<T><<<x.numel()/x.size(-1),256,0,stream>>>(reinterpret_cast<const T*>(x.const_data_ptr()),residual.defined()?reinterpret_cast<const T*>(residual.const_data_ptr()):nullptr,reinterpret_cast<const T*>(weight.const_data_ptr()),reinterpret_cast<T*>(output.mutable_data_ptr()),reinterpret_cast<T*>(sum.mutable_data_ptr()),x.size(-1),epsilon)
  if(x.scalar_type()==at::kHalf){TEI_RMS_DISPATCH(__half);}
  else if(x.scalar_type()==at::kBFloat16){TEI_RMS_DISPATCH(__nv_bfloat16);}
  else if(x.scalar_type()==at::kFloat){TEI_RMS_DISPATCH(float);}
  else TORCH_CHECK(false,"Unsupported fused RMSNorm dtype");
#undef TEI_RMS_DISPATCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {output,sum};
}
at::Tensor fused_bias_gelu(const at::Tensor& x,const at::Tensor& bias,bool approximate) {
  TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.dim()>0&&x.size(-1)>0,"Fused GELU requires contiguous CUDA input");
  validate(x,bias,"bias",false);c10::cuda::CUDAGuard guard(x.device());auto output=at::empty_like(x);if(x.numel()==0)return output;
  auto stream=at::cuda::getCurrentCUDAStream(x.device().index());
  if(x.scalar_type()==at::kHalf)run_gelu<__half>(x,bias,output,approximate,stream);
  else if(x.scalar_type()==at::kBFloat16)run_gelu<__nv_bfloat16>(x,bias,output,approximate,stream);
  else if(x.scalar_type()==at::kFloat)run_gelu<float>(x,bias,output,approximate,stream);
  else TORCH_CHECK(false,"Unsupported fused GELU dtype");
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
at::Tensor fused_gated_activation(const at::Tensor& x,int32_t activation,bool gate_first) {
  TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.dim()>0&&x.size(-1)>0&&x.size(-1)%2==0,"Fused gated activation requires contiguous CUDA gate/up input");
  TORCH_CHECK(activation>=0&&activation<=5,"Unsupported fused gated activation");
  c10::cuda::CUDAGuard guard(x.device());auto shape=x.sizes().vec();shape.back()/=2;auto output=at::empty(shape,x.options());if(output.numel()==0)return output;
  auto stream=at::cuda::getCurrentCUDAStream(x.device().index());
#define TEI_GATE(T) do { \
  if(shape.back()%4==0 && reinterpret_cast<uintptr_t>(x.const_data_ptr())%(4*x.element_size())==0) gated_vector4_kernel<T><<<(output.numel()/4+255)/256,256,0,stream>>>(reinterpret_cast<const T*>(x.const_data_ptr()),reinterpret_cast<T*>(output.mutable_data_ptr()),output.numel(),shape.back(),activation,gate_first); \
  else gated_kernel<T><<<(output.numel()+255)/256,256,0,stream>>>(reinterpret_cast<const T*>(x.const_data_ptr()),reinterpret_cast<T*>(output.mutable_data_ptr()),output.numel(),shape.back(),activation,gate_first); \
} while(false)
  if(x.scalar_type()==at::kHalf){TEI_GATE(__half);}else if(x.scalar_type()==at::kBFloat16){TEI_GATE(__nv_bfloat16);}else if(x.scalar_type()==at::kFloat){TEI_GATE(float);}else TORCH_CHECK(false,"Unsupported gated activation dtype");
#undef TEI_GATE
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
std::pair<at::Tensor,at::Tensor> fused_rotary(const at::Tensor& q,const at::Tensor& k,const at::Tensor& cosine,const at::Tensor& sine,bool contract_first_product) {
  TORCH_CHECK(q.is_cuda()&&k.is_cuda()&&q.dim()==3&&k.dim()==3&&q.size(0)==k.size(0)&&q.size(2)==k.size(2)&&q.stride(2)==1&&k.stride(2)==1&&q.stride(1)==q.size(2)&&k.stride(1)==k.size(2),"Invalid packed rotary Q/K tensors");
  TORCH_CHECK(k.device()==q.device()&&k.scalar_type()==q.scalar_type(),"Rotary Q/K dtype/device mismatch");
  TORCH_CHECK(cosine.dim()>0&&cosine.size(-1)>0&&2*cosine.size(-1)<=q.size(-1)&&cosine.numel()==q.size(0)*cosine.size(-1)&&sine.sizes()==cosine.sizes(),"Invalid rotary frequency dimensions");
  for(const auto& t:{cosine,sine})TORCH_CHECK(t.device()==q.device()&&t.scalar_type()==q.scalar_type()&&t.is_contiguous(),"Rotary frequencies must be contiguous and match Q/K device/dtype");
  c10::cuda::CUDAGuard guard(q.device());auto qr=q.size(0)*q.size(1),kr=k.size(0)*k.size(1),dim=q.size(-1);
  auto output=at::empty({qr+kr,dim},q.options());auto stream=at::cuda::getCurrentCUDAStream(q.device().index());
  if(output.numel()>0){
#define TEI_ROPE(T) rotary_kernel<T><<<(output.numel()+255)/256,256,0,stream>>>(reinterpret_cast<const T*>(q.const_data_ptr()),reinterpret_cast<const T*>(k.const_data_ptr()),reinterpret_cast<const T*>(cosine.const_data_ptr()),reinterpret_cast<const T*>(sine.const_data_ptr()),reinterpret_cast<T*>(output.mutable_data_ptr()),qr,kr,q.size(1),k.size(1),q.stride(0),k.stride(0),dim,cosine.size(-1),contract_first_product)
    if(q.scalar_type()==at::kHalf){TEI_ROPE(__half);}else if(q.scalar_type()==at::kBFloat16){TEI_ROPE(__nv_bfloat16);}else if(q.scalar_type()==at::kFloat){TEI_ROPE(float);}else TORCH_CHECK(false,"Unsupported rotary dtype");
#undef TEI_ROPE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output.narrow(0,0,qr).view({q.size(0),q.size(1),dim}),output.narrow(0,qr,kr).view({k.size(0),k.size(1),dim})};
}
void encoder_rotary_inplace(const at::Tensor& q,const at::Tensor& k,const at::Tensor& cosine,const at::Tensor& sine) {
  TORCH_CHECK(q.is_cuda()&&q.dim()==3&&k.dim()==3&&q.size(0)==k.size(0)&&q.size(2)==k.size(2),"Invalid packed encoder Q/K");
  TORCH_CHECK(q.device()==k.device()&&q.scalar_type()==k.scalar_type(),"Encoder Q/K dtype/device mismatch");
  for(const auto& t:{q,k})TORCH_CHECK(t.stride(2)==1&&t.stride(1)==t.size(2)&&t.stride(0)>=t.size(1)*t.size(2),"Encoder rotary requires nonoverlapping packed head views");
  TORCH_CHECK(cosine.dim()>0&&cosine.size(-1)>0&&2*cosine.size(-1)<=q.size(-1)&&cosine.numel()==q.size(0)*cosine.size(-1)&&sine.sizes()==cosine.sizes(),"Invalid encoder rotary frequencies");
  for(const auto& t:{cosine,sine})TORCH_CHECK(t.device()==q.device()&&t.scalar_type()==q.scalar_type()&&t.is_contiguous(),"Encoder rotary frequencies must match Q/K");
  c10::cuda::CUDAGuard guard(q.device());const auto count=q.size(0)*(q.size(1)+k.size(1))*cosine.size(-1);
  if(!count)return;
  auto stream=at::cuda::getCurrentCUDAStream(q.device().index());
#define TEI_ENCODER_ROPE(T) encoder_rotary_inplace_kernel<T><<<(count+255)/256,256,0,stream>>>(reinterpret_cast<T*>(q.mutable_data_ptr()),reinterpret_cast<T*>(k.mutable_data_ptr()),reinterpret_cast<const T*>(cosine.const_data_ptr()),reinterpret_cast<const T*>(sine.const_data_ptr()),q.size(1),k.size(1),q.stride(0),k.stride(0),q.size(0),q.size(2),cosine.size(-1))
  if(q.scalar_type()==at::kHalf){TEI_ENCODER_ROPE(__half);}else if(q.scalar_type()==at::kBFloat16){TEI_ENCODER_ROPE(__nv_bfloat16);}else if(q.scalar_type()==at::kFloat){TEI_ENCODER_ROPE(float);}else TORCH_CHECK(false,"Unsupported encoder rotary dtype");
#undef TEI_ENCODER_ROPE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
at::Tensor fused_gelu(const at::Tensor& x,bool approximate){return fused_bias_gelu(x,at::Tensor(),approximate);}
std::pair<at::Tensor,at::Tensor> fused_qk_norm_rope(const at::Tensor& q,const at::Tensor& k,const at::Tensor& qw,const at::Tensor& kw,const at::Tensor& cosine,const at::Tensor& sine,double epsilon) {
  TORCH_CHECK(q.is_cuda()&&k.is_cuda()&&q.dim()==3&&k.dim()==3&&q.size(0)==k.size(0)&&q.size(2)==k.size(2),"Invalid packed Q/K tensors");
  TORCH_CHECK(q.device()==k.device()&&q.scalar_type()==k.scalar_type(),"Q/K device and dtype must match");
  auto half=q.size(2)/2;
  if(q.size(2)!=128 || (q.scalar_type()!=at::kHalf&&q.scalar_type()!=at::kBFloat16)) {
    auto rotate=[&](const at::Tensor& x,const at::Tensor& weight){auto n=std::get<0>(at::_fused_rms_norm(x.contiguous(),{x.size(-1)},weight,epsilon));auto a=n.narrow(-1,0,half),b=n.narrow(-1,half,half);return at::cat({a*cosine-b*sine,b*cosine+a*sine},-1);};
    return {rotate(q,qw),rotate(k,kw)};
  }
  TORCH_CHECK(q.stride(2)==1&&k.stride(2)==1&&q.stride(1)==128&&k.stride(1)==128,"Q/K head dimensions must be contiguous");
  TORCH_CHECK(qw.numel()==128&&kw.numel()==128&&qw.is_contiguous()&&kw.is_contiguous(),"Invalid Q/K norm weights");
  TORCH_CHECK(cosine.is_contiguous()&&sine.is_contiguous()&&cosine.numel()==q.size(0)*64&&sine.numel()==q.size(0)*64,"Invalid packed cosine/sine tensors");
  for(const auto& t:{qw,kw,cosine,sine})TORCH_CHECK(t.device()==q.device()&&t.scalar_type()==q.scalar_type(),"Q/K norm and RoPE tensor device/dtype mismatch");
  TORCH_CHECK(q.size(0)<=INT32_MAX&&q.stride(0)<=INT32_MAX&&k.stride(0)<=INT32_MAX,"Q/K buffer exceeds kernel range");
  c10::cuda::CUDAGuard guard(q.device());
  const int64_t qrows=q.size(0)*q.size(1),krows=k.size(0)*k.size(1);
  auto output=at::empty({qrows+krows,128},q.options());auto stream=at::cuda::getCurrentCUDAStream(q.device().index());
  auto blocks=std::min<int64_t>((qrows+krows+7)/8,65535);
  if(blocks>0) {
    if(q.scalar_type()==at::kHalf) {
      auto p=[](const at::Tensor& t){return reinterpret_cast<const __half*>(t.const_data_ptr());};
      qk_norm_rope_f16<<<blocks,128,0,stream>>>(p(q),p(k),p(qw),p(kw),p(cosine),p(sine),reinterpret_cast<__half*>(output.mutable_data_ptr()),q.size(0),q.size(1),k.size(1),q.stride(0),k.stride(0),epsilon);
    } else {
      auto p=[](const at::Tensor& t){return reinterpret_cast<const __nv_bfloat16*>(t.const_data_ptr());};
      qk_bf16_bridge<<<blocks,128,0,stream>>>(p(q),p(k),p(qw),p(kw),p(cosine),p(sine),reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),q.size(0),q.size(1),k.size(1),q.stride(0),k.stride(0),epsilon);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output.narrow(0,0,qrows).view({q.size(0),q.size(1),128}),output.narrow(0,qrows,krows).view({k.size(0),k.size(1),128})};
}
}

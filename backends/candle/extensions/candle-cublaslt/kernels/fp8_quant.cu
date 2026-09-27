#include <cuda_runtime.h>
#if CUDART_VERSION < 12090
#error "experimental-fp8 requires CUDA toolkit 12.9 or newer"
#endif
#include <cuda_fp16.h>
#include <cuda_fp8.h>
// Match the evaluated per-token recipe: FP32 absmax times the FP32 reciprocal
// of 448, followed by correctly-rounded division and saturating E4M3 conversion.
template<int K,int T> __device__ void packed_quant(const half* x,__nv_fp8_e4m3* y,float* scales){
 __shared__ float partial[T/32];int row=blockIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32;
 float4 values[(K+T*4-1)/(T*4)];float mx=0;
 #pragma unroll
 for(int j=0;j<(K+T*4-1)/(T*4);j++){
  int i=(threadIdx.x+j*T)*4;float4 v=make_float4(0,0,0,0);
  if(i<K){uint2 raw=((const uint2*)(x+size_t(row)*K))[i/4];auto lo=__half22float2(*reinterpret_cast<half2*>(&raw.x));auto hi=__half22float2(*reinterpret_cast<half2*>(&raw.y));v=make_float4(lo.x,lo.y,hi.x,hi.y);}
  values[j]=v;mx=fmaxf(mx,fmaxf(fmaxf(fabsf(v.x),fabsf(v.y)),fmaxf(fabsf(v.z),fabsf(v.w))));
 }
 for(int d=16;d;d/=2)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,d));
 if(lane==0)partial[warp]=mx;__syncthreads();
 if(warp==0){mx=lane<T/32?partial[lane]:0;for(int d=16;d;d/=2)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,d));if(lane==0)partial[0]=__fmul_rn(fmaxf(mx,1e-12f),1.f/448.f);}
 __syncthreads();float s=partial[0];if(threadIdx.x==0)scales[row]=s;
 #pragma unroll
 for(int j=0;j<(K+T*4-1)/(T*4);j++){
  int i=(threadIdx.x+j*T)*4;auto v=values[j];
  if(i<K){v.x=__fdiv_rn(v.x,s);v.y=__fdiv_rn(v.y,s);v.z=__fdiv_rn(v.z,s);v.w=__fdiv_rn(v.w,s);((__nv_fp8x4_e4m3*)(y+size_t(row)*K))[i/4]=__nv_fp8x4_e4m3(v);}
 }
}
#define SPECIAL(K,T) extern "C" __global__ void quant_f16_##K(const half*x,__nv_fp8_e4m3*y,float*s){packed_quant<K,T>(x,y,s);}
SPECIAL(768,64) SPECIAL(1024,64) SPECIAL(1152,64) SPECIAL(3072,128)
SPECIAL(4096,128) SPECIAL(8192,128) SPECIAL(12288,128)
extern "C" __global__ void quant_f16_generic(const half*x,__nv_fp8_e4m3*y,float*scales,int k){
 __shared__ float partial[4];int row=blockIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32;float mx=0;
 for(int i=threadIdx.x;i<k;i+=128)mx=fmaxf(mx,fabsf(float(x[size_t(row)*k+i])));
 for(int d=16;d;d/=2)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,d));
 if(lane==0)partial[warp]=mx;__syncthreads();
 if(warp==0){mx=lane<4?partial[lane]:0;for(int d=16;d;d/=2)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,d));if(lane==0)partial[0]=__fmul_rn(fmaxf(mx,1e-12f),1.f/448.f);}
 __syncthreads();float s=partial[0];if(threadIdx.x==0)scales[row]=s;
 for(int i=threadIdx.x;i<k;i+=128)y[size_t(row)*k+i]=__nv_fp8_e4m3(__fdiv_rn(float(x[size_t(row)*k+i]),s));
}

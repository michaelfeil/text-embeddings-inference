/*
Adapted from candle-cublaslt/kernels/fp8_quant.cu under its MIT license.
Original row-scale and correctly-rounded division arithmetic is retained.

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
*/
#include "helpers/fp8_linear.h"
#include "kernels/gated_value.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <climits>
namespace tei {
namespace {
template <typename T>
__device__ void tei_fp8_quant_row(const T *x, __nv_fp8_e4m3 *y, float *scales, int k) {
  __shared__ float partial[4];
  int row = blockIdx.x, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  float mx = 0;
  for (int i = threadIdx.x; i < k; i += 128)
    mx = fmaxf(mx, fabsf(float(x[i])));
  for (int d = 16; d; d /= 2)
    mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, d));
  if (lane == 0)
    partial[warp] = mx;
  __syncthreads();
  if (warp == 0) {
    mx = lane < 4 ? partial[lane] : 0;
    for (int d = 16; d; d /= 2)
      mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, d));
    if (lane == 0)
      partial[0] = __fmul_rn(fmaxf(mx, 1e-12f), 1.f / 448.f);
  }
  __syncthreads();
  float s = partial[0];
  if (threadIdx.x == 0)
    scales[row] = s;
  for (int i = threadIdx.x; i < k; i += 128)
    y[size_t(row) * k + i] =
        __nv_fp8_e4m3(__fdiv_rn(float(x[i]), s));
}

// Keep original model-rounded Half/BF16 values in real-width shared storage. The
// unchanged row quantizer consumes them exactly as it consumed the old buffer.
template<typename T>
__global__ void gated_quant_kernel(const T* gate_up,__nv_fp8_e4m3* y,float* scales,int width) {
  extern __shared__ unsigned char storage[];
  auto* values=reinterpret_cast<T*>(storage);
  const auto* row=gate_up+size_t(blockIdx.x)*2*width;
  for(int i=threadIdx.x;i<width;i+=128)
    values[i]=detail::gated_value<T>(float(row[i]),float(row[width+i]),4);
  __syncthreads();
  tei_fp8_quant_row(values,y,scales,width);
}

template<typename T>
__global__ void quant_kernel(const T* x,__nv_fp8_e4m3* y,float* scales,int width) {
  tei_fp8_quant_row(x+size_t(blockIdx.x)*width,y,scales,width);
}
}
std::pair<at::Tensor,at::Tensor> fp8_quantize_rows(const at::Tensor& input) {
  TORCH_CHECK(input.is_cuda() && input.dim()==2 && input.is_contiguous()
    && (input.scalar_type()==at::kHalf || input.scalar_type()==at::kBFloat16),
    "FP8 row quantization requires contiguous CUDA float16/bfloat16 matrices");
  TORCH_CHECK(input.size(1)>0 && input.size(1)<=INT_MAX && input.size(0)<=INT_MAX,
    "Invalid FP8 row quantization geometry");
  c10::cuda::CUDAGuard guard(input.device());
  auto output=at::empty(input.sizes(),input.options().dtype(at::kFloat8_e4m3fn));
  auto scales=at::empty({input.size(0),1},input.options().dtype(at::kFloat));
  if(input.size(0)) {
    auto stream=at::cuda::getCurrentCUDAStream(input.device().index());
    auto* target=reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr());
    if(input.scalar_type()==at::kHalf)
      quant_kernel<<<input.size(0),128,0,stream>>>(reinterpret_cast<const half*>(input.data_ptr()),target,scales.data_ptr<float>(),static_cast<int>(input.size(1)));
    else
      quant_kernel<<<input.size(0),128,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),target,scales.data_ptr<float>(),static_cast<int>(input.size(1)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output,scales};
}
std::pair<at::Tensor,at::Tensor> fp8_quantize_gated_rows(const at::Tensor& input) {
  TORCH_CHECK(input.is_cuda() && input.dim()==2 && input.is_contiguous() &&
    (input.scalar_type()==at::kHalf || input.scalar_type()==at::kBFloat16),
    "FP8 gated row quantization requires contiguous CUDA Half/BF16 gate/up matrices");
  TORCH_CHECK(input.size(1)>0 && input.size(1)%2==0 && input.size(0)<=INT_MAX,
    "Invalid actual gated FP8 row geometry");
  const int64_t width=input.size(1)/2;
  TORCH_CHECK(width*sizeof(half)<=48*1024-4*sizeof(float),
    "Gated FP8 row exceeds bounded real-width shared storage");
  c10::cuda::CUDAGuard guard(input.device());
  auto output=at::empty({input.size(0),width},input.options().dtype(at::kFloat8_e4m3fn));
  auto scales=at::empty({input.size(0),1},input.options().dtype(at::kFloat));
  if(input.size(0)) {
    auto stream=at::cuda::getCurrentCUDAStream(input.device().index());
    auto* target=reinterpret_cast<__nv_fp8_e4m3*>(output.mutable_data_ptr());
    if(input.scalar_type()==at::kHalf)
      gated_quant_kernel<<<input.size(0),128,width*sizeof(half),stream>>>(
        reinterpret_cast<const half*>(input.const_data_ptr()),target,scales.data_ptr<float>(),int(width));
    else
      gated_quant_kernel<<<input.size(0),128,width*sizeof(__nv_bfloat16),stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.const_data_ptr()),target,scales.data_ptr<float>(),int(width));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output,scales};
}

}

#include "fp8_linear.h"
#include <ATen/cuda/CUDAContext.h>
#include "fp8_reference_test.h"
#include <cstdlib>
#include <c10/core/InferenceMode.h>
#include <iostream>

int main() {
  // Match the original independently implemented recipe's workspace budget.
  setenv("CUBLASLT_WORKSPACE_SIZE","32768",1);
  c10::InferenceMode inference;
  at::manual_seed(7249);
  at::globalContext().setAllowTF32CuBLAS(false);
  for(auto dtype:{at::kHalf,at::kBFloat16}) {
    auto options=at::TensorOptions().device(at::kCUDA,0).dtype(dtype);
    auto weight=at::randn({512,256},options)*.05;
    tei::Fp8Linear linear(weight);
    for(int64_t rows:{1,3,7,17,35,128}) {
      auto input=at::randn({rows,256},options)*.3;
      // Exact zero and sub-floor rows exercise the scale clamp, including
      // fp16 underflow. Row count remains the actual number of tokens.
      input.select(0,0).zero_();
      if(rows>1)input.select(0,1).fill_(1e-14);
      auto quantized=tei::fp8_quantize_rows(input);
      auto cpu=input.to(at::kCPU).to(at::kFloat);
      auto scale=at::amax(cpu.abs(),{1},true).clamp_min(1e-12)*(1.f/448.f);
      auto reference=(cpu/scale).clamp(-448.,448.).to(at::kFloat8_e4m3fn);
      TORCH_CHECK(at::equal(quantized.second.to(at::kCPU),scale),"FP8 row-scale arithmetic differs from independent CPU oracle");
      TORCH_CHECK(at::equal(quantized.first.to(at::kCPU).view(at::kByte),reference.view(at::kByte)),
        "FP8 quantized bytes differ from independent CPU oracle");
      TORCH_CHECK(quantized.first.sizes()==input.sizes() && quantized.second.size(0)==rows,
        "FP8 quantization padded actual token rows");
      auto output=linear.forward(input);
      auto expected=fp8_original_recipe(quantized.first,linear.matrix(),quantized.second,linear.scales(),dtype);
      TORCH_CHECK(at::equal(output,expected),
        "Native Torch FP8 GEMM differs from original independent cuBLASLt recipe: rows=",rows," dtype=",dtype);
      TORCH_CHECK(output.sizes()==at::IntArrayRef({rows,512}) && output.scalar_type()==dtype,
        "FP8 GEMM changed actual row count or model dtype");
      auto bias=at::randn({512},options)*.01;
      TORCH_CHECK(at::equal(linear.forward(input,bias),output+bias),"FP8 bias lost separate model-dtype rounding");
    }
    auto empty=linear.forward(at::empty({0,256},options));
    TORCH_CHECK(empty.sizes()==at::IntArrayRef({0,512}),"FP8 empty rows not preserved");
    std::cout<<"FP8 row-scale/bytes, odd real rows, model-dtype GEMM and bias: "<<dtype<<" passed\n";
  }
  for(auto shape:{std::pair<int64_t,int64_t>{511,256},{512,255}}) {
    bool rejected=false;
    try{tei::Fp8Linear invalid(at::zeros({shape.first,shape.second},at::TensorOptions().device(at::kCUDA,0).dtype(at::kHalf)));}
    catch(const c10::Error&){rejected=true;}
    TORCH_CHECK(rejected,"Unsupported FP8 projection width silently padded");
  }
}

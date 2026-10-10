// Native CUDA test for the exact Candle normalization adapter.
#include "encoder_norm.h"
#include <c10/core/InferenceMode.h>
#include <iostream>

int main(int argc,char** argv) {
  c10::InferenceMode inference;
  for(auto dtype:{at::kHalf,at::kBFloat16}) {
    for(int64_t width:{8,32,40,256,768,1024,1152,1536,2048}) {
      auto options=at::TensorOptions().device(c10::Device(argc>1?argv[1]:"cuda:0")).dtype(dtype);
      auto x=at::randn({7,width},options),residual=at::randn_like(x);
      auto weight=at::randn({width},options)*0.1+1;
      for(bool add:{false,true}) {
        auto result=tei::encoder_layer_norm(x,weight,1e-5,add?residual:at::Tensor());
        auto sum=add?x+residual:x;
        auto expected=at::layer_norm(sum.to(at::kFloat),{width},weight.to(at::kFloat),at::Tensor(),1e-5).to(dtype);
        auto error=(result.first-expected).abs().max().item<float>();
        TORCH_CHECK(error<=(dtype==at::kHalf?0.003:0.03125),"Encoder norm error ",error," at width ",width);
        if(add) {
          TORCH_CHECK(at::equal(result.second,sum),"Residual sum must round exactly to model dtype");
        } else {
          TORCH_CHECK(!result.second.defined(),"Plain normalization must not allocate residual output");
        }
      }
    }
  }
  std::cout<<"Exact encoder CUDA normalization tests passed\n";
}

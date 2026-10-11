// Native CUDA test for the exact Candle normalization adapter.
#include "kernels/encoder_norm.h"
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <cstring>
#include <c10/cuda/CUDAGuard.h>
#include "tests/fixtures/tei-bert-fp8-learned-norm-golden.h"

int main(int argc,char** argv) {
  c10::InferenceMode inference;
  for(auto dtype:{at::kHalf,at::kBFloat16}) {
    auto opts=at::TensorOptions().device(c10::Device(argc>1?argv[1]:"cuda:0")).dtype(dtype);
    auto base=at::full({2,768},32,opts);
    auto delta=(at::arange(768,opts.dtype(at::kFloat)).remainder(2)*.008-.004).to(dtype).repeat({2,1});
    auto gamma=at::ones({768},opts),beta=at::full({768},.25,opts);
    auto exact=tei::encoder_layer_norm(base,gamma,1e-5,delta,beta,false);
    auto unrounded=base.to(at::kFloat)+delta.to(at::kFloat);
    auto oracle=at::layer_norm(unrounded,{768},gamma.to(at::kFloat),beta.to(at::kFloat),1e-5).to(dtype);
    TORCH_CHECK((exact.first-oracle).abs().max().item<float>()<=(dtype==at::kHalf?.003:.03125),"Unrounded residual statistics or beta differs");
    TORCH_CHECK(at::equal(exact.second,base+delta),"Returned residual sum must retain model dtype");
    auto only=tei::encoder_layer_norm_only(base,gamma,1e-5,delta,beta,false);
    TORCH_CHECK((only-oracle).abs().max().item<float>()<=(dtype==at::kHalf?.003:.03125),"Normalized-only unrounded statistics or beta differs from independent FP32 oracle");
    TORCH_CHECK(at::equal(only,exact.first),"Suppressing residual output changed exact arithmetic");
    auto rounded=tei::encoder_layer_norm(base,gamma,1e-5,delta,beta,true);
    TORCH_CHECK((exact.first-rounded.first).abs().max().item<float>()>.5,"Rounding discriminator did not exercise distinct normalization contracts");
    {
      auto x=at::randn({17,384},opts),residual=at::randn_like(x)*.2;
      auto scale=at::randn({384},opts)*.1+1,bias=at::randn({384},opts)*.1;
      auto stats=x.to(at::kFloat)+residual.to(at::kFloat);
      auto reference=at::layer_norm(stats,{384},scale.to(at::kFloat),bias.to(at::kFloat),1e-12).to(dtype);
      auto actual=tei::encoder_layer_norm_only(x,scale,1e-12,residual,bias,false);
      TORCH_CHECK((actual-reference).abs().max().item<float>()<=(dtype==at::kHalf?.003:.03125),"Width384 biased unrounded norm differs from independent FP32 reference");
    }
    for(int64_t width:{8,32,40,256,384,768,1024,1152,1536,2048}) {
      auto options=at::TensorOptions().device(c10::Device(argc>1?argv[1]:"cuda:0")).dtype(dtype);
      auto x=at::randn({7,width},options),residual=at::randn_like(x);
      auto weight=at::randn({width},options)*0.1+1;
      for(bool add:{false,true}) {
        auto result=tei::encoder_layer_norm(x,weight,1e-5,add?residual:at::Tensor());
        auto sum=add?x+residual:x;
        auto expected=at::layer_norm(sum.to(at::kFloat),{width},weight.to(at::kFloat),at::Tensor(),1e-5).to(dtype);
        auto normalized_only=tei::encoder_layer_norm_only(x,weight,1e-5,add?residual:at::Tensor());
        TORCH_CHECK((normalized_only-expected).abs().max().item<float>()<=(dtype==at::kHalf?0.003:0.03125),"Normalized-only differs from independent FP32 oracle at width ",width);
        TORCH_CHECK(at::equal(normalized_only,result.first),"Normalized-only changed rounded normalization at width ",width);
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
  // Independent expected bits are from actual original Candle run_ln, not this adapter.
  auto options=at::TensorOptions().device(c10::Device(argc>1?argv[1]:"cuda:0")).dtype(at::kHalf);
  auto load_bits=[&](const uint16_t* bits,bool row){auto cpu=at::empty(row?std::vector<int64_t>{1,768}:std::vector<int64_t>{768},at::kHalf);std::memcpy(cpu.mutable_data_ptr(),bits,1536);return cpu.to(options.device());};
  for(const auto& bits : {std::array<const uint16_t*,5>{norm_0_x,norm_0_residual,norm_0_gamma,norm_0_beta,norm_0_expected},std::array<const uint16_t*,5>{norm_1_x,norm_1_residual,norm_1_gamma,norm_1_beta,norm_1_expected}}) {
    auto x=load_bits(bits[0],true),residual=load_bits(bits[1],true),gamma=load_bits(bits[2],false),beta=load_bits(bits[3],false),expected=load_bits(bits[4],true);
    if(c10::cuda::device_count()>1)c10::cuda::set_device(x.device().index()==0?1:0);
    const auto previous_device=c10::cuda::current_device();
    auto actual=tei::encoder_layer_norm_only(x,gamma,1e-12,residual,beta,false);
    TORCH_CHECK(c10::cuda::current_device()==previous_device,"Normalization changed caller current device");
    TORCH_CHECK(at::equal(actual,expected),"Learned residual norm differs from actual original Candle golden bits");
    auto rounded=tei::encoder_layer_norm_only(x,gamma,1e-12,residual,beta,true);
    TORCH_CHECK(!at::equal(rounded,expected),"Learned operands must distinguish unrounded residual statistics");
  }
  std::cout<<"Exact encoder CUDA normalization tests passed\n";
}

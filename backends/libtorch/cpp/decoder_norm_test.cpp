// Independent CUDA checks for residual precision and BF16 normalization.
#include "decoder_norm.h"
#include <c10/core/InferenceMode.h>
#include <iostream>
int main(int argc,char** argv) {
 c10::InferenceMode inference;at::manual_seed(43);
 auto options=at::TensorOptions().dtype(at::kBFloat16).device(c10::Device(argc>1?argv[1]:"cuda:0"));
 for(int64_t rows:{1,7,257}) {
  auto x=(at::rand({rows,2048},options)-.5),r=(at::rand_like(x)-.5),w=at::ones({2048},options);
  for(bool residual:{false,true})for(double eps:{1e-6,1e-3}) {
   auto result=tei::decoder_exact_rms_norm(x,residual?r:at::Tensor(),w,eps);
   auto unrounded=residual?x.to(at::kFloat)+r.to(at::kFloat):x.to(at::kFloat);
   auto expected=(unrounded*at::rsqrt(unrounded.square().mean(-1,true)+eps)).to(at::kBFloat16);
   TORCH_CHECK((result.first-expected).abs().max().item<float>()<=.015625,"Exact RMS differs by more than one BF16 output quantum from independent FP32 statistics");
   TORCH_CHECK(at::equal(result.second,residual?x+r:x),"RMS residual must round only at its output store");
   if(!residual)TORCH_CHECK(result.second.is_alias_of(x),"Plain RMS must retain the input residual without a copy");
  }
 }
 auto x=at::ones({1,2048},options);x.narrow(1,1024,1024).fill_(2);
 auto r=at::full_like(x,.005),w=at::full({2048},64.,options);
 auto result=tei::decoder_exact_rms_norm(x,r,w,1e-6);
 auto full=x.to(at::kFloat)+r.to(at::kFloat);
 auto expected=(full*at::rsqrt(full.square().mean(-1,true)+1e-6)*w.to(at::kFloat)).to(at::kBFloat16);
 auto rounded=(x+r).to(at::kFloat);auto wrong=(rounded*at::rsqrt(rounded.square().mean(-1,true)+1e-6)*w.to(at::kFloat)).to(at::kBFloat16);
 TORCH_CHECK(at::equal(result.first,expected),"Known residual-rounding regression must match independent FP32 oracle exactly");
 TORCH_CHECK((result.first-wrong).abs().max().item<float>()>=.125,"Fixture must distinguish pre-rounded residual statistics");
 bool rejected=false;try{tei::decoder_exact_rms_norm(x.to(at::kHalf),r.to(at::kHalf),w.to(at::kHalf),1e-6);}catch(const c10::Error&){rejected=true;}TORCH_CHECK(rejected,"Unvalidated dtype must be rejected");
 std::cout<<"Exact routed decoder RMS tests passed\n";
}

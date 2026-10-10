#include "fast_kernels.h"
#include <ATen/ops/_fused_rms_norm.h>
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <cmath>
void close(const at::Tensor& x,const at::Tensor& expected,double tolerance,const char* label){
 auto error=(x.to(at::kFloat)-expected.to(at::kFloat)).abs().max().item<float>();
 TORCH_CHECK(error<=tolerance,label," max abs=",error," tolerance=",tolerance);
}
// Round the double oracle directly to the model format. ATen's CPU double
// conversion passes through float, which double-rounds rare HFMA midpoint cases.
at::Tensor round_model_double(const at::Tensor& values,at::ScalarType dtype) {
 auto input=values.contiguous();auto output=at::empty(input.sizes(),input.options().dtype(dtype));
 const int mantissa=dtype==at::kHalf?10:7,bias=dtype==at::kHalf?15:127;
 const int maximum=dtype==at::kHalf?0x7bff:0x7f7f;
 auto decode=[&](int bits){int exponent=bits>>mantissa,fraction=bits&((1<<mantissa)-1);
  return std::ldexp(double(exponent?(1<<mantissa)+fraction:fraction),(exponent?exponent:1)-bias-mantissa);};
 const auto* source=input.const_data_ptr<double>();auto* dest=reinterpret_cast<uint16_t*>(output.mutable_data_ptr());
 for(int64_t i=0;i<input.numel();++i){double v=source[i],a=std::abs(v);TORCH_CHECK(std::isfinite(v)&&a<=decode(maximum),"Oracle requires finite representable range");
  int lo=0,hi=maximum;while(lo<hi){int mid=(lo+hi)/2;if(decode(mid)<a)lo=mid+1;else hi=mid;}
  int bits=lo;if(lo>0){double below=a-decode(lo-1),above=decode(lo)-a;if(below<above||(below==above&&((lo-1)&1)==0))bits=lo-1;}
  dest[i]=uint16_t(bits|(std::signbit(v)?0x8000:0));
 }
 return output;
}
int main(){c10::InferenceMode guard;
 for(auto dtype:{at::kHalf,at::kBFloat16,at::kFloat}) {
  auto opts=at::TensorOptions().device(at::kCUDA).dtype(dtype);
  for(int width:{7,16,63,128,257,513,768,769,1024,1152,2048,4096,8192}) {
   auto x=at::randn({33,width},opts),r=at::randn_like(x),r2=at::randn_like(x);
   auto w=at::randn({width},opts),b=at::randn({width},opts);
   auto tol=dtype==at::kBFloat16?0.04:dtype==at::kHalf?0.006:0.00001;
   close(tei::fused_add_layer_norm(x,r,w,b,1e-5),at::layer_norm(x+r,{width},w,b,1e-5),tol,"residual LN");
   close(tei::fused_add_layer_norm(x,r,w,b,1e-5,r2),at::layer_norm(x+r+r2,{width},w,b,1e-5),tol,"two residual LN");
   auto rms=tei::fused_add_rms_norm(x,r,w,1e-5);
   close(rms.second,x+r,0,"RMS residual output");
   auto unrounded=x.to(at::kFloat)+r.to(at::kFloat);
   auto rms_reference=(unrounded*at::rsqrt(unrounded.square().mean(-1,true)+1e-5))*w.to(at::kFloat);
   close(rms.first,rms_reference.to(dtype),tol,"unrounded residual RMSNorm");
   auto initial=tei::fused_add_rms_norm(x,{},w,1e-5);
   close(initial.second,x,0,"Initial RMS residual aliases original input");
   TORCH_CHECK(initial.second.data_ptr()==x.data_ptr(),"Initial RMS residual must reuse input storage");
   auto both=at::randn({33,2*width},opts);
   for(int code=0;code<4;++code)for(bool gate_first:{false,true}) {
     auto gate=both.narrow(-1,gate_first?0:width,width),up=both.narrow(-1,gate_first?width:0,width);
     auto activated=code==0?at::silu(gate):code==1?at::gelu(gate,"none"):code==2?at::gelu(gate,"tanh"):at::relu(gate);
     close(tei::fused_gated_activation(both,code,gate_first),activated*up,tol,"gated activation");
   }
   for(bool gate_first:{false,true}) {
     auto gate=both.narrow(-1,gate_first?0:width,width),up=both.narrow(-1,gate_first?width:0,width);
     auto scalar=[&](double value){return at::full({},value,opts);};
     auto square=gate*gate,cube=square*gate;
     // Candle/NVCC contracts this expression into a modeldtype FMA.
     // FP64 exactly represents these bounded half/BF16 operands' product
     // and sum; round once instead of inserting an extra product rounding.
     auto alpha=(gate.to(at::kDouble)+scalar(.044715).to(at::kDouble)*cube.to(at::kDouble)).to(dtype);
     auto tangent=at::tanh(scalar(.7978845608028654)*alpha);
     auto precise=(scalar(.5)*gate)*(scalar(1)+tangent);
     close(tei::fused_gated_activation(both,4,gate_first),precise*up,tol,"Candle precise gated GELU");
   }
   for(bool gate_first:{false,true}) {
    auto gate=both.narrow(-1,gate_first?0:width,width),up=both.narrow(-1,gate_first?width:0,width);
    auto exponential=at::exp(-gate),denominator=at::ones_like(gate)+exponential;
    close(tei::fused_gated_activation(both,5,gate_first),(gate/denominator)*up,tol,"Candle precise gated SiLU");
   }
   if(width%4==0) {
    auto storage=at::empty({both.numel()+1},opts);
    auto unaligned=storage.narrow(0,1,both.numel()).view_as(both);
    unaligned.copy_(both);
    for(int code=0;code<=5;++code)for(bool gate_first:{false,true})
      close(tei::fused_gated_activation(both,code,gate_first),tei::fused_gated_activation(unaligned,code,gate_first),0,"vector4/scalar gated activation parity");
   }
   for(bool approx:{false,true}) {
    close(tei::fused_gelu(x,approx),at::gelu(x,approx?"tanh":"none"),tol,"GELU");
    close(tei::fused_bias_gelu(x,b,approx),at::gelu(x+b,approx?"tanh":"none"),tol,"bias GELU");
   }
  }
  if(dtype!=at::kFloat){
   auto qkv=at::randn({7,8,128},opts);
   auto q=qkv.narrow(1,0,4),k=qkv.narrow(1,4,2);
   auto qw=at::randn({128},opts),kw=at::randn({128},opts);
   auto angle=at::randn({7,1,64},opts);auto cosine=at::cos(angle),sine=at::sin(angle);
   auto rotary=tei::fused_rotary(q,k,cosine,sine);
   auto rope=[&](const at::Tensor& x){auto a=x.narrow(-1,0,64),b=x.narrow(-1,64,64);return at::cat({a*cosine-b*sine,b*cosine+a*sine},-1);};
   close(rotary.first,rope(q),0,"Q rotary");close(rotary.second,rope(k),0,"K rotary");
   // A double reference preserves the exact half/BF16 HFMA result and
   // catches FP32-to-model-dtype double rounding at midpoint values.
   auto contracted_ref=[&](const at::Tensor& x){
    auto cpu=x.to(at::kCPU,at::kDouble),c=cosine.to(at::kCPU,at::kDouble),s=sine.to(at::kCPU,at::kDouble);
    auto a=cpu.narrow(-1,0,64),b=cpu.narrow(-1,64,64);
    auto bs=(b*s).to(dtype).to(at::kDouble),as=(a*s).to(dtype).to(at::kDouble);
    return round_model_double(at::cat({a*c-bs,b*c+as},-1),dtype).to(x.device());
   };
   auto contracted=tei::fused_rotary(q,k,cosine,sine,true);
   close(contracted.first,contracted_ref(q),0,"Q contracted rotary");
   close(contracted.second,contracted_ref(k),0,"K contracted rotary");
   auto inplace_qkv=qkv.clone(),inplace_q=inplace_qkv.narrow(1,0,4),inplace_k=inplace_qkv.narrow(1,4,2);
   tei::encoder_rotary_inplace(inplace_q,inplace_k,cosine,sine);
   close(inplace_q,contracted.first,0,"Q inplace encoder rotary");
   close(inplace_k,contracted.second,0,"K inplace encoder rotary");
   close(inplace_qkv.narrow(1,6,2),qkv.narrow(1,6,2),0,"Inplace encoder rotary preserves V");
   // Pitched GQA heads, incomplete launch blocks and partial-rotary fallback.
   for(int dim:{64,128,256})for(bool partial:{false,true})for(bool contract:{false,true}) {
    auto packed=at::randn({17,16,dim},opts),original=packed.clone();
    auto qs=packed.narrow(1,0,12),ks=packed.narrow(1,12,2);
    const int half=partial?dim/4:dim/2;
    auto c=at::randn({17,1,half},opts),s=at::randn_like(c);
    auto result=tei::fused_rotary(qs,ks,c,s,contract);
    auto reference=[&](const at::Tensor& input) {
     auto f=input.to(at::kCPU,at::kDouble),cd=c.to(at::kCPU,at::kDouble),sd=s.to(at::kCPU,at::kDouble);
     auto a=f.narrow(-1,0,half),b=f.narrow(-1,half,half);
     auto bs=(b*sd).to(dtype).to(at::kDouble),as=(a*sd).to(dtype).to(at::kDouble);
     auto ac=a*cd,bc=b*cd;
     if(!contract){ac=ac.to(dtype).to(at::kDouble);bc=bc.to(dtype).to(at::kDouble);}
     auto rotated=at::cat({ac-bs,bc+as},-1);
     if(partial)rotated=at::cat({rotated,f.narrow(-1,2*half,dim-2*half)},-1);
     return round_model_double(rotated,dtype).to(input.device());
    };
    close(result.first,reference(qs),0,"Q paired/partial rotary independent rounded oracle");
    close(result.second,reference(ks),0,"K paired/partial rotary independent rounded oracle");
    close(packed,original,0,"Out-of-place rotary preserves all QKV storage");
   }
   auto actual=tei::fused_qk_norm_rope(q,k,qw,kw,cosine,sine,1e-6);
   auto ref=[&](const at::Tensor& x,const at::Tensor& weight){auto n=std::get<0>(at::_fused_rms_norm(x.contiguous(),{128},weight,1e-6));auto a=n.narrow(-1,0,64),b=n.narrow(-1,64,64);return at::cat({a*cosine-b*sine,b*cosine+a*sine},-1);};
   close(actual.first,ref(q,qw),dtype==at::kBFloat16?0.0625:0.008,"Q norm+RoPE");
   close(actual.second,ref(k,kw),dtype==at::kBFloat16?0.0625:0.008,"K norm+RoPE");
  }
  std::cout<<dtype<<" fused CUDA operator parity passed\n";
 }
 {
  auto opts=at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  auto x=at::ones({2,256},opts);x.narrow(-1,128,128).fill_(2);
  auto residual=at::full_like(x,.005),weight=at::full({256},64.,opts);
  auto exact=x.to(at::kFloat)+residual.to(at::kFloat);
  auto reference=(exact*at::rsqrt(exact.square().mean(-1,true)+1e-5)*weight.to(at::kFloat)).to(at::kBFloat16);
  auto rounded=x+residual;
  auto wrong=std::get<0>(at::_fused_rms_norm(rounded,{256},weight,1e-5));
  TORCH_CHECK((wrong-reference).abs().max().item<float>()>=.125,"Residual rounding fixture must distinguish semantics");
  close(tei::fused_add_rms_norm(x,residual,weight,1e-5).first,reference,0,"RMS statistics retain FP32 residual sum");
 }
}

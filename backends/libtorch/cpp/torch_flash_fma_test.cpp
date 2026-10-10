// Independent exact-length reference for the private native Torch Flash adapter.
#include "torch_flash_fma.h"
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <vector>
#include <cmath>

int main(int argc,char** argv) {
  c10::InferenceMode guard;
  const c10::Device device(argc>1 ? argv[1] : "cuda:0");
  at::manual_seed(9917);at::set_num_threads(1);
  for(auto dtype:{at::kHalf,at::kBFloat16})for(int64_t dim:{64,128,256})
  for(bool grouped:{false,true})for(bool causal:{false,true})for(int64_t window:{-1,2})
  for(auto lengths:{std::vector<int32_t>{3,13,7},std::vector<int32_t>{32,129,17}}) {
    const int64_t heads=4,key_heads=grouped?2:4;
    std::vector<int32_t> offsets{0};int64_t total=0,max_sequence=0;
    for(auto n:lengths){total+=n;max_sequence=std::max<int64_t>(max_sequence,n);offsets.push_back(total);}
    auto options=at::TensorOptions().device(device).dtype(dtype);
    // Fresh fused QKV storage tests noncontiguous token/head views, including GQA.
    auto combined=at::randn({total,heads+2*key_heads,dim},options);
    auto q=combined.narrow(1,0,heads),k=combined.narrow(1,heads,key_heads),v=combined.narrow(1,heads+key_heads,key_heads);
    auto cu=at::tensor(offsets,at::TensorOptions().dtype(at::kInt)).to(device);
    const double scale=1./std::sqrt(dim);
    auto output=tei::torch_flash_fma(q,k,v,cu,max_sequence,scale,causal,window,window);
    TORCH_CHECK(at::isfinite(output).all().item<bool>(),"Nonfinite native Torch Flash output");
    double worst=0;
    for(size_t row=0;row+1<offsets.size();++row) {
      const int64_t start=offsets[row],n=offsets[row+1]-start;
      auto cpu=[&](const at::Tensor& x){return x.narrow(0,start,n).to(at::TensorOptions().device(at::kCPU).dtype(at::kDouble)).transpose(0,1);};
      auto qr=cpu(q),kr=cpu(k),vr=cpu(v);
      if(grouped){kr=kr.repeat_interleave(heads/key_heads,0);vr=vr.repeat_interleave(heads/key_heads,0);}
      auto index=at::arange(n,at::kLong),delta=index.unsqueeze(0)-index.unsqueeze(1);
      auto allowed=at::ones({n,n},at::kBool);
      if(causal)allowed.logical_and_(delta<=0);
      if(window>=0)allowed.logical_and_(delta>=-window).logical_and_(delta<=window);
      auto probabilities=at::softmax((at::matmul(qr,kr.transpose(1,2))*scale).masked_fill(allowed.logical_not(),-INFINITY),-1);
      auto expected=at::matmul(probabilities,vr).transpose(0,1);
      worst=std::max(worst,(output.narrow(0,start,n).to(at::TensorOptions().device(at::kCPU).dtype(at::kDouble))-expected).abs().max().item<double>());
    }
    TORCH_CHECK(worst<(dtype==at::kHalf?.004:.04),"Native Torch Flash reference mismatch: ",worst);
    auto changed=v.clone();changed.narrow(0,0,lengths[0]).add_(20);
    auto isolated=tei::torch_flash_fma(q,k,changed,cu,max_sequence,scale,causal,window,window);
    TORCH_CHECK(at::isfinite(isolated).all().item<bool>() && at::equal(output.narrow(0,lengths[0],total-lengths[0]),isolated.narrow(0,lengths[0],total-lengths[0])),"Packed sequence isolation failed");
    std::cout<<"dtype="<<dtype<<" total="<<total<<" dim="<<dim<<" grouped="<<grouped<<" causal="<<causal<<" window="<<window<<" maxabs="<<worst<<'\n';
  }
}

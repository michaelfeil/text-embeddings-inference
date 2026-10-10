// Independent exact-length reference for the private native Torch Flash adapter.
#include "torch_flash_fma.h"
#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAContext.h>
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
  // The exact-width512 launch is intentionally restricted; callers retain
  // efficient attention for unsupported devices, masks and activation types.
  auto wide_options=at::TensorOptions().device(device).dtype(at::kBFloat16);
  auto probe=at::zeros({3,1,512},wide_options);
  const auto* properties=at::cuda::getDeviceProperties(device.index());
  const bool hopper=properties->major==9 && properties->minor==0 && properties->sharedMemPerBlockOptin>=128*1024;
  TORCH_CHECK(tei::torch_flash_fma_supported(probe,probe,probe,false,-1,-1)==hopper,"Wide-head architecture guard mismatch");
  TORCH_CHECK(!tei::torch_flash_fma_supported(probe,probe,probe,true,-1,-1),"Wide-head causal mask must retain fallback");
  TORCH_CHECK(!tei::torch_flash_fma_supported(probe,probe,probe,false,2,2),"Wide-head local mask must retain fallback");
  auto half_probe=probe.to(at::kHalf),cpu_probe=probe.cpu();
  TORCH_CHECK(!tei::torch_flash_fma_supported(half_probe,half_probe,half_probe,false,-1,-1),"Wide-head FP16 must retain fallback");
  TORCH_CHECK(!tei::torch_flash_fma_supported(cpu_probe,cpu_probe,cpu_probe,false,-1,-1),"Wide-head CPU must retain fallback");
  if(hopper)for(bool grouped:{false,true})for(auto lengths:{std::vector<int32_t>{3,17,9},std::vector<int32_t>{32,129,17}}) {
    const int64_t heads=4,key_heads=grouped?1:4;std::vector<int32_t> offsets{0};int64_t total=0,maximum=0;
    for(auto n:lengths){total+=n;maximum=std::max<int64_t>(maximum,n);offsets.push_back(total);}
    auto combined=at::randn({total,heads+2*key_heads,512},wide_options)*.1;
    auto q=combined.narrow(1,0,heads),k=combined.narrow(1,heads,key_heads),v=combined.narrow(1,heads+key_heads,key_heads);
    auto cu=at::tensor(offsets,at::TensorOptions().dtype(at::kInt)).to(device);
    auto output=tei::torch_flash_fma(q,k,v,cu,maximum,1.,false,-1,-1);
    TORCH_CHECK(output.sizes()==q.sizes()&&at::isfinite(output).all().item<bool>(),"Wide-head packed result geometry/finiteness mismatch");
    double worst=0;
    for(size_t row=0;row+1<offsets.size();++row) {
      int64_t start=offsets[row],n=offsets[row+1]-start;
      auto cpu=[&](const at::Tensor& x){return x.narrow(0,start,n).to(at::TensorOptions().device(at::kCPU).dtype(at::kDouble)).transpose(0,1);};
      auto qr=cpu(q),kr=cpu(k),vr=cpu(v);
      if(grouped){kr=kr.repeat_interleave(heads/key_heads,0);vr=vr.repeat_interleave(heads/key_heads,0);}
      auto expected=at::matmul(at::softmax(at::matmul(qr,kr.transpose(1,2)),-1),vr).transpose(0,1);
      worst=std::max(worst,(output.narrow(0,start,n).cpu().to(at::kDouble)-expected).abs().max().item<double>());
    }
    TORCH_CHECK(worst<.003,"Wide-head independent FP64 ragged/GQA mismatch: ",worst);
    auto changed=v.clone();changed.narrow(0,0,lengths[0]).add_(1.);
    auto isolated=tei::torch_flash_fma(q,k,changed,cu,maximum,1.,false,-1,-1);
    TORCH_CHECK(at::equal(output.narrow(0,lengths[0],total-lengths[0]),isolated.narrow(0,lengths[0],total-lengths[0])),"Wide-head packed sequence isolation failed");
    std::cout<<"BF16 head512 ragged/GQA total="<<total<<" grouped="<<grouped<<" maxabs="<<worst<<'\n';
  }
}

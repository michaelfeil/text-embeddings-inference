// Independent composed BF16 product-rounding oracle for native packed Gemma rotary.
#include <ATen/ATen.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "qwen35_kernels.h"
at::Tensor reference(const at::Tensor& x,const at::Tensor& cos,const at::Tensor& sin,const at::Tensor& pos) {
 auto half=x.size(2)/2;auto a=x.slice(2,0,half),b=x.slice(2,half);
 auto c=cos.index_select(0,pos).unsqueeze(1),s=sin.index_select(0,pos).unsqueeze(1);
 return at::cat({a*c-b*s,b*c+a*s},2);
}
int main(int argc,char** argv) {
 c10::Device device(argc>1?argv[1]:"cuda:0");at::manual_seed(7312);
 auto bf=at::TensorOptions().device(device).dtype(at::kBFloat16);
 for(int64_t tokens:{1,3,33,129})for(int64_t heads:{1,2,16})for(int64_t width:{8,72,256,512}) {
  auto storage=at::randn({tokens,heads+3,width},bf);auto x=storage.narrow(1,1,heads);auto angles=at::randn({257,width/2},bf.dtype(at::kFloat));
  auto cos=angles.cos().to(bf),sin=angles.sin().to(bf);
  auto pos=(at::arange(tokens,at::TensorOptions().dtype(at::kLong).device(device))*17).remainder(257);
  auto actual=tei::gemma_rotary_cuda(x,cos,sin,pos),expected=reference(x,cos,sin,pos);
  TORCH_CHECK(at::equal(actual,expected),"Staged rotary mismatch T/H/D ",tokens,"/",heads,"/",width);
 }
 std::cout<<"Gemma BF16 staged rotary passed 48 true packed shapes, positions and fused strides\n";
}

#include "qwen35_reference.h"
#include "qwen35_kernels.h"
#include <ATen/ATen.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

at::Tensor independent_norm(const at::Tensor& x,const at::Tensor& gamma,float eps) {
  auto f=x.to(at::kFloat).contiguous(),g=gamma.to(at::kFloat).contiguous();
  auto result=at::empty_like(f);auto width=x.size(-1),rows=x.numel()/width;
  auto input=f.const_data_ptr<float>(),weight=g.const_data_ptr<float>();auto out=result.mutable_data_ptr<float>();
  int threads=1;while(threads<width && threads<1024)threads*=2;
  for(int64_t row=0;row<rows;++row) {
    std::vector<float> sums(threads,0.f);
    for(int lane=0;lane<threads;++lane)for(int64_t col=lane;col<width;col+=threads) {
      volatile float square=input[row*width+col]*input[row*width+col];
      volatile float add=sums[lane]+square;sums[lane]=add;
    }
    for(int stride=threads/2;stride;stride/=2)for(int lane=0;lane<stride;++lane) {
      volatile float add=sums[lane]+sums[lane+stride];sums[lane]=add;
    }
    volatile float mean=sums[0]*static_cast<float>(1./static_cast<double>(width));
    volatile float variance=mean+eps;volatile float inv=1.f/std::sqrt(variance);
    for(int64_t col=0;col<width;++col) {
      volatile float normalized=input[row*width+col]*inv;
      volatile float scale=weight[col]+1.f;volatile float value=normalized*scale;
      out[row*width+col]=value;
    }
  }
  return result.to(at::kBFloat16);
}

int main(int argc,char** argv) {
  c10::Device device(argc>1?argv[1]:"cuda:0");at::manual_seed(717);
  for(int width:{7,256,384,2048,4096})for(bool strided:{false,true}) {
    auto base=(at::randn({2,3,width*(strided?2:1)})*3).to(at::kBFloat16);
    if(strided)base.narrow(-1,width,width).fill_(NAN);
    auto x=base.narrow(-1,0,width),gamma=(at::randn({width})*.2).to(at::kBFloat16);
    auto cuda_base=base.to(device),cuda_x=cuda_base.narrow(-1,0,width);
    auto actual=tei::qwen35_reference_norm_cuda(cuda_x,gamma.to(device),1e-6).cpu();
    TORCH_CHECK(at::equal(actual,independent_norm(x,gamma,1e-6)),"Independent original binary-tree norm oracle failed width=",width," strided=",strided);
  }
  auto rejects=[&](const at::Tensor& x,const at::Tensor& gamma) {
    bool rejected=false;
    try { (void)tei::qwen35_reference_norm_cuda(x,gamma,1e-6); }
    catch(const c10::Error&) { rejected=true; }
    TORCH_CHECK(rejected,"Invalid norm operands must reject before launch");
  };
  auto opts=at::TensorOptions().device(device).dtype(at::kBFloat16);
  rejects(at::empty({2,0},opts),at::empty({0},opts));
  rejects(at::ones({2,7},opts),at::ones({7},opts.device(at::kCPU)));
  rejects(at::ones({2,7},opts),at::ones({1,7},opts));
  std::ifstream file(std::string(Q35_REFERENCE_FIXTURE_DIR)+"/learned_head.dat",std::ios::binary);
  char magic[8];file.read(magic,8);TORCH_CHECK(std::string(magic,8)=="TEIQ35R1","Invalid learned fixture");
  uint32_t header[4];file.read(reinterpret_cast<char*>(header),sizeof(header));
  TORCH_CHECK(header[0]==32 && header[1]==1 && header[2]==1 && header[3]==4,"Invalid fixture geometry");
  auto tensor=[&](std::vector<int64_t> shape,at::ScalarType dtype=at::kBFloat16) {
    auto t=at::empty(shape,at::TensorOptions().dtype(dtype));file.read(reinterpret_cast<char*>(t.mutable_data_ptr()),t.nbytes());TORCH_CHECK(file.good(),"Truncated learned fixture");return t.to(device);
  };
  auto q=tensor({32,384}),z=tensor({32,128}),ab=tensor({32,2}),conv=tensor({384,4});
  auto alog=tensor({1},at::kFloat),dt=tensor({1},at::kFloat),weight=tensor({128}),expected=tensor({32,128});
  int32_t offsets[3]={0,32,64};tei::PackedInput input;input.offsets=offsets;input.batch=1;input.max_sequence=32;
  input.cumulative=at::from_blob(offsets,{2},at::TensorOptions().dtype(at::kInt)).clone().to(device);
  auto actual=tei::qwen35_reference_delta_cuda(q,z,ab,conv,alog,dt,weight,input,1,1,1e-6);
  TORCH_CHECK(at::equal(actual,expected),"Actual original Candle learned Delta oracle failed");
  // The original trace contains a BF16 rounding boundary at token15,column21.
  // This makes the regression sensitive to restoring default FMA contraction.
  auto contracted=tei::qwen35_delta_cuda(q,z,ab,conv,alog,dt,weight,input,1,1,1e-6);
  TORCH_CHECK(!at::equal(contracted,expected),"Learned fixture must discriminate original no-FMA arithmetic");
  input.batch=2;input.cumulative=at::from_blob(offsets,{3},at::TensorOptions().dtype(at::kInt)).clone().to(device);
  auto twice=tei::qwen35_reference_delta_cuda(at::cat({q,q},0),at::cat({z,z},0),at::cat({ab,ab},0),conv,alog,dt,weight,input,1,1,1e-6);
  TORCH_CHECK(at::equal(twice,at::cat({expected,expected},0)),"Packed sequence recurrence state leaked");
  auto poisoned=tei::qwen35_reference_delta_cuda(at::cat({q,at::full_like(q,NAN)},0),at::cat({z,z},0),at::cat({ab,ab},0),conv,alog,dt,weight,input,1,1,1e-6);
  TORCH_CHECK(at::equal(poisoned.narrow(0,0,32),expected),"Independent sequence contaminated by later NaNs");
  std::cout<<"Independent original norms, learned Delta no-FMA discriminator and packed sequence isolation passed\n";
}

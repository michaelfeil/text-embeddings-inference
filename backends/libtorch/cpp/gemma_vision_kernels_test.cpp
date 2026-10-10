// Independent arithmetic reference for the actual Candle vision contracts.
#include "qwen35_kernels.h"
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <cmath>
#include <iostream>
#include <vector>

int main(int argc,char** argv) {
  c10::InferenceMode guard;
  const c10::Device device(argc>1?argv[1]:"cuda:0");
  at::set_num_threads(1);at::manual_seed(42261);
  for(int64_t width:{4,64,768,2048}) {
    auto input=at::randn({7,3,width},at::TensorOptions().device(device).dtype(at::kBFloat16));
    auto scale=at::randn({width},input.options()).to(at::kFloat);
    auto actual=tei::gemma_vision_norm_cuda(input,scale,1e-6);
    auto host=input.cpu().to(at::kFloat).contiguous(),weights=scale.cpu();
    auto expected=at::empty_like(host);auto x=host.const_data_ptr<float>();auto alpha=weights.const_data_ptr<float>();auto y=expected.mutable_data_ptr<float>();
    int64_t lanes=1;while(lanes<width&&lanes<1024)lanes*=2;
    for(int64_t row=0;row<21;++row) {
      std::vector<float> sums(lanes);
      for(int64_t lane=0;lane<lanes;++lane)for(int64_t col=lane;col<width;col+=lanes) {
        volatile float squared=x[row*width+col]*x[row*width+col];
        sums[lane]=sums[lane]+squared;
      }
      for(int64_t stride=lanes/2;stride;stride/=2)for(int64_t lane=0;lane<stride;++lane)sums[lane]=sums[lane]+sums[lane+stride];
      volatile float mean=sums[0]*float(1./double(width));
      const float denominator=std::sqrt(float(mean+1e-6f));
      for(int64_t col=0;col<width;++col){volatile float normalized=x[row*width+col]/denominator;y[row*width+col]=normalized*alpha[col];}
    }
    TORCH_CHECK(at::equal(actual.cpu(),expected.to(at::kBFloat16)),"Vision FP32 norm differs from independent shared reduction-tree reference width",width);
  }
  for(auto dimensions:{std::vector<int64_t>{3,3,9},std::vector<int64_t>{6,9,64},std::vector<int64_t>{3,6,768}}) {
    const int64_t height=dimensions[0],width=dimensions[1],hidden=dimensions[2],kernel=3;
    auto input=at::randn({height*width,hidden},at::TensorOptions().device(device).dtype(at::kBFloat16))*13.;
    auto actual=tei::gemma_vision_pool_cuda(input,height,width,kernel);
    auto host=input.cpu().to(at::kFloat).contiguous();auto expected=at::empty({height*width/9,hidden},at::kFloat);
    auto x=host.const_data_ptr<float>();auto y=expected.mutable_data_ptr<float>();
    const float reciprocal=float(1./9.),scale=float(std::sqrt(double(hidden)));
    for(int64_t by=0;by<height/3;++by)for(int64_t bx=0;bx<width/3;++bx)for(int64_t col=0;col<hidden;++col) {
      float sum=0.;
      for(int64_t row=0;row<3;++row)for(int64_t column=0;column<3;++column) {
        const int64_t patch=(by*3+row)*width+bx*3+column;
        volatile float scaled=x[patch*hidden+col]*reciprocal;sum=sum+scaled;
      }
      const float rounded=float(c10::BFloat16(sum));
      y[(by*(width/3)+bx)*hidden+col]=rounded*scale;
    }
    TORCH_CHECK(at::equal(actual.cpu(),expected),"Vision pooling differs from independent sequential scatter-add reference");
  }
  // An independent host float powf reference for audio's distinct inverse
  // normalization, including widths above one reduction block.
  at::manual_seed(72943);
  for(int64_t width:{4,64,1024,1536,2048}) {
    auto input=at::randn({7,3,width},at::TensorOptions().device(device).dtype(at::kBFloat16));
    auto scale=at::randn({width},input.options()).to(at::kFloat);
    auto actual=tei::gemma_audio_norm_cuda(input,scale,1e-6);
    auto host=input.cpu().to(at::kFloat).contiguous(),weights=scale.cpu();
    auto expected=at::empty_like(host);auto x=host.const_data_ptr<float>();auto alpha=weights.const_data_ptr<float>();auto y=expected.mutable_data_ptr<float>();
    int64_t lanes=1;while(lanes<width&&lanes<1024)lanes*=2;
    for(int64_t row=0;row<21;++row) {
      std::vector<float> sums(lanes);
      for(int64_t lane=0;lane<lanes;++lane)for(int64_t col=lane;col<width;col+=lanes) {
        volatile float squared=x[row*width+col]*x[row*width+col];sums[lane]=sums[lane]+squared;
      }
      for(int64_t stride=lanes/2;stride;stride/=2)for(int64_t lane=0;lane<stride;++lane)sums[lane]=sums[lane]+sums[lane+stride];
      volatile float mean=sums[0]*float(1./double(width));
      const float inverse=std::pow(float(mean+1e-6f),-.5f);
      for(int64_t col=0;col<width;++col){volatile float normalized=x[row*width+col]*inverse;y[row*width+col]=normalized*alpha[col];}
    }
    TORCH_CHECK(at::equal(actual.cpu(),expected.to(at::kBFloat16)),"Audio FP32 norm differs from independent reduction/pow reference width",width);
  }
  // The text reference stages mean and epsilon separately because Candle
  // compiles Gemma RMSNorm with --fmad=false. Keep this independent case
  // separate from the original vision/audio inputs and random seeds.
  at::manual_seed(42261);
  for(int64_t width:{2816}) {
    auto input=at::randn({257,1,width},at::TensorOptions().device(device).dtype(at::kBFloat16));
    auto scale=at::randn({width},input.options()).to(at::kFloat);
    auto actual=tei::gemma_norm_cuda(input,scale,1e-6,true);
    auto host=input.cpu().to(at::kFloat).contiguous(),weights=scale.cpu();
    auto expected=at::empty_like(host);auto x=host.const_data_ptr<float>();auto alpha=weights.const_data_ptr<float>();auto y=expected.mutable_data_ptr<float>();
    int64_t lanes=1;while(lanes<width&&lanes<1024)lanes*=2;
    for(int64_t row=0;row<257;++row) {
      std::vector<float> sums(lanes);
      for(int64_t lane=0;lane<lanes;++lane)for(int64_t col=lane;col<width;col+=lanes) {
        volatile float squared=x[row*width+col]*x[row*width+col];
        sums[lane]=sums[lane]+squared;
      }
      for(int64_t stride=lanes/2;stride;stride/=2)for(int64_t lane=0;lane<stride;++lane)sums[lane]=sums[lane]+sums[lane+stride];
      volatile float mean=sums[0]*float(1./double(width));
      const float denominator=std::sqrt(float(mean+1e-6f));
      for(int64_t col=0;col<width;++col){volatile float normalized=x[row*width+col]/denominator;y[row*width+col]=normalized*alpha[col];}
    }
    TORCH_CHECK(at::equal(actual.cpu(),expected.to(at::kBFloat16)),"Text reference FP32 norm differs from independent shared reduction-tree reference width",width);
  }
  std::cout<<"Vision norm/pooling, audio runtime-pow and text reference norm match independent references bitwise\n";
}

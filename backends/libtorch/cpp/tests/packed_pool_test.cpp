// SPDX-License-Identifier: Apache-2.0
#include "batching/packed_pool.h"
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <cmath>
#include <limits>
template<class T> void verify(at::ScalarType dtype) {
  std::vector<int32_t> lengths{1,3,33,128,513,1025},offsets{0};
  for(auto length:lengths)offsets.push_back(offsets.back()+length);
  for(int64_t width:{int64_t(9),int64_t(768)}) {
    auto input=(at::sin(at::arange(int64_t(offsets.back())*width,at::kFloat)*.13)*.7)
      .view({offsets.back(),width}).to(dtype);
    std::vector<int32_t> selected{5,1,4,0,2,3},spans;
    for(auto sequence:selected){spans.push_back(offsets[sequence]);spans.push_back(lengths[sequence]);}
    auto boundaries=at::tensor(spans,at::TensorOptions().dtype(at::kInt)).view({int64_t(selected.size()),2}).to(at::kCUDA);
    auto actual=tei::packed_mean_pool(input.to(at::kCUDA),boundaries).cpu();
    auto expected=at::empty({int64_t(selected.size()),width},input.options());
    auto source=input.template const_data_ptr<T>();
    auto target=expected.template data_ptr<T>();
    for(size_t row=0;row<selected.size();++row) {
      auto sequence=selected[row],length=lengths[sequence];
      int threads=1;while(threads<length&&threads<1024)threads*=2;
      for(int64_t column=0;column<width;++column) {
        std::vector<T> partial(threads,T(0.f));
        for(int lane=0;lane<threads;++lane)
          for(int token=lane;token<length;token+=threads)
            partial[lane]=T(float(partial[lane])+float(source[(int64_t(offsets[sequence])+token)*width+column]));
        for(int step=threads/2;step;step/=2)
          for(int lane=0;lane<step;++lane)partial[lane]=T(float(partial[lane])+float(partial[lane+step]));
        auto reciprocal=T(float(1./double(length)));
        target[row*width+column]=T(float(partial[0])*float(reciprocal));
      }
    }
    TORCH_CHECK(at::equal(actual,expected),"Packed mean pooling differs from independent rounded reduction tree");
    auto cumulative=at::tensor(offsets,at::TensorOptions().dtype(at::kInt)).to(at::kCUDA);
    auto full=tei::packed_mean_pool_cumulative(input.to(at::kCUDA),cumulative).cpu();
    for(size_t row=0;row<selected.size();++row)
      TORCH_CHECK(at::equal(full[selected[row]],expected[row]),
        "Cumulative packed mean differs from independent rounded reduction tree");
  }
}
void verify_half_overflow() {
  constexpr int64_t length=2048,width=8;
  auto input=at::empty({length,width},at::TensorOptions().dtype(at::kHalf));
  auto* values=input.data_ptr<at::Half>();
  const auto inf=std::numeric_limits<float>::infinity();
  for(int64_t row=0;row<length;++row) {
    const float features[]={64.f,(row%32<16?2048.f:-2048.f),
      row==0?std::numeric_limits<float>::quiet_NaN():1.f,
      row==0?inf:1.f,row==0?-inf:1.f,.25f,
      row==0?inf:row==1?-inf:1.f,-0.f};
    for(int64_t column=0;column<width;++column)values[row*width+column]=features[column];
  }
  auto hidden=input.to(at::kCUDA);
  auto spans=at::tensor({0,int(length)},at::kInt).view({1,2}).to(at::kCUDA);
  auto cumulative=at::tensor({0,int(length)},at::kInt).to(at::kCUDA);
  for(const auto& result:{tei::packed_mean_pool(hidden,spans).cpu(),
                          tei::packed_mean_pool_cumulative(hidden,cumulative).cpu()}) {
    const auto* actual=result.const_data_ptr<at::Half>();
    TORCH_CHECK(actual[0]==at::Half(64.f)&&actual[1]==at::Half(0.f),
      "Finite constant/cancelling means must survive Half sum overflow");
    TORCH_CHECK(std::isnan(float(actual[2]))&&std::isinf(float(actual[3]))&&
      float(actual[3])>0&&std::isinf(float(actual[4]))&&float(actual[4])<0&&
      std::isnan(float(actual[6])),"Genuine NaN/Inf inputs must retain nonfinite propagation");
    TORCH_CHECK(actual[5].x==at::Half(.25f).x&&actual[7].x==at::Half(0.f).x,
      "Finite normal coordinates must retain their original bits");
  }
  // Plain F32 sequential addition can drift above Half's finite boundary too.
  auto maximum=at::full({32768,1},65504.f,input.options()).to(at::kCUDA);
  auto long_spans=at::tensor({0,32768},at::kInt).view({1,2}).to(at::kCUDA);
  auto long_cumulative=at::tensor({0,32768},at::kInt).to(at::kCUDA);
  for(const auto& result:{tei::packed_mean_pool(maximum,long_spans).cpu(),
                          tei::packed_mean_pool_cumulative(maximum,long_cumulative).cpu()})
    TORCH_CHECK(result.const_data_ptr<at::Half>()[0].x==at::Half(65504.f).x,
      "Mean of maximum finite Half values must stay finite");
}
int main() {
  c10::InferenceMode inference;
  verify<at::Half>(at::kHalf);verify<at::BFloat16>(at::kBFloat16);
  verify_half_overflow();
  std::cout<<"Packed mean pooling Half/BF16 exact trees, ragged spans and selection order passed\n";
}

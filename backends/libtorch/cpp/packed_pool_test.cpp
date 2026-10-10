// SPDX-License-Identifier: Apache-2.0
#include "packed_pool.h"
#include <c10/core/InferenceMode.h>
#include <iostream>
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
  }
}
int main() {
  c10::InferenceMode inference;
  verify<at::Half>(at::kHalf);verify<at::BFloat16>(at::kBFloat16);
  std::cout<<"Packed mean pooling Half/BF16 exact trees, ragged spans and selection order passed\n";
}

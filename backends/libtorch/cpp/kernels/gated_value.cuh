// SPDX-License-Identifier: Apache-2.0 AND MIT
// Shared original gated arithmetic; unchanged expressions and model rounding.
/*
Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

*/
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
namespace tei { namespace detail { namespace {
template<class T> __device__ __forceinline__ T candle_model_exp(T value){return T(expf(float(value)));}
template<> __device__ __forceinline__ __half candle_model_exp(__half value){return hexp(value);}
template<> __device__ __forceinline__ __nv_bfloat16 candle_model_exp(__nv_bfloat16 value){return hexp(value);}
template<class T> __device__ __forceinline__ T gated_value(float gate,float up,int activation) {
  float activated;
  if(activation==0)activated=gate/(1.f+expf(-gate));
  else if(activation==5) {
    T value=T(gate),exponential=candle_model_exp(T(-value));
    T denominator=T(T(1)+exponential);
    activated=float(T(value/denominator));
  }
  else if(activation==1)activated=.5f*gate*(1.f+erff(gate*.7071067811865475f));
  else if(activation==2)activated=.5f*gate*(1.f+tanhf(.7978845608028654f*(gate+.044715f*gate*gate*gate)));
  else if(activation==4) {
    // Match MIT-licensed Candle kernels/gated_activation.cu: each GELU
    // intermediate rounds to model dtype, including the tanh input/value.
    T x=T(gate),square=T(x*x),cube=T(square*x);
    T alpha=T(x+T(.044715)*cube);
    T argument=T(T(.7978845608028654)*alpha);
    T tangent=T(tanhf(float(argument)));
    activated=float(T(T(T(.5)*x)*T(T(1)+tangent)));
  }
  else activated=fmaxf(gate,0.f);
  return T(float(T(activated))*up);
}
}}}

#pragma once
#include "batching/prefix_cache.h"
#include "models/model.h"
inline void check_prefix_cache(tei::Model& model,const tei::PackedInput& input,int operation=0,
  double rtol=1e-5,double atol=2e-5) {
  tei::PrefixCache cache(16<<20,32,8);
  auto reference=operation?model.predict(input,operation==2):model.forward(input);
  auto first=cache.run(model,input,operation);
  TORCH_CHECK(at::allclose(first,reference,rtol,atol),"Cold prefix-cache execution changed model outputs");
  first.zero_(); // Caller ownership must not mutate a resident snapshot.
  auto second=cache.run(model,input,operation);
  TORCH_CHECK(at::allclose(second,reference,rtol,atol),"Warm prefix-cache execution changed model outputs, maxabs=",(second-reference).abs().max().item<float>());
  if(model.supports_prefix_cache(operation)) {TORCH_CHECK(cache.stats().hits>0,"Eligible model never reused prefix state");}
  else {TORCH_CHECK(cache.stats().bypasses==2&&cache.stats().hits==0,"Ineligible execution reused prefix state");}
  TORCH_CHECK(cache.bytes()<=16<<20,"Prefix-cache resident budget exceeded");
}

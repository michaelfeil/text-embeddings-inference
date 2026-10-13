#pragma once
#include <ATen/ops/_foreach_copy.h>
#include <array>
#include <map>
#include <vector>

namespace tei {
// Foreach's fast copy route requires a homogeneous tensor list. Graph inputs
// mix integer metadata with model values, so group them before dispatch.
inline void tensor_copies(const std::vector<at::Tensor>& targets,const std::vector<at::Tensor>& sources) {
  using Group=std::pair<std::vector<at::Tensor>,std::vector<at::Tensor>>;
  std::map<std::array<int,3>,Group> groups;
  TORCH_CHECK(targets.size()==sources.size(),"Invalid tensor copy batch");
  for(size_t i=0;i<targets.size();++i) {
    const auto device=targets[i].device();
    auto& group=groups[{int(device.type()),int(device.index()),int(targets[i].scalar_type())}];
    group.first.push_back(targets[i]);group.second.push_back(sources[i]);
  }
  for(auto& [key,group]:groups)at::_foreach_copy_(group.first,group.second,true);
}
}

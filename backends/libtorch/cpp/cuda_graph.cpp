// SPDX-License-Identifier: Apache-2.0
#include "cuda_graph.h"
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <list>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#endif
namespace tei {
struct CudaGraphCache::Impl {
  size_t limit;
  int64_t max_tokens;
#ifdef TEI_TORCH_CUDA_KERNELS
  struct Entry {
    std::vector<int32_t> offsets;
    PackedInput input;
    at::Tensor output;
    std::unique_ptr<at::cuda::CUDAGraph> graph;
    std::list<std::string>::iterator lru;
  };
  std::unordered_map<std::string, std::unique_ptr<Entry>> entries;
  std::list<std::string> lru;
  std::unordered_set<std::string> failed;
#endif
  Impl(size_t count,int64_t tokens):limit(count),max_tokens(tokens){}
};
CudaGraphCache::CudaGraphCache(size_t count,int64_t tokens):impl(std::make_unique<Impl>(count,tokens)){}
CudaGraphCache::~CudaGraphCache()=default;
size_t CudaGraphCache::size() const {
#ifdef TEI_TORCH_CUDA_KERNELS
  return impl->entries.size();
#else
  return 0;
#endif
}
at::Tensor CudaGraphCache::forward(const Model& model,const PackedInput& original) {
#ifdef TEI_TORCH_CUDA_KERNELS
  auto input=original;
  if(!input.ids.is_cuda()||!impl->limit||input.ids.numel()>impl->max_tokens||input.ids.numel()==0)
    return model.forward(input);
  c10::cuda::CUDAGuard device(input.ids.device());
  model.prepare_input(input);
  // Include offsets, max length, device and scalar types: no sequence metadata
  // from one batch can leak into another graph with the same total token count.
  std::ostringstream key;
  key<<&model<<':'<<input.ids.device()<<':'<<input.ids.scalar_type()<<':'<<input.positions.scalar_type()<<':'<<input.types.scalar_type()<<':'<<input.cumulative.scalar_type()<<':'<<input.max_sequence<<':';
  for(int64_t i=0;i<=input.batch;++i)key<<input.offsets[i]<<',';
  auto tensor_key=[&](const at::Tensor& tensor) {
    key<<';'<<tensor.defined();
    if(tensor.defined())key<<':'<<tensor.device()<<':'<<tensor.scalar_type()<<':'<<tensor.sizes();
  };
  key<<";images="<<input.images.size();
  for(const auto& image:input.images) {
    tensor_key(image.pixels);
    for(auto dimension:image.grid_thw)key<<':'<<dimension;
    key<<':'<<image.merge_size<<':'<<image.token_start<<':'<<image.token_count<<':'<<image.sequence_start;
  }
  key<<";audios="<<input.audios.size();
  for(const auto& audio:input.audios) {
    tensor_key(audio.features);tensor_key(audio.validity);tensor_key(audio.selected_frames);
    key<<':'<<audio.token_start<<':'<<audio.token_count;
  }
  tensor_key(input.multimodal_positions);
  auto id=key.str();
  auto found=impl->entries.find(id);
  if(found!=impl->entries.end()) {
    auto& entry=*found->second;
    impl->lru.erase(entry.lru);impl->lru.push_front(id);entry.lru=impl->lru.begin();
    entry.input.ids.copy_(input.ids);entry.input.types.copy_(input.types);
    entry.input.positions.copy_(input.positions);entry.input.cumulative.copy_(input.cumulative);
    for(size_t i=0;i<input.images.size();++i)entry.input.images[i].pixels.copy_(input.images[i].pixels);
    for(size_t i=0;i<input.audios.size();++i) {
      entry.input.audios[i].features.copy_(input.audios[i].features);
      entry.input.audios[i].validity.copy_(input.audios[i].validity);
      if(input.audios[i].selected_frames.defined())entry.input.audios[i].selected_frames.copy_(input.audios[i].selected_frames);
    }
    if(input.multimodal_positions.defined())entry.input.multimodal_positions.copy_(input.multimodal_positions);
    entry.graph->replay();
    return entry.output;
  }
  if(impl->failed.count(id))return model.forward(input);
  auto entry=std::make_unique<Impl::Entry>();
  entry->offsets.assign(input.offsets,input.offsets+input.batch+1);
  entry->input={input.ids.clone(),input.types.clone(),input.positions.clone(),input.cumulative.clone(),entry->offsets.data(),input.batch,input.max_sequence};
  for(auto image:input.images){image.pixels=image.pixels.clone();entry->input.images.push_back(std::move(image));}
  for(auto audio:input.audios) {
    audio.features=audio.features.clone();audio.validity=audio.validity.clone();
    if(audio.selected_frames.defined())audio.selected_frames=audio.selected_frames.clone();
    entry->input.audios.push_back(std::move(audio));
  }
  if(input.multimodal_positions.defined())entry->input.multimodal_positions=input.multimodal_positions.clone();
  auto current=c10::cuda::getCurrentCUDAStream(input.ids.device().index());
  auto capture_stream=c10::cuda::getStreamFromPool(false,input.ids.device().index());
  at::cuda::CUDAEvent input_ready;input_ready.record(current);input_ready.block(capture_stream);
  bool capturing=false;
  try {
    {
      c10::cuda::CUDAStreamGuard stream(capture_stream);
      // Initialize GEMM and attention handles before entering capture.
      for(int i=0;i<3;++i)entry->output=model.forward(entry->input);
      entry->graph=std::make_unique<at::cuda::CUDAGraph>();
      entry->graph->capture_begin({0,0},cudaStreamCaptureModeThreadLocal);capturing=true;
      entry->output=model.forward(entry->input);
      entry->graph->capture_end();capturing=false;
    }
    at::cuda::CUDAEvent warmed;warmed.record(capture_stream);warmed.block(current);
    entry->graph->replay();
  } catch(const std::exception&) {
    // Unsupported operators fall back to eager inference. Capture happens on a
    // dedicated stream so an invalidated capture never touches the caller stream.
    if(capturing) {
      try {c10::cuda::CUDAStreamGuard stream(capture_stream);entry->graph->capture_end();}catch(...){}
    }
    entry.reset();
    cudaGetLastError();
    // Bound failed shapes too; prevent an unbounded negative cache.
    if(impl->failed.size()>=impl->limit)impl->failed.clear();
    impl->failed.insert(id);
    return model.forward(input);
  }
  if(impl->entries.size()>=impl->limit) {
    impl->entries.erase(impl->lru.back());impl->lru.pop_back();
  }
  auto result=entry->output;
  impl->lru.push_front(id);entry->lru=impl->lru.begin();
  impl->entries.emplace(std::move(id),std::move(entry));
  return result;
#else
  return model.forward(original);
#endif
}
}

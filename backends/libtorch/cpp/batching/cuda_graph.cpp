// SPDX-License-Identifier: Apache-2.0
#include "batching/cuda_graph.h"
#include "batching/prefix_cache.h"
#include "batching/tensor_copy.h"
#include <string>
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/ops/_foreach_copy.h>
#include <list>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#endif
namespace tei {
struct CudaGraphCache::Impl {
  size_t limit;
  int64_t max_tokens;
  size_t reuse, max_captures, consecutive=0;
  uint64_t max_bytes;
  std::string last_shape;
  bool budget_exhausted=false;
  Stats counters;
#ifdef TEI_TORCH_CUDA_KERNELS
  struct Entry {
    std::vector<int32_t> offsets;
    PackedInput input;
    std::vector<int32_t> original_offsets;
    PackedInput original;
    std::unique_ptr<PrefixContext> prefix;
    // Resident snapshots are immutable. Remember the last copied owner without
    // extending its cache lifetime; address equality after weak locking cannot
    // confuse a retired owner with a new allocation at the same address.
    std::vector<std::weak_ptr<PrefixEntry>> parents;
    at::Tensor output;
    std::unique_ptr<at::cuda::CUDAGraph> graph;
    std::list<std::string>::iterator lru;
    uint64_t bytes=0;
    at::cuda::CUDAEvent consumed;
    bool recorded=false;
    ~Entry(){if(prefix&&recorded)try{consumed.synchronize();}catch(...) {}}
  };
  std::unordered_map<std::string, std::unique_ptr<Entry>> entries;
  std::list<std::string> lru;
  std::unordered_set<std::string> failed;
#endif
  Impl(size_t count,int64_t tokens,size_t uses,uint64_t bytes,size_t captures)
    :limit(count),max_tokens(tokens),reuse(uses),max_captures(captures),max_bytes(bytes) {
    TORCH_CHECK(uses>0 && bytes>0,"CUDA graph reuse and byte budgets must be positive");
  }
};
#ifdef TEI_TORCH_CUDA_KERNELS
static PackedInput clone_graph_input(const PackedInput& input,std::vector<int32_t>& offsets) {
  offsets.assign(input.offsets,input.offsets+input.batch+1);
  PackedInput owned{input.ids.clone(),input.types.clone(),input.positions.clone(),input.cumulative.clone(),
    offsets.data(),input.batch,input.max_sequence};
  for(auto image:input.images) {
    if(image.prepared_features.defined()){image.prepared_features=image.prepared_features.clone();image.pixels=at::Tensor();}
    else image.pixels=image.pixels.clone();
    owned.images.push_back(std::move(image));
  }
  for(auto audio:input.audios) {
    audio.features=audio.features.clone();audio.validity=audio.validity.clone();
    if(audio.selected_frames.defined())audio.selected_frames=audio.selected_frames.clone();
    owned.audios.push_back(std::move(audio));
  }
  if(input.multimodal_positions.defined())owned.multimodal_positions=input.multimodal_positions.clone();
  if(input.compact_ids.defined()) {
    owned.compact_ids=input.compact_ids.clone();owned.compact_positions=input.compact_positions.clone();owned.compact_types=input.compact_types.clone();
    owned.scatter_unfold=input.scatter_unfold.clone();owned.fold_gather=input.fold_gather.clone();
  }
  return owned;
}
static void graph_input_copies(const PackedInput& target,const PackedInput& source,
  std::vector<at::Tensor>& targets,std::vector<at::Tensor>& sources) {
  auto copy=[&](const at::Tensor& a,const at::Tensor& b){if(a.defined()){targets.push_back(a);sources.push_back(b);}};
  copy(target.ids,source.ids);copy(target.types,source.types);copy(target.positions,source.positions);copy(target.cumulative,source.cumulative);
  for(size_t i=0;i<source.images.size();++i) {
    if(target.images[i].prepared_features.defined())copy(target.images[i].prepared_features,source.images[i].prepared_features);
    else copy(target.images[i].pixels,source.images[i].pixels);
  }
  for(size_t i=0;i<source.audios.size();++i) {
    copy(target.audios[i].features,source.audios[i].features);copy(target.audios[i].validity,source.audios[i].validity);
    copy(target.audios[i].selected_frames,source.audios[i].selected_frames);
  }
  copy(target.multimodal_positions,source.multimodal_positions);
  copy(target.compact_ids,source.compact_ids);copy(target.compact_positions,source.compact_positions);copy(target.compact_types,source.compact_types);
  copy(target.scatter_unfold,source.scatter_unfold);copy(target.fold_gather,source.fold_gather);
}
#endif
CudaGraphCache::CudaGraphCache(size_t count,int64_t tokens,size_t uses,uint64_t bytes,size_t captures)
  :impl(std::make_unique<Impl>(count,tokens,uses,bytes,captures)){}
CudaGraphCache::~CudaGraphCache()=default;
CudaGraphCache::Stats CudaGraphCache::stats() const { return impl->counters; }
size_t CudaGraphCache::size() const {
#ifdef TEI_TORCH_CUDA_KERNELS
  return impl->entries.size();
#else
  return 0;
#endif
}
at::Tensor CudaGraphCache::forward(const Model& model,const PackedInput& original, int operation) {
  return forward(model,original,operation,nullptr);
}
at::Tensor CudaGraphCache::forward(const Model& model,const PackedInput& original,int operation,const MathPolicy* known_policy) {
  TORCH_CHECK(operation >= 0 && operation <= 2, "Invalid CUDA graph operation");
  auto run = [&](const PackedInput& input) {
    return operation == 0 ? model.forward(input) : model.predict(input, operation == 2);
  };
#ifdef TEI_TORCH_CUDA_KERNELS
  auto input=original;
  c10::cuda::OptionalCUDAGuard device;
  if(input.ids.is_cuda())device.set_device(input.ids.device());
  model.prepare_input(input);
  // Pool reservation accounting requires the native caching allocator.
  if(c10::cuda::CUDACachingAllocator::name()!="native")return run(input);
  if(!input.ids.is_cuda()||!impl->limit||input.ids.numel()>impl->max_tokens||input.ids.numel()==0)
    return run(input);
  if(input.prefix&&model.prefix_checkpoint_only())return run(input);
  // Include offsets, max length, device and scalar types: no sequence metadata
  // from one batch can leak into another graph with the same total token count.
  std::ostringstream key;
  const auto policy=known_policy?*known_policy:input.prefix?input.prefix->math_policy:execution_math_policy(input.ids.device());
  // Keys are process-local strings; append the fixed policy bytes once instead
  // of formatting every setting, including for full-request/miss graphs.
  key.write(reinterpret_cast<const char*>(policy.data()),sizeof(policy));
  key<<operation<<':'<<model.prefix_namespace()<<':'<<input.ids.device()<<':'<<input.ids.scalar_type()<<':'<<input.positions.scalar_type()<<':'<<input.types.scalar_type()<<':'<<input.cumulative.scalar_type()<<':'<<input.max_sequence<<':';
  for(int64_t i=0;i<=input.batch;++i)key<<input.offsets[i]<<',';
  auto tensor_key=[&](const at::Tensor& tensor,int64_t rows=-1) {
    key<<';'<<tensor.defined();
    if(tensor.defined()) {
      key<<':'<<tensor.device()<<':'<<tensor.scalar_type()<<':';
      if(rows<0)key<<tensor.sizes();
      else {
        TORCH_CHECK(tensor.dim()>0&&rows<=tensor.size(0),"Invalid graph prefix span");
        key<<'[';
        for(int64_t dim=0;dim<tensor.dim();++dim)
          key<<(dim?", ":"")<<(dim?tensor.size(dim):rows);
        key<<']';
      }
    }
  };
  auto describe_extras=[&](const PackedInput& value) {
  key<<";images="<<value.images.size();
  for(const auto& image:value.images) {
    tensor_key(image.prepared_features);
    if(!image.prepared_features.defined())tensor_key(image.pixels);
    for(auto dimension:image.grid_thw)key<<':'<<dimension;
    key<<':'<<image.merge_size<<':'<<image.token_start<<':'<<image.token_count<<':'<<image.sequence_start;
  }
  key<<";audios="<<value.audios.size();
  for(const auto& audio:value.audios) {
    tensor_key(audio.features);tensor_key(audio.validity);tensor_key(audio.selected_frames);
    key<<':'<<audio.token_start<<':'<<audio.token_count;
  }
  tensor_key(value.multimodal_positions);
  tensor_key(value.compact_ids);tensor_key(value.compact_positions);tensor_key(value.compact_types);
  tensor_key(value.scatter_unfold);tensor_key(value.fold_gather);
  };
  describe_extras(input);
  key<<";prefix="<<bool(input.prefix);
  if(input.prefix) {
    const auto& prefix=*input.prefix;
    key<<':'<<prefix.original.max_sequence;
    for(int64_t row=0;row<=prefix.original.batch;++row)key<<':'<<prefix.original.offsets[row];
    for(auto hit:prefix.hits)key<<':'<<hit;
    tensor_key(prefix.original.ids);tensor_key(prefix.original.multimodal_positions);
    tensor_key(prefix.original.compact_ids);tensor_key(prefix.original.scatter_unfold);
    describe_extras(prefix.original);
    for(size_t row=0;row<prefix.readers.size();++row)if(prefix.hits[row]) {
      tensor_key(prefix.readers[row]->output,prefix.hits[row]);
      for(const auto& [layer,state]:prefix.readers[row]->layers) {
        key<<':'<<layer;tensor_key(state.key,prefix.hits[row]);tensor_key(state.value,prefix.hits[row]);
      }
    }
  }
  auto id=key.str();
  if(id==impl->last_shape) {
    if(impl->consecutive<impl->reuse)++impl->consecutive;
  } else {impl->last_shape=id;impl->consecutive=1;}
  auto found=impl->entries.find(id);
  if(found!=impl->entries.end()) {
    ++impl->counters.hits;
    auto& entry=*found->second;
    impl->lru.erase(entry.lru);impl->lru.push_front(id);entry.lru=impl->lru.begin();
    std::vector<at::Tensor> targets,sources;graph_input_copies(entry.input,input,targets,sources);
    if(entry.prefix) {
      graph_input_copies(entry.original,input.prefix->original,targets,sources);
      for(size_t row=0;row<entry.prefix->readers.size();++row)if(entry.prefix->hits[row]) {
        const auto& actual=input.prefix->readers[row];const auto& owned=entry.prefix->readers[row];
        if(entry.parents[row].lock()==actual)continue;
        entry.parents[row].reset();
        targets.push_back(owned->output);sources.push_back(actual->output.narrow(0,0,entry.prefix->hits[row]));
        for(const auto& [layer,state]:owned->layers) {
          const auto& fresh=actual->layers.at(layer);
          targets.push_back(state.key);sources.push_back(fresh.key.narrow(0,0,entry.prefix->hits[row]));
          targets.push_back(state.value);sources.push_back(fresh.value.narrow(0,0,entry.prefix->hits[row]));
        }
      }
    }
    tensor_copies(targets,sources);
    if(entry.prefix)for(size_t row=0;row<entry.parents.size();++row)
      entry.parents[row]=input.prefix->readers[row];
    entry.graph->replay();
    if(entry.prefix) {
      ++impl->counters.prefix_hits;input.prefix->publish_graph(*entry.prefix);
      auto output=entry.output.clone();entry.consumed.record(c10::cuda::getCurrentCUDAStream(input.ids.get_device()));entry.recorded=true;return output;
    }
    return entry.output;
  }
  ++impl->counters.misses;
  // Bound private-pool creation, including failed captures. Alternating cold
  // shapes never pass consecutive-use admission.
  if(impl->budget_exhausted || impl->counters.attempts>=impl->max_captures
    || impl->consecutive<impl->reuse || impl->failed.count(id))return run(input);
  if(input.prefix) {
    // Reject a known oversized prefix arena before cloning inputs or starting
    // capture. Actual private-pool reservation accounting still follows below.
    auto minimum=model.prefix_payload_bytes(input.prefix->original.ids.numel());
    for(size_t row=0;row<input.prefix->readers.size();++row)if(input.prefix->hits[row]) {
      const auto& reader=input.prefix->readers[row];const auto hits=input.prefix->hits[row];
      minimum=prefix_storage_sum(minimum,reader->output.narrow(0,0,hits).nbytes());
      for(const auto& [layer,state]:reader->layers) {
        minimum=prefix_storage_sum(minimum,state.key.narrow(0,0,hits).nbytes());
        minimum=prefix_storage_sum(minimum,state.value.narrow(0,0,hits).nbytes());
      }
    }
    if(minimum>impl->max_bytes-impl->counters.admitted_bytes)return run(input);
  }
  ++impl->counters.attempts;
  auto entry=std::make_unique<Impl::Entry>();
  entry->input=clone_graph_input(input,entry->offsets);
  if(input.prefix) {
    const auto& original=*input.prefix;
    entry->original=clone_graph_input(original.original,entry->original_offsets);
    entry->prefix=std::make_unique<PrefixContext>(PrefixContext{entry->original,original.offsets,original.hits,{},
      std::vector<std::shared_ptr<PrefixEntry>>(original.original.batch),original.max_bytes,std::make_shared<PrefixBudget>()});
    entry->prefix->linear=original.linear;entry->prefix->math_policy=original.math_policy;entry->prefix->collect_graph=true;
    entry->prefix->graph_layers.resize(original.original.batch);entry->prefix->graph_outputs.resize(original.original.batch);
    for(size_t row=0;row<original.readers.size();++row) {
      std::shared_ptr<PrefixEntry> owned;
      if(original.hits[row]) {
        owned=std::make_shared<PrefixEntry>();const auto& source=original.readers[row];
        owned->output=source->output.narrow(0,0,original.hits[row]).clone();
        for(const auto& [layer,state]:source->layers)owned->layers[layer]=PrefixLayer{
          state.key.narrow(0,0,original.hits[row]).clone(),state.value.narrow(0,0,original.hits[row]).clone(),{}, {}};
      }
      entry->prefix->readers.push_back(std::move(owned));
      entry->parents.push_back(original.readers[row]);
    }
    entry->input.prefix=entry->prefix.get();
  }
  auto current=c10::cuda::getCurrentCUDAStream(input.ids.device().index());
  auto capture_stream=c10::cuda::getStreamFromPool(false,input.ids.device().index());
  at::cuda::CUDAEvent input_ready;input_ready.record(current);input_ready.block(capture_stream);
  bool capturing=false;
  try {
    {
      c10::cuda::CUDAStreamGuard stream(capture_stream);
      // Initialize GEMM and attention handles before entering capture.
      for(int i=0;i<3;++i)entry->output=run(entry->input);
      if(entry->prefix) {
        // Warmups allocate full K/V arenas. Bind verified parent views directly
        // into them so replay writes only suffix rows, without concatenation.
        for(size_t row=0;row<entry->prefix->readers.size();++row)if(entry->prefix->hits[row]) {
          auto& reader=entry->prefix->readers[row];
          for(auto& [layer,state]:reader->layers) {
            c10::cuda::CUDACachingAllocator::recordStream(state.key.storage().data_ptr(),capture_stream);
            c10::cuda::CUDACachingAllocator::recordStream(state.value.storage().data_ptr(),capture_stream);
            const auto& full=entry->prefix->graph_layers[row].at(layer);
            state.key=full.key.narrow(0,0,entry->prefix->hits[row]);
            state.value=full.value.narrow(0,0,entry->prefix->hits[row]);
          }
        }
        for(auto& output:entry->prefix->graph_outputs)output=at::Tensor();
      }
      entry->graph=std::make_unique<at::cuda::CUDAGraph>();
      entry->graph->capture_begin({0,0},cudaStreamCaptureModeThreadLocal);capturing=true;
      entry->output=run(entry->input);
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
    // Eager warmups may still use entry storage on the dedicated stream even
    // when capture fails. Complete them before releasing that storage/pool.
    capture_stream.synchronize();
    entry.reset();
    cudaGetLastError();
    // The lifetime attempt budget also bounds this negative cache.
    impl->failed.insert(id);
    return run(input);
  }
  // Charge reserved private-pool segments and static inputs separately.
  // Cumulative admission charges survive eviction: reservation can outlive graphs.
  for(const auto& segment:c10::cuda::CUDACachingAllocator::snapshot(entry->graph->pool(),false).segments)
    if(segment.device==input.ids.get_device() && segment.owner_private_pool_id==entry->graph->pool())entry->bytes+=segment.total_size;
  auto charge=[&](const at::Tensor& t){if(t.defined() && t.is_cuda())entry->bytes+=t.nbytes();};
  charge(entry->input.ids);charge(entry->input.types);charge(entry->input.positions);charge(entry->input.cumulative);
  for(const auto& image:entry->input.images){charge(image.prepared_features);charge(image.pixels);}
  for(const auto& audio:entry->input.audios){charge(audio.features);charge(audio.validity);charge(audio.selected_frames);}
  charge(entry->input.multimodal_positions);charge(entry->input.compact_ids);charge(entry->input.compact_positions);
  charge(entry->input.compact_types);charge(entry->input.scatter_unfold);charge(entry->input.fold_gather);
  if(entry->prefix) {
    charge(entry->original.ids);charge(entry->original.types);charge(entry->original.positions);charge(entry->original.cumulative);
    charge(entry->original.multimodal_positions);charge(entry->original.compact_ids);charge(entry->original.compact_positions);
    charge(entry->original.compact_types);charge(entry->original.scatter_unfold);charge(entry->original.fold_gather);
    for(const auto& image:entry->original.images){charge(image.prepared_features);charge(image.pixels);}
    for(const auto& audio:entry->original.audios){charge(audio.features);charge(audio.validity);charge(audio.selected_frames);}
    for(const auto& reader:entry->prefix->readers)if(reader) {
      charge(reader->output);
    }
    for(const auto& row:entry->prefix->graph_layers)
      for(const auto& [layer,state]:row){charge(state.key);charge(state.value);}
  }
  if(entry->bytes>impl->max_bytes-impl->counters.admitted_bytes) {
    // Allow only one oversized trial; drain replay before pool destruction.
    current.synchronize();impl->budget_exhausted=true;entry.reset();return run(input);
  }
  if(impl->entries.size()>=impl->limit) {
    // Output copies/leases are ordered on the caller stream. Drain its work
    // before destroying an entry and its private pool.
    current.synchronize();
    impl->counters.live_bytes-=impl->entries.at(impl->lru.back())->bytes;
    impl->entries.erase(impl->lru.back());impl->lru.pop_back();++impl->counters.evictions;
  }
  ++impl->counters.captures;impl->counters.live_bytes+=entry->bytes;
  impl->counters.admitted_bytes+=entry->bytes;
  auto result=entry->output;
  if(entry->prefix) {
    ++impl->counters.prefix_captures;input.prefix->publish_graph(*entry->prefix);result=result.clone();
    entry->consumed.record(current);entry->recorded=true;
  }
  impl->lru.push_front(id);entry->lru=impl->lru.begin();
  impl->entries.emplace(std::move(id),std::move(entry));
  return result;
#else
  auto input=original;model.prepare_input(input);return run(input);
#endif
}
}

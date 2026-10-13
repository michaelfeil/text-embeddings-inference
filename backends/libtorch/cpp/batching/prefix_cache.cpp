#include "batching/prefix_cache.h"
#include "batching/radix.h"
#include "models/model.h"
#include "helpers/prefix_linear.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <ATen/ops/_flash_attention_forward.h>
#include <ATen/ops/_efficient_attention_forward.h>
#include <ATen/ops/_cudnn_attention_forward.h>
#ifdef TEI_TORCH_CUDA_KERNELS
#include "kernels/torch_flash_fma.h"
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace tei {
PrefixEntry::~PrefixEntry() {
  if(budget) {budget->live_bytes-=bytes;budget->reserved_bytes-=reserved_bytes;}
}
const PackedInput& prefix_logical_input(const PackedInput& input) {
  return input.prefix?input.prefix->original:input;
}
at::Tensor prefix_restore(const PackedInput& input,const at::Tensor& hidden) {
  return input.prefix?input.prefix->restore(hidden):hidden;
}
at::Tensor prefix_linear(const at::Tensor& input,const at::Tensor& weight,const at::Tensor& bias,const PackedInput& packed) {
  if(packed.prefix&&packed.prefix->linear)
    return packed.prefix->linear->forward(input,weight,bias,radix_ids(packed.prefix->original).numel());
  return at::linear(input,weight,bias);
}
const PrefixLayer* PrefixContext::previous(int64_t row,int64_t layer) const {
  if(!readers[row])return nullptr;
  auto found=readers[row]->layers.find(layer);
  return found==readers[row]->layers.end()?nullptr:&found->second;
}
bool PrefixContext::store(int64_t row,int64_t layer,const PrefixLayer& source) {
  auto& entry=writers[row];if(!entry)return false;
  // Charge the map node as well as tensor payloads. Views are cloned below so
  // an admitted prefix cannot keep a larger batch allocation alive.
  uint64_t needed=sizeof(std::pair<const int64_t,PrefixLayer>)+4*sizeof(void*);
  for(const auto& t:{source.key,source.value,source.recurrent,source.convolution})
    if(t.defined())needed+=t.numel()*t.element_size();
  if(needed>entry->reserved_bytes) { entry.reset();return false; }
  PrefixLayer owned;
  auto clone=[](const at::Tensor& t){return t.defined()?t.clone():at::Tensor();};
  owned.key=clone(source.key);owned.value=clone(source.value);
  owned.recurrent=clone(source.recurrent);owned.convolution=clone(source.convolution);
  TORCH_CHECK(!entry->layers.count(layer),"Prefix layer captured more than once");
  entry->layers.emplace(layer,std::move(owned));entry->bytes+=needed;budget->live_bytes+=needed;
  entry->reserved_bytes-=needed;budget->reserved_bytes-=needed;
  return true;
}
PrefixContext::KV PrefixContext::attention(int64_t layer,const at::Tensor& key,const at::Tensor& value,bool capture) {
  std::vector<at::Tensor> keys,values;
  for(int64_t row=0;row<original.batch;++row) {
    int64_t length=offsets[row+1]-offsets[row],full=length+hits[row];
    auto k=key.narrow(0,offsets[row],length),v=value.narrow(0,offsets[row],length);
    if(hits[row]) {
      auto old=previous(row,layer);
      TORCH_CHECK(old&&old->key.defined()&&old->key.size(0)>=hits[row],"Missing prefix attention state");
      k=at::cat({old->key.narrow(0,0,hits[row]),k},0);
      v=at::cat({old->value.narrow(0,0,hits[row]),v},0);
    }
    if(capture&&writers[row]) {
      auto saved=int64_t(writers[row]->identity.tokens.size());
      store(row,layer,PrefixLayer{k.narrow(0,0,saved),v.narrow(0,0,saved),{}, {}});
    }
    TORCH_CHECK(full==original.offsets[row+1]-original.offsets[row],"Cached keys must preserve the full logical context");
    keys.push_back(k);values.push_back(v);
  }
  // Full K/V boundaries are unchanged by suffix execution. Reuse the existing
  // device metadata instead of synchronizing a host upload at every layer.
  // A single sequence also needs no second copy after prefix/suffix assembly.
  return {keys.size()==1?keys[0].contiguous():at::cat(keys,0),
    values.size()==1?values[0].contiguous():at::cat(values,0),original.cumulative,original.max_sequence};
}
at::Tensor prefix_attention(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
  const PackedInput& input,int64_t layer,double scale,int64_t left,int64_t right,
  bool exact,bool capture,bool cudnn) {
  auto kv=input.prefix->attention(layer,k,v,capture);
  if(q.is_cuda()) {
    if(cudnn) {
      // Keep the original packed cuDNN geometry and causal provider. An
      // explicit dense suffix bias changes cuDNN's reduction recipe enough
      // to alter learned logits. Prefix queries are discarded, so zero rows
      // can stand in for them without retaining or projecting cached Q.
      // This replays attention rows, while QKV/MLP still process only suffixes.
      const auto& original=input.prefix->original;
      auto shape=q.sizes().vec();shape[0]=kv.key.size(0);
      auto queries=at::zeros(shape,q.options());
      for(int64_t row=0;row<input.batch;++row) {
        auto count=input.offsets[row+1]-input.offsets[row];
        queries.narrow(0,original.offsets[row]+input.prefix->hits[row],count)
          .copy_(q.narrow(0,input.offsets[row],count));
      }
      auto full=std::get<0>(at::_cudnn_attention_forward(queries,kv.key,kv.value,std::nullopt,
        original.cumulative,original.cumulative,original.max_sequence,original.max_sequence,
        true,0.,true,false,scale));
      std::vector<at::Tensor> outputs;
      for(int64_t row=0;row<input.batch;++row) {
        auto count=input.offsets[row+1]-input.offsets[row];
        outputs.push_back(full.narrow(0,original.offsets[row]+input.prefix->hits[row],count));
      }
      return outputs.size()==1?outputs[0]:at::cat(outputs,0);
    }
#ifdef TEI_TORCH_CUDA_KERNELS
    if(!exact&&input.max_sequence==1&&left<0&&right<0&&(q.size(2)==64||q.size(2)==128)
      &&torch_flash_fma_supported(q,kv.key,kv.value,true,left,right))
      return torch_flash_unfused_varlen(q,kv.key,kv.value,input.cumulative,kv.cumulative,
        input.max_sequence,kv.max_length,scale);
    if(exact&&torch_flash_fma_supported(q,kv.key,kv.value,true,left,right))
      return torch_flash_fma_varlen(q,kv.key,kv.value,input.cumulative,kv.cumulative,
        input.max_sequence,kv.max_length,scale,true,left,right);
#endif
    if(q.size(2)<=256)return std::get<0>(at::_flash_attention_forward(q,kv.key,kv.value,
      input.cumulative,kv.cumulative,input.max_sequence,kv.max_length,0.,true,false,scale,left,right));
    TORCH_CHECK(left<0,"Cached wide-head local attention requires a qualified kernel");
    auto heads=q.size(1)/kv.key.size(1);
    auto keys=heads==1?kv.key:kv.key.repeat_interleave(heads,1);
    auto values=heads==1?kv.value:kv.value.repeat_interleave(heads,1);
    // The pinned efficient-attention enum uses 2 for bottom-right causality.
    return std::get<0>(at::_efficient_attention_forward(q.unsqueeze(0),keys.unsqueeze(0),values.unsqueeze(0),
      std::nullopt,input.cumulative,kv.cumulative,input.max_sequence,kv.max_length,0.,2,false,scale)).squeeze(0);
  }
  std::vector<at::Tensor> outputs;
  for(int64_t row=0;row<input.batch;++row) {
    auto nq=input.offsets[row+1]-input.offsets[row];
    auto nk=input.prefix->original.offsets[row+1]-input.prefix->original.offsets[row];
    auto qs=q.narrow(0,input.offsets[row],nq).transpose(0,1).unsqueeze(0);
    auto ks=kv.key.narrow(0,input.prefix->original.offsets[row],nk).transpose(0,1).unsqueeze(0);
    auto vs=kv.value.narrow(0,input.prefix->original.offsets[row],nk).transpose(0,1).unsqueeze(0);
    auto repeat=q.size(1)/kv.key.size(1);
    if(repeat!=1){ks=ks.repeat_interleave(repeat,1);vs=vs.repeat_interleave(repeat,1);}
    auto qi=at::arange(nq,qs.options().dtype(at::kLong))+input.prefix->hits[row];
    auto ki=at::arange(nk,qs.options().dtype(at::kLong));auto delta=qi.unsqueeze(1)-ki.unsqueeze(0);
    auto mask=delta>=0;
    if(left>=0)mask=mask&(delta<=left);
    if(right>=0)mask=mask&(delta>=-right);
    outputs.push_back(at::scaled_dot_product_attention(qs,ks,vs,mask,0.,false,scale).squeeze(0).transpose(0,1));
  }
  return at::cat(outputs,0);
}
void PrefixContext::checkpoint(int64_t row,int64_t layer,const at::Tensor& state,const at::Tensor& history) {
  store(row,layer,PrefixLayer{{},{},state,history});
}
at::Tensor PrefixContext::restore(const at::Tensor& suffix) {
  TORCH_CHECK(suffix.dim()==2&&suffix.size(0)==offsets.back(),"Invalid cached suffix output");
  std::vector<at::Tensor> rows;
  for(int64_t row=0;row<original.batch;++row) {
    auto value=suffix.narrow(0,offsets[row],offsets[row+1]-offsets[row]);
    if(hits[row])value=at::cat({readers[row]->output.narrow(0,0,hits[row]),value},0);
    auto& entry=writers[row];
    if(entry) {
      auto retained=value.narrow(0,0,entry->identity.tokens.size());
      auto bytes=retained.numel()*retained.element_size();
      if(bytes<=entry->reserved_bytes) {
        entry->output=retained.clone();entry->bytes+=bytes;budget->live_bytes+=bytes;
        entry->reserved_bytes-=bytes;budget->reserved_bytes-=bytes;
      } else entry.reset();
    }
    rows.push_back(value);
  }
  return rows.size()==1?rows[0]:at::cat(rows,0);
}
PrefixRecurrent prefix_recurrent(const PackedInput& input,const at::Tensor& qkv,
  int64_t layer,int64_t heads,int64_t kernel) {
  PrefixRecurrent buffers;if(!input.prefix)return buffers;
  std::vector<int64_t> sources(input.batch,0);
  std::vector<int32_t> lengths(input.batch,0),positions(input.batch,-1),destinations(input.batch,-1);
  buffers.history=at::zeros({input.batch,kernel-1,qkv.size(1)},qkv.options());
  for(int64_t row=0;row<input.batch;++row) {
    auto old=input.prefix->previous(row,layer);
    if(input.prefix->hits[row]) {
      TORCH_CHECK(old&&old->recurrent.defined()&&old->convolution.defined(),"Missing cached DeltaNet state/history");
      TORCH_CHECK(old->recurrent.scalar_type()==at::kFloat&&old->recurrent.is_contiguous()
        &&old->recurrent.sizes()==at::IntArrayRef({heads,128,128}),"Invalid DeltaNet cached state geometry");
      sources[row]=reinterpret_cast<int64_t>(old->recurrent.const_data_ptr<float>());
      lengths[row]=old->convolution.size(0);
      TORCH_CHECK(lengths[row]<=kernel-1,"Cached convolution history exceeds kernel width");
      if(lengths[row])buffers.history[row].narrow(0,kernel-1-lengths[row],lengths[row]).copy_(old->convolution);
    }
    if(input.prefix->writers[row]) {
      positions[row]=int64_t(input.prefix->writers[row]->identity.tokens.size())-input.prefix->hits[row];
      TORCH_CHECK(positions[row]>0&&positions[row]<=input.offsets[row+1]-input.offsets[row],"Invalid DeltaNet checkpoint boundary");
      destinations[row]=buffers.rows.size();buffers.rows.push_back(row);
    }
  }
  buffers.initial_sources=at::tensor(sources,at::TensorOptions().dtype(at::kLong)).to(qkv.device());
  buffers.history_lengths=at::tensor(lengths,at::TensorOptions().dtype(at::kInt)).to(qkv.device());
  buffers.checkpoint_positions=at::tensor(positions,at::TensorOptions().dtype(at::kInt)).to(qkv.device());
  buffers.checkpoint_rows=at::tensor(destinations,at::TensorOptions().dtype(at::kInt)).to(qkv.device());
  if(!buffers.rows.empty())buffers.checkpoints=at::empty({int64_t(buffers.rows.size()),heads,128,128},qkv.options().dtype(at::kFloat));
  return buffers;
}
void prefix_recurrent_commit(const PackedInput& input,const at::Tensor& qkv,
  int64_t layer,int64_t kernel,const PrefixRecurrent& buffers) {
  for(size_t slot=0;slot<buffers.rows.size();++slot) {
    auto row=buffers.rows[slot];if(!input.prefix->writers[row])continue;
    auto count=int64_t(input.prefix->writers[row]->identity.tokens.size())-input.prefix->hits[row];
    auto history=qkv.narrow(0,input.offsets[row],count);
    auto previous=input.prefix->previous(row,layer);
    if(previous&&previous->convolution.defined())history=at::cat({previous->convolution,history},0);
    history=history.narrow(0,std::max<int64_t>(0,history.size(0)-(kernel-1)),std::min<int64_t>(history.size(0),kernel-1));
    input.prefix->checkpoint(row,layer,buffers.checkpoints[slot],history);
  }
}
PrefixCache::PrefixCache(uint64_t bytes,int64_t max_prefix,size_t count)
  :limit(bytes),maximum(max_prefix),capacity(count) {
  TORCH_CHECK(maximum>0&&capacity>0,"Invalid prefix cache bounds");
  linear=std::make_unique<PrefixLinearPlans>();
}
PrefixCache::~PrefixCache()=default;
void PrefixCache::clear() {std::lock_guard guard(mutex);entries.clear();linear->clear();}
PrefixCache::Stats PrefixCache::stats() const {std::lock_guard guard(mutex);return counters;}
uint64_t PrefixCache::bytes() const {std::lock_guard guard(mutex);return budget->live_bytes;}
static PrefixIdentity identity(const PackedInput& input,int64_t row) {
  const auto& ids=input.ids;const auto& positions=input.positions;const auto& types=input.types;
  const auto& axes=input.multimodal_positions;
  for(const auto& values:{ids,positions,types})TORCH_CHECK(values.device().is_cpu()
    &&values.scalar_type()==at::kLong&&values.is_contiguous()&&values.dim()==1
    &&values.numel()==ids.numel(),"Invalid host prefix identity metadata");
  if(axes.defined())TORCH_CHECK(axes.device().is_cpu()&&axes.scalar_type()==at::kLong
    &&axes.is_contiguous()&&axes.sizes()==at::IntArrayRef({3,ids.numel()}),"Invalid host prefix rotary axes");
  auto token_ids=ids.const_data_ptr<int64_t>(),token_positions=positions.const_data_ptr<int64_t>(),token_types=types.const_data_ptr<int64_t>();
  const auto* rotary=axes.defined()?axes.const_data_ptr<int64_t>():nullptr;
  auto start=input.offsets[row],end=input.offsets[row+1];PrefixIdentity key;
  // Image processing can select a residual path for every row in a batch,
  // including text-only rows. Keep that execution context in their identity.
  key.image_context=!input.images.empty();
  key.tokens.reserve(end-start);
  for(int64_t t=start;t<end;++t) {
    PrefixToken token;token.values={token_ids[t],token_positions[t],token_types[t],0,0,0};
    if(rotary)for(int i=0;i<3;++i)token.values[3+i]=rotary[i*ids.numel()+t];
    key.tokens.push_back(token);
  }
  for(const auto& image:input.images)if(image.token_start>=start&&image.token_start<end) {
    PrefixMedia media;media.start=image.token_start-start;media.count=image.token_count;
    media.layout={image.grid_thw[0],image.grid_thw[1],image.grid_thw[2],image.merge_size};
    media.layout.push_back(int64_t(image.pixels.scalar_type()));
    for(auto size:image.pixels.sizes())media.layout.push_back(size);
    media.hashed=image.has_feature_key;
    if(media.hashed)media.digest=image.feature_key;
    else {
      auto pixels=image.pixels.cpu().contiguous();
      auto bytes=pixels.numel()*pixels.element_size();media.content.resize(bytes);
      std::memcpy(media.content.data(),pixels.const_data_ptr(),bytes);
    }
    key.media.push_back(std::move(media));
  }
  return key;
}
static int64_t common(const PrefixIdentity& a,const PrefixIdentity& b,int64_t limit) {
  if(a.image_context!=b.image_context)return 0;
  int64_t n=0;while(n<limit&&n<int64_t(a.tokens.size())&&n<int64_t(b.tokens.size())&&a.tokens[n]==b.tokens[n])++n;
  for(const auto* key:{&a,&b})for(const auto& media:key->media)if(media.start<n) {
    auto& other=key==&a?b:a;
    auto match=std::find(other.media.begin(),other.media.end(),media);
    if(match==other.media.end()||media.start+media.count>n)n=std::min(n,media.start);
  }
  return n;
}
at::Tensor PrefixCache::run(const Model& model,const PackedInput& original,int operation) {
  std::lock_guard guard(mutex);
  auto run=[&](const PackedInput& input){return operation?model.predict(input,operation==2):model.forward(input);};
  if(!limit||!model.supports_prefix_cache(operation)||!original.audios.empty()) {++counters.bypasses;return run(original);}
  try {
  const auto& global=at::globalContext();
  decltype(math_policy) policy={int64_t(global.allowTF32CuBLAS()),int64_t(global.allowFP16ReductionCuBLAS()),
    int64_t(global.allowBF16ReductionCuBLAS()),int64_t(global.allowFP16AccumulationCuBLAS()),
    int64_t(global.float32MatmulPrecision()),int64_t(at::globalContext().blasPreferredBackend()),
    int64_t(global.deterministicAlgorithms()),int64_t(global.allowTF32CuDNN()),int64_t(global.allowFP16ReductionCPU())};
#ifdef TEI_TORCH_CUDA_KERNELS
  if(original.ids.is_cuda()) {
    policy[9]=global._SMCarveout_EXPERIMENTAL().value_or(-1);
    policy[10]=at::cuda::getChosenWorkspaceSize();policy[11]=at::cuda::getCUDABlasLtWorkspaceSize();
  }
#endif
  policy[12]=global.userEnabledFlashSDP();policy[13]=global.userEnabledFA3SDP();
  policy[14]=global.userEnabledMemEfficientSDP();policy[15]=global.userEnabledMathSDP();
  policy[16]=global.userEnabledCuDNNSDP();policy[17]=global.allowFP16BF16ReductionMathSDP();
  policy[18]=global.userEnabledOverrideableSDP();
  auto priority=at::globalContext().sDPPriorityOrder();
  for(size_t i=0;i<priority.size();++i)policy[19+i]=int64_t(priority[i]);
  // Global deterministicAlgorithms does not cover cuDNN's algorithm controls.
  constexpr auto cudnn_policy=19+at::num_sdp_backends;
  policy[cudnn_policy]=global.deterministicCuDNN();
  policy[cudnn_policy+1]=global.benchmarkCuDNN();
  policy[cudnn_policy+2]=global.benchmarkLimitCuDNN();
  if(model_epoch!=model.prefix_namespace()||math_policy!=policy) {
    entries.clear();linear->clear();model_epoch=model.prefix_namespace();math_policy=policy;
  }
  TORCH_CHECK(original.batch>0&&original.offsets&&original.offsets[0]==0
    &&original.offsets[original.batch]==original.ids.numel(),"Invalid full prefix-cache input boundaries");
  for(int64_t row=0;row<original.batch;++row)
    TORCH_CHECK(original.offsets[row+1]>original.offsets[row]
      &&original.offsets[row+1]-original.offsets[row]<=original.max_sequence,"Invalid full prefix-cache sequence length");
  auto host=original;
  host.ids=original.host_ids.defined()?original.host_ids:original.ids.cpu();
  host.types=original.host_types.defined()?original.host_types:original.types.cpu();
  host.positions=original.host_positions.defined()?original.host_positions:original.positions.cpu();
  if(original.multimodal_positions.defined())host.multimodal_positions=original.host_multimodal_positions.defined()
    ?original.host_multimodal_positions:original.multimodal_positions.cpu();
  PrefixContext context{original,{}, {}, {}, {},limit,budget};context.offsets={0};
  context.linear=linear.get();
  std::vector<at::Tensor> ids,positions,types,axes;std::vector<int64_t> original_rows;
  for(int64_t row=0;row<original.batch;++row) {
    auto key=identity(host,row);int64_t best=0;std::shared_ptr<PrefixEntry> source;
    for(const auto& entry:entries)if(entry->operation==operation&&entry->ready()) {
      auto n=common(key,entry->identity,std::min<int64_t>(maximum,key.tokens.size()-1));
      if(model.prefix_checkpoint_only()&&n!=int64_t(entry->identity.tokens.size()))continue;
      if(n>best){best=n;source=entry;}
    }
    if(best){++counters.hits;counters.reused_tokens+=best;source->used=++clock;}else ++counters.misses;
#ifdef TEI_TORCH_CUDA_KERNELS
    if(source&&original.ids.is_cuda()) {
      auto stream=at::cuda::getCurrentCUDAStream(original.ids.get_device());
      auto retain=[&](const at::Tensor& tensor){if(tensor.defined())c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(),stream);};
      retain(source->output);
      for(const auto& [id,state]:source->layers)for(const auto& tensor:{state.key,state.value,state.recurrent,state.convolution})retain(tensor);
    }
#endif
    context.hits.push_back(best);context.readers.push_back(source);
    auto start=original.offsets[row]+best,length=original.offsets[row+1]-start;
    ids.push_back(original.ids.narrow(0,start,length));positions.push_back(original.positions.narrow(0,start,length));types.push_back(original.types.narrow(0,start,length));
    if(original.multimodal_positions.defined())axes.push_back(original.multimodal_positions.narrow(1,start,length));
    context.offsets.push_back(context.offsets.back()+length);
    for(int64_t t=start;t<original.offsets[row+1];++t)original_rows.push_back(t);
    std::shared_ptr<PrefixEntry> writer;
    auto retained=std::min<int64_t>(maximum,key.tokens.size());
    if(model.prefix_checkpoint_only()) {
      retained=std::min<int64_t>(maximum,key.tokens.size()-1);
      if(retained>=64)retained=retained/64*64;
    }
    for(const auto& media:key.media)if(media.start<retained&&media.start+media.count>retained)retained=media.start;
    key.tokens.resize(retained);
    key.tokens.shrink_to_fit();
    key.media.erase(std::remove_if(key.media.begin(),key.media.end(),[&](const auto& media){return media.start>=retained;}),key.media.end());
    key.media.shrink_to_fit();
    bool resident_exact=std::any_of(entries.begin(),entries.end(),[&](const auto& entry){return entry->operation==operation&&entry->identity.tokens.size()==key.tokens.size()&&common(key,entry->identity,key.tokens.size())==int64_t(key.tokens.size());});
    uint64_t metadata=sizeof(PrefixEntry)+key.tokens.capacity()*sizeof(PrefixToken)+key.media.capacity()*sizeof(PrefixMedia);
    for(const auto& media:key.media)metadata+=media.layout.capacity()*sizeof(int64_t)+media.content.capacity();
    if(retained>best&&!resident_exact) {
      writer=std::make_shared<PrefixEntry>();writer->identity=std::move(key);writer->operation=operation;writer->used=++clock;
      writer->bytes=metadata;
    }
    context.writers.push_back(std::move(writer));
  }
  // Look up every row before evicting, then reserve each complete snapshot.
  // Oversized recurrent checkpoints never allocate capture buffers. Reader
  // payloads remain charged even if their lookup slot is evicted during a run.
  size_t admitting=0;
  for(auto& writer:context.writers)if(writer) {
    const auto payload=model.prefix_payload_bytes(writer->identity.tokens.size());
    const auto nodes=prefix_storage_bytes(model.prefix_layers(),sizeof(std::pair<const int64_t,PrefixLayer>)+4*sizeof(void*),1);
    const auto reservation=prefix_storage_sum(payload,nodes);
    const auto needed=prefix_storage_sum(writer->bytes,reservation);
    if(!payload||needed>limit) {writer.reset();continue;}
    auto fits=[&]{return budget->live_bytes<=limit&&budget->reserved_bytes<=limit-budget->live_bytes
      &&needed<=limit-budget->live_bytes-budget->reserved_bytes;};
    while(!entries.empty()&&(!fits()||entries.size()+admitting>=capacity)) {
      auto victim=entries.end();
      for(auto it=entries.begin();it!=entries.end();++it)
        if(it->use_count()==1&&(victim==entries.end()||(*it)->used<(*victim)->used))victim=it;
      if(victim==entries.end()) {
        if(!fits())break;
        victim=std::min_element(entries.begin(),entries.end(),[](const auto& a,const auto& b){return a->used<b->used;});
      }
      entries.erase(victim);++counters.evictions;
    }
    if(!fits()||entries.size()+admitting>=capacity) {writer.reset();continue;}
    writer->budget=budget;writer->reserved_bytes=reservation;
    budget->live_bytes+=writer->bytes;budget->reserved_bytes+=reservation;++admitting;
  }
  if(!admitting&&std::none_of(context.hits.begin(),context.hits.end(),[](auto hit){return hit!=0;})) {
    ++counters.bypasses;return run(original);
  }
  PackedInput input=original;
  // Cold capture keeps the caller's original metadata and device buffers.
  // Only a real hit needs packed suffix rows and a new Radix mapping.
  if(std::any_of(context.hits.begin(),context.hits.end(),[](auto hit){return hit!=0;})) {
    input.ids=ids.size()==1?ids[0]:at::cat(ids);
    input.positions=positions.size()==1?positions[0]:at::cat(positions);
    input.types=types.size()==1?types[0]:at::cat(types);
    input.cumulative=at::tensor(context.offsets,at::TensorOptions().dtype(at::kInt)).to(input.ids.device());input.offsets=context.offsets.data();input.max_sequence=0;
    for(int64_t row=0;row<original.batch;++row)input.max_sequence=std::max<int64_t>(input.max_sequence,context.offsets[row+1]-context.offsets[row]);
    if(!axes.empty())input.multimodal_positions=at::cat(axes,1);
    input.images.clear();
    for(const auto& image:original.images) {
      auto found=std::upper_bound(original.offsets,original.offsets+original.batch+1,image.token_start);
      auto row=found-original.offsets-1;
      if(image.token_start+image.token_count<=original.offsets[row]+context.hits[row])continue;
      auto copy=image;copy.token_start=context.offsets[row]+image.token_start-original.offsets[row]-context.hits[row];copy.sequence_start=context.offsets[row];input.images.push_back(std::move(copy));
    }
    if(radix_enabled(original)) {
      auto scatter=original.host_scatter_unfold.defined()?original.host_scatter_unfold:original.scatter_unfold.cpu();
      TORCH_CHECK(scatter.device().is_cpu()&&scatter.scalar_type()==at::kLong&&scatter.is_contiguous()
        &&scatter.sizes()==at::IntArrayRef({original.ids.numel()}),"Invalid host prefix Radix mapping");
      const auto* scatter_rows=scatter.const_data_ptr<int64_t>();
      std::unordered_map<int64_t,int64_t> remap;
      std::vector<int64_t> compact,unfold,fold;
      for(int64_t row=0;row<int64_t(original_rows.size());++row) {
        auto old=scatter_rows[original_rows[row]];auto [it,inserted]=remap.emplace(old,remap.size());
        if(inserted){compact.push_back(row);fold.push_back(row);}unfold.push_back(it->second);
      }
      auto index=at::tensor(compact,at::TensorOptions().dtype(at::kLong)).to(input.ids.device());
      input.compact_ids=input.ids.index_select(0,index);input.compact_positions=input.positions.index_select(0,index);input.compact_types=input.types.index_select(0,index);
      input.scatter_unfold=at::tensor(unfold,at::TensorOptions().dtype(at::kLong)).to(input.ids.device());input.fold_gather=at::tensor(fold,at::TensorOptions().dtype(at::kLong)).to(input.ids.device());
    }
  }
  input.prefix=&context;
    auto result=run(input);
#ifdef TEI_TORCH_CUDA_KERNELS
    std::shared_ptr<at::cuda::CUDAEvent> event;
    if(admitting&&input.ids.is_cuda()){event=std::make_shared<at::cuda::CUDAEvent>();event->record(at::cuda::getCurrentCUDAStream(input.ids.device().index()));}
#endif
    for(auto& entry:context.writers)if(entry&&entry->output.defined()&&int64_t(entry->layers.size())==model.prefix_layers()) {
      budget->reserved_bytes-=entry->reserved_bytes;entry->reserved_bytes=0;
#ifdef TEI_TORCH_CUDA_KERNELS
      if(event) {
        const auto producer=at::cuda::getCurrentCUDAStream(input.ids.get_device());
        // All snapshot writes and the event are queued before publication.
        // Stream ordering therefore permits immediate producer-stream reuse;
        // another stream can consume the entry only after global completion.
        entry->ready=[event,producer]{return at::cuda::getCurrentCUDAStream(producer.device_index())==producer||event->query();};
      }
#endif
      entries.erase(std::remove_if(entries.begin(),entries.end(),[&](const auto& old){return old->operation==operation&&common(old->identity,entry->identity,entry->identity.tokens.size())==int64_t(entry->identity.tokens.size())&&old->identity.tokens.size()==entry->identity.tokens.size();}),entries.end());
      if(entries.size()<capacity){entries.push_back(entry);++counters.admissions;}
    }
    return result;
  } catch(...) {entries.clear();linear->clear();throw;}
}
}

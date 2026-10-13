#include "batching/prefix_cache.h"
#include "prefix_cache_test.h"
#include "helpers/prefix_linear.h"
#include "bridge.h"
// Native packed decoder regression tests; no Python runtime or padded tokens.
#include "models/decoder_models.h"
#include "batching/radix.h"
#include "batching/cuda_graph.h"
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <atomic>
#include <barrier>
#include <thread>
#include <ATen/Parallel.h>
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/cuda/CUDAGuard.h>
#include <chrono>
#endif
using namespace tei;
namespace {
Options config(const std::string& family, bool moe = false) {
  Options cfg;
  cfg.values = {{"model_type",family},{"hidden_size","16"},{"num_attention_heads","4"},
    {"num_key_value_heads","2"},{"head_dim","4"},{"intermediate_size","24"},
    {"num_hidden_layers","2"},{"vocab_size","32"},{"max_position_embeddings","32"},
    {"hidden_act","silu"},{"rms_norm_eps","0.000001"}};
  if (moe) { cfg.values["num_experts"]="3";cfg.values["num_experts_per_tok"]="2";cfg.values["moe_intermediate_size"]="12"; }
  return cfg;
}
Weights weights(const Options& cfg, bool fused = false) {
  Weights w;
  auto add = [&](const std::string& name, std::initializer_list<int64_t> shape, bool norm = false) {
    int64_t count=1; for(auto n:shape)count*=n;
    float offset=0;for(auto c:name)offset+=c;
    auto data = at::sin(at::arange(count,at::TensorOptions().dtype(at::kFloat))*.13 + offset*.01)*.15;
    if(norm)data+=1;
    w[name]=data.view(at::IntArrayRef(shape));
  };
  add("model.embed_tokens.weight",{32,16});add("model.norm.weight",{16},true);
  for(int layer=0;layer<2;++layer){
    auto p="model.layers."+std::to_string(layer)+".";
    add(p+"input_layernorm.weight",{16},true);add(p+"post_attention_layernorm.weight",{16},true);
    add(p+"self_attn.q_proj.weight",{16,16});add(p+"self_attn.k_proj.weight",{8,16});add(p+"self_attn.v_proj.weight",{8,16});add(p+"self_attn.o_proj.weight",{16,16});
    if(cfg.string("model_type")=="qwen2")for(auto name:{"q_proj","k_proj","v_proj"})add(p+"self_attn."+name+".bias",{std::string(name)=="q_proj"?16:8});
    if(cfg.string("model_type")=="qwen3" || cfg.string("model_type")=="qwen3_moe") {add(p+"self_attn.q_norm.weight",{4},true);add(p+"self_attn.k_norm.weight",{4},true);}
    if(!cfg.integer("num_experts")){add(p+"mlp.gate_proj.weight",{24,16});add(p+"mlp.up_proj.weight",{24,16});add(p+"mlp.down_proj.weight",{16,24});}
    else {
      add(p+"mlp.gate.weight",{3,16});std::vector<at::Tensor> gu,down;
      for(int e=0;e<3;++e){auto ep=p+"mlp.experts."+std::to_string(e)+".";add(ep+"gate_proj.weight",{12,16});add(ep+"up_proj.weight",{12,16});add(ep+"down_proj.weight",{16,12});gu.push_back(at::cat({w.at(ep+"gate_proj.weight"),w.at(ep+"up_proj.weight")},0));down.push_back(w.at(ep+"down_proj.weight"));}
      if(fused){w[p+"mlp.experts.gate_up_proj"]=at::stack(gu);w[p+"mlp.experts.down_proj"]=at::stack(down);}
    }
  }
  return w;
}
at::Tensor run(Model& model, std::vector<int64_t> ids, std::vector<int32_t> offsets) {
  std::vector<int64_t> positions;
  int64_t maximum=0;
  for(size_t row=0;row+1<offsets.size();++row){auto len=offsets[row+1]-offsets[row];maximum=std::max(maximum,int64_t(len));for(int n=0;n<len;++n)positions.push_back(n);}
  auto opts=at::TensorOptions().dtype(at::kLong);
  PackedInput input{at::from_blob(ids.data(),{int64_t(ids.size())},opts),at::zeros({int64_t(ids.size())},opts),at::from_blob(positions.data(),{int64_t(positions.size())},opts),at::from_blob(offsets.data(),{int64_t(offsets.size())},opts.dtype(at::kInt)),offsets.data(),int64_t(offsets.size()-1),maximum};
  return model.forward(input);
}
void check_close(const at::Tensor& a,const at::Tensor& b){auto error=(a-b).abs().max().item<float>();TORCH_CHECK(error<2e-5,"Decoder parity error ",error);}
void radix_decoder_test(Model& model) {
 auto options=at::TensorOptions().dtype(at::kLong);
 std::vector<int32_t> offsets={0,3,7,10};
 PackedInput input{at::tensor({3,4,5,3,4,6,7,3,4,5},options),at::zeros({10},options),
   at::tensor({0,1,2,0,1,2,3,0,1,2},options),at::tensor({0,3,7,10},options.dtype(at::kInt)),offsets.data(),3,4};
 auto expected=model.forward(input);
 input.compact_ids=at::tensor({3,4,5,6,7},options);input.compact_positions=at::tensor({0,1,2,2,3},options);
 input.compact_types=at::zeros({5},options);input.scatter_unfold=at::tensor({0,1,2,0,1,3,4,0,1,2},options);
 input.fold_gather=at::tensor({0,1,2,5,6},options);
 validate_radix(input,true);
 if(!model.supports_radix()) {
   bool rejected=false;try{model.forward(input);}catch(const c10::Error&){rejected=true;}
   TORCH_CHECK(rejected,"Unsupported bidirectional folding was silently ignored");return;
 }
 check_close(model.forward(input),expected);
 check_prefix_cache(model,input);
 TORCH_CHECK(radix_ids(input).numel()==5&&input.ids.numel()==10,"Radix retained duplicate token rows");
 // Both tails contain token 9 at position 3, but only the first two requests
 // have the same complete causal history after resuming cached prefixes.
 std::vector<int32_t> tails={0,4,8,12};
 PackedInput branched{at::tensor({3,4,5,9,3,4,5,9,6,4,5,9},options),at::zeros({12},options),
  at::tensor({0,1,2,3,0,1,2,3,0,1,2,3},options),at::tensor(tails,options.dtype(at::kInt)),tails.data(),3,4};
 branched.compact_ids=at::tensor({3,4,5,9,6,4,5,9},options);
 branched.compact_positions=at::tensor({0,1,2,3,0,1,2,3},options);
 branched.compact_types=at::zeros({8},options);
 branched.scatter_unfold=at::tensor({0,1,2,3,0,1,2,3,4,5,6,7},options);
 branched.fold_gather=at::tensor({0,1,2,3,8,9,10,11},options);
 validate_radix(branched,true);check_prefix_cache(model,branched);
}
void cached_decoder_test(Model& model) {
 PrefixCache cache(1<<20,64,8);
 auto cached=[&](std::vector<int64_t> ids,std::vector<int32_t> offsets) {
  auto options=at::TensorOptions().dtype(at::kLong);std::vector<int64_t> positions;
  for(size_t row=0;row+1<offsets.size();++row)for(int64_t t=0;t<offsets[row+1]-offsets[row];++t)positions.push_back(t);
  auto tensor=at::tensor(ids,options);PackedInput input{tensor,at::zeros_like(tensor),at::tensor(positions,options),
   at::tensor(offsets,at::TensorOptions().dtype(at::kInt)),offsets.data(),int64_t(offsets.size()-1),0};
  for(size_t row=0;row+1<offsets.size();++row)input.max_sequence=std::max<int64_t>(input.max_sequence,offsets[row+1]-offsets[row]);
  return cache.run(model,input,0);
 };
 auto original=cached({3,4,5,3,4,8,9},{0,3,7}).clone();
 check_close(original,run(model,{3,4,5,3,4,8,9},{0,3,7}));
 check_close(cached({3,4,5,3,4,8,9},{0,3,7}),original);
 check_close(cached({3,4,5,9,3,4,8},{0,4,7}),run(model,{3,4,5,9,3,4,8},{0,4,7}));
 // Equal suffixes/positions have different complete parent contexts.
 check_close(cached({6,4,5,9,3,4,5,9},{0,4,8}),run(model,{6,4,5,9,3,4,5,9},{0,4,8}));
 TORCH_CHECK(cache.stats().hits>0&&cache.stats().reused_tokens>0,"Prefix manager never reused actual causal state");
 TORCH_CHECK(cache.bytes()<=1<<20,"Prefix manager exceeded tensor budget");
 check_close(original,run(model,{3,4,5,3,4,8,9},{0,3,7}));
 cache.clear();TORCH_CHECK(cache.bytes()==0,"Cleared prefix entries retain resident tensors");
}
void cache_admission_test(Model& model) {
 struct Observed final:Model {
  Model& inner;mutable bool capturing=false;mutable size_t calls=0;bool fail=false;
  explicit Observed(Model& model):inner(model) {}
  void ready() override {}
  bool supports_prefix_cache(int operation) const override {return inner.supports_prefix_cache(operation);}
  int64_t prefix_layers() const override {return inner.prefix_layers();}
  uint64_t prefix_payload_bytes(int64_t tokens) const override {return inner.prefix_payload_bytes(tokens);}
  at::Tensor forward(const PackedInput& input) const override {
   ++calls;
   capturing=input.prefix&&std::any_of(input.prefix->writers.begin(),input.prefix->writers.end(),[](const auto& writer){return bool(writer);});
   auto result=inner.forward(input);TORCH_CHECK(!fail,"Injected failure after snapshot creation");return result;
  }
 } observed(model);
 auto options=at::TensorOptions().dtype(at::kLong);std::vector<int32_t> ends={0,4};
 PackedInput input{at::tensor({3,4,5,6},options),at::zeros({4},options),at::arange(4,options),
  at::tensor(ends,options.dtype(at::kInt)),ends.data(),1,4};
 auto reference=model.forward(input);
 PrefixCache measure(1<<20,64,1);check_close(measure.run(observed,input,0),reference);
 const auto footprint=measure.bytes();TORCH_CHECK(footprint>0&&observed.capturing,"Snapshot footprint was not measured");
 bool unsupported=false;try{measure.run(observed,input,1);}catch(const c10::Error&){unsupported=true;}
 TORCH_CHECK(unsupported&&measure.bytes()==footprint,"Unsupported task discarded valid backbone state");
 PrefixCache too_small(footprint-1,64,1);
 check_close(too_small.run(observed,input,0),reference);
 TORCH_CHECK(!observed.capturing&&too_small.bytes()==0&&too_small.stats().admissions==0,
  "Insufficient capacity still allocated partial snapshots");
 PrefixCache exact(footprint,64,1);check_close(exact.run(observed,input,0),reference);
 TORCH_CHECK(observed.capturing&&exact.bytes()==footprint,"Exact byte budget rejected a complete snapshot");
 check_close(exact.run(observed,input,0),reference);TORCH_CHECK(exact.stats().hits==1,"Exact budget prevented reuse");
 Observed other(model);auto hits=measure.stats().hits;
 check_close(measure.run(other,input,0),reference);
 TORCH_CHECK(measure.stats().hits==hits,"A different loaded model instance reused old prefix state");
 check_close(measure.run(observed,input,0),reference);
 const auto tf32=at::globalContext().allowTF32CuBLAS();
 at::globalContext().setAllowTF32CuBLAS(!tf32);
 auto changed=measure.run(observed,input,0);
 at::globalContext().setAllowTF32CuBLAS(tf32);
 check_close(changed,reference);
 TORCH_CHECK(measure.stats().hits==hits,"Arithmetic policy change retained old prefix state");
 check_close(measure.run(observed,input,0),reference);
 hits=measure.stats().hits;auto flash=at::globalContext().userEnabledFlashSDP();
 at::globalContext().setSDPUseFlash(!flash);
 auto attention_changed=measure.run(observed,input,0);
 at::globalContext().setSDPUseFlash(flash);
 check_close(attention_changed,reference);
 TORCH_CHECK(measure.stats().hits==hits,"Attention backend policy retained old prefix state");
 auto check_cudnn_policy=[&](auto change,auto restore) {
  check_close(measure.run(observed,input,0),reference);
  const auto before=measure.stats().hits;
  change();at::Tensor changed;
  try {changed=measure.run(observed,input,0);}catch(...) {restore();throw;}
  restore();check_close(changed,reference);
  TORCH_CHECK(measure.stats().hits==before,"cuDNN algorithm policy reused old prefix state");
  check_close(measure.run(observed,input,0),reference);
  const auto restored=measure.stats().hits;
  check_close(measure.run(observed,input,0),reference);
  TORCH_CHECK(measure.stats().hits==restored+1,"Restored cuDNN policy prevented normal prefix reuse");
 };
 auto& global=at::globalContext();
 const auto deterministic=global.deterministicCuDNN();
 check_cudnn_policy([&]{global.setDeterministicCuDNN(!deterministic);},
                   [&]{global.setDeterministicCuDNN(deterministic);});
 const auto benchmark=global.benchmarkCuDNN();
 check_cudnn_policy([&]{global.setBenchmarkCuDNN(!benchmark);},
                   [&]{global.setBenchmarkCuDNN(benchmark);});
 const auto benchmark_limit=global.benchmarkLimitCuDNN();
 check_cudnn_policy([&]{global.setBenchmarkLimitCuDNN(benchmark_limit==0?1:0);},
                   [&]{global.setBenchmarkLimitCuDNN(benchmark_limit);});
 input.ids=at::tensor({8,4,5,6},options);
 observed.fail=true;bool failed=false;try{measure.run(observed,input,0);}catch(const c10::Error&){failed=true;}
 TORCH_CHECK(failed&&observed.capturing&&measure.bytes()==0,"Failure retained resident or reserved cache memory");
 input.ids=at::tensor({3,4,5,6},options);observed.fail=false;check_close(measure.run(observed,input,0),reference);
 // A pinned reader may lose its lookup slot while its immutable payload stays
 // alive. Capacity one must still admit a longer checkpoint after the run.
 auto extended=[&](int64_t tokens) {
  ends[1]=tokens;input.ids=at::arange(3,3+tokens,options);input.types=at::zeros_like(input.ids);
  input.positions=at::arange(tokens,options);input.cumulative=at::tensor(ends,options.dtype(at::kInt));input.max_sequence=tokens;
  check_close(measure.run(observed,input,0),model.forward(input));
 };
 extended(5);auto reused=measure.stats().reused_tokens;extended(6);
 TORCH_CHECK(measure.stats().reused_tokens-reused==5,"Capacity-one cache did not advance its retained prefix");
 // Production admission does not capture one-off inputs. Exact full results
 // avoid the backbone but still own writable output storage independently.
 ends[1]=4;input.ids=at::tensor({3,4,5,6},options);input.types=at::zeros_like(input.ids);
 input.positions=at::arange(4,options);input.cumulative=at::tensor(ends,options.dtype(at::kInt));input.max_sequence=4;
 PrefixCache delayed(1<<20,64,8,2);CudaGraphCache graphs;
 check_close(delayed.run(observed,input,0,&graphs),reference);
 TORCH_CHECK(!observed.capturing&&delayed.bytes()==0,"First observation captured a one-off prefix");
 check_close(delayed.run(observed,input,0,&graphs),reference);
 TORCH_CHECK(observed.capturing&&delayed.bytes()>0,"Repeated causal stem was never admitted");
 auto before_complete=observed.calls;
 auto owned=delayed.run(observed,input,0,&graphs);check_close(owned,reference);owned.zero_();
 check_close(delayed.run(observed,input,0,&graphs),reference);
 TORCH_CHECK(observed.calls==before_complete,"Exact complete input reran the backbone");
 input.max_sequence=5;check_close(delayed.run(observed,input,0,&graphs),reference);
 TORCH_CHECK(observed.calls==before_complete+1,"Complete-result reuse ignored original attention geometry");
 delayed.clear();input.max_sequence=4;
 for(int64_t token=8;token<24;++token) {
  input.ids=at::tensor(std::vector<int64_t>{token,4,5,6},options);
  check_close(delayed.run(observed,input,0,&graphs),model.forward(input));
 }
 TORCH_CHECK(delayed.bytes()==0&&delayed.stats().admissions==1,"One-off churn consumed cache tensor storage");
}
void cache_concurrency_test(Model& model) {
 auto options=at::TensorOptions().dtype(at::kLong);
 std::vector<at::Tensor> references;
 for(int64_t token=3;token<12;++token)references.push_back(run(model,{token,4,5,6},{0,4}));
 PrefixCache cache(1<<20,64,1);std::barrier start(3);
 std::exception_ptr failure;std::mutex errors;
 auto save_failure=[&]{std::lock_guard lock(errors);if(!failure)failure=std::current_exception();};
 auto worker=[&](int shift) {
  c10::InferenceMode inference;start.arrive_and_wait();
  try {
   std::vector<int32_t> offsets={0,4};
   for(int i=0;i<200;++i) {
    const auto token=3+(i/3+shift)%9;
    PackedInput input{at::tensor(std::vector<int64_t>{token,4,5,6},options),at::zeros({4},options),
     at::arange(4,options),at::tensor(offsets,options.dtype(at::kInt)),offsets.data(),1,4};
    auto value=cache.run(model,input,0);check_close(value,references[token-3]);
    value.fill_(123); // Mutating caller storage must never change a cache reader.
   }
  }catch(...){save_failure();}
 };
 std::jthread first(worker,0),second(worker,4);
 std::jthread maintenance([&] {
  c10::InferenceMode inference;start.arrive_and_wait();
  try {for(int i=0;i<600;++i) {
   if(i%7==0)cache.clear();
   TORCH_CHECK(cache.bytes()<=1<<20,"Concurrent clear/stats exceeded the cache budget");
   static_cast<void>(cache.stats());std::this_thread::yield();
  }}catch(...){save_failure();}
 });
 first.join();second.join();maintenance.join();if(failure)std::rethrow_exception(failure);
 cache.clear();TORCH_CHECK(cache.bytes()==0,"Concurrent readers retained cleared cache storage");
}
#ifdef TEI_TORCH_CUDA_KERNELS
void cache_cold_maximum_test(const c10::Device& device) {
 const auto major=at::cuda::getDeviceProperties(device.index())->major;
 if((major!=9&&major!=10)||at::detail::getCUDAHooks().versionRuntimeCuDNN()<91800)return;
 auto cfg=config("qwen2");
 cfg.values["hidden_size"]="128";cfg.values["num_attention_heads"]="2";
 cfg.values["num_key_value_heads"]="1";cfg.values["head_dim"]="64";
 cfg.values["intermediate_size"]="192";cfg.values["max_position_embeddings"]="512";
 cfg.values["_cudnn_varlen"]="true";
 auto generator=at::detail::createCPUGenerator(713);
 Weights tensors;
 auto add=[&](const std::string& name,std::initializer_list<int64_t> shape,bool norm=false) {
  auto value=norm?at::ones(at::IntArrayRef(shape),at::TensorOptions().dtype(at::kFloat))
    :at::randn(at::IntArrayRef(shape),generator,at::TensorOptions().dtype(at::kFloat))*.03;
  tensors[name]=value.to(device,at::kBFloat16);
 };
 add("model.embed_tokens.weight",{32,128});add("model.norm.weight",{128},true);
 for(int64_t layer=0;layer<2;++layer) {
  auto p="model.layers."+std::to_string(layer)+".";
  add(p+"input_layernorm.weight",{128},true);add(p+"post_attention_layernorm.weight",{128},true);
  for(auto projection:{"q_proj","k_proj","v_proj"}) {
   auto width=std::string(projection)=="q_proj"?128:64;
   add(p+"self_attn."+projection+".weight",{width,128});
   add(p+"self_attn."+projection+".bias",{width});
  }
  add(p+"self_attn.o_proj.weight",{128,128});
  add(p+"mlp.gate_proj.weight",{192,128});add(p+"mlp.up_proj.weight",{192,128});
  add(p+"mlp.down_proj.weight",{128,192});
 }
 auto model=create_decoder(cfg,tensors,device,at::kBFloat16);model->ready();
 auto options=at::TensorOptions().device(device).dtype(at::kLong);
 std::vector<int32_t> offsets{0,256,512};
 PackedInput input{at::arange(512,options).remainder(31)+1,at::zeros({512},options),
  at::arange(256,options).repeat({2}),at::tensor(offsets,options.dtype(at::kInt)),offsets.data(),2,512};
 auto reference=model->forward(input);PrefixCache cache(2<<20,512,4);
 auto captured=cache.run(*model,input,0);
 TORCH_CHECK(at::equal(captured,reference),"Cold capture changed caller's attention bound/provider, maxabs=",
  (captured-reference).abs().max().item<float>());
 // A cached suffix must preserve packed cuDNN's geometry too. Both identical
 // rows hit; then one row changes its last 13 tokens, leaving a mixed suffix.
 auto warm=cache.run(*model,input,0);
 TORCH_CHECK(at::equal(warm,reference),"Warm cuDNN changed outputs, maxabs=",
  (warm-reference).abs().max().item<float>());
 input.ids=input.ids.clone();input.ids.narrow(0,499,13).fill_(17);
 reference=model->forward(input);auto partial=cache.run(*model,input,0);
 TORCH_CHECK(at::equal(partial,reference),"Mixed cuDNN suffix changed outputs, maxabs=",
  (partial-reference).abs().max().item<float>());
 TORCH_CHECK(cache.stats().hits>=4,"cuDNN controls did not exercise prefix reuse");
}
void linear_workspace_test(const c10::Device& device) {
 // This SGEMM shape's valid suffix recipe needs 1 MiB plus alignment padding.
 // The original provider permits it, so the Lt default must not force GEMV.
 if(at::cuda::getDeviceProperties(device.index())->major!=9)return;
 c10::cuda::CUDAGuard guard(device);
 auto& global=at::globalContext();
 const auto old_precision=global.float32MatmulPrecision();
 const auto old_backend=global.blasPreferredBackend();
 const auto old_blas=at::cuda::getChosenWorkspaceSize(),old_lt=at::cuda::getCUDABlasLtWorkspaceSize();
 struct Restore {
  at::Context& context;at::Float32MatmulPrecision precision;at::BlasBackend backend;size_t blas,lt;
  ~Restore(){context.setFloat32MatmulPrecision(precision==at::Float32MatmulPrecision::HIGHEST?"highest":precision==at::Float32MatmulPrecision::HIGH?"high":"medium");context.setBlasPreferredBackend(backend);at::cuda::setChosenWorkspaceSize(blas);at::cuda::setCUDABlasLtWorkspaceSize(lt);}
 } restore{global,old_precision,old_backend,old_blas,old_lt};
 global.setFloat32MatmulPrecision("highest");global.setBlasPreferredBackend(at::BlasBackend::Cublas);
 at::cuda::setChosenWorkspaceSize(32<<20);at::cuda::setCUDABlasLtWorkspaceSize(1<<20);
 auto options=at::TensorOptions().device(device).dtype(at::kFloat);
 auto input=at::randn({2048,2816},options),weight=at::randn({128,2816},options);
 auto reference=at::linear(input,weight).narrow(0,2047,1);
 at::cuda::getCurrentCUDAStream(device.index()).synchronize();
 auto producer=c10::cuda::getStreamFromPool(false,device.index());at::Tensor actual;
 {
  c10::cuda::CUDAStreamGuard stream(producer);PrefixLinearPlans plans;
  actual=plans.forward(input.narrow(0,2047,1),weight,{},2048,true);
  plans.clear();
  // Scratch can be recycled after submission; all GPU uses remain ordered.
  for(int i=0;i<8;++i)at::empty({(1<<20)+512},options.dtype(at::kByte)).fill_(123);
 }
 producer.synchronize();
 TORCH_CHECK(at::equal(actual,reference),"Original SGEMM recipe fell back under the Lt workspace cutoff: ",(actual-reference).abs().max().item<float>());
}
void cache_stream_test(Model& model,PackedInput input) {
 input.host_ids=input.ids.cpu();input.host_types=input.types.cpu();input.host_positions=input.positions.cpu();
 auto expected=model.forward(input);at::cuda::getCurrentCUDAStream(input.ids.get_device()).synchronize();
 PrefixCache cache(1<<20,64,8);
 auto producer=c10::cuda::getStreamFromPool(false,input.ids.get_device());
 auto consumer=c10::cuda::getStreamFromPool(false,input.ids.get_device());
 {
  c10::cuda::CUDAStreamGuard guard(producer);
  cache.run(model,input,0);producer.synchronize();
 }
 at::Tensor actual;
 {
  c10::cuda::CUDAStreamGuard guard(consumer);
  actual=cache.run(model,input,0);
  TORCH_CHECK(cache.stats().hits==input.batch,"Ready producer entries were not consumed across streams");
  cache.clear();TORCH_CHECK(cache.bytes()==0,"Clear retained a cache entry after cross-stream submission");
 }
 {
  c10::cuda::CUDAStreamGuard guard(producer);
  // Recycle similarly sized allocations on the original allocation stream
  // while the consumer owns reads of the evicted immutable snapshots.
  for(int i=0;i<16;++i)at::empty_like(expected).fill_(123);
 }
 consumer.synchronize();producer.synchronize();
 TORCH_CHECK(at::allclose(actual,expected,.01,.02),"Eviction corrupted cross-stream cached output");
 struct Gate {
  std::atomic<bool> released=false;
  c10::cuda::CUDAStream stream;
  explicit Gate(c10::cuda::CUDAStream value):stream(value) {}
  void open(){released.store(true);released.notify_all();}
  ~Gate(){open();stream.synchronize();}
 } gate(producer);
 struct Delayed final:Model {
  Model& inner;Gate& gate;at::cuda::CUDAEvent& proof;bool delay=true;
  Delayed(Model& model,Gate& value,at::cuda::CUDAEvent& event):inner(model),gate(value),proof(event) {}
  void ready() override {}
  bool supports_prefix_cache(int operation) const override {return inner.supports_prefix_cache(operation);}
  int64_t prefix_layers() const override {return inner.prefix_layers();}
  uint64_t prefix_payload_bytes(int64_t tokens) const override {return inner.prefix_payload_bytes(tokens);}
  at::Tensor forward(const PackedInput& input) const override {
   auto result=inner.forward(input);
   if(delay) {
    auto wait=[](void* data){static_cast<Gate*>(data)->released.wait(false);};
    C10_CUDA_CHECK(cudaLaunchHostFunc(gate.stream.stream(),wait,&gate));
    proof.record(gate.stream);
   }
   return result;
  }
 };
 at::cuda::CUDAEvent proof;Delayed delayed(model,gate,proof);PrefixCache pending(1<<20,64,8);
 {
  c10::cuda::CUDAStreamGuard guard(producer);
  pending.run(delayed,input,0);
  // The proof event precedes cache publication's event. A closed gate keeps
  // both pending, so this tests actual queued state rather than a ready hit.
  const bool incomplete=!proof.query();
  std::jthread release([&]{std::this_thread::sleep_for(std::chrono::milliseconds(50));gate.open();});
  TORCH_CHECK(incomplete,"Fixture did not retain pending producer work");
  delayed.delay=false;auto resumed=pending.run(delayed,input,0);
  TORCH_CHECK(pending.stats().hits==input.batch,"Pending producer-stream state was recomputed");
  producer.synchronize();
  TORCH_CHECK(at::allclose(resumed,expected,.01,.02),"Producer-stream pending reuse changed outputs");
 }
}
#endif
void multimodal_test(){
 auto cfg=config("qwen3");auto w=weights(cfg);auto model=create_decoder(cfg,w,c10::Device("cpu"),at::kFloat);model->ready();
 auto opts=at::TensorOptions().dtype(at::kLong);
 std::vector<int64_t> ids={3,4,5,6,7},pos={0,1,2,0,1};std::vector<int32_t> offsets={0,3,5};
 PackedInput input{at::from_blob(ids.data(),{5},opts),at::zeros({5},opts),at::from_blob(pos.data(),{5},opts),at::from_blob(offsets.data(),{3},opts.dtype(at::kInt)),offsets.data(),2,3};
 auto frequencies=at::tensor({1.f,.01f});auto angles=input.positions.to(at::kFloat).unsqueeze(1)*frequencies;
 auto cosine=at::cos(angles),sine=at::sin(angles);
 check_close(model->forward(input),decoder_multimodal_forward(*model,input,{}, {},cosine,sine));
 auto indices=at::tensor({int64_t(1)},opts),visual=at::ones({1,16})*.25;
 auto injected=decoder_multimodal_forward(*model,input,indices,visual,cosine,sine);
 auto zero=at::zeros_like(visual);
 check_close(injected,decoder_multimodal_forward(*model,input,indices,visual,cosine,sine,{zero,zero}));
 auto deep=decoder_multimodal_forward(*model,input,indices,visual,cosine,sine,{at::arange(16).to(at::kFloat).view({1,16})*.07});
 TORCH_CHECK((deep-injected).abs().max().item<float>()>1e-4,"Deepstack feature injection did not affect text states");
 check_close(deep.narrow(0,3,2),model->forward(input).narrow(0,3,2));
 check_close(deep.narrow(0,0,1),injected.narrow(0,0,1));
 std::cout<<"Qwen3 external embeddings, multimodal RoPE and deepstack packed isolation passed\n";
}
void probe_cache_test() {
 auto cfg=config("qwen3");auto source=weights(cfg);
 TeiConfig native{16,4,2,24,32,32,1,1e-6,3};
 auto handle=tei_create(&native,"cpu",0);TORCH_CHECK(handle,tei_error());
 struct Cleanup{void* handle;~Cleanup(){tei_destroy(handle);}} cleanup{handle};
 for(const auto& [key,value]:cfg.values)TORCH_CHECK(tei_option(handle,key.c_str(),value.c_str())==0,tei_error());
 TORCH_CHECK(tei_option(handle,"_prefix_cache_max_bytes","1048576")==0,tei_error());
 for(const auto& [name,tensor]:source)TORCH_CHECK(tei_weight(handle,name.c_str(),
  reinterpret_cast<const uint8_t*>(tensor.const_data_ptr()),tensor.sizes().data(),tensor.dim(),0)==0,tei_error());
 TORCH_CHECK(tei_ready(handle)==0,tei_error());
 int64_t ids[]={1,2,3},types[]={0,0,0},positions[]={0,1,2},selection[]={0};int32_t cumulative[]={0,3};float result[16];
 auto forward=[&]{TORCH_CHECK(tei_forward(handle,ids,types,positions,cumulative,1,3,2,selection,1,nullptr,0,
  result,16,nullptr,nullptr,0,nullptr,0)==0,tei_error());};
 uint64_t before[7],after[7];forward();
 TORCH_CHECK(tei_prefix_cache_stats(handle,before,7)==0&&before[3]==0,"One-off user input bypassed admission policy");
 forward();
 TORCH_CHECK(tei_prefix_cache_stats(handle,before,7)==0&&before[3]==1,"User request did not fill prefix cache");
 TORCH_CHECK(tei_prefix_cache_set_probe(handle,1)==0,"Probe begin did not return prior flag");forward();
 TORCH_CHECK(tei_prefix_cache_stats(handle,after,7)==0,tei_error());
 TORCH_CHECK(std::equal(std::begin(before),std::end(before),std::begin(after)),"Probe consumed or populated user cache state");
 TORCH_CHECK(tei_prefix_cache_set_probe(handle,0)==1,"Probe end did not restore prior flag");forward();
 TORCH_CHECK(tei_prefix_cache_stats(handle,after,7)==0&&after[0]==before[0]+1,"Probe bypass leaked into next user request");
 std::barrier start(2);std::exception_ptr failure;std::mutex errors;
 auto save_failure=[&]{std::lock_guard lock(errors);if(!failure)failure=std::current_exception();};
 std::jthread inference([&] {
  start.arrive_and_wait();
  try {for(int i=0;i<100;++i) {
   float owned[16];
   TORCH_CHECK(tei_forward(handle,ids,types,positions,cumulative,1,3,2,selection,1,nullptr,0,
    owned,16,nullptr,nullptr,0,nullptr,0)==0,tei_error());
   TORCH_CHECK(std::equal(std::begin(owned),std::end(owned),std::begin(result)),"Concurrent native probe changed user output");
  }}catch(...){save_failure();}
 });
 std::jthread probes([&] {
  start.arrive_and_wait();
  try {for(int i=0;i<600;++i) {
   TORCH_CHECK(tei_prefix_cache_set_probe(handle,i%2)>=0,tei_error());
   uint64_t current[7];TORCH_CHECK(tei_prefix_cache_stats(handle,current,7)==0,tei_error());
   if(i%7==0)TORCH_CHECK(tei_prefix_cache_clear(handle)==0,tei_error());
  }}catch(...){save_failure();}
 });
 inference.join();probes.join();if(failure)std::rethrow_exception(failure);
 TORCH_CHECK(tei_prefix_cache_set_probe(handle,0)>=0,tei_error());
}
}
int main(int argc,char** argv){c10::InferenceMode inference;at::set_num_threads(1);
 {
  // Preselected, non-normalized FP32 routes preserve learned expert scales
  // and their accumulation dtype, including a different activation callback.
  auto opts=at::TensorOptions().dtype(at::kBFloat16);
  auto x=at::arange(16,at::TensorOptions().dtype(at::kFloat)).view({2,8}).to(opts);
  auto gate_up=at::zeros({3,16,8},opts),down=at::zeros({3,8,8},opts);
  for(int64_t e=0;e<3;++e){gate_up[e].narrow(0,0,8).copy_(at::eye(8,opts));gate_up[e].narrow(0,8,8).fill_((e+1)/8.);down[e].copy_(at::eye(8,opts));}
  auto ids=at::tensor({2,0,1,2},at::TensorOptions().dtype(at::kLong)).view({2,2});
  auto routing=at::tensor({.123f,1.501f,.379f,.941f},at::TensorOptions().dtype(at::kFloat)).view({2,2});
  auto activation=[](const at::Tensor& both){return at::relu(both.narrow(-1,0,8))*both.narrow(-1,8,8);};
  auto actual=decoder_moe_dispatch(x,gate_up,down,ids,routing,activation);
  auto expected=at::zeros({2,8},at::TensorOptions().dtype(at::kFloat));
  for(int64_t row=0;row<2;++row)for(int64_t slot=0;slot<2;++slot){auto e=ids[row][slot].item<int64_t>();auto y=at::linear(activation(at::linear(x[row].unsqueeze(0),gate_up[e])),down[e]);expected[row].add_(y.squeeze(0).to(at::kFloat)*routing[row][slot]);}
  TORCH_CHECK(actual.scalar_type()==at::kFloat,"Preselected MoE lost FP32 accumulation");check_close(actual,expected);
 }

 {
  auto cfg=config("qwen2");cfg.values["architectures.0"]="Qwen2ForSequenceClassification";
  cfg.values["num_labels"]="3";cfg.values["pad_token_id"]="0";
  auto tensors=weights(cfg);tensors["score.weight"]=at::arange(48,at::kFloat).view({3,16})*.013;
  tensors["score.bias"]=at::tensor({.1f,-.2f,.3f});
  auto model=create_decoder(cfg,tensors,c10::Device("cpu"),at::kFloat);model->ready();
  TORCH_CHECK(model->supports_prefix_cache(1),"Loaded causal classifier did not advertise cache support");
  auto options=at::TensorOptions().dtype(at::kLong);std::vector<int32_t> offsets{0,3,7};
  PackedInput input{at::tensor({3,4,0,3,6,7,0},options),at::zeros({7},options),
   at::tensor({0,1,2,0,1,2,3},options),at::tensor(offsets,options.dtype(at::kInt)),offsets.data(),2,4};
  check_prefix_cache(*model,input,1);
 }
 for(auto family:{"llama","mistral","qwen2","qwen3","qwen3_moe"}){
  auto cfg=config(family,std::string(family)=="qwen3_moe");auto w=weights(cfg);auto initial=w;auto model=create_decoder(cfg,initial,c10::Device("cpu"),at::kFloat);model->ready();
  TORCH_CHECK(!model->supports_prefix_cache(1),"Backbone without a score head advertised classifier cache support");
  radix_decoder_test(*model);
  cached_decoder_test(*model);
  if(std::string(family)=="qwen3") {cache_admission_test(*model);cache_concurrency_test(*model);}
  if(argc>1) {
   auto gpu_cfg=cfg;gpu_cfg.values["num_attention_heads"]="2";gpu_cfg.values["num_key_value_heads"]="1";gpu_cfg.values["head_dim"]="8";
   auto gpu_weights=w;
   for(auto& [name,value]:gpu_weights)if(name.ends_with("q_norm.weight")||name.ends_with("k_norm.weight"))value=value.repeat({2});
   for(auto& [name,value]:gpu_weights)value=value.to(c10::Device(argv[1]),at::kBFloat16);
   auto gpu=create_decoder(gpu_cfg,gpu_weights,c10::Device(argv[1]),at::kBFloat16);gpu->ready();
   auto opts=at::TensorOptions().device(c10::Device(argv[1])).dtype(at::kLong);
   std::vector<int32_t> ends={0,3,7};
   PackedInput input{at::tensor({3,4,5,3,4,8,9},opts),at::zeros({7},opts),
    at::tensor({0,1,2,0,1,2,3},opts),at::tensor(ends,opts.dtype(at::kInt)),ends.data(),2,4};
   check_prefix_cache(*gpu,input,0,.01,.02);
#ifdef TEI_TORCH_CUDA_KERNELS
   if(std::string(family)=="qwen3")cache_stream_test(*gpu,input);
   if(std::string(family)=="qwen2")cache_cold_maximum_test(c10::Device(argv[1]));
#endif
  }
  auto packed=run(*model,{3,4,5,6,7,8,9},{0,3,7});check_close(packed,at::cat({run(*model,{3,4,5},{0,3}),run(*model,{6,7,8,9},{0,4})},0));
  check_close(run(*model,{3,4},{0,2}),run(*model,{3,4,5},{0,3}).narrow(0,0,2));
  if(std::string(family)=="qwen2"||std::string(family)=="qwen3"){
   auto scaledcfg=cfg;scaledcfg.values["rope_scaling.rope_type"]="llama3";scaledcfg.values["rope_scaling.factor"]="4";scaledcfg.values["rope_scaling.high_freq_factor"]="4";
   auto scaledweights=weights(scaledcfg);auto scaled=create_decoder(scaledcfg,scaledweights,c10::Device("cpu"),at::kFloat);scaled->ready();
   check_close(packed,run(*scaled,{3,4,5,6,7,8,9},{0,3,7}));
  }
  if(std::string(family)=="qwen3_moe"){
   auto unsupportedcfg=cfg;unsupportedcfg.values["rope_scaling.factor"]="4";auto unsupportedweights=weights(unsupportedcfg);auto unsupported=create_decoder(unsupportedcfg,unsupportedweights,c10::Device("cpu"),at::kFloat);
   bool rejected=false;try{unsupported->ready();}catch(const c10::Error&){rejected=true;}TORCH_CHECK(rejected,"Scaled Qwen3 MoE configuration was silently accepted");
  }
  if(std::string(family)=="qwen3_moe"){auto fused=weights(cfg,true);auto other=create_decoder(cfg,fused,c10::Device("cpu"),at::kFloat);other->ready();check_close(packed,run(*other,{3,4,5,6,7,8,9},{0,3,7}));}
  cfg.values["use_bidirectional_attention"]="true";if(std::string(family)=="qwen2")cfg.values["is_causal"]="false";
  auto bidi=create_decoder(cfg,w,c10::Device("cpu"),at::kFloat);bidi->ready();auto b=run(*bidi,{3,4,5},{0,3});TORCH_CHECK((b.narrow(0,0,2)-run(*bidi,{3,4},{0,2})).abs().max().item<float>()>1e-5,"Bidirectional attention was accidentally causal");
  radix_decoder_test(*bidi);
  TORCH_CHECK(!bidi->supports_radix(),"Bidirectional decoder incorrectly advertises prefix folding");
  check_close(run(*bidi,{3,4,5,6,7},{0,3,5}),at::cat({b,run(*bidi,{6,7},{0,2})},0));
  std::cout<<family<<" packed boundaries, causality and bidirectional attention passed\n";
 }
 multimodal_test();
 probe_cache_test();
#ifdef TEI_TORCH_CUDA_KERNELS
 if(argc>1)linear_workspace_test(c10::Device(argv[1]));
#endif
}

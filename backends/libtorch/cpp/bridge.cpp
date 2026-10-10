#include "bridge.h"
#include "model.h"
#ifdef TEI_TORCH_CUDA_KERNELS
#include "fast_kernels.h"
#include "packed_pool.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/detail/CUDAHooksInterface.h>
#endif
#include "decoder_models.h"
#include "decision_models.h"
#include "dense_models.h"
#include "encoder_models.h"
#include "gemma_models.h"
#include "multimodal_models.h"
#include "cuda_graph.h"
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/ops/_flash_attention_forward.h>
#include <ATen/ops/_addmm_activation.h>
#include <ATen/ops/_cudnn_attention_forward.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>

namespace {
thread_local std::string error;
std::once_flag precision_initialized;
void initialize_precision() {
  std::call_once(precision_initialized, [] {
    const auto* value = std::getenv("TEI_TORCH_FULL_PRECISION_GEMM");
    if (value && (std::strcmp(value, "true") == 0 || std::strcmp(value, "1") == 0)) {
      at::globalContext().setAllowFP16ReductionCuBLAS(false);
      at::globalContext().setAllowBF16ReductionCuBLAS(false);
      at::globalContext().setAllowFP16AccumulationCuBLAS(false);
      at::globalContext().setAllowTF32CuBLAS(false);
      at::globalContext().setAllowTF32CuDNN(false);
    }
  });
}
at::ScalarType scalar(int32_t dtype) {
  switch (dtype) {
    case 0: return at::kFloat;
    case 1: return at::kHalf;
    case 2: return at::kBFloat16;
    default: throw std::runtime_error("Unsupported tensor dtype");
  }
}
c10::Device device(const char* name) {
  if (std::strcmp(name, "auto") == 0)
    return c10::Device(at::globalContext().hasCUDA() && at::getNumGPUs() > 0 ? "cuda:0" : "cpu");
  return c10::Device(name);
}
struct Bert : tei::Model {
  TeiConfig config;
  c10::Device device;
  at::ScalarType dtype;
  std::unordered_map<std::string, at::Tensor> weights;
  std::string prefix;
  tei::Options options;
  bool roberta = false;
  bool cudnn_varlen = false;
  Bert(TeiConfig cfg, c10::Device dev, at::ScalarType type): config(cfg), device(dev), dtype(type) {}
  const at::Tensor& w(const std::string& name) const {
    auto entry = weights.find(prefix + name);
    TORCH_CHECK(entry != weights.end(), "Missing BERT weight: ", prefix, name);
    return entry->second;
  }
  at::Tensor linear(const at::Tensor& x, const std::string& name) const {
    return at::linear(x, w(name + ".weight"), w(name + ".bias"));
  }
  at::Tensor activated_linear(const at::Tensor& x,const std::string& name) const {
    // Candle HiddenAct::Gelu is tanh GELU. Its CUDA linear fuses the same
    // cuBLASLt epilogue; avoid writing then rereading the full intermediate.
    if(x.is_cuda())return at::_addmm_activation(w(name+".bias"),x,w(name+".weight").t(),1,1,config.activation!=2);
    auto projected=linear(x,name);
    return config.activation==2?at::relu(projected):at::gelu(projected,"tanh");
  }
  at::Tensor norm(const at::Tensor& x, const std::string& name) const {
    return at::layer_norm(x, {config.hidden}, w(name + ".weight"), w(name + ".bias"), config.epsilon);
  }
  at::Tensor residual_norm(const at::Tensor& x, const at::Tensor& residual, const std::string& name) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    if (x.is_cuda()) return tei::fused_add_layer_norm(x, residual, w(name + ".weight"), w(name + ".bias"), config.epsilon);
#endif
    return norm(x + residual, name);
  }
  void check(const std::string& name, std::initializer_list<int64_t> shape) const {
    TORCH_CHECK(w(name).sizes() == at::IntArrayRef(shape), "Invalid shape for ", name);
  }
  void ready() override {
    TORCH_CHECK(device.is_cpu() || device.is_cuda(),
      "LibTorch 2.14.1 varlen Flash Attention requires CUDA. CPU supports an unpadded reference path; MPS/XPU are not implemented.");
    if (device.is_cuda()) {
      TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "CUDA varlen attention requires float16 or bfloat16");
#ifdef TEI_TORCH_CUDA_KERNELS
      const auto major = at::cuda::getDeviceProperties(device.index())->major;
      cudnn_varlen = options.boolean("_cudnn_varlen", false) && (major == 9 || major == 10)
        && at::detail::getCUDAHooks().versionRuntimeCuDNN() >= 91800;
#endif
      const auto head = config.hidden / config.heads;
      TORCH_CHECK(head % 8 == 0 && head <= 256, "CUDA varlen attention requires a head dimension divisible by 8, at most 256");
    }
    roberta = options.string("model_type", "bert") != "bert";
    prefix = "";
    for (const auto* stem : {"bert.", "roberta.", "xlm-roberta.", "camembert.", ""}) {
      if (weights.count(std::string(stem) + "embeddings.word_embeddings.weight")) { prefix = stem; break; }
    }
    check("embeddings.word_embeddings.weight", {config.vocab, config.hidden});
    check("embeddings.position_embeddings.weight", {config.positions, config.hidden});
    check("embeddings.token_type_embeddings.weight", {config.types, config.hidden});
    auto norm_check = [&](const std::string& p) {
      check(p + ".weight", {config.hidden}); check(p + ".bias", {config.hidden});
    };
    auto linear_check = [&](const std::string& p, int64_t out, int64_t in) {
      check(p + ".weight", {out, in}); check(p + ".bias", {out});
    };
    norm_check("embeddings.LayerNorm");
    for (int64_t i = 0; i < config.layers; ++i) {
      std::string p = "encoder.layer." + std::to_string(i) + ".";
      for (auto qkv : {"query", "key", "value"})
        linear_check(p + "attention.self." + qkv, config.hidden, config.hidden);
      weights[prefix + p + "attention.self.qkv.weight"] = at::cat({w(p + "attention.self.query.weight"), w(p + "attention.self.key.weight"), w(p + "attention.self.value.weight")}, 0);
      weights[prefix + p + "attention.self.qkv.bias"] = at::cat({w(p + "attention.self.query.bias"), w(p + "attention.self.key.bias"), w(p + "attention.self.value.bias")}, 0);
      linear_check(p + "attention.output.dense", config.hidden, config.hidden);
      norm_check(p + "attention.output.LayerNorm");
      linear_check(p + "intermediate.dense", config.intermediate, config.hidden);
      linear_check(p + "output.dense", config.hidden, config.intermediate);
      norm_check(p + "output.LayerNorm");
    }
  }
  int64_t classification_width() const override {
    auto name = weights.count("classifier.out_proj.weight") ? "classifier.out_proj.weight" : "classifier.weight";
    auto it = weights.find(name); return it == weights.end() ? 0 : it->second.size(0);
  }
  at::Tensor root_linear(const at::Tensor& h, const std::string& name) const {
    return at::linear(h, tei::weight(weights, name + ".weight"), tei::weight(weights, name + ".bias"));
  }
  at::Tensor predict(const tei::PackedInput& input, bool tokenwise) const override {
    auto h = forward(input);
    if (!tokenwise) {
      std::vector<int64_t> first(input.offsets, input.offsets + input.batch);
      auto indices = at::tensor(first, at::TensorOptions().dtype(at::kLong)).to(device);
      h = h.index_select(0, indices);
    }
    if (weights.count("classifier.dense.weight")) {
      return root_linear(at::tanh(root_linear(h, "classifier.dense")), "classifier.out_proj");
    }
    if (!tokenwise && weights.count(prefix + "pooler.dense.weight")) h = at::tanh(linear(h, "pooler.dense"));
    return root_linear(h, "classifier");
  }
  at::Tensor token_scores(const at::Tensor& h) const override {
    at::Tensor transformed, matrix, bias;
    if (roberta) {
      transformed = at::gelu(root_linear(h, "lm_head.dense"), "tanh");
      transformed = at::layer_norm(transformed, {config.hidden}, tei::weight(weights, "lm_head.layer_norm.weight"), tei::weight(weights, "lm_head.layer_norm.bias"), config.epsilon);
      matrix = weights.count("lm_head.decoder.weight") ? weights.at("lm_head.decoder.weight") : w("embeddings.word_embeddings.weight");
      bias = tei::weight(weights, "lm_head.bias");
    } else {
      transformed = root_linear(h, "cls.predictions.transform.dense");
      transformed = config.activation == 2 ? at::relu(transformed) : at::gelu(transformed, "tanh");
      transformed = at::layer_norm(transformed, {config.hidden}, tei::weight(weights, "cls.predictions.transform.LayerNorm.weight"), tei::weight(weights, "cls.predictions.transform.LayerNorm.bias"), config.epsilon);
      matrix = weights.count("cls.predictions.decoder.weight") ? weights.at("cls.predictions.decoder.weight") : w("embeddings.word_embeddings.weight");
      bias = tei::weight(weights, "cls.predictions.bias");
    }
    return at::log1p(at::relu(at::linear(transformed, matrix, bias)));
  }
  int64_t output_width() const override { return config.hidden; }
  at::Tensor forward(const tei::PackedInput& input) const override {
    const auto& ids = input.ids; const auto& types = input.types;
    const auto& positions = input.positions; const auto& cumulative = input.cumulative;
    const auto* ends = input.offsets; const auto batch = input.batch, max_sequence = input.max_sequence;
    auto word = at::embedding(w("embeddings.word_embeddings.weight"), ids);
    auto token_type = at::embedding(w("embeddings.token_type_embeddings.weight"), types);
    auto position = at::embedding(w("embeddings.position_embeddings.weight"), positions);
    at::Tensor h;
#ifdef TEI_TORCH_CUDA_KERNELS
    if (word.is_cuda()) h = tei::fused_add_layer_norm(word, token_type,
      w("embeddings.LayerNorm.weight"), w("embeddings.LayerNorm.bias"), config.epsilon, position);
    else
#endif
    h = norm(word + token_type + position, "embeddings.LayerNorm");
    const int64_t tokens = ids.size(0), head = config.hidden / config.heads;
    for (int64_t i = 0; i < config.layers; ++i) {
      std::string p = "encoder.layer." + std::to_string(i) + ".";
      auto qkv = linear(h, p + "attention.self.qkv").view({tokens, 3, config.heads, head});
      auto q = qkv.select(1, 0), k = qkv.select(1, 1), v = qkv.select(1, 2);
      at::Tensor attended;
      if (device.is_cuda()) {
        // The same ATen operator used by torch.nn.attention.varlen.varlen_attn's
        // Flash Attention path. Q/K/V stay [total_tokens, heads, head_dim].
        // Cumulative INT32 offsets are on-device. No padded batch or mask exists.
        // cuDNN helps long sequences; FlashAttention is faster on the measured
        // ragged batches with maximum length 256, despite a similar token total.
        if (cudnn_varlen && max_sequence >= 512) {
          attended = std::get<0>(at::_cudnn_attention_forward(q, k, v, std::nullopt,
            cumulative, cumulative, max_sequence, max_sequence, true, 0.0, false, false));
        } else {
          attended = std::get<0>(at::_flash_attention_forward(
            q, k, v, cumulative, cumulative, max_sequence, max_sequence,
            0.0, false, false, std::nullopt, -1, -1));
        }
      } else {
        // CPU-only reference path: each actual sequence uses SDPA at its exact length.
        // The GPU path never falls back to this loop or to dense padded attention.
        std::vector<at::Tensor> sequences;
        for (int64_t row = 0; row < batch; ++row) {
          auto slice = [&](const at::Tensor& t) {
            return t.narrow(0, ends[row], ends[row + 1] - ends[row]).transpose(0, 1).unsqueeze(0);
          };
          sequences.push_back(at::scaled_dot_product_attention(slice(q), slice(k), slice(v),
            std::nullopt, 0.0, false).squeeze(0).transpose(0, 1));
        }
        attended = at::cat(sequences, 0);
      }
      attended = attended.contiguous().view({tokens, config.hidden});
      h = residual_norm(linear(attended, p + "attention.output.dense"), h, p + "attention.output.LayerNorm");
      auto mid = activated_linear(h, p + "intermediate.dense");
      h = residual_norm(linear(mid, p + "output.dense"), h, p + "output.LayerNorm");
    }
    return h;
  }
};
struct Engine {
  TeiConfig config;
  c10::Device device;
  at::ScalarType dtype;
  tei::Options options;
  tei::Weights weights;
  std::unique_ptr<tei::Model> model;
  std::unique_ptr<tei::DenseChain> dense;
  std::unique_ptr<tei::DecisionHead> decision;
  std::unique_ptr<tei::CudaGraphCache> graphs;
  bool media_graphs=false;
  Engine(TeiConfig cfg, c10::Device dev, at::ScalarType type) : config(cfg), device(dev), dtype(type) {}
  void ready() {
    dense = std::make_unique<tei::DenseChain>(options, weights);
    auto family = options.string("model_type", "bert");
    auto decision_kind=options.string("_decision.kind");
    if(!decision_kind.empty()) {
      tei::Options head_config;tei::Weights head_weights;
      constexpr std::string_view config_prefix="_decision.config.",weight_prefix="__tei_decision.";
      for(const auto& [key,value]:options.values)if(key.starts_with(config_prefix))head_config.values[key.substr(config_prefix.size())]=value;
      for(auto it=weights.begin();it!=weights.end();) {
        if(it->first.starts_with(weight_prefix)){head_weights[it->first.substr(weight_prefix.size())]=it->second;it=weights.erase(it);}else ++it;
      }
      decision=tei::create_decision_head(decision_kind,head_config,options,decision_kind=="laya"?weights:head_weights,weights);
      if(decision_kind=="laya") {
        TORCH_CHECK(family=="modernbert","Laya requires a ModernBERT encoder");tei::Weights encoder;
        for(const auto& [key,value]:weights)if(key.starts_with("encoder."))encoder[key.substr(8)]=value;
        TORCH_CHECK(!encoder.empty(),"Laya checkpoint contains no encoder.* weights");weights=std::move(encoder);
      }
    }
    model = tei::create_encoder(options, weights, device, dtype);
    if (!model) model = tei::create_decoder(options, weights, device, dtype);
    if (!model) model = tei::create_gemma(options, weights, device, dtype);
    if (!model) model = tei::create_multimodal(options, weights, device, dtype);
    if (!model && (family == "bert" || family == "roberta" || family == "xlm-roberta" || family == "camembert")) {
      auto bert = std::make_unique<Bert>(config, device, dtype);
      bert->options = options;
      bert->weights = std::move(weights);
      model = std::move(bert);
    }
    TORCH_CHECK(model, "Unsupported native Torch model family: ", family);
    model->ready();
    TORCH_CHECK(model->output_width() > 0, "Model must declare a positive output width");
    dense->output_width(options.string("_pooling") == "splade" ? config.vocab : model->output_width());
    media_graphs=options.boolean("_media_cuda_graphs",false);
    if (device.is_cuda() && options.boolean("_cuda_graphs", false)
        && options.integer("num_experts", options.integer("text_config.num_experts", 0)) == 0
        && options.integer("moe_every_n_layers", options.integer("text_config.moe_every_n_layers", 0)) == 0
        && !options.boolean("enable_moe_block", options.boolean("text_config.enable_moe_block", false))
        && (family == "bert" || family == "roberta"
        || family == "xlm-roberta" || family == "camembert" || family == "distilbert" || family == "modernbert"
        || family == "gte" || family == "nomic_bert" || family == "llama" || family == "mistral"
        || family == "qwen2" || family == "qwen3" || family == "gemma3" || family == "gemma3_text"
        || family == "gemma4" || family == "gemma4_text" || family == "embedding_gemma2"
        || (media_graphs && (family == "qwen3_vl" || family == "qwen3_vl_text" || family == "qwen3_5" || family == "qwen3_5_text")))) graphs = std::make_unique<tei::CudaGraphCache>(4, options.integer("_cuda_graph_max_tokens", 4096));
  }
};
tei::PackedInput packed_input(Engine& model,const int64_t* ids,const int64_t* types,
 const int64_t* positions,const int32_t* cumulative,int64_t batch,int64_t max_sequence,
 const int64_t* media_positions,const TeiImage* images,size_t image_count,const TeiAudio* audios,size_t audio_count) {
 TORCH_CHECK(batch>0&&cumulative&&cumulative[0]==0,"Invalid native packed batch");
 for(int64_t i=0;i<batch;++i)TORCH_CHECK(cumulative[i+1]>cumulative[i],"Native packed sequences must be nonempty");
 auto options=at::TensorOptions().dtype(at::kLong).device(at::kCPU);const int64_t tokens=cumulative[batch];
 auto input=[&](const int64_t* data){TORCH_CHECK(data,"Missing native token buffer");return at::from_blob(const_cast<int64_t*>(data),{tokens},options).to(model.device);};
 auto cu=at::from_blob(const_cast<int32_t*>(cumulative),{batch+1},options.dtype(at::kInt)).to(model.device);
 tei::PackedInput packed{input(ids),input(types),input(positions),cu,cumulative,batch,max_sequence};
 TORCH_CHECK(!image_count||model.model->supports_images(),"This native model does not yet support images");
 TORCH_CHECK(!audio_count||model.model->supports_audio(),"This native model does not yet support audio");
 if(media_positions)packed.multimodal_positions=at::from_blob(const_cast<int64_t*>(media_positions),{3,tokens},options).to(model.device);
 for(size_t i=0;i<image_count;++i){const auto& item=images[i];
  const auto image_dtype=model.options.string("model_type").starts_with("qwen3_5")?at::kHalf:model.dtype;
  auto pixels=at::from_blob(const_cast<float*>(item.pixels),{item.rows,item.patch_dim},options.dtype(at::kFloat)).to(model.device,image_dtype);
  packed.images.push_back({pixels,{item.grid[0],item.grid[1],item.grid[2]},item.merge_size,item.token_start,item.token_count,item.sequence_start});
 }
 for(size_t i=0;i<audio_count;++i){const auto& item=audios[i];
  auto features=at::from_blob(const_cast<float*>(item.values),{item.frames,item.feature_size},options.dtype(at::kFloat)).to(model.device);
  auto validity=at::from_blob(const_cast<uint8_t*>(item.mask),{item.frames},options.dtype(at::kByte)).to(model.device);
  packed.audios.push_back({features,validity,item.token_start,item.token_count});
 }return packed;
}
std::vector<tei::DecisionRequest> decision_requests(const TeiDecision* requests,size_t count) {
 TORCH_CHECK(requests||!count,"Missing decision metadata buffer");std::vector<tei::DecisionRequest> result;result.reserve(count);
 for(size_t i=0;i<count;++i){const auto& request=requests[i];TORCH_CHECK(request.kind>=0&&request.kind<=3,"Invalid decision metadata kind");
  tei::DecisionRequest converted{static_cast<tei::DecisionRequest::Kind>(request.kind)};converted.question_type=request.question_type;
  TORCH_CHECK((!request.marker_count||request.markers)&&(!request.token_count||request.token_ids)&&(!request.field_count||request.fields),"Missing decision metadata values");
  if(request.marker_count)converted.markers.assign(request.markers,request.markers+request.marker_count);
  if(request.token_count)converted.token_ids.assign(request.token_ids,request.token_ids+request.token_count);
  for(size_t j=0;j<request.field_count;++j){const auto& field=request.fields[j];TORCH_CHECK(!field.option_count||field.options,"Missing Clef option spans");tei::DecisionField f{field.kind,{field.question_start,field.question_end},{}};
   for(size_t k=0;k<field.option_count;++k)f.options.emplace_back(field.options[2*k],field.options[2*k+1]);converted.fields.push_back(std::move(f));
  }result.push_back(std::move(converted));
 }return result;
}
template<class F> int32_t checked(F&& f) {
  try { c10::InferenceMode guard; f(); return 0; }
  catch (const std::exception& ex) { error = ex.what(); return -1; }
  catch (...) { error = "Unknown native exception"; return -1; }
}
}
extern "C" {
const char* tei_error() { return error.c_str(); }
int64_t tei_device_count(const char* name) {
  int64_t count = -1;
  checked([&] {
    auto dev = device(name);
    count = c10::impl::getDeviceGuardImpl(dev.type())->deviceCount();
  });
  return count;
}
void* tei_create(const TeiConfig* cfg, const char* name, int32_t dtype) {
  Engine* model = nullptr;
  checked([&] { initialize_precision(); model = new Engine(*cfg, device(name), scalar(dtype)); });
  return model;
}
int32_t tei_weight(void* handle, const char* name, const uint8_t* data,
                   const int64_t* shape, size_t rank, int32_t dtype) {
  return checked([&] {
    auto& model = *static_cast<Engine*>(handle);
    auto tensor = at::empty(at::IntArrayRef(shape, rank), at::TensorOptions().dtype(scalar(dtype)).device(at::kCPU));
    std::memcpy(tensor.data_ptr(), data, tensor.nbytes());
    const std::string key(name);
    // DeltaNet decay parameters are FP32 in Candle and in the checkpoint's compute contract.
    const bool fp32_audio_filter = key.find("audio_tower.") != std::string::npos
      && ((key.find("subsample_conv_projection.") != std::string::npos && key.ends_with(".conv.weight"))
        || key.ends_with(".lconv1d.depthwise_conv1d.weight"));
    auto target_dtype = key.ends_with(".A_log") || key.ends_with(".dt_bias") || fp32_audio_filter ? at::kFloat : model.dtype;
    const auto family = model.options.string("model_type");
    if (key.starts_with("model.visual.") && family.starts_with("qwen3_5")) target_dtype = at::kHalf;
    if(key.starts_with("__tei_decision.")&&model.options.string("_decision.kind")=="pplx")target_dtype=at::kBFloat16;
    TORCH_CHECK(model.weights.emplace(name, tensor.to(model.device, target_dtype)).second,
                "Duplicate weight: ", name);
  });
}
int32_t tei_option(void* handle, const char* key, const char* value) {
  return checked([&] { static_cast<Engine*>(handle)->options.values[key] = value; });
}
int64_t tei_output_width(void* handle) {
  int64_t width = -1;
  checked([&] { width = static_cast<Engine*>(handle)->model->output_width(); });
  return width;
}
int64_t tei_graph_count(void* handle) {
  const auto& engine = *static_cast<Engine*>(handle);
  return engine.graphs ? engine.graphs->size() : 0;
}
int64_t tei_classification_width(void* handle) {
  int64_t width = -1;
  checked([&] { width = static_cast<Engine*>(handle)->model->classification_width(); });
  return width;
}
int64_t tei_pooled_width(void* handle) {
  int64_t width = -1;
  checked([&] {
    const auto& engine = *static_cast<Engine*>(handle);
    width = engine.dense->output_width(engine.options.string("_pooling") == "splade" ? engine.config.vocab : engine.model->output_width());
  });
  return width;
}
int32_t tei_ready(void* handle) { return checked([&] { static_cast<Engine*>(handle)->ready(); }); }
int32_t tei_decision_counts(void* handle,const TeiDecision* requests,size_t count,int64_t* counts) {
 return checked([&]{auto& engine=*static_cast<Engine*>(handle);TORCH_CHECK(engine.decision,"Model was not loaded for typed decisions");
  auto converted=decision_requests(requests,count);TORCH_CHECK(counts||!count,"Missing decision counts buffer");
  for(size_t i=0;i<count;++i)counts[i]=engine.decision->output_count(converted[i]);
 });
}
int32_t tei_decide(void* handle,const int64_t* ids,const int64_t* types,const int64_t* positions,
 const int32_t* cumulative,int64_t batch,int64_t max_sequence,const TeiDecision* requests,
 float* logits,size_t capacity,float* actions,const int64_t* media_positions,
 const TeiImage* images,size_t image_count,const TeiAudio* audios,size_t audio_count) {
 return checked([&]{auto& engine=*static_cast<Engine*>(handle);TORCH_CHECK(engine.decision&&batch>0,"Model was not loaded for typed decisions");
  auto converted=decision_requests(requests,batch);size_t expected=0;
  for(const auto& request:converted){auto count=engine.decision->output_count(request);TORCH_CHECK(count>0&&static_cast<size_t>(count)<=capacity-expected,"Decision output exceeds provided capacity");expected+=count;}
  TORCH_CHECK((logits||!expected)&&actions,"Missing decision output buffer");
  auto packed=packed_input(engine,ids,types,positions,cumulative,batch,max_sequence,media_positions,images,image_count,audios,audio_count);
  auto hidden=engine.graphs&&(engine.media_graphs||(!image_count&&!audio_count&&!media_positions))?engine.graphs->forward(*engine.model,packed):engine.model->forward(packed);
  auto results=engine.decision->forward(hidden,packed,converted);TORCH_CHECK(results.size()==size_t(batch),"Decision head output count mismatch");
  std::vector<at::Tensor> values;for(size_t i=0;i<results.size();++i){TORCH_CHECK(results[i].logits.numel()==engine.decision->output_count(converted[i]),"Decision head output width mismatch");values.push_back(results[i].logits.flatten());}
  std::vector<at::Tensor> action_values;for(const auto& result:results)action_values.push_back(result.action_probability);
  values.push_back(at::stack(action_values));auto output=at::cat(values).to(at::kCPU,at::kFloat).contiguous();
  std::memcpy(logits,output.data_ptr<float>(),expected*sizeof(float));std::memcpy(actions,output.data_ptr<float>()+expected,batch*sizeof(float));
 });
}
int32_t tei_forward(void* handle, const int64_t* ids, const int64_t* types,
                    const int64_t* positions, const int32_t* cumulative,
                    int64_t batch, int64_t max_sequence, int32_t pool,
                    const int64_t* pooled, size_t pooled_count,
                    const int64_t* raw, size_t raw_count, float* output, size_t capacity,
                    const int64_t* media_positions, const TeiImage* images, size_t image_count,
                    const TeiAudio* audios, size_t audio_count) {
  return checked([&] {
    auto& model = *static_cast<Engine*>(handle);
    auto packed=packed_input(model,ids,types,positions,cumulative,batch,max_sequence,
      media_positions,images,image_count,audios,audio_count);
    auto h = pool >= 4 ? model.model->predict(packed, pool == 5)
      : model.graphs && (model.media_graphs || (!image_count && !audio_count && !media_positions))
        ? model.graphs->forward(*model.model, packed) : model.model->forward(packed);
    std::vector<at::Tensor> outputs;
    bool mean_pooled=false;
#ifdef TEI_TORCH_CUDA_KERNELS
    if(pool==1&&pooled_count&&h.is_cuda()&&h.is_contiguous()
       &&(h.scalar_type()==at::kHalf||h.scalar_type()==at::kBFloat16)&&pooled_count<=65535) {
      std::vector<int32_t> spans;spans.reserve(pooled_count*2);
      for(size_t i=0;i<pooled_count;++i) {
        spans.push_back(cumulative[pooled[i]]);
        spans.push_back(cumulative[pooled[i]+1]-cumulative[pooled[i]]);
      }
      auto host=at::from_blob(spans.data(),{int64_t(pooled_count),2},at::TensorOptions().dtype(at::kInt));
      outputs.push_back(tei::packed_mean_pool(h,host.to(h.device())));
      mean_pooled=true;
    }
#endif
    for (size_t i = 0; i < pooled_count && !mean_pooled; ++i) {
      const auto row = pooled[i];
      const auto start = cumulative[row], length = cumulative[row + 1] - start;
      if (pool == 4) outputs.push_back(h.narrow(0, row, 1));
      else if (pool == 3) outputs.push_back(std::get<0>(model.model->token_scores(h.narrow(0, start, length)).max(0, true)));
      else if (pool == 0) outputs.push_back(h.narrow(0, start, 1));
      else if (pool == 1) outputs.push_back(h.narrow(0, start, length).mean(0, true, at::kFloat));
      else outputs.push_back(h.narrow(0, start + length - 1, 1));
    }
    if (pool < 4 && !model.dense->empty() && !outputs.empty()) {
      auto pooled_output = model.dense->forward(at::cat(outputs, 0));
      outputs.clear();
      outputs.push_back(pooled_output);
    }
    for (size_t i = 0; i < raw_count; ++i)
      outputs.push_back(h.narrow(0, cumulative[raw[i]], cumulative[raw[i] + 1] - cumulative[raw[i]]));
    if (!outputs.empty()) {
      size_t offset = 0;
      std::vector<at::Tensor> group;
      auto flush = [&] {
        if (group.empty()) return;
        auto cpu = at::cat(group, 0).to(at::kCPU, at::kFloat).contiguous();
        TORCH_CHECK(static_cast<size_t>(cpu.numel()) <= capacity - offset, "Native output exceeds provided capacity");
        std::memcpy(output + offset, cpu.data_ptr<float>(), cpu.nbytes());
        offset += cpu.numel();
        group.clear();
      };
      for (const auto& tensor : outputs) {
        if (!group.empty() && group.back().size(-1) != tensor.size(-1)) flush();
        group.push_back(tensor);
      }
      flush();
    }
  });
}
void tei_destroy(void* handle) { delete static_cast<Engine*>(handle); }
}

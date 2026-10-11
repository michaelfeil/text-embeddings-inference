#include "helpers/attention.h"
#include "models/encoder_models.h"
#include "helpers/fp8_linear.h"
#ifdef TEI_TORCH_CUDA_KERNELS
#include "kernels/fast_kernels.h"
#include "kernels/encoder_norm.h"
#include "kernels/encoder_bias.h"
#include "kernels/encoder_activation.h"
#include "batching/packed_pool.h"
#include "kernels/torch_flash_fma.h"
#endif
#include <ATen/ops/_flash_attention_forward.h>
#include <ATen/ops/_addmm_activation.h>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace tei {
namespace {
enum class Family { Distil, Modern, Nomic, Gte, Jina, JinaCode };

struct Encoder final : Model {
  Options o;
  Weights& weights;
  c10::Device device;
  at::ScalarType dtype;
  Family family;
  int64_t hidden, heads, layers;
  double eps;
  std::string prefix, activation;
  bool exact_norm = false;
  bool exact_moe = false;
  struct Layer {
    at::Tensor qkv_weight, qkv_bias;
    at::Tensor gate_weight, gate_bias;
  };
  std::vector<Layer> fused;
  std::unordered_map<std::string,Fp8Linear> fp8_mlp;
  std::shared_ptr<Fp8RowwiseExecutor> fp8_executor;
  std::shared_ptr<OrdinaryLinearExecutor> ordinary_executor;
  at::Tensor global_cos, global_sin, local_cos, local_sin;
  std::optional<at::Tensor> alibi;

  Encoder(const Options& opt, Weights& w, c10::Device dev, at::ScalarType dt, Family f)
    : o(opt), weights(w), device(dev), dtype(dt), family(f) {
    const bool distil = f == Family::Distil, nomic = f == Family::Nomic;
    hidden = o.integer(distil ? "dim" : nomic ? "n_embd" : "hidden_size", 768);
    heads = o.integer(distil ? "n_heads" : nomic ? "n_head" : "num_attention_heads", 12);
    layers = o.integer(distil ? "n_layers" : nomic ? "n_layer" : "num_hidden_layers", 12);
    eps = o.real(distil ? "unused" : nomic ? "layer_norm_epsilon" : f == Family::Modern ? "norm_eps" : "layer_norm_eps", 1e-12);
    activation = o.string(distil ? "activation" : nomic ? "activation_function" : f == Family::Modern ? "hidden_activation" : "hidden_act", "gelu");
    // Candle HiddenAct deserializes this alias as Gelu. Normalize before the
    // Nomic dense/gated checkpoint-layout selection and fused dispatch checks.
    if (nomic && activation == "gelu_pytorch_tanh") activation = "gelu";
    exact_moe=nomic&&o.integer("num_experts",0)>0;
    const char* norm_mode=std::getenv("TEI_TORCH_EXACT_ENCODER_NORM");
    // Preserve Candle normalization and rounded residuals for deep rotary encoders.
    // An explicit zero opts into the ATen diagnostic path.
    exact_norm=f==Family::Modern && (!norm_mode||std::string(norm_mode)=="1"||std::string(norm_mode)=="true");
    TORCH_CHECK(!o.boolean("_fp8_dynamic",false) || family==Family::Modern,
      "Dynamic FP8 is unsupported for this encoder family");
  }
  int64_t output_width() const override { return hidden; }
  int64_t classification_width() const override {
    if (family != Family::Modern && family != Family::Gte && family != Family::Jina) return 0;
    const auto it = weights.find("classifier.weight");
    return it == weights.end() ? 0 : it->second.size(0);
  }
  bool has(const std::string& n) const { return weights.count(prefix + n); }
  const at::Tensor& w(const std::string& n) const {
    auto it = weights.find(prefix + n);
    TORCH_CHECK(it != weights.end(), "Missing encoder weight: ", prefix, n);
    return it->second;
  }
  at::Tensor bias(const std::string& n) const { return has(n + ".bias") ? w(n + ".bias") : at::Tensor(); }
  at::Tensor linear(const at::Tensor& x, const std::string& n) const {
    auto quantized=fp8_mlp.find(n);
    if(quantized!=fp8_mlp.end())return quantized->second.forward(x,bias(n));
    if(ordinary_executor && !bias(n).defined()) {
      auto result=ordinary_executor->forward(x,w(n+".weight"));
      if(result.defined())return result;
    }
    return at::linear(x, w(n + ".weight"), bias(n));
  }
  at::Tensor activated_linear(const at::Tensor& x, const std::string& n) const {
    const auto b = bias(n);
    if (x.is_cuda() && b.defined() && (activation == "gelu" || activation == "relu")) {
      // Candle MlpLinear uses the cuBLASLt bias/activation epilogue for these heads.
      return at::_addmm_activation(b, x, w(n + ".weight").t(), 1, 1, activation == "gelu");
    }
    return act(linear(x, n));
  }
  at::Tensor norm(const at::Tensor& x, const std::string& n) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    if(exact_moe&&x.is_cuda())return encoder_layer_norm(x,w(n+".weight"),eps,at::Tensor(),bias(n),false).first;
    if(exact_norm&&family==Family::Modern&&x.is_cuda())return encoder_layer_norm(x,w(n+".weight"),eps).first;
#endif
    return at::layer_norm(x, {hidden}, w(n + (has(n + ".weight") ? ".weight" : ".gamma")), family == Family::Modern ? at::Tensor() : bias(n), eps);
  }
  at::Tensor residual_norm(const at::Tensor& x, const at::Tensor& residual, const std::string& n) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    if(exact_moe&&x.is_cuda())return encoder_layer_norm(x,w(n+".weight"),eps,residual,bias(n),false).first;
    if (x.is_cuda()) return fused_add_layer_norm(x, residual,
      w(n + (has(n + ".weight") ? ".weight" : ".gamma")),
      family == Family::Modern ? at::Tensor() : bias(n), eps);
#endif
    return norm(x + residual, n);
  }
  at::Tensor act(const at::Tensor& x) const {
    if (activation == "relu") return at::relu(x);
    if (activation == "silu" || activation == "swiglu") return at::silu(x);
    if (activation == "tanh") return at::tanh(x);
    TORCH_CHECK(activation == "gelu" || activation == "gelu_new" || activation == "gelu_fast" || activation == "gelu_pytorch_tanh" || activation == "gelu_approx", "Unsupported encoder activation: ", activation);
    // Candle HiddenAct::Gelu and ModernBertActivation::Gelu both use tanh GELU.
    return at::gelu(x, "tanh");
  }
  at::Tensor gated(const at::Tensor& x, bool gate_first=true) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    if(x.is_cuda() && activation!="tanh") return fused_gated_activation(x,
      activation=="relu"?3:(activation=="silu"||activation=="swiglu")?0:4,gate_first);
#endif
    const auto width=x.size(-1)/2;
    auto first=x.narrow(-1,0,width),second=x.narrow(-1,width,width);
    return gate_first?act(first)*second:first*act(second);
  }
  at::Tensor modern_mlp_output(const at::Tensor& middle,const std::string& p) const {
    const auto name=p+"mlp.Wo";
#ifdef TEI_TORCH_CUDA_KERNELS
    const bool gelu=activation!="relu" && activation!="silu" && activation!="swiglu" && activation!="tanh";
    if(family==Family::Modern && gelu && middle.is_cuda() &&
      (middle.scalar_type()==at::kHalf || middle.scalar_type()==at::kBFloat16) && !fp8_mlp.empty()) {
      auto projection=fp8_mlp.find(name);
      if(projection!=fp8_mlp.end() && projection->second.can_fuse_gated_quantization()) {
        auto quantized=fp8_quantize_gated_rows(middle);
        return projection->second.forward_quantized(quantized.first,quantized.second,bias(name));
      }
    }
#endif
    return linear(gated(middle),name);
  }
  std::string layer_name(int64_t i) const {
    if (family == Family::Distil) return (has("transformer.layer.0.attention.q_lin.weight") ? "transformer.layer." : "encoder.layer.") + std::to_string(i) + ".";
    if (family == Family::Modern) return "layers." + std::to_string(i) + ".";
    if (family == Family::Nomic) return "encoder.layers." + std::to_string(i) + ".";
    return "encoder.layer." + std::to_string(i) + ".";
  }
  void rotary_cache(double theta, at::Tensor& cosine, at::Tensor& sine) {
    const int64_t d = hidden / heads;
    const int64_t length = o.integer(family == Family::Nomic ? "n_positions" : "max_position_embeddings", 8192);
    auto fp = at::TensorOptions().device(device).dtype(at::kFloat);
    // Candle constructs base inverse frequencies on CPU with float powf.
    // An exp/log rewrite differs enough after half conversion to accumulate
    // measurable drift through deep rotary encoders.
    std::vector<float> frequencies;
    for (int64_t i = 0; i < d; i += 2)
      frequencies.push_back(1.f / std::pow(static_cast<float>(theta), static_cast<float>(i) / d));
    auto inv = at::from_blob(frequencies.data(), {d / 2}, at::TensorOptions().dtype(at::kFloat)).to(device).clone();
    if (family == Family::Gte && o.values.count("rope_scaling.factor")) {
      const auto factor = o.real("rope_scaling.factor", 1.0);
      TORCH_CHECK(factor > 0, "GTE rope scaling factor must be positive");
      if (o.values.count("rope_scaling.high_freq_factor")) {
        const auto high = o.real("rope_scaling.high_freq_factor", 4), low = o.real("rope_scaling.low_freq_factor", 1);
        const auto original = o.real("rope_scaling.original_max_position_embeddings", 8192);
        TORCH_CHECK(high > low && low > 0 && original > 0, "Invalid Llama3 rotary scaling");
        auto wavelength = (2.0 * M_PI) / inv;
        auto smooth = (original / wavelength - low) / (high - low);
        auto mixed = (1.0 - smooth) * inv / factor + smooth * inv;
        inv = at::where(wavelength < original / high, inv, at::where(wavelength > original / low, inv / factor, mixed));
      } else {
        inv = at::exp(at::arange(0, d, 2, fp) * (-std::log(theta * factor) / d)) / std::pow(factor, 2.0 / d);
      }
    }
    auto phase = at::arange(length, fp).unsqueeze(1) * inv.unsqueeze(0);
    cosine = at::cos(phase).to(dtype);
    sine = at::sin(phase).to(dtype);
  }
  void ready() override {
    ordinary_executor.reset();
    if(family==Family::Modern && device.is_cuda() && (dtype==at::kHalf || dtype==at::kBFloat16) &&
      hidden==768 && heads==12 && o.integer("intermediate_size",1152)==1152 &&
      !o.boolean("_fp8_dynamic",false))
      ordinary_executor=std::make_shared<OrdinaryLinearExecutor>();
    TORCH_CHECK(hidden > 0 && heads > 0 && hidden % heads == 0 && layers > 0, "Invalid encoder dimensions");
    TORCH_CHECK(activation=="gelu"||activation=="gelu_new"||activation=="gelu_fast"||activation=="gelu_pytorch_tanh"||activation=="gelu_approx"||activation=="relu"||activation=="silu"||activation=="swiglu"||activation=="tanh", "Unsupported encoder activation: ",activation);
    TORCH_CHECK(device.is_cpu() || device.is_cuda(), "Packed encoders currently support CPU and CUDA");
    if (device.is_cuda()) TORCH_CHECK((dtype == at::kHalf || dtype == at::kBFloat16) && (hidden / heads) % 8 == 0 && hidden / heads <= 256, "CUDA varlen requires half/bfloat16 and head dimension divisible by eight, at most 256");
    if (family == Family::Distil) prefix = weights.count("distilbert.embeddings.word_embeddings.weight") ? "distilbert." : "";
    if (family == Family::Modern) prefix = weights.count("model.embeddings.tok_embeddings.weight") ? "model." : "";
    if (family == Family::Gte) prefix = weights.count("new.embeddings.word_embeddings.weight") ? "new." : "";
    if (family == Family::Jina || family == Family::JinaCode) {
      prefix = weights.count("bert.embeddings.word_embeddings.weight") ? "bert." : "";
      const auto position = o.string("position_embedding_type", "absolute");
      TORCH_CHECK(position == "absolute" || position == "alibi", "Unsupported Jina position embedding: ", position);
      if (position == "alibi") {
        std::vector<float> slopes;
        const int64_t power = int64_t{1} << static_cast<int>(std::floor(std::log2(heads)));
        const double start = std::pow(2., -std::pow(2., -(std::log2(power) - 3)));
        for (int64_t h = 0; h < power; ++h) slopes.push_back(std::pow(start, h + 1));
        const double extra = std::pow(2., -std::pow(2., -(std::log2(2 * power) - 3)));
        for (int64_t h = 0; h < heads - power; ++h) slopes.push_back(std::pow(extra, 2 * h + 1));
        alibi = at::from_blob(slopes.data(), {heads}, at::TensorOptions().dtype(at::kFloat)).to(device).clone();
      }
    }
    if (family == Family::Nomic) {
      TORCH_CHECK(!o.boolean("prenorm", false) && o.real("rotary_emb_fraction", 1.0) == 1.0 && !o.boolean("rotary_emb_interleaved", false), "Nomic supports postnorm, full noninterleaved rotary (same as Candle)");
      if (o.integer("moe_every_n_layers", 0) > 0) {
        TORCH_CHECK(o.integer("num_experts", 0) > 0 && o.integer("moe_top_k", 0) > 0 && o.integer("moe_top_k", 0) <= o.integer("num_experts", 0), "Invalid Nomic MoE expert count/top-k");
      }
      rotary_cache(o.real("rotary_emb_base", 1000), global_cos, global_sin);
      if (o.values.count("rotary_scaling_factor")) {
        const double factor = o.real("rotary_scaling_factor", 1), d = hidden / heads;
        const double base = std::pow(o.real("rotary_emb_base", 1000) * (factor * o.integer("n_positions", 8192) / o.integer("max_trained_positions", 2048) - (factor - 1)), static_cast<int>(d / (d - 2)));
        rotary_cache(base, local_cos, local_sin);
      }
    }
    if (family == Family::Modern) {
      TORCH_CHECK(o.integer("global_attn_every_n_layers", 3) > 0, "ModernBERT global attention interval must be positive");
      rotary_cache(o.real("global_rope_theta", 160000), global_cos, global_sin);
      rotary_cache(o.real("local_rope_theta", 10000), local_cos, local_sin);
    }
    if (family == Family::Gte) {
      TORCH_CHECK(o.string("position_embedding_type", "rope") == "rope", "GTE packed model supports rotary positions only");
      TORCH_CHECK(!o.boolean("logn_attention_scale", false) && !o.boolean("logn_attention_clip1", false), "GTE logn attention is not supported by Candle packed model");
      rotary_cache(o.real("rope_theta", 10000), global_cos, global_sin);
    }
    w(family == Family::Modern ? "embeddings.tok_embeddings.weight" : "embeddings.word_embeddings.weight");
    if(family==Family::Modern && o.boolean("_fp8_dynamic",false))
      fp8_executor=std::make_shared<Fp8RowwiseExecutor>();
    fused.resize(layers);
    for (int64_t i = 0; i < layers; ++i) {
      const auto p = layer_name(i);
      if (family == Family::Distil || family == Family::Jina || family == Family::JinaCode) {
        const auto a = p + (family == Family::Distil ? "attention." : "attention.self.");
        const std::string q = a + (family == Family::Distil ? "q_lin" : "query"), k = a + (family == Family::Distil ? "k_lin" : "key"), v = a + (family == Family::Distil ? "v_lin" : "value");
        fused[i].qkv_weight = at::cat({w(q + ".weight"), w(k + ".weight"), w(v + ".weight")}, 0);
        if (has(q + ".bias")) fused[i].qkv_bias = at::cat({w(q + ".bias"), w(k + ".bias"), w(v + ".bias")}, 0);
      } else {
        const auto a = p + (family == Family::Modern ? "attn.Wqkv" : family == Family::Nomic ? "attn.Wqkv" : "attention.qkv_proj");
        fused[i].qkv_weight = w(a + ".weight");
        fused[i].qkv_bias = bias(a);
      }
      if(family==Family::Nomic && activation!="gelu" && !(o.integer("moe_every_n_layers",0)>0 && i%o.integer("moe_every_n_layers",1)==1)) {
        const auto gate=p+"mlp.fc12",up=p+"mlp.fc11";
        fused[i].gate_weight=at::cat({w(gate+".weight"),w(up+".weight")},0);
        if(has(gate+".bias"))fused[i].gate_bias=at::cat({w(gate+".bias"),w(up+".bias")},0);
      }
      if(o.boolean("_fp8_dynamic",false)) {
        for(const auto& name:{p+"mlp.Wi",p+"mlp.Wo"}) {
          fp8_mlp.try_emplace(name,w(name+".weight"),family==Family::Modern,fp8_executor,name==p+"mlp.Wo");
          weights.erase(prefix+name+".weight");
        }
      }
    }
  }
  at::Tensor rotate(const at::Tensor& x, const at::Tensor& c, const at::Tensor& s) const {
    const auto d = x.size(-1) / 2;
    auto left = x.narrow(-1, 0, d), right = x.narrow(-1, d, d);
    return at::cat({left * c - right * s, right * c + left * s}, -1);
  }
  at::Tensor nomic_moe(const at::Tensor& h, const std::string& p) const {
    // Route packed tokens without expanding or padding any sequence. Only the
    // selected token rows are sent to each expert, matching Candle's dispatch.
    auto logits = linear(h, p + "mlp.router.layer").to(at::kFloat);
    auto top = at::softmax(logits, -1).topk(o.integer("moe_top_k", 2), -1);
    auto scores = std::get<0>(top).to(dtype), experts = std::get<1>(top);
    const auto count = o.integer("num_experts", 0), width = o.integer("n_inner", 0);
    auto first = w(p + "mlp.experts.mlp.w1").view({count, width, hidden});
    auto second = w(p + "mlp.experts.mlp.w2").view({count, width, hidden});
    auto output = at::zeros_like(h);
    for (int64_t e = 0; e < count; ++e) {
      auto matches = at::nonzero(experts == e);
      if (!matches.size(0)) continue;
      auto rows = matches.select(1, 0), slots = matches.select(1, 1);
      auto expert_hidden=at::linear(h.index_select(0,rows),first[e]);
      at::Tensor expert_act;
#ifdef TEI_TORCH_CUDA_KERNELS
      if(expert_hidden.is_cuda()&&activation=="gelu")expert_act=encoder_gelu(expert_hidden);
      else
#endif
      expert_act=act(expert_hidden);
      auto contribution=at::matmul(expert_act,second[e]);
      auto selected = scores.index_select(0, rows).gather(1, slots.unsqueeze(1));
      output.index_add_(0, rows, contribution * selected);
    }
    return output + w(p + "mlp.experts.bias");
  }
  at::Tensor attend(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const PackedInput& input, int64_t window = -1, const std::optional<at::Tensor>& bias = std::nullopt) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    if((family==Family::Modern||exact_moe) && !bias && torch_flash_fma_supported(q,k,v))
      return torch_flash_fma(q,k,v,input.cumulative,input.max_sequence,
        1.0/std::sqrt(static_cast<double>(q.size(-1))),false,window,window);
#endif
    return packed_attention(q, k, v, input, 1.0 / std::sqrt(hidden / heads), false, window, window, alibi, bias);
  }
  at::Tensor forward(const PackedInput& input) const override {
    at::Tensor h, modern_residual;
    if (family == Family::Modern) h = norm(at::embedding(w("embeddings.tok_embeddings.weight"), input.ids), "embeddings.norm");
    else {
      h = at::embedding(w("embeddings.word_embeddings.weight"), input.ids);
      if (has("embeddings.token_type_embeddings.weight")) h = h + at::embedding(w("embeddings.token_type_embeddings.weight"), input.types);
      if (family == Family::Distil || ((family == Family::Jina || family == Family::JinaCode) && !alibi)) h = h + at::embedding(w("embeddings.position_embeddings.weight"), input.positions);
      h = norm(h, family == Family::Nomic ? "emb_ln" : "embeddings.LayerNorm");
    }
    // All Jina layers share slopes and local positions, so materialize their
    // packed bias once per request rather than once per layer.
    std::optional<at::Tensor> attention_bias;
    if(alibi && h.is_cuda())attention_bias=packed_alibi_bias(h.view({h.size(0),heads,hidden/heads}),input,*alibi);
    bool scaled = family == Family::Nomic && local_cos.defined() && input.max_sequence > o.integer("max_trained_positions", 2048);
    at::Tensor cos_global, sin_global, cos_local, sin_local;
    if (global_cos.defined()) {
      cos_global = global_cos.index_select(0, input.positions).unsqueeze(1);
      sin_global = global_sin.index_select(0, input.positions).unsqueeze(1);
    }
    if (local_cos.defined()) {
      cos_local = local_cos.index_select(0, input.positions).unsqueeze(1);
      sin_local = local_sin.index_select(0, input.positions).unsqueeze(1);
    }
    for (int64_t i = 0; i < layers; ++i) {
      const auto p = layer_name(i);
      const bool local = family == Family::Modern && i % o.integer("global_attn_every_n_layers", 3) != 0;
      auto source = h;
#ifdef TEI_TORCH_CUDA_KERNELS
      if(exact_norm&&family==Family::Modern&&h.is_cuda()) {
        if(i==0)modern_residual=h;
        else {
          auto normalized=encoder_layer_norm(h,w(p+"attn_norm.weight"),eps,modern_residual);
          source=normalized.first;modern_residual=normalized.second;
        }
      } else
#endif
      if(family==Family::Modern&&i>0)source=norm(h,p+"attn_norm");
      at::Tensor qkv;
      if(ordinary_executor && !fused[i].qkv_bias.defined())
        qkv=ordinary_executor->forward(source,fused[i].qkv_weight);
      if(!qkv.defined())qkv=at::linear(source,fused[i].qkv_weight,fused[i].qkv_bias);
      qkv=qkv.view({h.size(0),3*heads,hidden/heads});
      auto q = qkv.narrow(1, 0, heads), k = qkv.narrow(1, heads, heads), v = qkv.narrow(1, 2 * heads, heads);
      if (family == Family::Modern || family == Family::Nomic || family == Family::Gte) {
        const auto& c = local || scaled ? cos_local : cos_global;
        const auto& s = local || scaled ? sin_local : sin_global;
#ifdef TEI_TORCH_CUDA_KERNELS
        if(q.is_cuda()) {
          if(family==Family::Modern) {
            // Q/K occupy distinct head ranges in this fresh linear QKV output.
            // No later operation reads unrotated Q/K; the disjoint V view stays intact.
            encoder_rotary_inplace(q,k,c,s);
          } else {auto rotated=fused_rotary(q,k,c,s,true);q=rotated.first;k=rotated.second;}
        }
        else
#endif
        {q = rotate(q, c, s); k = rotate(k, c, s);}
      }
      if (family == Family::JinaCode) {
        q = norm(q.reshape({h.size(0), hidden}), p + "attention.self.layer_norm_q").view_as(q);
        k = norm(k.reshape({h.size(0), hidden}), p + "attention.self.layer_norm_k").view_as(k);
      }
      auto a = attend(q, k, v, input, local ? o.integer("local_attention", 128) / 2 : -1, attention_bias).contiguous().view({h.size(0), hidden});
      const auto output = p + (family == Family::Distil ? "attention.out_lin" : family == Family::Modern ? "attn.Wo" : family == Family::Nomic ? "attn.out_proj" : family == Family::Gte ? "attention.o_proj" : "attention.output.dense");
      auto residual = h;
      auto projected = linear(a, output);
      if (family == Family::Modern) {
#ifdef TEI_TORCH_CUDA_KERNELS
        if(exact_norm&&h.is_cuda()) {
          auto normalized=encoder_layer_norm(projected,w(p+"mlp_norm.weight"),eps,modern_residual);
          modern_residual=normalized.second;
          h=modern_mlp_output(linear(normalized.first,p+"mlp.Wi"),p);
        } else
#endif
        {
        h = h + projected;
        auto middle = linear(norm(h, p + "mlp_norm"), p + "mlp.Wi");
        h = h + modern_mlp_output(middle,p);
        }
      } else if (family == Family::Distil) {
        h = residual_norm(projected, h, p + "sa_layer_norm");
        h = residual_norm(linear(activated_linear(h, p + "ffn.lin1"), p + "ffn.lin2"), h, p + "output_layer_norm");
      } else if (family == Family::Nomic) {
        h = residual_norm(projected, h, p + "norm1");
        at::Tensor middle;
        const auto every = o.integer("moe_every_n_layers", 0);
        if (every > 0 && i % every == 1) h = residual_norm(nomic_moe(h, p), h, p + "norm2");
        else {
          if (activation == "gelu") middle = activated_linear(h, p + "mlp.fc1");
          else middle = gated(at::linear(h,fused[i].gate_weight,fused[i].gate_bias));
          h = residual_norm(linear(middle, p + "mlp.fc2"), h, p + "norm2");
        }
      } else if (family == Family::Gte) {
        h = residual_norm(projected, h, p + "attn_ln");
        auto middle = linear(h, p + "mlp.up_gate_proj");
        h = residual_norm(linear(gated(middle,false), p + "mlp.down_proj"), h, p + "mlp_ln");
      } else {
        h = residual_norm(projected, h, p + "attention.output.LayerNorm");
        if (family == Family::JinaCode) h = residual_norm(h, residual, p + "layer_norm_1");
        auto middle = linear(h, p + (family == Family::JinaCode ? "mlp.up_gated_layer" : "mlp.gated_layers"));
        auto activated = gated(middle,family!=Family::JinaCode);
        h = residual_norm(linear(activated, p + (family == Family::JinaCode ? "mlp.down_layer" : "mlp.wo")), h, p + (family == Family::JinaCode ? "layer_norm_2" : "mlp.layernorm"));
      }
    }
    if (family == Family::Modern) {
#ifdef TEI_TORCH_CUDA_KERNELS
      if(exact_norm&&h.is_cuda())h=encoder_layer_norm(h,w("final_norm.weight"),eps,modern_residual).first;
      else
#endif
      h = norm(h, "final_norm");
    }
    return h;
  }
  at::Tensor root_linear(const at::Tensor& h, const std::string& n) const {
    auto b = weights.find(n + ".bias");
    return at::linear(h, weight(weights, n + ".weight"), b == weights.end() ? at::Tensor() : b->second);
  }
  at::Tensor token_scores(const at::Tensor& h) const override {
    TORCH_CHECK(family == Family::Distil, "SPLADE is only supported for DistilBERT in this encoder family");
    auto transformed = act(root_linear(h, "vocab_transform"));
    transformed = at::layer_norm(transformed, {hidden}, weight(weights, "vocab_layer_norm.weight"), weight(weights, "vocab_layer_norm.bias"), 1e-12);
    const auto projector = weights.find("vocab_projector.weight");
    const auto& matrix = projector == weights.end() ? w("embeddings.word_embeddings.weight") : projector->second;
    return at::log1p(at::relu(at::linear(transformed, matrix, weight(weights, "vocab_projector.bias"))));
  }
  at::Tensor predict(const PackedInput& input, bool tokenwise) const override {
    TORCH_CHECK(family == Family::Modern || family == Family::Gte || family == Family::Jina, "This encoder does not support classification (same as Candle)");
    TORCH_CHECK(!tokenwise || family != Family::Gte, "GTE does not support token classification (same as Candle)");
    auto h = forward(input);
    if (tokenwise) return root_linear(h, "classifier");
    std::vector<at::Tensor> rows;
    const auto pool = family == Family::Modern ? o.string("classifier_pooling", "cls") : "cls";
#ifdef TEI_TORCH_CUDA_KERNELS
    if (family == Family::Modern && pool == "mean" && h.is_cuda()
        && h.is_contiguous() && (h.scalar_type() == at::kHalf || h.scalar_type() == at::kBFloat16)
        && hidden > 0 && hidden <= std::numeric_limits<uint32_t>::max()
        && input.batch > 0 && input.batch <= 65535
        && input.cumulative.defined() && input.cumulative.device() == h.device()
        && input.cumulative.scalar_type() == at::kInt && input.cumulative.is_contiguous()
        && input.cumulative.dim() == 1
        && input.cumulative.numel() == input.batch + 1) {
      // Reuse Candle's packed model-dtype sum tree and rounded reciprocal.
      // Device cumulative offsets are replay inputs, so this metadata remains
      // capture-safe when actual token contents or sequence lengths change.
      h = packed_mean_pool_cumulative(h, input.cumulative);
    } else
#endif
    {
      for (int64_t row = 0; row < input.batch; ++row) {
        const int64_t start = input.offsets[row], length = input.offsets[row + 1] - start;
        if (pool == "mean") rows.push_back(h.narrow(0, start, length).mean(0, true).to(dtype));
        else if (pool == "last_token" || pool == "last-token") rows.push_back(h.narrow(0, start + length - 1, 1));
        else {
          TORCH_CHECK(pool == "cls", "Unsupported classifier pooling: ", pool);
          rows.push_back(h.narrow(0, start, 1));
        }
      }
      h = at::cat(rows, 0);
    }
    if (family == Family::Modern) {
      // Candle intentionally omits a dense bias and normalizes with no bias.
      h = at::linear(h, weight(weights, "head.dense.weight"));
      const auto activation = o.string("classifier_activation", "gelu");
      if (activation == "relu") h = at::relu(h);
      else if (activation == "silu") h = at::silu(h);
      else { TORCH_CHECK(activation == "gelu" || activation == "gelu_pytorch_tanh", "Unsupported ModernBERT classifier activation"); h = at::gelu(h, "tanh"); }
      h = at::layer_norm(h, {hidden}, weight(weights, "head.norm.weight"), at::Tensor(), eps);
    } else {
      auto matrix_matches = [&](const std::string& key) {
        const auto matrix = weights.find(key + ".weight");
        return matrix != weights.end() && matrix->second.dim() == 2
          && matrix->second.size(0) == hidden && matrix->second.size(1) == hidden;
      };
      auto bias_matches = [&](const std::string& key) {
        const auto value = weights.find(key + ".bias");
        return value != weights.end() && value->second.dim() == 1 && value->second.numel() == hidden;
      };
      if (family == Family::Jina) {
        // Candle Jina checks bert.pooler independently of embedding prefix.
        const std::string key = "bert.pooler.dense";
        if (matrix_matches(key)) {
          TORCH_CHECK(bias_matches(key), "Invalid Jina classifier pooler bias");
          h = at::tanh(root_linear(h, key));
        }
      } else {
        // Candle GTE tries a complete new.pooler, then a complete root pooler.
        const std::string key = matrix_matches("new.pooler.dense") && bias_matches("new.pooler.dense")
          ? "new.pooler.dense" : "pooler.dense";
        if (matrix_matches(key) && bias_matches(key)) h = at::tanh(root_linear(h, key));
      }
    }
    return root_linear(h, "classifier");
  }
};

// MPNet computation follows Hugging Face Transformers modeling_mpnet.py.
// Copyright 2018 The HuggingFace Inc. team, Microsoft Corporation.
// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
// Licensed under Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0
struct Mpnet final : Model {
  Options o;
  Weights& weights;
  c10::Device device;
  at::ScalarType dtype;
  int64_t hidden, heads, layers;
  std::string prefix;
  std::vector<at::Tensor> qkv_weights, qkv_biases;
  Mpnet(const Options& options, Weights& w, c10::Device dev, at::ScalarType dt)
    : o(options), weights(w), device(dev), dtype(dt), hidden(o.integer("hidden_size",768)),
      heads(o.integer("num_attention_heads",12)), layers(o.integer("num_hidden_layers",12)) {}
  const at::Tensor& w(const std::string& key) const { return weight(weights,prefix+key); }
  at::Tensor linear(const at::Tensor& h, const std::string& n) const { return at::linear(h,w(n+".weight"),w(n+".bias")); }
  at::Tensor norm(const at::Tensor& h, const std::string& n) const { return at::layer_norm(h,{hidden},w(n+".weight"),w(n+".bias"),o.real("layer_norm_eps",1e-12)); }
  int64_t output_width() const override { return hidden; }
  void ready() override {
    TORCH_CHECK(hidden > 0 && heads > 0 && hidden%heads==0 && layers>0,"Invalid MPNet dimensions");
    TORCH_CHECK(device.is_cpu() || device.is_cuda(),"MPNet supports CPU and CUDA");
    prefix = weights.count("mpnet.embeddings.word_embeddings.weight") ? "mpnet." : "";
    w("embeddings.word_embeddings.weight");w("embeddings.position_embeddings.weight");
    TORCH_CHECK(w("encoder.relative_attention_bias.weight").size(0)>=32,"MPNet needs at least 32 relative position buckets");
    for(int64_t i=0;i<layers;++i) {
      const auto p="encoder.layer."+std::to_string(i)+".attention.attn.";
      qkv_weights.push_back(at::cat({w(p+"q.weight"),w(p+"k.weight"),w(p+"v.weight")},0));
      qkv_biases.push_back(at::cat({w(p+"q.bias"),w(p+"k.bias"),w(p+"v.bias")},0));
    }
  }
  at::Tensor position_bias(int64_t length) const {
    auto idx=at::arange(length,at::TensorOptions().device(device).dtype(at::kLong));
    auto relative=idx.unsqueeze(0)-idx.unsqueeze(1), distance=relative.abs();
    // HF MPNet's encoder compute_position_bias defaults to 32 buckets.
    auto large=(at::log(distance.clamp_min(1).to(at::kFloat)/8.)/std::log(128./8.)*8.).to(at::kLong)+8;
    auto bucket=(relative>0).to(at::kLong)*16+at::where(distance<8,distance,large.clamp_max(15));
    return at::embedding(w("encoder.relative_attention_bias.weight"),bucket).permute({2,0,1}).unsqueeze(0);
  }
  at::Tensor forward(const PackedInput& input) const override {
    auto h=norm(at::embedding(w("embeddings.word_embeddings.weight"),input.ids)
      +at::embedding(w("embeddings.position_embeddings.weight"),input.positions+2),"embeddings.LayerNorm");
    std::vector<at::Tensor> biases;
    for(int64_t row=0;row<input.batch;++row) {
      const int64_t n=input.offsets[row+1]-input.offsets[row];
      auto bias=position_bias(n);
      if(device.is_cuda()) {
        const int64_t stride=((n+7)/8)*8;
        auto aligned=at::empty_strided({1,heads,n,n},{heads*n*stride,n*stride,stride,1},h.options());
        aligned.copy_(bias);bias=aligned;
      }
      biases.push_back(bias);
    }
    for(int64_t layer=0;layer<layers;++layer) {
      const auto p="encoder.layer."+std::to_string(layer)+".";
      auto qkv=at::linear(h,qkv_weights[layer],qkv_biases[layer]).view({h.size(0),3*heads,hidden/heads});
      auto q=qkv.narrow(1,0,heads),k=qkv.narrow(1,heads,heads),v=qkv.narrow(1,2*heads,heads);
      std::vector<at::Tensor> results;
      for(int64_t row=0;row<input.batch;++row) {
        const int64_t start=input.offsets[row],n=input.offsets[row+1]-start;
        auto slice=[&](const at::Tensor& t){return t.narrow(0,start,n).unsqueeze(0);};
        if(device.is_cuda()) {
          auto cu=input.cumulative.narrow(0,row,2)-start;
          results.push_back(std::get<0>(at::_efficient_attention_forward(slice(q),slice(k),slice(v),biases[row],cu,cu,n,n,0.,0,false,1./std::sqrt(hidden/heads))).squeeze(0));
        } else {
          results.push_back(at::scaled_dot_product_attention(slice(q).transpose(1,2),slice(k).transpose(1,2),slice(v).transpose(1,2),biases[row],0.,false).squeeze(0).transpose(0,1));
        }
      }
      auto attended=at::cat(results,0).contiguous().view_as(h);
      h=norm(h+linear(attended,p+"attention.attn.o"),p+"attention.LayerNorm");
      auto intermediate=linear(h,p+"intermediate.dense");
      const auto activation=o.string("hidden_act","gelu");
      if(activation=="relu") intermediate=at::relu(intermediate);
      else if(activation=="silu") intermediate=at::silu(intermediate);
      else {TORCH_CHECK(activation=="gelu" || activation=="gelu_pytorch_tanh","Unsupported MPNet activation");intermediate=at::gelu(intermediate,activation=="gelu"?"none":"tanh");}
      h=norm(h+linear(intermediate,p+"output.dense"),p+"output.LayerNorm");
    }
    return h;
  }
};

struct Deberta final : Model {
  Options o; Weights& weights; c10::Device device; at::ScalarType dtype;
  int64_t hidden,heads,dim,layers,span,max_relative; bool c2p,p2c;
  std::string prefix;
  std::vector<at::Tensor> qkv_weight,qkv_bias,position_key,position_query;
  Deberta(const Options& options,Weights& w,c10::Device dev,at::ScalarType dt)
    :o(options),weights(w),device(dev),dtype(dt),hidden(o.integer("hidden_size",768)),
     heads(o.integer("num_attention_heads",12)),dim(o.integer("attention_head_size",heads>0?hidden/heads:0)),
     layers(o.integer("num_hidden_layers",12)) {
    max_relative=o.integer("max_relative_positions",-1);
    if(max_relative<1)max_relative=o.integer("max_position_embeddings",512);
    span=o.integer("position_buckets",-1);if(span<1)span=max_relative;
    const auto flags=o.string("pos_att_type","");c2p=flags.find("c2p")!=std::string::npos;p2c=flags.find("p2c")!=std::string::npos;
  }
  const at::Tensor& w(const std::string& n) const {return weight(weights,prefix+n);}
  bool has(const std::string& n) const {return weights.count(prefix+n);}
  at::Tensor linear(const at::Tensor& x,const std::string& n) const {return at::linear(x,w(n+".weight"),has(n+".bias")?w(n+".bias"):at::Tensor());}
  // Candle DeBERTa rounds its model-dtype residual sum before standalone
  // LayerNorm. Keep that contract while avoiding a separate add launch.
  bool exact_cuda_norm() const {return device.is_cuda() && hidden==768 && (dtype==at::kHalf||dtype==at::kBFloat16);}
  at::Tensor norm(const at::Tensor& x,const std::string& n) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    if(exact_cuda_norm())return encoder_layer_norm_only(x,w(n+".weight"),o.real("layer_norm_eps",1e-7),at::Tensor(),w(n+".bias"),true);
#endif
    return at::layer_norm(x,{hidden},w(n+".weight"),w(n+".bias"),o.real("layer_norm_eps",1e-7));
  }
  at::Tensor residual_norm(const at::Tensor& x,const at::Tensor& residual,const std::string& n) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    if(exact_cuda_norm())return encoder_layer_norm_only(x,w(n+".weight"),o.real("layer_norm_eps",1e-7),residual,w(n+".bias"),true);
#endif
    return norm(x+residual,n);
  }
  at::Tensor activation(const at::Tensor& x,const std::string& kind) const {
    if(kind=="relu")return at::relu(x);if(kind=="silu")return at::silu(x);if(kind=="tanh")return at::tanh(x);
    TORCH_CHECK(kind=="gelu" || kind=="gelu_new" || kind=="gelu_pytorch_tanh","Unsupported DeBERTa activation: ",kind);
    return at::gelu(x,kind=="gelu"?"none":"tanh");
  }
  int64_t output_width()const override{return hidden;}
  int64_t classification_width()const override {auto it=weights.find("classifier.weight");return it==weights.end()?0:it->second.size(0);}
  void ready()override {
    TORCH_CHECK(hidden>0&&heads>0&&dim>0&&layers>0&&span>0,"Invalid DeBERTa configuration");
    TORCH_CHECK(device.is_cpu()||device.is_cuda(),"DeBERTa supports CPU and CUDA");
    prefix=weights.count("deberta.embeddings.word_embeddings.weight")?"deberta.":"";
    auto relative=o.boolean("relative_attention",false)?w("encoder.rel_embeddings.weight"):at::Tensor();
    const auto normalization=o.string("norm_rel_ebd","none");
    TORCH_CHECK(normalization.empty() || normalization=="none" || normalization=="layer_norm" || normalization=="none|layer_norm" || normalization=="layer_norm|none", "Unsupported DeBERTa relative embedding normalization: ",normalization);
    if(relative.defined()&&normalization.find("layer_norm")!=std::string::npos)relative=norm(relative,"encoder.LayerNorm");
    for(int64_t layer=0;layer<layers;++layer) {
      const auto p="encoder.layer."+std::to_string(layer)+".attention.self.";
      qkv_weight.push_back(at::cat({w(p+"query_proj.weight"),w(p+"key_proj.weight"),w(p+"value_proj.weight")},0));
      qkv_bias.push_back(at::cat({w(p+"query_proj.bias"),w(p+"key_proj.bias"),w(p+"value_proj.bias")},0));
      const auto share=o.boolean("share_att_key",false);
      position_key.push_back(relative.defined()&&c2p?linear(relative,p+(share?"key_proj":"pos_key_proj")).view({2*span,heads,dim}).permute({1,2,0}).contiguous():at::Tensor());
      position_query.push_back(relative.defined()&&p2c?linear(relative,p+(share?"query_proj":"pos_query_proj")).view({2*span,heads,dim}).permute({1,2,0}).contiguous():at::Tensor());
    }
    const auto kernel=o.integer("conv_kernel_size",0),groups=o.integer("conv_groups",1);
    TORCH_CHECK(kernel==0||(kernel>0&&kernel%2==1&&groups>0&&hidden%groups==0),"Invalid DeBERTa convolution geometry");
  }
  at::Tensor buckets(int64_t length)const {
    auto idx=at::arange(length,at::TensorOptions().device(device).dtype(at::kLong));
    auto delta=idx.unsqueeze(1)-idx.unsqueeze(0);
    if(o.integer("position_buckets",-1)>0) {
      const double mid=span/2.;TORCH_CHECK(span>=4&&max_relative>mid+1,"Invalid logarithmic DeBERTa buckets");
      auto distance=delta.abs().to(at::kFloat);
      auto logarithmic=at::ceil(at::log(distance.clamp_min(1)/mid)/std::log((max_relative-1)/mid)*(mid-1))+mid;
      delta=at::where(distance>mid,logarithmic.to(at::kLong)*delta.sign(),delta);
    }
    return (delta+span).clamp(0,2*span-1);
  }
  at::Tensor forward(const PackedInput& input)const override {
    auto h=at::embedding(w("embeddings.word_embeddings.weight"),input.ids);
    if(o.boolean("position_biased_input",true))h=h+at::embedding(w("embeddings.position_embeddings.weight"),input.positions);
    if(o.integer("type_vocab_size",0)>0)h=h+at::embedding(w("embeddings.token_type_embeddings.weight"),input.types);
    if(has("embeddings.embed_proj.weight"))h=linear(h,"embeddings.embed_proj");
    h=norm(h,"embeddings.LayerNorm");auto embedding=h;
    std::vector<at::Tensor> indices;
    // Bucket values depend only on local i-j. Use the actual longest real
    // sequence's table for CUDA, with direct top-left indexing in bias fill.
    // This metadata shares no token rows and introduces no padded activations.
    at::Tensor shared_indices;
#ifdef TEI_TORCH_CUDA_KERNELS
    if(device.is_cuda()&&(dtype==at::kHalf||dtype==at::kBFloat16))shared_indices=buckets(input.max_sequence);
    else
#endif
    {
      std::unordered_map<int64_t,at::Tensor> indices_by_length;
      for(int64_t row=0;row<input.batch;++row) {
        const auto length=input.offsets[row+1]-input.offsets[row];
        auto found=indices_by_length.find(length);
        if(found==indices_by_length.end())found=indices_by_length.emplace(length,buckets(length)).first;
        indices.push_back(found->second);
      }
    }
    auto scale=at::full({},std::sqrt((1+c2p+p2c)*dim),h.options());
    for(int64_t layer=0;layer<layers;++layer) {
      const auto p="encoder.layer."+std::to_string(layer)+".";
      auto qkv=at::linear(h,qkv_weight[layer],qkv_bias[layer]).view({h.size(0),3*heads,dim});
      auto q=qkv.narrow(1,0,heads),k=qkv.narrow(1,heads,heads),v=qkv.narrow(1,2*heads,heads);
      auto keys=k/scale;
      at::Tensor attended;
      if(device.is_cuda()) {
        // The packed Torch kernel addresses bias with local query/key coordinates
        // and advances stride(0) per sequence, independently of packed QKV offsets.
        // Bound storage for both these actual tiles and the logical tensor view.
        const int64_t nmax=input.max_sequence,stride=((nmax+7)/8)*8;
        const int64_t tile=nmax*stride,batch_stride=heads*tile;
        auto packed_bias=packed_sequence_bias(q,input);
        at::Tensor content_key,position_content;
        if(position_key[layer].defined())content_key=at::matmul(q.transpose(0,1),position_key[layer])/scale;
        if(position_query[layer].defined())position_content=at::matmul(k.transpose(0,1),position_query[layer])/scale;
#ifdef TEI_TORCH_CUDA_KERNELS
        if(shared_indices.defined()) {
          packed_bias_fill(packed_bias,content_key,position_content,shared_indices,at::Tensor(),input.cumulative,nmax);
        } else
#endif
        {
        for(int64_t row=0;row<input.batch;++row) {
          const int64_t start=input.offsets[row],n=input.offsets[row+1]-start;
          auto bias=at::zeros({heads,n,n},h.options());
          if(content_key.defined())bias=bias+content_key.narrow(1,start,n).gather(2,indices[row].unsqueeze(0).expand({heads,n,n}));
          if(position_content.defined())bias=bias+position_content.narrow(1,start,n).gather(2,indices[row].transpose(0,1).unsqueeze(0).expand({heads,n,n})).transpose(1,2);
          packed_bias.as_strided({heads,n,n},{tile,stride,1},row*batch_stride).copy_(bias);
        }
        }
        attended=std::get<0>(at::_efficient_attention_forward(q.unsqueeze(0),keys.unsqueeze(0),v.unsqueeze(0),packed_bias,input.cumulative,input.cumulative,nmax,nmax,0.,0,false,1.)).squeeze(0);
      } else {
        std::vector<at::Tensor> result;
        for(int64_t row=0;row<input.batch;++row) {
          const int64_t start=input.offsets[row],n=input.offsets[row+1]-start;
          auto qlocal=q.narrow(0,start,n).transpose(0,1),klocal=k.narrow(0,start,n).transpose(0,1);
          auto bias=at::zeros({heads,n,n},h.options());
          if(position_key[layer].defined())bias=bias+(at::matmul(qlocal,position_key[layer])/scale).gather(2,indices[row].unsqueeze(0).expand({heads,n,n}));
          if(position_query[layer].defined())bias=bias+(at::matmul(klocal,position_query[layer])/scale).gather(2,indices[row].transpose(0,1).unsqueeze(0).expand({heads,n,n})).transpose(1,2);
          auto slice=[&](const at::Tensor& t){return t.narrow(0,start,n).unsqueeze(0);};
          result.push_back(at::scaled_dot_product_attention(slice(q).transpose(1,2),slice(keys).transpose(1,2),slice(v).transpose(1,2),bias.unsqueeze(0),0.,false,1.).squeeze(0).transpose(0,1));
        }
        attended=at::cat(result,0);
      }
      h=residual_norm(linear(attended.contiguous().view({h.size(0),heads*dim}),p+"attention.output.dense"),h,p+"attention.output.LayerNorm");
      h=residual_norm(linear(activation(linear(h,p+"intermediate.dense"),o.string("hidden_act","gelu")),p+"output.dense"),h,p+"output.LayerNorm");
      if(layer==0&&o.integer("conv_kernel_size",0)>0) {
        std::vector<at::Tensor> pieces;
        for(int64_t row=0;row<input.batch;++row) {
          const int64_t start=input.offsets[row],n=input.offsets[row+1]-start;
          auto x=embedding.narrow(0,start,n).transpose(0,1).unsqueeze(0);
          auto convolved=at::conv1d(x,w("encoder.conv.conv.weight"),w("encoder.conv.conv.bias"),{1},{o.integer("conv_kernel_size",0)/2},{1},o.integer("conv_groups",1));
          pieces.push_back(activation(convolved,o.string("conv_act","tanh")).squeeze(0).transpose(0,1));
        }
        h=norm(h+at::cat(pieces,0),"encoder.conv.LayerNorm");
      }
    }
    return h;
  }
  at::Tensor predict(const PackedInput& input,bool tokenwise)const override {
    TORCH_CHECK(classification_width() > 0, "DeBERTa classifier not loaded");
    bool token_checkpoint = false;
    for (int64_t i = 0; ; ++i) {
      const auto entry = o.values.find("architectures." + std::to_string(i));
      if (entry == o.values.end()) break;
      token_checkpoint = token_checkpoint || entry->second.ends_with("ForTokenClassification");
    }
    TORCH_CHECK(tokenwise == token_checkpoint,
      tokenwise ? "token prediction requires a token classifier checkpoint"
                : "sequence prediction requires a sequence classifier checkpoint");
    if (!tokenwise) TORCH_CHECK(weights.count("pooler.dense.weight") && weights.count("pooler.dense.bias"),
      "sequence prediction requires a sequence classifier checkpoint");
    auto h=forward(input);
    if(!tokenwise) {
      std::vector<at::Tensor> rows;for(int64_t row=0;row<input.batch;++row)rows.push_back(h.narrow(0,input.offsets[row],1));h=at::cat(rows,0);
      if(weights.count("pooler.dense.weight"))h=activation(at::linear(h,weight(weights,"pooler.dense.weight"),weight(weights,"pooler.dense.bias")),o.string("pooler_hidden_act","gelu"));
    }
    return at::linear(h,weight(weights,"classifier.weight"),weight(weights,"classifier.bias"));
  }
};
}

std::unique_ptr<Model> create_encoder(const Options& options, Weights& weights, c10::Device device, at::ScalarType dtype) {
  const auto type = options.string("model_type", "bert");
  if (type == "distilbert") return std::make_unique<Encoder>(options, weights, device, dtype, Family::Distil);
  if (type == "modernbert") return std::make_unique<Encoder>(options, weights, device, dtype, Family::Modern);
  if (type == "nomic_bert" || type == "nomic-bert") return std::make_unique<Encoder>(options, weights, device, dtype, Family::Nomic);
  if (type == "new" || type == "gte") return std::make_unique<Encoder>(options, weights, device, dtype, Family::Gte);
  if (type == "mpnet") return std::make_unique<Mpnet>(options, weights, device, dtype);
  if (type == "deberta-v2") return std::make_unique<Deberta>(options, weights, device, dtype);
  const auto name = options.string("_name_or_path", "") + options.string("auto_map.AutoConfig", "");
  if (type == "bert" && name.find("jina-bert-v2-qk-post-norm") != std::string::npos) return std::make_unique<Encoder>(options, weights, device, dtype, Family::JinaCode);
  if (type == "bert" && name.find("jina-bert-implementation") != std::string::npos) return std::make_unique<Encoder>(options, weights, device, dtype, Family::Jina);
  return nullptr;
}
}

#pragma once
#include "bridge.h"
#include "models/model.h"
#include "helpers/options.h"
#include "helpers/fp8_linear.h"
#include "helpers/attention.h"
#ifdef TEI_TORCH_CUDA_KERNELS
#include "kernels/fast_kernels.h"
#include "kernels/encoder_norm.h"
#include "kernels/encoder_activation.h"
#include "kernels/torch_flash_fma.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/detail/CUDAHooksInterface.h>
#endif
#include <ATen/ops/_addmm_activation.h>
#include <ATen/ops/_cudnn_attention_forward.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace {
struct Bert : tei::Model {
  TeiConfig config;
  c10::Device device;
  at::ScalarType dtype;
  std::unordered_map<std::string, at::Tensor> weights;
  std::string prefix;
  tei::Options options;
  std::unordered_map<std::string,tei::Fp8Linear> fp8_mlp;
  bool roberta = false;
  bool cudnn_varlen = false;
  bool private_flash_embedding = false;
  bool private_flash_roberta_embedding = false;
  bool exact_biased_norm = false;
  Bert(TeiConfig cfg, c10::Device dev, at::ScalarType type): config(cfg), device(dev), dtype(type) {}
  const at::Tensor& w(const std::string& name) const {
    auto entry = weights.find(prefix + name);
    TORCH_CHECK(entry != weights.end(), "Missing BERT weight: ", prefix, name);
    return entry->second;
  }
  at::Tensor linear(const at::Tensor& x, const std::string& name) const {
    auto quantized=fp8_mlp.find(name);
    if(quantized!=fp8_mlp.end())return quantized->second.forward(x,w(name+".bias"));
    return at::linear(x, w(name + ".weight"), w(name + ".bias"));
  }
  at::Tensor activated_linear(const at::Tensor& x,const std::string& name) const {
    if(fp8_mlp.count(name)) {
#ifdef TEI_TORCH_CUDA_KERNELS
      if(private_flash_embedding && options.boolean("_fp8_dynamic",false)
          && config.activation!=2 && config.activation!=3) {
        // Preserve model-dtype GEMM rounding, then model-dtype bias rounding,
        // then Candle unary GELU, without a separate bias launch.
        return tei::encoder_bias_gelu(fp8_mlp.at(name).forward(x),w(name+".bias"));
      }
#endif
      auto projected=linear(x,name);
#ifdef TEI_TORCH_CUDA_KERNELS
      if(config.activation!=2 && config.activation!=3)return tei::encoder_gelu(projected);
#endif
      return config.activation==3?projected/(at::exp(-projected)+1):at::relu(projected);
    }
    // Candle HiddenAct::Gelu is tanh GELU. Its CUDA linear fuses the same
    // cuBLASLt epilogue; avoid writing then rereading the full intermediate.
    if(x.is_cuda()&&config.activation!=3)return at::_addmm_activation(w(name+".bias"),x,w(name+".weight").t(),1,1,config.activation!=2);
    auto projected=linear(x,name);
    return config.activation==3?at::silu(projected):config.activation==2?at::relu(projected):at::gelu(projected,"tanh");
  }
  at::Tensor norm(const at::Tensor& x, const std::string& name) const {
    return at::layer_norm(x, {config.hidden}, w(name + ".weight"), w(name + ".bias"), config.epsilon);
  }
  at::Tensor residual_norm(const at::Tensor& x, const at::Tensor& residual, const std::string& name) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    if (x.is_cuda() && (exact_biased_norm || config.hidden == 384 || (private_flash_embedding && options.boolean("_fp8_dynamic", false))))
      return tei::encoder_layer_norm_only(x, w(name + ".weight"), config.epsilon, residual, w(name + ".bias"), false);
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
#ifdef TEI_TORCH_CUDA_KERNELS
    // Qualified H100 FP16 BERT embedding geometry. Keep classifier/SPLADE and
    // RoBERTa paths unchanged; their trained heads have separate qualification.
    private_flash_embedding = device.is_cuda() && dtype == at::kHalf && !roberta
      && config.hidden == 768 && config.heads == 12 && (config.activation == 0 || config.activation == 1)
      && at::cuda::getDeviceProperties(device.index())->major == 9
      && classification_width() == 0 && options.string("_pooling") != "splade";
#endif
#ifdef TEI_TORCH_CUDA_KERNELS
    // BERT and its aliases share Candle's biased LayerNorm primitive: residual
    // statistics use the unrounded FP32 sum. Kernel eligibility is independent
    // of model alias, pooling and head architecture; CPU/other CUDA scopes stay unchanged.
    exact_biased_norm = device.is_cuda() && dtype == at::kHalf && config.hidden == 768
      && at::cuda::getDeviceProperties(device.index())->major == 9;
    // Initial measured RoBERTa eager scope: H100 FP16 hidden768/head64 mean
    // embeddings. Keep dynamic FP8, trained heads and other pools unchanged.
    private_flash_roberta_embedding = device.is_cuda() && dtype == at::kHalf
      && options.string("model_type") == "roberta" && config.hidden == 768 && config.heads == 12
      && (config.activation == 0 || config.activation == 1)
      && at::cuda::getDeviceProperties(device.index())->major == 9
      && classification_width() == 0 && options.string("_embedding_pool") == "mean"
      && !options.boolean("_fp8_dynamic", false);
#endif
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
      if(options.boolean("_fp8_dynamic",false)) {
        for(const auto& name:{p+"intermediate.dense",p+"output.dense"}) {
          fp8_mlp.try_emplace(name,w(name+".weight"));
          weights.erase(prefix+name+".weight");
        }
      }
    }
  }
  int64_t classification_width() const override {
    auto name = weights.count("classifier.out_proj.weight") ? "classifier.out_proj.weight" : "classifier.weight";
    auto it = weights.find(name); return it == weights.end() ? 0 : it->second.size(0);
  }
  at::Tensor root_linear(const at::Tensor& h, const std::string& name) const {
    return at::linear(h, tei::weight(weights, name + ".weight"), tei::weight(weights, name + ".bias"));
  }
  at::Tensor head_activation(const at::Tensor& h, const at::Tensor& matrix,
                             const at::Tensor& bias, int activation) const {
    if (h.is_cuda() && h.dim() == 2 && bias.defined() && bias.dim() == 1
        && (h.scalar_type() == at::kHalf || h.scalar_type() == at::kBFloat16)
        && activation != 3)
      return at::_addmm_activation(bias, h, matrix.t(), 1, 1, activation != 2);
    auto projected = at::linear(h, matrix, bias);
    return activation == 3 ? at::silu(projected) : activation == 2 ? at::relu(projected) : at::gelu(projected, "tanh");
  }
  at::Tensor predict(const tei::PackedInput& input, bool tokenwise) const override {
    auto h = forward(input);
    if (!tokenwise) {
      auto indices = input.cumulative.narrow(0, 0, input.batch).to(at::kLong);
      h = h.index_select(0, indices);
    }
    if (weights.count("classifier.dense.weight")) {
      return root_linear(at::tanh(root_linear(h, "classifier.dense")), "classifier.out_proj");
    }
    if (!tokenwise) {
      // Candle's optional classifier pooler namespace is independent of the backbone.
      const auto pooler = weights.find("bert.pooler.dense.weight");
      if (pooler != weights.end() && pooler->second.dim() == 2
          && pooler->second.size(0) == config.hidden && pooler->second.size(1) == config.hidden) {
        const auto bias = weights.find("bert.pooler.dense.bias");
        TORCH_CHECK(bias != weights.end() && bias->second.dim() == 1 && bias->second.numel() == config.hidden,
                    "Invalid BERT classifier pooler bias");
        h = at::tanh(root_linear(h, "bert.pooler.dense"));
      }
    }
    return root_linear(h, "classifier");
  }
  at::Tensor token_scores(const at::Tensor& h) const override {
    at::Tensor transformed, matrix, bias;
    if (roberta) {
      transformed = head_activation(h, tei::weight(weights, "lm_head.dense.weight"), tei::weight(weights, "lm_head.dense.bias"), 1);
      transformed = at::layer_norm(transformed, {config.hidden}, tei::weight(weights, "lm_head.layer_norm.weight"), tei::weight(weights, "lm_head.layer_norm.bias"), config.epsilon);
      matrix = weights.count("lm_head.decoder.weight") ? weights.at("lm_head.decoder.weight") : w("embeddings.word_embeddings.weight");
      bias = tei::weight(weights, "lm_head.bias");
    } else {
      transformed = head_activation(h, tei::weight(weights, "cls.predictions.transform.dense.weight"),
                                    tei::weight(weights, "cls.predictions.transform.dense.bias"), config.activation);
      transformed = at::layer_norm(transformed, {config.hidden}, tei::weight(weights, "cls.predictions.transform.LayerNorm.weight"), tei::weight(weights, "cls.predictions.transform.LayerNorm.bias"), config.epsilon);
      matrix = weights.count("cls.predictions.decoder.weight") ? weights.at("cls.predictions.decoder.weight") : w("embeddings.word_embeddings.weight");
      bias = tei::weight(weights, "cls.predictions.bias");
    }
    return at::log1p(head_activation(transformed, matrix, bias, 2));
  }
  int64_t output_width() const override { return config.hidden; }
  at::Tensor forward(const tei::PackedInput& input) const override {
    const auto& ids = input.ids; const auto& types = input.types;
    const auto& positions = input.positions; const auto& cumulative = input.cumulative;
    const auto* ends = input.offsets; const auto batch = input.batch, max_sequence = input.max_sequence;
    at::Tensor h;
#ifdef TEI_TORCH_CUDA_KERNELS
    if (private_flash_roberta_embedding && !exact_biased_norm) {
      h = tei::fused_embedding_layer_norm(ids, types, positions,
        w("embeddings.word_embeddings.weight"), w("embeddings.token_type_embeddings.weight"),
        w("embeddings.position_embeddings.weight"), w("embeddings.LayerNorm.weight"),
        w("embeddings.LayerNorm.bias"), config.epsilon);
    } else
#endif
    {
    auto word = at::embedding(w("embeddings.word_embeddings.weight"), ids);
    auto token_type = at::embedding(w("embeddings.token_type_embeddings.weight"), types);
    auto position = at::embedding(w("embeddings.position_embeddings.weight"), positions);
#ifdef TEI_TORCH_CUDA_KERNELS
    if (word.is_cuda() && (exact_biased_norm || (private_flash_embedding && options.boolean("_fp8_dynamic", false))))
      h = tei::encoder_layer_norm_only(word + token_type,
        w("embeddings.LayerNorm.weight"), config.epsilon, position, w("embeddings.LayerNorm.bias"), false);
    else if (word.is_cuda() && config.hidden == 384)
      h = tei::encoder_layer_norm_only(word + token_type + position,
        w("embeddings.LayerNorm.weight"), config.epsilon, at::Tensor(), w("embeddings.LayerNorm.bias"), false);
    else if (word.is_cuda()) h = tei::fused_add_layer_norm(word, token_type,
      w("embeddings.LayerNorm.weight"), w("embeddings.LayerNorm.bias"), config.epsilon, position);
    else
#endif
    h = norm(word + token_type + position, "embeddings.LayerNorm");
    }
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
#ifdef TEI_TORCH_CUDA_KERNELS
        // Native Torch packed Flash avoids cuDNN setup overhead for moderate
        // token totals. FP8 needs the fused Torch arithmetic at every total
        // to preserve quantization inputs; normal FP16 larger totals retain
        // the measured cuDNN advantage.
        if ((private_flash_embedding || private_flash_roberta_embedding)
            && (tokens <= 4096 || (private_flash_embedding && options.boolean("_fp8_dynamic", false)))
            && tei::torch_flash_fma_supported(q, k, v)) {
          attended = tei::torch_flash_fma(q, k, v, cumulative, max_sequence,
            1.0 / std::sqrt(static_cast<double>(head)), false);
        } else
#endif
        if (cudnn_varlen && max_sequence >= 512) {
          attended = std::get<0>(at::_cudnn_attention_forward(q, k, v, std::nullopt,
            cumulative, cumulative, max_sequence, max_sequence, false, 0.0, false, false));
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
}

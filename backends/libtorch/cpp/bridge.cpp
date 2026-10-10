#include "bridge.h"
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/ops/_flash_attention_forward.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>

namespace {
thread_local std::string error;
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
struct Bert {
  TeiConfig config;
  c10::Device device;
  at::ScalarType dtype;
  std::unordered_map<std::string, at::Tensor> weights;
  std::string prefix;
  Bert(TeiConfig cfg, c10::Device dev, at::ScalarType type): config(cfg), device(dev), dtype(type) {}
  const at::Tensor& w(const std::string& name) const {
    auto entry = weights.find(prefix + name);
    TORCH_CHECK(entry != weights.end(), "Missing BERT weight: ", prefix, name);
    return entry->second;
  }
  at::Tensor linear(const at::Tensor& x, const std::string& name) const {
    return at::linear(x, w(name + ".weight"), w(name + ".bias"));
  }
  at::Tensor norm(const at::Tensor& x, const std::string& name) const {
    return at::layer_norm(x, {config.hidden}, w(name + ".weight"), w(name + ".bias"), config.epsilon);
  }
  void check(const std::string& name, std::initializer_list<int64_t> shape) const {
    TORCH_CHECK(w(name).sizes() == at::IntArrayRef(shape), "Invalid shape for ", name);
  }
  void ready() {
    TORCH_CHECK(device.is_cpu() || device.is_cuda(),
      "LibTorch 2.14.1 varlen Flash Attention requires CUDA. CPU supports an unpadded reference path; MPS/XPU are not implemented.");
    if (device.is_cuda()) {
      TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "CUDA varlen attention requires float16 or bfloat16");
      const auto head = config.hidden / config.heads;
      TORCH_CHECK(head % 8 == 0 && head <= 256, "CUDA varlen attention requires a head dimension divisible by 8, at most 256");
    }
    prefix = weights.count("bert.embeddings.word_embeddings.weight") ? "bert." : "";
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
      linear_check(p + "attention.output.dense", config.hidden, config.hidden);
      norm_check(p + "attention.output.LayerNorm");
      linear_check(p + "intermediate.dense", config.intermediate, config.hidden);
      linear_check(p + "output.dense", config.hidden, config.intermediate);
      norm_check(p + "output.LayerNorm");
    }
  }
  at::Tensor forward(const at::Tensor& ids, const at::Tensor& types,
                     const at::Tensor& positions, const at::Tensor& cumulative,
                     const int32_t* ends, int64_t batch, int64_t max_sequence) const {
    auto h = norm(at::embedding(w("embeddings.word_embeddings.weight"), ids)
                + at::embedding(w("embeddings.token_type_embeddings.weight"), types)
                + at::embedding(w("embeddings.position_embeddings.weight"), positions),
                "embeddings.LayerNorm");
    const int64_t tokens = ids.size(0), head = config.hidden / config.heads;
    for (int64_t i = 0; i < config.layers; ++i) {
      std::string p = "encoder.layer." + std::to_string(i) + ".";
      auto qkv = [&](const char* name) {
        return linear(h, p + "attention.self." + name)
          .view({tokens, config.heads, head});
      };
      auto q = qkv("query"), k = qkv("key"), v = qkv("value");
      at::Tensor attended;
      if (device.is_cuda()) {
        // The same ATen operator used by torch.nn.attention.varlen.varlen_attn's
        // Flash Attention path. Q/K/V stay [total_tokens, heads, head_dim].
        // Cumulative INT32 offsets are on-device. No padded batch or mask exists.
        attended = std::get<0>(at::_flash_attention_forward(
          q, k, v, cumulative, cumulative, max_sequence, max_sequence,
          0.0, false, false, std::nullopt, -1, -1));
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
      h = norm(h + linear(attended, p + "attention.output.dense"), p + "attention.output.LayerNorm");
      auto mid = linear(h, p + "intermediate.dense");
      mid = config.activation == 2 ? at::relu(mid) : at::gelu(mid, config.activation == 1 ? "tanh" : "none");
      h = norm(h + linear(mid, p + "output.dense"), p + "output.LayerNorm");
    }
    return h;
  }
};
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
  Bert* model = nullptr;
  checked([&] { model = new Bert(*cfg, device(name), scalar(dtype)); });
  return model;
}
int32_t tei_weight(void* handle, const char* name, const uint8_t* data,
                   const int64_t* shape, size_t rank, int32_t dtype) {
  return checked([&] {
    auto& model = *static_cast<Bert*>(handle);
    auto tensor = at::empty(at::IntArrayRef(shape, rank), at::TensorOptions().dtype(scalar(dtype)).device(at::kCPU));
    std::memcpy(tensor.data_ptr(), data, tensor.nbytes());
    TORCH_CHECK(model.weights.emplace(name, tensor.to(model.device, model.dtype)).second,
                "Duplicate weight: ", name);
  });
}
int32_t tei_ready(void* handle) { return checked([&] { static_cast<Bert*>(handle)->ready(); }); }
int32_t tei_forward(void* handle, const int64_t* ids, const int64_t* types,
                    const int64_t* positions, const int32_t* cumulative,
                    int64_t batch, int64_t max_sequence, int32_t pool,
                    const int64_t* pooled, size_t pooled_count,
                    const int64_t* raw, size_t raw_count, float* output) {
  return checked([&] {
    auto& model = *static_cast<Bert*>(handle);
    auto options = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
    const int64_t tokens = cumulative[batch];
    auto input = [&](const int64_t* data) {
      return at::from_blob(const_cast<int64_t*>(data), {tokens}, options).to(model.device);
    };
    auto cu = at::from_blob(const_cast<int32_t*>(cumulative), {batch + 1}, options.dtype(at::kInt)).to(model.device);
    auto h = model.forward(input(ids), input(types), input(positions), cu, cumulative, batch, max_sequence);
    std::vector<at::Tensor> outputs;
    for (size_t i = 0; i < pooled_count; ++i) {
      const auto row = pooled[i];
      const auto start = cumulative[row], length = cumulative[row + 1] - start;
      if (pool == 0) outputs.push_back(h.narrow(0, start, 1));
      else if (pool == 1) outputs.push_back(h.narrow(0, start, length).mean(0, true, at::kFloat));
      else outputs.push_back(h.narrow(0, start + length - 1, 1));
    }
    for (size_t i = 0; i < raw_count; ++i)
      outputs.push_back(h.narrow(0, cumulative[raw[i]], cumulative[raw[i] + 1] - cumulative[raw[i]]));
    if (!outputs.empty()) {
      auto cpu = at::cat(outputs, 0).to(at::kCPU, at::kFloat).contiguous();
      std::memcpy(output, cpu.data_ptr<float>(), cpu.nbytes());
    }
  });
}
void tei_destroy(void* handle) { delete static_cast<Bert*>(handle); }
}

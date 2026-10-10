// Packed batch invariants for native encoders. This fixture needs no Python or
// downloaded checkpoint. Run with --cuda to exercise Torch varlen FlashAttention.
#include "encoder_models.h"
#include <c10/core/InferenceMode.h>
#include <iostream>

namespace {
constexpr int64_t H = 32, I = 48;
void add(tei::Weights& w, const std::string& name, std::vector<int64_t> dims,
         c10::Device device, at::ScalarType dtype, bool identity = false) {
  auto value = identity ? at::ones(dims) : at::randn(dims) * .03;
  w[name] = value.to(device, dtype);
}
void linear(tei::Weights& w, const std::string& n, int64_t output, int64_t input,
            c10::Device device, at::ScalarType dtype, bool bias = true) {
  add(w, n + ".weight", {output, input}, device, dtype);
  if (bias) add(w, n + ".bias", {output}, device, dtype);
}
void norm(tei::Weights& w, const std::string& n, c10::Device device, at::ScalarType dtype, bool bias = true) {
  add(w, n + ".weight", {H}, device, dtype, true);
  if (bias) add(w, n + ".bias", {H}, device, dtype);
}
tei::Options config(const std::string& type) {
  tei::Options o;
  o.values = {{"model_type", type}, {"hidden_size", "32"}, {"dim", "32"}, {"n_embd", "32"},
    {"num_attention_heads", "4"}, {"n_heads", "4"}, {"n_head", "4"},
    {"num_hidden_layers", "2"}, {"n_layers", "2"}, {"n_layer", "2"},
    {"max_position_embeddings", "32"}, {"n_positions", "32"}, {"global_attn_every_n_layers", "2"},
    {"local_attention", "4"}, {"activation_function", "gelu"}, {"type_vocab_size", "2"}};
  if (type == "jina" || type == "jina-code") {
    o.values["model_type"] = "bert";
    o.values["_name_or_path"] = type == "jina" ? "jinaai/jina-bert-implementation" : "jinaai/jina-bert-v2-qk-post-norm";
    o.values["position_embedding_type"] = "alibi";
  }
  return o;
}
tei::Weights weights(const std::string& type, c10::Device d, at::ScalarType dt) {
  tei::Weights w;
  add(w, type == "modernbert" ? "embeddings.tok_embeddings.weight" : "embeddings.word_embeddings.weight", {64,H}, d, dt);
  norm(w, type == "modernbert" ? "embeddings.norm" : type == "nomic_bert" ? "emb_ln" : "embeddings.LayerNorm", d, dt, type != "modernbert");
  if (type != "modernbert" && type != "distilbert") add(w, "embeddings.token_type_embeddings.weight", {2,H},d,dt);
  if (type == "distilbert" || type=="mpnet") add(w,"embeddings.position_embeddings.weight",{32,H},d,dt);
  if(type=="mpnet") add(w,"encoder.relative_attention_bias.weight",{32,4},d,dt);
  for (int i=0; i<2; ++i) {
    std::string p = (type=="modernbert" ? "layers." : type=="nomic_bert" ? "encoder.layers." : type=="distilbert" ? "transformer.layer." : "encoder.layer.") + std::to_string(i) + ".";
    if(type=="mpnet") {
      for(auto q:{"q","k","v","o"}) linear(w,p+"attention.attn."+q,H,H,d,dt);
      norm(w,p+"attention.LayerNorm",d,dt);norm(w,p+"output.LayerNorm",d,dt);
      linear(w,p+"intermediate.dense",I,H,d,dt);linear(w,p+"output.dense",H,I,d,dt);
    } else if (type=="distilbert") {
      for (auto q : {"q_lin","k_lin","v_lin","out_lin"}) linear(w,p+"attention."+q,H,H,d,dt);
      linear(w,p+"ffn.lin1",I,H,d,dt); linear(w,p+"ffn.lin2",H,I,d,dt);
      norm(w,p+"sa_layer_norm",d,dt); norm(w,p+"output_layer_norm",d,dt);
    } else if (type=="modernbert") {
      linear(w,p+"attn.Wqkv",3*H,H,d,dt,false); linear(w,p+"attn.Wo",H,H,d,dt,false);
      linear(w,p+"mlp.Wi",2*I,H,d,dt,false); linear(w,p+"mlp.Wo",H,I,d,dt,false);
      if(i) norm(w,p+"attn_norm",d,dt,false);
      norm(w,p+"mlp_norm",d,dt,false);
    } else if (type=="nomic_bert") {
      linear(w,p+"attn.Wqkv",3*H,H,d,dt); linear(w,p+"attn.out_proj",H,H,d,dt);
      linear(w,p+"mlp.fc1",I,H,d,dt); linear(w,p+"mlp.fc2",H,I,d,dt);
      norm(w,p+"norm1",d,dt); norm(w,p+"norm2",d,dt);
    } else if (type=="new") {
      linear(w,p+"attention.qkv_proj",3*H,H,d,dt); linear(w,p+"attention.o_proj",H,H,d,dt);
      linear(w,p+"mlp.up_gate_proj",2*I,H,d,dt,false); linear(w,p+"mlp.down_proj",H,I,d,dt);
      norm(w,p+"attn_ln",d,dt); norm(w,p+"mlp_ln",d,dt);
    } else {
      for(auto q:{"query","key","value"}) linear(w,p+"attention.self."+q,H,H,d,dt);
      linear(w,p+"attention.output.dense",H,H,d,dt); norm(w,p+"attention.output.LayerNorm",d,dt);
      if(type=="jina-code") {
        norm(w,p+"attention.self.layer_norm_q",d,dt);norm(w,p+"attention.self.layer_norm_k",d,dt);
        norm(w,p+"layer_norm_1",d,dt);norm(w,p+"layer_norm_2",d,dt);
        linear(w,p+"mlp.up_gated_layer",2*I,H,d,dt,false);linear(w,p+"mlp.down_layer",H,I,d,dt);
      } else {
        linear(w,p+"mlp.gated_layers",2*I,H,d,dt,false);linear(w,p+"mlp.wo",H,I,d,dt);
        norm(w,p+"mlp.layernorm",d,dt);
      }
    }
  }
  if(type=="modernbert") norm(w,"final_norm",d,dt,false);
  return w;
}
}
int main(int argc, char** argv) {
  c10::InferenceMode guard;
  at::manual_seed(13);
  c10::Device device(argc>1 ? "cuda:0" : "cpu");
  auto dtype = device.is_cuda() ? at::kHalf : at::kFloat;
  auto opts=at::TensorOptions().device(device).dtype(at::kLong);
  int32_t offsets[]={0,3,8}, short_offsets[]={0,3},long_offsets[]={0,5};
  auto ids=at::tensor({2,3,4,5,6,7,8,9},opts), types=at::zeros({8},opts), positions=at::tensor({0,1,2,0,1,2,3,4},opts);
  auto cu=at::tensor({0,3,8},opts.dtype(at::kInt));
  for(auto type:{"distilbert","modernbert","nomic_bert","new","jina","jina-code","mpnet"}) {
    auto o=config(type);auto w=weights(type,device,dtype);auto model=tei::create_encoder(o,w,device,dtype);model->ready();
    auto packed=model->forward({ids,types,positions,cu,offsets,2,5});
    std::vector<at::Tensor> separate;
    for(int b=0;b<2;++b) {
      auto start=offsets[b],n=offsets[b+1]-start;
      auto local_cu=at::tensor({0,n},opts.dtype(at::kInt));
      separate.push_back(model->forward({ids.narrow(0,start,n),types.narrow(0,start,n),positions.narrow(0,start,n),local_cu,b?long_offsets:short_offsets,1,n}));
    }
    auto error=(packed-at::cat(separate)).abs().max().item<float>();
    TORCH_CHECK(at::isfinite(packed).all().item<bool>(),"Nonfinite output for ",type);
    TORCH_CHECK(error < (device.is_cuda() ? .003 : 1e-5),"Packed sequence contamination for ",type,": ",error);
    std::cout<<type<<" packed-vs-independent max error "<<error<<"\n";
  }
}

// Native packed decoder regression tests; no Python runtime or padded tokens.
#include "decoder_models.h"
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <ATen/Parallel.h>
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
}
int main(){c10::InferenceMode inference;at::set_num_threads(1);
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

 for(auto family:{"llama","mistral","qwen2","qwen3","qwen3_moe"}){
  auto cfg=config(family,std::string(family)=="qwen3_moe");auto w=weights(cfg);auto initial=w;auto model=create_decoder(cfg,initial,c10::Device("cpu"),at::kFloat);model->ready();
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
  check_close(run(*bidi,{3,4,5,6,7},{0,3,5}),at::cat({b,run(*bidi,{6,7},{0,2})},0));
  std::cout<<family<<" packed boundaries, causality and bidirectional attention passed\n";
 }
 multimodal_test();
}

#include "multimodal_models.h"
#include <iostream>
using namespace tei;
int main(int argc,char** argv) {
  at::manual_seed(1451);
  Options cfg;
  cfg.values={{"model_type","qwen3_5"},{"hidden_size","16"},{"num_attention_heads","2"},
    {"num_key_value_heads","1"},{"head_dim","8"},{"num_hidden_layers","2"},
    {"max_position_embeddings","32"},{"rope_parameters.partial_rotary_factor",".5"},
    {"linear_num_key_heads","1"},{"linear_num_value_heads","1"},
    {"layer_types.0","linear_attention"},{"layer_types.1","full_attention"}};
  Weights weights;
  auto put=[&](const std::string& name,std::initializer_list<int64_t> shape,bool zero=false) {
    weights[name]=zero?at::zeros(shape):at::randn(shape)*.05;
  };
  put("model.embed_tokens.weight",{32,16});put("model.norm.weight",{16},true);
  for(int i=0;i<2;++i) {
    auto p="model.layers."+std::to_string(i)+".";
    put(p+"input_layernorm.weight",{16},true);put(p+"post_attention_layernorm.weight",{16},true);
    put(p+"mlp.gate_proj.weight",{24,16});put(p+"mlp.up_proj.weight",{24,16});put(p+"mlp.down_proj.weight",{16,24});
    if(i==0) {
      p+="linear_attn.";
      put(p+"in_proj_qkv.weight",{384,16});put(p+"in_proj_z.weight",{128,16});
      put(p+"in_proj_a.weight",{1,16});put(p+"in_proj_b.weight",{1,16});
      put(p+"conv1d.weight",{384,1,4});put(p+"A_log",{1});put(p+"dt_bias",{1});
      weights[p+"norm.weight"]=at::ones({128});put(p+"out_proj.weight",{16,128});
    } else {
      p+="self_attn.";
      put(p+"q_proj.weight",{32,16});put(p+"k_proj.weight",{8,16});put(p+"v_proj.weight",{8,16});put(p+"o_proj.weight",{16,16});
      put(p+"q_norm.weight",{8},true);put(p+"k_norm.weight",{8},true);
    }
  }
  auto model=create_multimodal(cfg,weights,c10::Device("cpu"),at::kFloat);model->ready();
  std::vector<int32_t> offsets={0,3,5},one={0,3},two={0,2};
  auto make=[](const std::vector<int64_t>& ids,const std::vector<int32_t>& ends) {
    auto tokens=at::from_blob(const_cast<int64_t*>(ids.data()),{static_cast<int64_t>(ids.size())},at::TensorOptions().dtype(at::kLong)).clone();
    std::vector<int64_t> positions;
    int64_t max=0;
    for(size_t i=1;i<ends.size();++i) { auto n=ends[i]-ends[i-1];max=std::max<int64_t>(max,n);for(int j=0;j<n;++j)positions.push_back(j); }
    auto pos=at::from_blob(positions.data(),{static_cast<int64_t>(positions.size())},at::TensorOptions().dtype(at::kLong)).clone();
    auto cu=at::from_blob(const_cast<int32_t*>(ends.data()),{static_cast<int64_t>(ends.size())},at::TensorOptions().dtype(at::kInt)).clone();
    return PackedInput{tokens,at::zeros_like(tokens),pos,cu,ends.data(),static_cast<int64_t>(ends.size()-1),max};
  };
  auto output=model->forward(make({1,2,3,4,5},offsets));
  auto independent=at::cat({model->forward(make({1,2,3},one)),model->forward(make({4,5},two))},0);
  TORCH_CHECK(at::allclose(output,independent,1e-5,1e-6),"DeltaNet packed recurrence leaked sequence state");
  auto changed=model->forward(make({31,2,3,4,5},offsets));
  TORCH_CHECK(at::allclose(output.narrow(0,3,2),changed.narrow(0,3,2),1e-5,1e-6),"DeltaNet state not reset at sequence boundary");
  std::cout<<"Qwen3.5 packed/single parity and recurrence reset passed\n";
  auto bidirectional_cfg=cfg;bidirectional_cfg.values["decision_attention_mode"]="noncausal_full_attention";
  auto bidirectional=create_multimodal(bidirectional_cfg,weights,c10::Device("cpu"),at::kFloat);bidirectional->ready();
  auto bi=bidirectional->forward(make({1,2,3,4,5},offsets));
  auto future=bidirectional->forward(make({1,2,31,4,5},offsets));
  TORCH_CHECK((bi.select(0,0)-future.select(0,0)).abs().max().item<float>()>1e-5,"Qwen3.5 noncausal full attention ignored future tokens");
  TORCH_CHECK(at::allclose(bi.narrow(0,3,2),future.narrow(0,3,2),1e-5,1e-6),"Noncausal Qwen3.5 crossed packed sequence boundaries");
  std::cout<<"Qwen3.5 noncausal-full attention decision mode and sequence isolation passed\n";
  auto moe_cfg=cfg;moe_cfg.values["num_experts"]="3";moe_cfg.values["num_experts_per_tok"]="2";
  auto moe_weights=weights;
  for(int i=0;i<2;++i) {
    auto p="model.layers."+std::to_string(i)+".mlp.";
    moe_weights[p+"gate.weight"]=at::zeros({3,16});
    auto gu=at::cat({weights.at(p+"gate_proj.weight"),weights.at(p+"up_proj.weight")},0);
    moe_weights[p+"experts.gate_up_proj"]=at::stack({gu,gu*1.5,gu*2.});
    auto down=weights.at(p+"down_proj.weight");moe_weights[p+"experts.down_proj"]=at::stack({down,down,down});
    for(auto name:{"gate_proj","up_proj","down_proj"})moe_weights[p+"shared_expert."+name+".weight"]=weights.at(p+name+".weight");
    moe_weights[p+"shared_expert_gate.weight"]=at::zeros({1,16});
  }
  auto moe=create_multimodal(moe_cfg,moe_weights,c10::Device("cpu"),at::kFloat);moe->ready();
  auto routed=moe->forward(make({1,2,3,4,5},offsets));
  auto routed_single=at::cat({moe->forward(make({1,2,3},one)),moe->forward(make({4,5},two))},0);
  TORCH_CHECK(at::allclose(routed,routed_single,1e-5,1e-6),"Qwen3.5 routed experts leak packed sequences");
  for(int i=0;i<2;++i)moe_weights["model.layers."+std::to_string(i)+".mlp.experts.gate_up_proj"].select(0,2).fill_(100.);
  TORCH_CHECK(at::allclose(routed,moe->forward(make({1,2,3,4,5},offsets)),1e-5,1e-6),"Qwen3.5 router ties must prefer lower expert IDs");
  std::cout<<"Qwen3.5 MoE/shared expert, stable routing ties and packed isolation passed\n";
  if(argc>1) {
    c10::Device device(argv[1]);
    for(auto& [key,value]:weights)value=value.to(at::kBFloat16);
    auto cpu=create_multimodal(cfg,weights,c10::Device("cpu"),at::kBFloat16);cpu->ready();
    Weights cuda_weights;
    for(const auto& [key,value]:weights)cuda_weights[key]=value.to(device);
    auto cuda=create_multimodal(cfg,cuda_weights,device,at::kBFloat16);cuda->ready();
    auto host=make({1,2,3,4,5},offsets),dev=host;
    dev.ids=host.ids.to(device);dev.types=host.types.to(device);dev.positions=host.positions.to(device);dev.cumulative=host.cumulative.to(device);
    auto reference=cpu->forward(host).to(at::kFloat),actual=cuda->forward(dev).cpu().to(at::kFloat);
    TORCH_CHECK(at::allclose(reference,actual,.025,.025),"Qwen3.5 native CUDA DeltaNet disagrees with CPU recurrence, maxabs=",(reference-actual).abs().max().item<float>());
    std::cout<<"Qwen3.5 CUDA DeltaNet/CPU reference parity maxabs="<<(reference-actual).abs().max().item<float>()<<"\n";
  }
}

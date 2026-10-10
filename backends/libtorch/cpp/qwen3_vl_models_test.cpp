// Native actual-length image/deepstack regression; no Python or token padding.
#include "multimodal_models.h"
#include <c10/core/InferenceMode.h>
#include <ATen/Parallel.h>
#include <iostream>
using namespace tei;
int main(int argc,char** argv) {
  c10::InferenceMode inference;at::set_num_threads(1);at::manual_seed(712);
  Options cfg;
  cfg.values={{"model_type","qwen3_vl"},{"text_config.hidden_size","16"},{"text_config.num_attention_heads","2"},
    {"text_config.num_key_value_heads","1"},{"text_config.head_dim","8"},{"text_config.num_hidden_layers","2"},
    {"text_config.vocab_size","32"},{"text_config.intermediate_size","24"},
    {"text_config.max_position_embeddings","32"},{"text_config.rope_theta","10000"},
    {"text_config.rope_scaling.mrope_interleaved","true"},{"text_config.rope_scaling.mrope_section.0","2"},
    {"text_config.rope_scaling.mrope_section.1","1"},{"text_config.rope_scaling.mrope_section.2","1"},
    {"vision_config.hidden_size","16"},{"vision_config.num_heads","2"},{"vision_config.depth","2"},
    {"vision_config.spatial_merge_size","2"},{"vision_config.hidden_act","gelu"},{"vision_config.deepstack_visual_indexes.0","0"}};
  Weights weights;
  auto add=[&](const std::string& name,std::initializer_list<int64_t> shape,bool norm=false) {
    weights[name]=norm?at::ones(shape):at::randn(shape)*.1;
  };
  auto tp=std::string("model.language_model.");
  add(tp+"embed_tokens.weight",{32,16});add(tp+"norm.weight",{16},true);
  for(int i=0;i<2;++i) {
    auto p=tp+"layers."+std::to_string(i)+".";
    add(p+"input_layernorm.weight",{16},true);add(p+"post_attention_layernorm.weight",{16},true);
    add(p+"self_attn.q_proj.weight",{16,16});add(p+"self_attn.k_proj.weight",{8,16});add(p+"self_attn.v_proj.weight",{8,16});add(p+"self_attn.o_proj.weight",{16,16});
    add(p+"self_attn.q_norm.weight",{8},true);add(p+"self_attn.k_norm.weight",{8},true);
    add(p+"mlp.gate_proj.weight",{24,16});add(p+"mlp.up_proj.weight",{24,16});add(p+"mlp.down_proj.weight",{16,24});
  }
  auto vp=std::string("model.visual.");
  auto ln=[&](const std::string& p,int64_t width){add(vp+p+".weight",{width},true);weights[vp+p+".bias"]=at::zeros({width});};
  auto fc=[&](const std::string& p,int64_t out,int64_t in){add(vp+p+".weight",{out,in});add(vp+p+".bias",{out});};
  add(vp+"patch_embed.proj.weight",{16,3,2,1,1});add(vp+"patch_embed.proj.bias",{16});add(vp+"pos_embed.weight",{4,16});
  for(int i=0;i<2;++i){auto p="blocks."+std::to_string(i)+".";ln(p+"norm1",16);ln(p+"norm2",16);fc(p+"attn.qkv",48,16);fc(p+"attn.proj",16,16);fc(p+"mlp.linear_fc1",24,16);fc(p+"mlp.linear_fc2",16,24);}
  for(const auto& p:{std::string("merger"),std::string("deepstack_merger_list.0")}){ln(p+".norm",p=="merger"?16:64);fc(p+".linear_fc1",64,64);fc(p+".linear_fc2",16,64);}
  auto model=create_multimodal(cfg,weights,c10::Device("cpu"),at::kFloat);model->ready();
  TORCH_CHECK(model->supports_images(),"Qwen3-VL image capability unavailable");
  std::vector<int64_t> ids={1,2,3,4,5,6},pos={0,1,2,3,0,1};std::vector<int32_t> offsets={0,4,6};
  auto options=at::TensorOptions().dtype(at::kLong);
  PackedInput input{at::from_blob(ids.data(),{6},options),at::zeros({6},options),at::from_blob(pos.data(),{6},options),
    at::from_blob(offsets.data(),{3},options.dtype(at::kInt)),offsets.data(),2,4};
  auto plain=model->forward(input);
  input.images.push_back({at::rand({4,6}),{1,2,2},2,1,1,0});
  input.multimodal_positions=input.positions.unsqueeze(0).repeat({3,1});
  auto visual=model->forward(input);
  TORCH_CHECK((visual-plain).abs().max().item<float>()>1e-4,"Image features not injected");
  TORCH_CHECK(at::allclose(visual.narrow(0,4,2),plain.narrow(0,4,2),1e-5,1e-6),"Images leaked between packed sequences");
  TORCH_CHECK(at::allclose(visual.narrow(0,0,1),plain.narrow(0,0,1),1e-5,1e-6),"Image affected causal prefix");
  input.images[0].pixels=at::zeros({4,6});auto changed=model->forward(input);
  TORCH_CHECK((changed.narrow(0,1,3)-visual.narrow(0,1,3)).abs().max().item<float>()>1e-4,"Image pixels ignored");
  TORCH_CHECK(at::allclose(changed.narrow(0,4,2),visual.narrow(0,4,2),1e-5,1e-6),"Changed image leaked between sequences");
  std::cout<<"Qwen3-VL natural patches, image injection, deepstack and packed isolation passed\n";
  auto q35cfg=cfg;q35cfg.values["model_type"]="qwen3_5";
  q35cfg.values.erase("vision_config.deepstack_visual_indexes.0");
  q35cfg.values["text_config.layer_types.0"]="full_attention";q35cfg.values["text_config.layer_types.1"]="full_attention";
  q35cfg.values["text_config.rope_parameters.mrope_interleaved"]="true";
  q35cfg.values["text_config.rope_parameters.partial_rotary_factor"]="1";
  for(int i=0;i<3;++i)q35cfg.values["text_config.rope_parameters.mrope_section."+std::to_string(i)]=cfg.values.at("text_config.rope_scaling.mrope_section."+std::to_string(i));
  auto q35weights=weights;
  for(int i=0;i<2;++i) {
    auto key=tp+"layers."+std::to_string(i)+".self_attn.q_proj.weight";
    // Qwen3.5 interleaves each head's query with its attention output gate.
    auto query=q35weights.at(key).view({2,8,16});
    q35weights[key]=at::cat({query,at::zeros_like(query)},1).reshape({32,16});
  }
  auto q35=create_multimodal(q35cfg,q35weights,c10::Device("cpu"),at::kFloat);q35->ready();
  auto q35plain=input;q35plain.images.clear();q35plain.multimodal_positions=at::Tensor();
  auto q35text=q35->forward(q35plain),q35image=q35->forward(input);
  TORCH_CHECK(q35->supports_images() && (q35image-q35text).abs().max().item<float>()>1e-4,"Qwen3.5 vision features not injected");
  TORCH_CHECK(at::allclose(q35image.narrow(0,4,2),q35text.narrow(0,4,2),1e-5,1e-6),"Qwen3.5 vision leaked between packed sequences");
  std::cout<<"Qwen3.5 FP16 vision, BF16-compatible injection and mRoPE packed isolation passed\n";
  if(argc>1) {
    c10::Device device(argv[1]);Weights hw,cw;
    for(const auto& [name,value]:weights){hw[name]=value.to(at::kHalf);cw[name]=hw[name].to(device);}
    auto reference=create_multimodal(cfg,hw,c10::Device("cpu"),at::kHalf);reference->ready();
    auto native=create_multimodal(cfg,cw,device,at::kHalf);native->ready();
    auto host=input,dev=input;host.images[0].pixels=host.images[0].pixels.to(at::kHalf);
    dev.ids=host.ids.to(device);dev.types=host.types.to(device);dev.positions=host.positions.to(device);dev.cumulative=host.cumulative.to(device);
    dev.multimodal_positions=host.multimodal_positions.to(device);dev.images[0].pixels=host.images[0].pixels.to(device);
    auto expected=reference->forward(host).to(at::kFloat),actual=native->forward(dev).cpu().to(at::kFloat);
    TORCH_CHECK(at::allclose(expected,actual,.03,.03),"Qwen3-VL CUDA/CPU image parity mismatch: ",(expected-actual).abs().max().item<float>());
    std::cout<<"Qwen3-VL CUDA/CPU float16 parity maxabs="<<(expected-actual).abs().max().item<float>()<<"\n";
  }
}

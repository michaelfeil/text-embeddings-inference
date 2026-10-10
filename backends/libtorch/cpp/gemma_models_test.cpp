// Packed sequence isolation and geometry tests for native Gemma implementations.
#include "gemma_models.h"
#include <iostream>

using namespace tei;
namespace {
struct Fixture {
  Options cfg;
  Weights weights;
  bool legacy, embedding2;
  Fixture(const std::string& family, int64_t head_dim=8) : legacy(family == "gemma3"), embedding2(family == "embedding_gemma2") {
    cfg.values = {{"model_type", family}, {"hidden_size", "16"}, {"num_attention_heads", "2"},
      {"num_key_value_heads", "1"}, {"head_dim", std::to_string(head_dim)}, {"global_head_dim", std::to_string(head_dim)},
      {"num_hidden_layers", "4"}, {"max_position_embeddings", "512"}, {"sliding_window", "2"},
      {"query_pre_attn_scalar", "8"}, {"_sliding_window_pattern", "2"},
      {"num_kv_shared_layers", legacy ? "0" : "2"}, {"hidden_activation", "gelu_pytorch_tanh"},
      {"hidden_size_per_layer_input", legacy ? "0" : "4"}};
    cfg.values["embedding_dim"]="12";
    auto put = [&](const std::string& name, std::initializer_list<int64_t> shape, bool norm = false) {
      weights.emplace(name, norm ? (legacy ? at::zeros(shape) : at::ones(shape)) : at::randn(shape) * .1);
    };
    put("embed_tokens.weight", {64,16}); put("norm.weight", {16},true);
    if (!legacy) {
      auto stem = embedding2 ? std::string("ple.") : std::string();
      put(stem + "per_layer_model_projection.weight", {16,16});
      put(stem + "per_layer_projection_norm.weight", {4},true);
      if (!embedding2) put("embed_tokens_per_layer.weight", {64,16});
      if (embedding2) put("embedding_projection.weight", {12,16});
    }
    for (int64_t i=0;i<4;++i) {
      cfg.values["layer_types." + std::to_string(i)] = (head_dim>256 || i % 2) ? "full_attention" : "sliding_attention";
      auto p = "layers." + std::to_string(i) + ".";
      for (const auto& name : {"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"}) put(p+name+".weight",{16},true);
      put(p+"self_attn.q_proj.weight", {2*head_dim,16}); put(p+"self_attn.q_norm.weight", {head_dim},true);
      if (legacy || i<2) {
        put(p+"self_attn.k_proj.weight", {head_dim,16}); put(p+"self_attn.v_proj.weight", {head_dim,16});
        put(p+"self_attn.k_norm.weight", {head_dim},true);
      }
      put(p+"self_attn.o_proj.weight", {16,2*head_dim});
      put(p+"mlp.gate_proj.weight", {24,16}); put(p+"mlp.up_proj.weight", {24,16}); put(p+"mlp.down_proj.weight", {16,24});
      if (!legacy) {
        weights[p+"layer_scalar"] = at::ones({1});
        auto stem = p + (embedding2 ? "ple_block." : "");
        put(stem+"per_layer_input_gate.weight", {4,16}); put(stem+"per_layer_projection.weight", {16,4}); put(stem+"post_per_layer_input_norm.weight", {16},true);
      }
    }
  }
};
PackedInput input(const std::vector<int64_t>& ids, const std::vector<int32_t>& offsets) {
  auto tokens = at::from_blob(const_cast<int64_t*>(ids.data()), {static_cast<int64_t>(ids.size())}, at::TensorOptions().dtype(at::kLong)).clone();
  std::vector<int64_t> positions;
  int64_t maximum=0;
  for (size_t i=1;i<offsets.size();++i) {
    auto len=offsets[i]-offsets[i-1]; maximum=std::max<int64_t>(maximum,len);
    for(int j=0;j<len;++j) positions.push_back(j);
  }
  auto pos = at::from_blob(positions.data(), {static_cast<int64_t>(positions.size())}, at::TensorOptions().dtype(at::kLong)).clone();
  return {tokens,at::zeros_like(tokens),pos,at::from_blob(const_cast<int32_t*>(offsets.data()), {static_cast<int64_t>(offsets.size())}, at::TensorOptions().dtype(at::kInt)).clone(),offsets.data(),static_cast<int64_t>(offsets.size()-1),maximum};
}
}
int main(int argc,char** argv) {
  at::manual_seed(418);
  for (const auto& family : {"gemma3","gemma4","embedding_gemma2"}) {
    Fixture f(family);
    auto model = create_gemma(f.cfg,f.weights,c10::Device("cpu"),at::kFloat);
    model->ready();
    const std::vector<int64_t> a={1,2,3,4,5}, b={6,7};
    std::vector<int64_t> packed=a; packed.insert(packed.end(),b.begin(),b.end());
    const std::vector<int32_t> ao={0,5},bo={0,2},po={0,5,7};
    auto output = model->forward(input(packed,po));
    auto independent = at::cat({model->forward(input(a,ao)),model->forward(input(b,bo))},0);
    TORCH_CHECK(at::allclose(output,independent,1e-5,1e-6),family," packed cross-sequence leakage");
    auto changed=packed; changed[0]=55;
    auto modified=model->forward(input(changed,po));
    TORCH_CHECK(at::allclose(output.narrow(0,5,2),modified.narrow(0,5,2),1e-5,1e-6),family," cross-sequence attention leaked");
    TORCH_CHECK(output.size(0)==7 && output.size(1)==(f.embedding2?12:16),family," unexpected packed output geometry");
    std::cout << family << ": packed/single parity and independent sequence isolation passed\n";
    if(std::string(family)=="gemma4") {
      Fixture mf(family);
      mf.cfg.values["enable_moe_block"]="true";mf.cfg.values["num_experts"]="3";mf.cfg.values["top_k_experts"]="2";
      for(int layer=0;layer<4;++layer) {
        auto p="layers."+std::to_string(layer)+".";
        mf.weights[p+"router.scale"]=at::ones({16});mf.weights[p+"router.proj.weight"]=at::zeros({3,16});
        mf.weights[p+"router.per_expert_scale"]=at::ones({3});
        mf.weights[p+"experts.gate_up_proj"]=at::randn({3,12,16})*.03;mf.weights[p+"experts.down_proj"]=at::randn({3,16,6})*.03;
        for(const auto& name:{"post_feedforward_layernorm_1","pre_feedforward_layernorm_2","post_feedforward_layernorm_2"})mf.weights[p+name+".weight"]=at::ones({16});
      }
      auto routed=create_gemma(mf.cfg,mf.weights,c10::Device("cpu"),at::kFloat);routed->ready();
      auto mo=routed->forward(input(packed,po));
      auto mi=at::cat({routed->forward(input(a,ao)),routed->forward(input(b,bo))},0);
      TORCH_CHECK(at::allclose(mo,mi,1e-5,1e-6),"Gemma4 MoE packed output differs from independent sequences");
      for(int layer=0;layer<4;++layer)mf.weights["layers."+std::to_string(layer)+".experts.down_proj"].select(0,2).fill_(100.);
      TORCH_CHECK(at::allclose(mo,routed->forward(input(packed,po)),1e-5,1e-6),"Gemma4 routing tie did not select lower expert IDs");
      for(int layer=0;layer<4;++layer)mf.weights["layers."+std::to_string(layer)+".router.per_expert_scale"]=at::tensor({0.f,1.f,1.f});
      auto scaled=routed->forward(input(packed,po));
      TORCH_CHECK(!at::allclose(mo,scaled,1e-5,1e-6),"Gemma4 learned expert output scales were ignored");
      TORCH_CHECK(at::allclose(scaled,at::cat({routed->forward(input(a,ao)),routed->forward(input(b,bo))},0),1e-5,1e-6),"Gemma4 scaled expert outputs leaked across sequences");
      std::cout<<"gemma4: native MoE packed isolation and stable routing ties passed\n";
    }
    if(!f.legacy) {
      f.cfg.values["vision_config.hidden_size"]="8";f.cfg.values["vision_config.num_attention_heads"]="2";
      f.cfg.values["vision_config.num_key_value_heads"]="1";f.cfg.values["vision_config.head_dim"]="4";
      f.cfg.values["vision_config.num_hidden_layers"]="1";
      auto put=[&](const std::string& name,std::initializer_list<int64_t> shape,bool norm=false) {
        f.weights[name]=norm?at::ones(shape):at::randn(shape)*.03;
      };
      put("vision_tower.patch_embedder.input_proj.weight",{8,768});
      put("vision_tower.patch_embedder.position_embedding_table",{2,16,8});
      auto p=std::string("vision_tower.encoder.layers.0.");
      for(const auto& name:{"input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm"})put(p+name+".weight",{8},true);
      put(p+"self_attn.q_proj.linear.weight",{8,8});put(p+"self_attn.k_proj.linear.weight",{4,8});
      put(p+"self_attn.v_proj.linear.weight",{4,8});put(p+"self_attn.o_proj.linear.weight",{8,8});
      put(p+"self_attn.q_norm.weight",{4},true);put(p+"self_attn.k_norm.weight",{4},true);
      put(p+"mlp.gate_proj.linear.weight",{12,8});put(p+"mlp.up_proj.linear.weight",{12,8});put(p+"mlp.down_proj.linear.weight",{8,12});
      put("embed_vision.embedding_projection.weight",{16,8});
      auto visual=create_gemma(f.cfg,f.weights,c10::Device("cpu"),at::kFloat);visual->ready();
      auto batch=input(packed,po);
      batch.images.push_back({at::rand({9,768}),{1,3,3},3,2,1,0});
      auto with_image=visual->forward(batch);
      TORCH_CHECK(!at::allclose(output.narrow(0,0,5),with_image.narrow(0,0,5)),family," vision features were ignored");
      TORCH_CHECK(at::allclose(output.narrow(0,5,2),with_image.narrow(0,5,2),1e-5,1e-6),family," image injection leaked across sequence boundaries");
      std::cout<<family<<": natural-resolution vision injection/isolation passed\n";
      Weights bf16_weights;
      for(const auto& [name,value]:f.weights)bf16_weights[name]=value.to(at::kBFloat16);
      auto bf16_visual=create_gemma(f.cfg,bf16_weights,c10::Device("cpu"),at::kBFloat16);bf16_visual->ready();
      auto precise=bf16_visual->forward(batch);
      auto canonical=batch;
      // Distinct FP32 pixels whose normalized projection inputs are identical
      // after BF16 rounding must produce identical complete model outputs.
      canonical.images[0].pixels=((batch.images[0].pixels-.5)*2.).to(at::kBFloat16).to(at::kFloat)*.5+.5;
      TORCH_CHECK(at::equal(precise,bf16_visual->forward(canonical)),family," vision normalized source pixels after premature activation cast");
      auto rounded=batch;rounded.images[0].pixels=batch.images[0].pixels.to(at::kBFloat16).to(at::kFloat);
      TORCH_CHECK(!at::equal(precise,bf16_visual->forward(rounded)),family," vision pixel precision regression fixture was insensitive to premature BF16 conversion");
      std::cout<<family<<": FP32 image normalization before BF16 projection passed\n";
      if(f.embedding2) {
        f.cfg.values["audio_config.hidden_size"]="8";f.cfg.values["audio_config.num_attention_heads"]="2";
        f.cfg.values["audio_config.num_hidden_layers"]="1";f.cfg.values["audio_config.attention_chunk_size"]="2";
        f.cfg.values["audio_config.attention_context_left"]="2";
        put("audio_tower.subsample_conv_projection.layer0.conv.weight",{2,1,3,3});
        put("audio_tower.subsample_conv_projection.layer1.conv.weight",{2,2,3,3});
        put("audio_tower.subsample_conv_projection.layer0.norm.weight",{2},true);
        put("audio_tower.subsample_conv_projection.layer1.norm.weight",{2},true);
        put("audio_tower.subsample_conv_projection.input_proj_linear.weight",{8,64});
        auto p=std::string("audio_tower.layers.0.");
        for(const auto& stem:{"feed_forward1.","feed_forward2."}) {
          put(p+stem+"pre_layer_norm.weight",{8},true);put(p+stem+"post_layer_norm.weight",{8},true);
          put(p+stem+"ffw_layer_1.linear.weight",{32,8});put(p+stem+"ffw_layer_2.linear.weight",{8,32});
        }
        put(p+"norm_pre_attn.weight",{8},true);put(p+"norm_post_attn.weight",{8},true);put(p+"norm_out.weight",{8},true);
        for(const auto& stem:{"q_proj","k_proj","v_proj","post","relative_k_proj"})put(p+"self_attn."+stem+".weight",{8,8});
        put(p+"self_attn.per_dim_scale",{4});
        put(p+"lconv1d.pre_layer_norm.weight",{8},true);put(p+"lconv1d.conv_norm.weight",{8},true);
        put(p+"lconv1d.linear_start.linear.weight",{16,8});put(p+"lconv1d.linear_end.linear.weight",{8,8});put(p+"lconv1d.depthwise_conv1d.weight",{8,1,3});
        put("embed_audio.embedding_projection.weight",{16,8});
        auto audio=create_gemma(f.cfg,f.weights,c10::Device("cpu"),at::kFloat);audio->ready();
        auto audio_batch=input(packed,po);
        audio_batch.audios.push_back({at::randn({8,128}),at::ones({8},at::TensorOptions().dtype(at::kByte)),1,2});
        auto with_audio=audio->forward(audio_batch);
        auto prepared=audio_batch;audio->prepare_input(prepared);
        TORCH_CHECK(at::allclose(with_audio,audio->forward(prepared),1e-5,1e-6),"Prepared Gemma audio indices changed output");
        auto masked=audio_batch;masked.audios[0].validity=masked.audios[0].validity.clone();
        masked.audios[0].validity.select(0,0).zero_();masked.audios[0].token_count=1;
        auto masked_reference=audio->forward(masked);audio->prepare_input(masked);
        TORCH_CHECK(masked.audios[0].selected_frames.numel()==1&&masked.audios[0].selected_frames[0].item<int64_t>()==1,"Gemma prepared audio indices did not preserve exact SSCP validity");
        TORCH_CHECK(at::allclose(masked_reference,audio->forward(masked),1e-5,1e-6),"Prepared masked Gemma audio changed selected feature");
        TORCH_CHECK(!at::allclose(output.narrow(0,0,5),with_audio.narrow(0,0,5)),"Gemma audio features were ignored");
        TORCH_CHECK(at::allclose(output.narrow(0,5,2),with_audio.narrow(0,5,2),1e-5,1e-6),"Gemma audio injection leaked across sequences");
        std::cout<<family<<": actual-length Conformer audio injection/isolation passed\n";
      }
    }
  }
  if(argc>1) {
    const c10::Device device(argv[1]);
    for(const auto& family:{"gemma4","embedding_gemma2"}) {
      Fixture f(family,512);
      f.cfg.values["num_hidden_layers"]="1";
      f.cfg.values["num_kv_shared_layers"]="0";
      f.cfg.values["hidden_size_per_layer_input"]="0";
      // Keep the synthetic attention well conditioned: untrained unit Q/K
      // norms at scale1 otherwise amplify BF16 reduction differences across4
      // random layers into unstable argmax decisions.
      for(auto& [key,value]:f.weights) {
        if(key.ends_with("q_norm.weight") || key.ends_with("k_norm.weight")) value=value*.05;
      }
      for(auto& [key,value]:f.weights) value=value.to(at::kBFloat16);
      auto cpu=create_gemma(f.cfg,f.weights,c10::Device("cpu"),at::kBFloat16);cpu->ready();
      Weights gpu_weights;
      for(const auto& [key,value]:f.weights) gpu_weights[key]=value.to(device);
      auto gpu=create_gemma(f.cfg,gpu_weights,device,at::kBFloat16);gpu->ready();
      for(int64_t length:{5,129}) {
        std::vector<int64_t> ids;
        for(int64_t i=0;i<length+2;++i)ids.push_back(i%63+1);
        const std::vector<int32_t> offsets={0,static_cast<int32_t>(length),static_cast<int32_t>(length+2)};
        auto host=input(ids,offsets),dev=host;
        dev.ids=host.ids.to(device);dev.types=host.types.to(device);dev.positions=host.positions.to(device);dev.cumulative=host.cumulative.to(device);
        auto reference=cpu->forward(host).to(at::kFloat),actual=gpu->forward(dev).cpu().to(at::kFloat);
        TORCH_CHECK(at::allclose(reference,actual,.04,.04),family," Torch varlen512-vs-CPU mismatch max_length",length," maxabs=",(reference-actual).abs().max().item<float>());
        std::cout<<family<<" Torch512 GPU/CPU parity max_length="<<length<<" maxabs="<<(reference-actual).abs().max().item<float>()<<"\n";
      }
    }
  }
}

#include "cuda_graph.h"
#include "decoder_models.h"
#include <c10/core/InferenceMode.h>
#include <iostream>
using namespace tei;
class Dense final:public Model {
public:
 void ready() override{}
 at::Tensor forward(const PackedInput& input) const override{
  return at::sin(input.ids.to(at::kFloat).unsqueeze(1)+input.positions.to(at::kFloat).unsqueeze(1)*.1)
    *at::arange(1,9,input.ids.options().dtype(at::kFloat)).unsqueeze(0);
 }
};
class Uncapturable final:public Model {
public:
 void ready() override{}
 at::Tensor forward(const PackedInput& input) const override{
  auto sum=input.ids.sum().item<int64_t>();
  return at::full({input.ids.size(0),8},sum,input.ids.options().dtype(at::kFloat));
 }
};
class Media final:public Model {
public:
 void ready() override{}
 void prepare_input(PackedInput& input) const override {
  for(auto& audio:input.audios)audio.selected_frames=at::nonzero(audio.validity).flatten();
 }
 at::Tensor forward(const PackedInput& input) const override {
  auto result=input.ids.to(at::kFloat);
  for(const auto& image:input.images)result=result+image.pixels.sum()+image.grid_thw[1]+image.token_start;
  for(const auto& audio:input.audios) {
   auto indices=audio.selected_frames.defined()?audio.selected_frames:at::nonzero(audio.validity).flatten();
   result=result+audio.features.index_select(0,indices).sum()+audio.token_start;
  }
  if(input.multimodal_positions.defined())result=result+input.multimodal_positions.sum();
  return result.unsqueeze(1);
 }
};
std::unique_ptr<Model> tiny_decoder() {
 Options cfg;cfg.values={{"model_type","qwen3"},{"hidden_size","16"},{"num_attention_heads","2"},{"num_key_value_heads","1"},{"head_dim","8"},{"intermediate_size","24"},{"num_hidden_layers","1"},{"vocab_size","32"},{"max_position_embeddings","32"}};
 Weights w;auto opts=at::TensorOptions().device(at::kCUDA).dtype(at::kHalf);
 auto add=[&](std::string name,std::initializer_list<int64_t> shape,bool norm=false){w[name]=norm?at::ones(at::IntArrayRef(shape),opts):at::randn(at::IntArrayRef(shape),opts)*.1;};
 add("model.embed_tokens.weight",{32,16});add("model.norm.weight",{16},true);
 auto p=std::string("model.layers.0.");add(p+"input_layernorm.weight",{16},true);add(p+"post_attention_layernorm.weight",{16},true);
 add(p+"self_attn.q_proj.weight",{16,16});add(p+"self_attn.k_proj.weight",{8,16});add(p+"self_attn.v_proj.weight",{8,16});add(p+"self_attn.o_proj.weight",{16,16});
 add(p+"self_attn.q_norm.weight",{8},true);add(p+"self_attn.k_norm.weight",{8},true);
 add(p+"mlp.gate_proj.weight",{24,16});add(p+"mlp.up_proj.weight",{24,16});add(p+"mlp.down_proj.weight",{16,24});
 auto result=create_decoder(cfg,w,c10::Device("cuda:0"),at::kHalf);result->ready();return result;
}
int main(){c10::InferenceMode guard;auto decoder=tiny_decoder();auto& model=*decoder;CudaGraphCache cache(2,8);
 for(auto offsets:std::vector<std::vector<int32_t>>{{0,3},{0,1,3},{0,2,3},{0,3}}){
  auto options=at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
  auto ids=at::tensor({3,4,5},options),pos=at::tensor({0,1,2},options);
  auto cu=at::from_blob(offsets.data(),{int64_t(offsets.size())},at::TensorOptions().dtype(at::kInt)).to(at::kCUDA);
  PackedInput input{ids,at::zeros_like(ids),pos,cu,offsets.data(),int64_t(offsets.size()-1),3};
  for(int run=0;run<3;++run){input.ids.add_(7);input.positions.add_(1);
   auto actual=cache.forward(model,input).clone(),expected=model.forward(input);
   TORCH_CHECK((actual-expected).abs().max().item<float>()<0.0001,"Graph replay used stale inputs");
  }
  TORCH_CHECK(cache.size()<=2,"Graph cache exceeds bound");
 }
 TORCH_CHECK(cache.size()==2,"Dense graphs did not capture");
 std::vector<int32_t> offsets{0,3};auto options=at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
 auto ids=at::tensor({3,4,5},options);auto cu=at::tensor({0,3},options.dtype(at::kInt));
 PackedInput input{ids,at::zeros_like(ids),at::zeros_like(ids),cu,offsets.data(),1,3};
 auto angles=input.positions.to(at::kFloat).unsqueeze(1)*at::tensor({1.f,.1f,.01f,.001f},options.dtype(at::kFloat));
 auto cosine=at::cos(angles).to(at::kHalf),sine=at::sin(angles).to(at::kHalf);
 auto direct=decoder_multimodal_forward(model,input,{}, {},cosine,sine);
 TORCH_CHECK((direct-model.forward(input)).abs().max().item<float>()<0.0001,"External RoPE changed ordinary decoder output");
 auto visual_indices=at::tensor({int64_t(1)},options),visual=at::ones({1,16},options.dtype(at::kHalf))*.25;
 auto injected=decoder_multimodal_forward(model,input,visual_indices,visual,cosine,sine);
 auto withzero=decoder_multimodal_forward(model,input,visual_indices,visual,cosine,sine,{at::zeros_like(visual)});
 TORCH_CHECK((injected-withzero).abs().max().item<float>()<0.0001,"Deepstack residual materialization changed decoder output");
 auto deepstack=at::arange(16,options.dtype(at::kHalf)).view({1,16})*.07;
 auto withdeep=decoder_multimodal_forward(model,input,visual_indices,visual,cosine,sine,{deepstack});
 TORCH_CHECK((withdeep-injected).abs().max().item<float>()>0.001,"CUDA deepstack features were ignored");
 TORCH_CHECK((withdeep[0]-injected[0]).abs().max().item<float>()<0.0001,"CUDA deepstack crossed causal token boundary");
 Uncapturable unsupported;CudaGraphCache fallback;
 for(int i=0;i<2;++i){auto actual=fallback.forward(unsupported,input);TORCH_CHECK(actual[0][0].item<float>()==12,"Graph fallback changed result");}
 TORCH_CHECK(fallback.size()==0,"Unsupported model unexpectedly captured");
 Media media;CudaGraphCache media_cache(4,8);
 input.images.push_back({at::ones({2,4},options.dtype(at::kFloat)),{1,2,2},1,0,1,0});
 input.audios.push_back({at::arange(12,options.dtype(at::kFloat)).view({3,4}),at::tensor({1,0,1},options).to(at::kBool),0,2});
 input.multimodal_positions=at::zeros({3,3},options);
 for(int run=0;run<3;++run) {
  input.images[0].pixels.add_(1);input.audios[0].features.add_(1);input.multimodal_positions.add_(1);
  if(run==1)input.audios[0].validity.copy_(at::tensor({0,1,1},options).to(at::kBool));
  auto actual=media_cache.forward(media,input).clone();
  TORCH_CHECK(at::equal(actual,media.forward(input)),"Media graph replay retained stale values or frame indices");
 }
 TORCH_CHECK(media_cache.size()==1,"Same exact media shapes did not reuse capture");
 input.images[0].token_start=1;
 TORCH_CHECK(at::equal(media_cache.forward(media,input).clone(),media.forward(input)),"Media metadata changed without recapture");
 TORCH_CHECK(media_cache.size()==2,"Media descriptors are missing from graph key");
 auto before=media_cache.size();
 TORCH_CHECK(at::equal(media_cache.forward(media,input).clone(),media.forward(input)),"Media descriptor graph replay failed");
 Dense other_model;
 TORCH_CHECK(at::equal(media_cache.forward(other_model,input).clone(),other_model.forward(input)),"Graph reused across different model instances");
 TORCH_CHECK(media_cache.size()==before+1,"Model instance is missing from graph key");
 std::cout<<"CUDA graph dynamic inputs, exact sequence keys, bounded eviction and eager fallback passed\n";
}

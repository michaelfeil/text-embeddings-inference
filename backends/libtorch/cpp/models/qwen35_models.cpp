#include "helpers/attention.h"
// Native adaptation of Candle Qwen3.5. Upstream notices retained in LICENSE/NOTICE.
#include "models/qwen35_models.h"
#include "models/qwen3_vl_models.h"
#include "models/decoder_models.h"
#include "kernels/qwen35_kernels.h"
#include "kernels/qwen35_reference.h"
#include "kernels/torch_flash_fma.h"
#include "kernels/moe_kernels.h"
#include <ATen/ops/_efficient_attention_forward.h>
#include <cmath>

namespace tei {
namespace {
// Adapted from Candle qwen3_vl/vision.rs; upstream Apache-2.0 notices retained.
class Qwen35 final : public Model {
  Options cfg;
  Weights weights;
  c10::Device device;
  at::ScalarType dtype;
  std::string prefix;
  int64_t hidden, heads, kv_heads, dim, layer_count, rotary;
  double epsilon;
  at::Tensor cos_cache, sin_cache;
  struct Mlp {at::Tensor gate_up,down,router,experts,expert_down,shared_gate_up,shared_down,shared_gate;};
  std::vector<Mlp> mlps;
  std::unique_ptr<Qwen3VL> visual;
  at::Tensor rotary_frequencies,rotary_axes;
  std::string key(const std::string& name) const { return cfg.values.count("text_config."+name) ? "text_config."+name : name; }
  int64_t integer(const std::string& name,int64_t fallback=0) const { return cfg.integer(key(name),fallback); }
  double real(const std::string& name,double fallback=0) const { return cfg.real(key(name),fallback); }
  std::string string(const std::string& name,const std::string& fallback="") const { return cfg.string(key(name),fallback); }
  const at::Tensor& w(const std::string& name) const { return weight(weights,prefix+name); }
  at::Tensor linear(const at::Tensor& x,const std::string& name) const { return at::linear(x,w(name+".weight")); }
  at::Tensor gated(const at::Tensor& gu) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    if(device.is_cuda() && dtype==at::kBFloat16)return gemma_gated_cuda(gu,false);
#endif
    auto gate=gu.narrow(-1,0,gu.size(-1)/2);
    return (gate/(at::ones({},gate.options())+at::exp(-gate)))*gu.narrow(-1,gu.size(-1)/2,gu.size(-1)/2);
  }
  at::Tensor feed_forward(const at::Tensor& x,int64_t layer) const {
    const auto& m=mlps[layer];
    if(!m.router.defined())return at::linear(gated(at::linear(x,m.gate_up)),m.down);
    auto top=integer("num_experts_per_tok",8);
    auto renormalize=cfg.boolean(key("norm_topk_prob"),true);
    const bool native=top==8 && device.is_cuda() && dtype==at::kBFloat16 &&
      (m.experts.size(0)==128 || m.experts.size(0)==256) && x.size(1)%8==0 && m.expert_down.size(2)%8==0;
    auto output=native?routed_moe_cuda(x,at::linear(x,m.router).to(at::kFloat),m.experts,m.expert_down,renormalize):at::Tensor();
    if(!output.defined())output=decoder_moe_forward(x,m.router,m.experts,m.expert_down,top,renormalize);
    auto shared=at::linear(gated(at::linear(x,m.shared_gate_up)),m.shared_down);
    auto logits=at::linear(x,m.shared_gate);
    auto gate=device.is_cuda() && dtype==at::kBFloat16 ?
      at::reciprocal(at::ones_like(logits)+at::exp(-logits)) : at::sigmoid(logits);
    return output+shared*gate;
  }
  at::Tensor norm(const at::Tensor& x,const std::string& name) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    // Deep routed models amplify individual BF16 differences. Preserve the
    // original Candle reduction tree before casting the normalized value.
    if(integer("num_experts",0)>0 && device.is_cuda() && dtype==at::kBFloat16)
      return qwen35_reference_norm_cuda(x,w(name+".weight"),epsilon);
#endif
    auto f=x.to(at::kFloat);
    return (f*at::reciprocal(at::sqrt(f.square().mean(-1,true)+epsilon))*(w(name+".weight").to(at::kFloat)+1.)).to(x.scalar_type());
  }
  at::Tensor rotate(const at::Tensor& x,const PackedInput& input) const {
    auto r=x.narrow(-1,0,rotary);
    auto a=r.narrow(-1,0,rotary/2),b=r.narrow(-1,rotary/2,rotary/2);
    at::Tensor cos,sin;
    if(input.multimodal_positions.defined()) {
      TORCH_CHECK(visual,"Qwen3.5 multimodal positions require vision configuration");
      auto angles=input.multimodal_positions.index_select(0,rotary_axes).transpose(0,1).to(at::kFloat)*rotary_frequencies;
      cos=angles.cos().to(dtype).unsqueeze(1);sin=angles.sin().to(dtype).unsqueeze(1);
    } else {cos=cos_cache.index_select(0,input.positions).unsqueeze(1);sin=sin_cache.index_select(0,input.positions).unsqueeze(1);}
    return at::cat({at::cat({a*cos-b*sin,b*cos+a*sin},-1),x.narrow(-1,rotary,dim-rotary)},-1).contiguous();
  }
  at::Tensor full_attention(const at::Tensor& x,const std::string& p,const PackedInput& input) const {
    auto tokens=x.size(0);
    auto qg=linear(x,p+"q_proj").view({tokens,heads,2*dim});
    auto q=rotate(norm(qg.narrow(-1,0,dim),p+"q_norm"),input);
    auto k=rotate(norm(linear(x,p+"k_proj").view({tokens,kv_heads,dim}),p+"k_norm"),input);
    auto v=linear(x,p+"v_proj").view({tokens,kv_heads,dim});
    auto mode=cfg.string("_decision_attention_mode",cfg.string("decision_attention_mode","causal"));
    auto causal=mode!="noncausal_full_attention" && !cfg.boolean("use_bidirectional_attention",false);
    const bool reference=integer("num_experts",0)>0 && device.is_cuda() && dtype==at::kBFloat16;
    at::Tensor attention;
#ifdef TEI_TORCH_CUDA_KERNELS
    if(reference && torch_flash_fma_supported(q,k,v))
      attention=torch_flash_fma(q,k,v,input.cumulative,input.max_sequence,1./std::sqrt(static_cast<double>(dim)),causal);
    else
#endif
      attention=packed_attention(q,k,v,input,1./std::sqrt(static_cast<double>(dim)),causal);
    auto a=attention.view({tokens,heads,dim});
    auto logits=qg.narrow(-1,dim,dim);
    auto gate=reference ? at::reciprocal(at::ones_like(logits)+at::exp(-logits)) : at::sigmoid(logits);
    return linear((a*gate).reshape({tokens,heads*dim}),p+"o_proj");
  }
  at::Tensor delta_cpu(const at::Tensor& qkv,const at::Tensor& z,const at::Tensor& ab,const std::string& p,const PackedInput& input) const {
    // Exact-length reference recurrence. No sequence padding and no persistent state.
    auto kh=integer("linear_num_key_heads"),vh=integer("linear_num_value_heads");
    auto channels=(2*kh+vh)*128;
    auto conv=w(p+"conv1d.weight").view({channels,-1});
    auto kernel=conv.size(-1);
    auto alog=w(p+"A_log").to(at::kFloat),dt=w(p+"dt_bias").to(at::kFloat);
    std::vector<at::Tensor> outputs;
    for(int64_t row=0;row<input.batch;++row) {
      auto start=input.offsets[row],length=input.offsets[row+1]-start;
      std::vector<at::Tensor> convolved;
      for(int64_t t=0;t<length;++t) {
        auto sum=at::zeros({channels},qkv.options().dtype(at::kFloat));
        for(int64_t j=0;j<kernel;++j) {
          auto source=t+j-kernel+1;
          if(source>=0) sum=sum+qkv.select(0,start+source).to(at::kFloat)*conv.select(1,j).to(at::kFloat);
        }
        convolved.push_back(at::silu(sum.to(dtype)).to(dtype));
      }
      auto values=at::stack(convolved,0);
      auto q=values.narrow(-1,0,kh*128).view({length,kh,128}).to(at::kFloat);
      auto k=values.narrow(-1,kh*128,kh*128).view({length,kh,128}).to(at::kFloat);
      auto v=values.narrow(-1,2*kh*128,vh*128).view({length,vh,128}).to(at::kFloat);
      q=q*at::rsqrt(q.square().sum(-1,true)+1e-6)/std::sqrt(128.);
      k=k*at::rsqrt(k.square().sum(-1,true)+1e-6);
      q=at::repeat_interleave(q,vh/kh,1); k=at::repeat_interleave(k,vh/kh,1);
      auto alpha=ab.narrow(0,start,length).narrow(-1,0,vh).to(at::kFloat)+dt;
      auto decay=at::exp(-at::exp(alog)*at::softplus(alpha));
      auto beta=at::sigmoid(ab.narrow(0,start,length).narrow(-1,vh,vh).to(at::kFloat)).to(dtype).to(at::kFloat);
      auto state=at::zeros({vh,128,128},q.options());
      for(int64_t t=0;t<length;++t) {
        auto kt=k.select(0,t),qt=q.select(0,t);
        state=state*decay.select(0,t).view({vh,1,1});
        auto memory=(state*kt.unsqueeze(-1)).sum(1);
        auto delta=(v.select(0,t)-memory)*beta.select(0,t).unsqueeze(-1);
        state=state+kt.unsqueeze(-1)*delta.unsqueeze(1);
        auto out=(state*qt.unsqueeze(-1)).sum(1).to(dtype);
        auto f=out.to(at::kFloat);
        out=(f*at::rsqrt(f.square().mean(-1,true)+epsilon)).to(dtype);
        out=(out*w(p+"norm.weight")).to(dtype);
        outputs.push_back((out.to(at::kFloat)*at::silu(z.select(0,start+t).view({vh,128}).to(at::kFloat))).to(dtype).reshape({vh*128}));
      }
    }
    return at::stack(outputs,0);
  }
  at::Tensor delta_attention(const at::Tensor& x,const std::string& p,const PackedInput& input) const {
    auto qkv=linear(x,p+"in_proj_qkv").contiguous(),z=linear(x,p+"in_proj_z").contiguous();
    auto ab=at::cat({linear(x,p+"in_proj_a"),linear(x,p+"in_proj_b")},-1).contiguous();
    at::Tensor output;
    if(device.is_cuda()) {
#ifdef TEI_TORCH_CUDA_KERNELS
      auto conv=w(p+"conv1d.weight").view({qkv.size(1),-1}).contiguous();
      auto alog=w(p+"A_log").to(at::kFloat).contiguous(),dt=w(p+"dt_bias").to(at::kFloat).contiguous();
      if(integer("num_experts",0)>0)
        output=qwen35_reference_delta_cuda(qkv,z,ab,conv,alog,dt,w(p+"norm.weight"),input,
          integer("linear_num_key_heads"),integer("linear_num_value_heads"),epsilon);
      else
        output=qwen35_delta_cuda(qkv,z,ab,conv,alog,dt,w(p+"norm.weight"),input,
          integer("linear_num_key_heads"),integer("linear_num_value_heads"),epsilon);
#else
      TORCH_CHECK(false,"Qwen3.5 requires the native DeltaNet CUDA kernel build");
#endif
    } else output=delta_cpu(qkv,z,ab,p,input);
    return linear(output,p+"out_proj");
  }
 public:
  Qwen35(const Options& config,Weights& tensors,c10::Device dev,at::ScalarType type)
    :cfg(config),weights(std::move(tensors)),device(dev),dtype(type) {
    hidden=integer("hidden_size"); heads=integer("num_attention_heads"); kv_heads=integer("num_key_value_heads");
    dim=integer("head_dim",hidden/(heads?heads:1)); layer_count=integer("num_hidden_layers"); epsilon=real("rms_norm_eps",1e-6);
    rotary=static_cast<int64_t>(dim*real("rope_parameters.partial_rotary_factor",1.));
    for(const auto& p:{"model.language_model.","language_model.","model.",""})
      if(weights.count(std::string(p)+"embed_tokens.weight")) { prefix=p; break; }
    if(cfg.integer("vision_config.depth",0)>0)visual=std::make_unique<Qwen3VL>(cfg,weights,device,at::kHalf,true);
  }
  int64_t output_width() const override { return hidden; }
  bool supports_images() const override {return bool(visual);}
  void ready() override {
    TORCH_CHECK(hidden>0 && heads>0 && kv_heads>0 && heads%kv_heads==0 && dim>0 && rotary>0 && rotary<=dim && rotary%2==0 && layer_count>0,"Invalid Qwen3.5 attention geometry");
    TORCH_CHECK(integer("linear_key_head_dim",128)==128 && integer("linear_value_head_dim",128)==128,"Qwen3.5 DeltaNet requires head dimension128");
    TORCH_CHECK(device.is_cpu() || device.is_cuda(),"Native Qwen3.5 supports CPU/CUDA");
    if(device.is_cuda()) TORCH_CHECK(dtype==at::kBFloat16,"Native Qwen3.5 CUDA DeltaNet requires BF16");
    auto theta=real("rope_parameters.rope_theta",10000.);
    TORCH_CHECK(theta>0.,"Qwen3.5 rotary theta must be positive");
    std::vector<float> inverse(rotary/2);
    for(int64_t i=0;i<rotary/2;++i)inverse[i]=1.f/std::pow(float(theta),float(2*i)/float(rotary));
    auto inv=at::from_blob(inverse.data(),{rotary/2},at::TensorOptions().dtype(at::kFloat)).clone().to(device);
    rotary_frequencies=inv;
    auto angles=at::arange(integer("max_position_embeddings",32768),inv.options()).unsqueeze(1)*inv.unsqueeze(0);
    cos_cache=angles.cos().to(dtype); sin_cache=angles.sin().to(dtype);
    w("embed_tokens.weight");w("norm.weight");
    if(visual) {
      TORCH_CHECK(cfg.boolean(key("rope_parameters.mrope_interleaved"),false),"Qwen3.5 vision requires interleaved mRoPE");
      int64_t sections=0;for(int i=0;i<3;++i)sections+=integer("rope_parameters.mrope_section."+std::to_string(i));
      TORCH_CHECK(sections==rotary/2 && !cfg.values.count("vision_config.deepstack_visual_indexes.0"),"Invalid Qwen3.5 vision mRoPE/deepstack configuration");
      auto s1=integer("rope_parameters.mrope_section.1"),s2=integer("rope_parameters.mrope_section.2");
      std::vector<int64_t> axes;
      for(int64_t j=0;j<rotary/2;++j)axes.push_back(j%3==1 && j<s1*3?1:j%3==2 && j<s2*3?2:0);
      rotary_axes=at::from_blob(axes.data(),{rotary/2},at::TensorOptions().dtype(at::kLong)).clone().to(device);
      visual->ready();
    }
    for(int64_t i=0;i<layer_count;++i) {
      auto kind=string("layer_types."+std::to_string(i));
      TORCH_CHECK(kind=="full_attention" || kind=="linear_attention","Invalid Qwen3.5 attention layer ",i);
      auto p="layers."+std::to_string(i)+".";
      w(p+"input_layernorm.weight");w(p+"post_attention_layernorm.weight");
      auto mp=p+"mlp.";Mlp m;std::vector<std::string> consumed;
      auto experts=integer("num_experts",0);
      if(experts==0) {
        m.gate_up=at::cat({w(mp+"gate_proj.weight"),w(mp+"up_proj.weight")},0);m.down=w(mp+"down_proj.weight");
        consumed={mp+"gate_proj.weight",mp+"up_proj.weight",mp+"down_proj.weight"};
      } else {
        auto top=integer("num_experts_per_tok",8);
        TORCH_CHECK(top>0 && top<=experts,"Invalid Qwen3.5 routed expert count");
        m.router=w(mp+"gate.weight");
        consumed.push_back(mp+"gate.weight");
        if(weights.count(prefix+mp+"experts.gate_up_proj")) {
          m.experts=w(mp+"experts.gate_up_proj");m.expert_down=w(mp+"experts.down_proj");
          consumed.push_back(mp+"experts.gate_up_proj");consumed.push_back(mp+"experts.down_proj");
        }
        else {
          std::vector<at::Tensor> gu,down;
          for(int64_t e=0;e<experts;++e){
            auto ep=mp+"experts."+std::to_string(e)+".";
            gu.push_back(at::cat({w(ep+"gate_proj.weight"),w(ep+"up_proj.weight")},0));down.push_back(w(ep+"down_proj.weight"));
            for(const auto& name:{"gate_proj.weight","up_proj.weight","down_proj.weight"})consumed.push_back(ep+name);
          }
          m.experts=at::stack(gu);m.expert_down=at::stack(down);
        }
        m.shared_gate_up=at::cat({w(mp+"shared_expert.gate_proj.weight"),w(mp+"shared_expert.up_proj.weight")},0);
        m.shared_down=w(mp+"shared_expert.down_proj.weight");m.shared_gate=w(mp+"shared_expert_gate.weight");
        for(const auto& name:{"shared_expert.gate_proj.weight","shared_expert.up_proj.weight","shared_expert.down_proj.weight","shared_expert_gate.weight"})consumed.push_back(mp+name);
      }
      mlps.push_back(std::move(m));
      // Every FFN tensor is now owned by its inference cache. The dense/shared
      // gate and up projections, and individually stacked experts, otherwise
      // retain a second full allocation after concatenation. Forward reads FFNs
      // exclusively through mlps; keep all attention/embedding/vision/head keys.
      for(const auto& name:consumed)weights.erase(prefix+name);
      if(kind=="linear_attention") {
        auto kh=integer("linear_num_key_heads"),vh=integer("linear_num_value_heads");
        TORCH_CHECK(kh>0 && vh>0 && vh%kh==0,"Invalid Qwen3.5 DeltaNet heads");
        w(p+"linear_attn.conv1d.weight");w(p+"linear_attn.A_log");w(p+"linear_attn.dt_bias");w(p+"linear_attn.norm.weight");
      }
    }
  }
  at::Tensor forward(const PackedInput& input) const override {
    auto states=at::embedding(w("embed_tokens.weight"),input.ids);
    for(const auto& image:input.images) {
      TORCH_CHECK(visual,"Qwen3.5 vision tower unavailable");
      auto features=visual->image_features(image).to(dtype);
      TORCH_CHECK(features.size(0)==image.token_count && features.size(1)==hidden,"Qwen3.5 visual token geometry mismatch");
      states.narrow(0,image.token_start,image.token_count).copy_(features);
    }
    for(int64_t i=0;i<layer_count;++i) {
      auto p="layers."+std::to_string(i)+".";
      auto normalized=norm(states,p+"input_layernorm");
      auto output=string("layer_types."+std::to_string(i))=="linear_attention" ?
        delta_attention(normalized,p+"linear_attn.",input) : full_attention(normalized,p+"self_attn.",input);
      states=states+output;
      normalized=norm(states,p+"post_attention_layernorm");
      states=states+feed_forward(normalized,i);
    }
    return norm(states,"norm");
  }
};
}
std::unique_ptr<Model> create_qwen35(const Options& cfg,Weights& weights,c10::Device device,at::ScalarType dtype) {
  auto family=cfg.string("model_type");
  if(family=="qwen3_5" || family=="qwen3_5_text" || family=="qwen3_5_moe" || family=="qwen3_5_moe_text")
    return std::make_unique<Qwen35>(cfg,weights,device,dtype);
  return nullptr;
}
}

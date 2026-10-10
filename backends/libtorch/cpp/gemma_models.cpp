// Native adaptations of backends/candle/src/models/{gemma3,gemma4,embedding_gemma2}.rs.
// Upstream Apache-2.0 notices remain in LICENSE and NOTICE.
#include "gemma_models.h"
#include "decoder_models.h"
#include "qwen35_kernels.h"
#include <ATen/ops/_flash_attention_forward.h>
#include <ATen/ops/_efficient_attention_forward.h>
#include <ATen/ops/_cudnn_attention_forward.h>
#include <cmath>
#include <array>
#include <limits>

namespace tei {
namespace {
class Gemma final : public Model {
  Options cfg;
  Weights& weights;
  c10::Device device;
  at::ScalarType dtype;
  std::string prefix, family;
  int64_t hidden, heads, layers, ple_width;
  double epsilon;
  bool legacy, embedding2, causal;
  struct Layer {
    std::string p;
    at::Tensor qkv, gate_up;
    int64_t dim, kv, window;
    bool sliding, shared, store, k_equals_v;
    at::Tensor cos, sin;
  };
  std::vector<Layer> stack;
  Weights norm_scales;
  at::Tensor vision_frequency;
  std::unordered_map<int64_t,at::Tensor> norm_ones;
  std::unordered_map<std::string,std::array<double,4>> audio_bounds;
  at::Tensor moe(const at::Tensor& residual,const at::Tensor& dense,const std::string& p) const {
    auto input=norm(residual,p+"pre_feedforward_layernorm_2");
    // Preserve the router's BF16 rounding before both root-size and learned scales.
    auto root=at::scalar_tensor(1./std::sqrt(double(hidden)),residual.options());
    auto routing=((norm(residual,"",true)*root)*w(p+"router.scale")).to(at::kFloat);
    auto logits=at::linear(routing,w(p+"router.proj.weight").to(at::kFloat));
    auto count=integer("num_experts",0),top=integer("top_k_experts",8);
    TORCH_CHECK(count>0 && top>0 && top<=count,"Invalid Gemma4 MoE routing geometry");
    // Stable descending order gives lower expert IDs precedence on tied logits.
    auto ids=std::get<1>(at::sort(logits,true,-1,true)).narrow(-1,0,top);
    auto probability=at::softmax(logits,-1).gather(-1,ids);
    probability=probability/probability.sum(-1,true);
    probability=probability*w(p+"router.per_expert_scale").to(at::kFloat).index_select(0,ids.reshape({-1})).view_as(ids);
    const auto& gate_up=w(p+"experts.gate_up_proj");const auto& down=w(p+"experts.down_proj");
    auto activation=[&](const at::Tensor& gu) {
#ifdef TEI_TORCH_CUDA_KERNELS
      auto act=string("hidden_activation","gelu_pytorch_tanh");
      if(device.is_cuda()&&dtype==at::kBFloat16&&act!="relu"&&act!="gelu")return gemma_gated_cuda(gu,act!="silu"&&act!="swish");
#endif
      auto width=gu.size(-1)/2;
      return activate(gu.narrow(-1,0,width))*gu.narrow(-1,width,width);
    };
    auto output=decoder_moe_dispatch(input,gate_up,down,ids,probability,activation);
    return norm(dense,p+"post_feedforward_layernorm_1")+norm(output.to(dtype),p+"post_feedforward_layernorm_2");
  }
  at::Tensor audio(const AudioInput& media) const {
    auto channels=cfg.integer("audio_config.hidden_size",1024);
    auto heads=cfg.integer("audio_config.num_attention_heads",cfg.integer("audio_config.conf_num_attention_heads",8));
    auto layers=cfg.integer("audio_config.num_hidden_layers",cfg.integer("audio_config.conf_num_hidden_layers",12));
    auto chunk=cfg.integer("audio_config.attention_chunk_size",12);
    auto past=cfg.integer("audio_config.attention_context_left",13)-1;
    auto future=cfg.integer("audio_config.attention_context_right",0);
    auto eps=cfg.real("audio_config.rms_norm_eps",1e-6),cap=cfg.real("audio_config.attention_logit_cap",50.);
    auto clip=std::min(cfg.real("audio_config.gradient_clipping",1e10),static_cast<double>(std::numeric_limits<float>::max()));
    TORCH_CHECK(media.features.dim()==2 && media.features.size(1)==128 && media.validity.numel()==media.features.size(0),"Invalid Gemma audio features");
    TORCH_CHECK(channels>0 && heads>0 && channels%heads==0 && chunk>0 && past>=0 && future>=0 && cap>0,"Invalid Gemma audio architecture");
    auto root=weights.count("model.audio_tower.subsample_conv_projection.layer0.conv.weight")?std::string("model."):std::string();
    auto aw=[&](const std::string& name)->const at::Tensor& { return weight(weights,root+name); };
    auto al=[&](const at::Tensor& value,const std::string& name) {
      auto base=root+name;
      auto input=value;
      auto bounds=audio_bounds.find(base);
      if(bounds!=audio_bounds.end()) input=input.clamp(bounds->second[0],bounds->second[1]);
      auto key=base+".weight";
      if(!weights.count(key))key=base+".linear.weight";
      std::optional<at::Tensor> bias;
      if(weights.count(base+".bias"))bias=weight(weights,base+".bias");
      auto output=at::linear(input.to(weight(weights,key).scalar_type()),weight(weights,key),bias);
      if(bounds!=audio_bounds.end())output=output.clamp(bounds->second[2],bounds->second[3]);
      return output;
    };
    auto an=[&](const at::Tensor& x,const std::string& name) {
      auto f=x.to(at::kFloat);
      return (f*at::pow(f.square().mean(-1,true)+eps,-.5)*weight(norm_scales,root+name+".weight")).to(x.scalar_type());
    };
    auto valid=media.validity.to(at::kBool);
    auto states=media.features.to(dtype).unsqueeze(0).unsqueeze(0);
    for(int block=0;block<2;++block) {
      auto p="audio_tower.subsample_conv_projection.layer"+std::to_string(block);
      auto filter=aw(p+".conv.weight").to(at::kFloat);
      auto stride=cfg.integer("audio_config.sscp_conv_stride_size."+std::to_string(block)+".0",2);
      // Convolution's mathematical zero boundary is intrinsic to the filter;
      // actual audio frame lengths are never rounded up or padded into batches.
      states=at::conv2d((states*valid.view({1,1,-1,1})).to(at::kFloat),filter,std::nullopt,{stride,stride},{filter.size(2)/2,1},{1,1},1).to(dtype);
      auto ix=at::arange(states.size(2),valid.options().dtype(at::kLong))*stride;
      valid=valid.index_select(0,ix.clamp_max(valid.size(0)-1));
      auto x=states.permute({0,2,3,1}).to(at::kFloat);
      auto centered=x-x.mean(-1,true);
      x=centered/at::sqrt(centered.square().mean(-1,true)+eps);
      if(weights.count(root+p+".norm.weight"))x=x*aw(p+".norm.weight").to(at::kFloat);
      states=at::relu(x.to(dtype).permute({0,3,1,2}));
    }
    states=states.permute({0,2,3,1}).reshape({states.size(2),-1});
    states=al(states,"audio_tower.subsample_conv_projection.input_proj_linear");

    auto time=states.size(0),dim=channels/heads;
    auto relative_span=(chunk+past+future)/2+1;
    auto timing=at::arange(relative_span-1,-1,-1,states.options().dtype(at::kFloat));
    auto inv=at::exp(at::arange(channels/2,timing.options())*(-std::log(10000.)/std::max<int64_t>(channels/2-1,1)));
    auto angles=timing.unsqueeze(1)*inv;
    auto signal=at::cat({angles.sin(),angles.cos()},-1).to(dtype);
    auto ff=[&](const at::Tensor& value,const std::string& p) {
      auto x=an(value.clamp(-clip,clip),p+"pre_layer_norm");

      x=al(x,p+"ffw_layer_1");
      x=at::silu(x.to(at::kFloat)).to(dtype);
      x=al(x,p+"ffw_layer_2").clamp(-clip,clip);
      x=an(x,p+"post_layer_norm");
      x=value+x*cfg.real("audio_config.residual_weight",.5);return x;
    };
    for(int64_t layer=0;layer<layers;++layer) {
      auto p="audio_tower.layers."+std::to_string(layer)+".";
      states=ff(states,p+"feed_forward1.");
      auto x=an(states.clamp(-clip,clip),p+"norm_pre_attn");
      auto q=al(x,p+"self_attn.q_proj").to(at::kFloat).view({time,heads,dim});
      auto k=al(x,p+"self_attn.k_proj").to(at::kFloat).view({time,heads,dim});
      auto v=al(x,p+"self_attn.v_proj").to(at::kFloat).view({time,heads,dim});
      auto scale=at::softplus(aw(p+"self_attn.per_dim_scale").to(at::kFloat)).to(dtype).to(at::kFloat);
      q=q*(1./std::sqrt(static_cast<double>(dim))/std::log(2.))*scale;
      k=k*(std::log(1.+std::exp(1.))/std::log(2.));
      auto relative=al(signal,p+"self_attn.relative_k_proj").to(at::kFloat).view({relative_span,heads,dim});
      std::vector<at::Tensor> blocks;
      for(int64_t start=0;start<time;start+=chunk) {
        auto count=std::min(chunk,time-start),key_start=std::max<int64_t>(0,start-past),key_end=std::min(time,start+count+future);
        auto queries=at::arange(start,start+count,q.options().dtype(at::kLong));
        auto keys=at::arange(key_start,key_end,q.options().dtype(at::kLong));
        auto delta=keys.unsqueeze(0)-queries.unsqueeze(1);
        auto qs=q.narrow(0,start,count);
        // The reference relative branch uses FP32 GEMM rather than a separate
        // product/reduction tree. Gather only the natural key offsets afterward.
        auto relative_logits=at::matmul(qs.permute({1,0,2}),relative.permute({1,2,0}));
        // Preserve the reference's flatten/reshape relative shift without
        // materializing its artificial boundary rows or columns. A shifted
        // element may come from the preceding query row; direct position
        // lookup changes the trained attention computation.
        auto context=chunk+past+future,span=relative.size(0);
        auto shifted=(queries-start).unsqueeze(1)*context+(keys-(start-past)).unsqueeze(0);
        auto source_row=at::floor_divide(shifted,context+1);
        auto source_column=at::remainder(shifted,context+1);
        auto rel_index=(source_row*span+source_column.clamp_max(span-1)).reshape({1,-1}).expand({heads,-1});
        auto shifted_relative=relative_logits.reshape({heads,-1}).gather(-1,rel_index).reshape({heads,count,key_end-key_start});
        shifted_relative=shifted_relative.masked_fill(source_column.ge(span).unsqueeze(0),0);
        auto logits=at::matmul(qs.permute({1,0,2}),k.narrow(0,key_start,key_end-key_start).permute({1,2,0}))+
          shifted_relative;
        logits=at::tanh(logits/cap)*cap;
        // The Conformer reference excludes the exact context horizon endpoint.
        auto allowed=(((delta<=0)&(delta>-past))|((delta>0)&(delta<future)))&
          valid.narrow(0,key_start,key_end-key_start).unsqueeze(0);
        logits=logits.masked_fill(allowed.logical_not().unsqueeze(0),cfg.real("audio_config.attention_invalid_logits_value",-1e9));
        blocks.push_back(at::matmul(at::softmax(logits,-1),v.narrow(0,key_start,key_end-key_start).permute({1,0,2})).permute({1,0,2}).reshape({count,channels}));
      }
      auto attended=al(at::cat(blocks,0).to(dtype),p+"self_attn.post").clamp(-clip,clip);
      states=states+an(attended,p+"norm_post_attn");

      x=al(an(states,p+"lconv1d.pre_layer_norm"),p+"lconv1d.linear_start");
      // Conformer's fused GLU has one BF16 rounding after FP32 sigmoid/product.
      auto gated=(x.narrow(-1,0,channels).to(at::kFloat)*
        at::sigmoid(x.narrow(-1,channels,channels).to(at::kFloat))).to(dtype);
      auto filter=aw(p+"lconv1d.depthwise_conv1d.weight").to(at::kFloat);
      // ATen's convolution boundary handles the causal receptive field without
      // appending any synthetic frames to the sequence tensor.
      auto conv=at::conv1d(gated.transpose(0,1).unsqueeze(0).to(at::kFloat),filter,std::nullopt,{1},{filter.size(-1)-1},{1},channels);
      conv=conv.narrow(-1,0,time).squeeze(0).transpose(0,1).to(dtype).clamp(-clip,clip);
      states=states+al(at::silu(an(conv,p+"lconv1d.conv_norm").to(at::kFloat)).to(dtype),p+"lconv1d.linear_end");

      states=an(ff(states,p+"feed_forward2.").clamp(-clip,clip),p+"norm_out");

    }
    if(weights.count(root+"audio_tower.output_proj.weight"))states=al(states,"audio_tower.output_proj");

    states=states*valid.unsqueeze(-1);
    auto f=states.to(at::kFloat);
    auto projected=al((f/at::sqrt(f.square().mean(-1,true)+eps)).to(dtype),"embed_audio.embedding_projection");
    auto selected=media.selected_frames.defined()?media.selected_frames:at::nonzero(valid).squeeze(-1);
    return projected.index_select(0,selected);
  }
  at::Tensor vision(const ImageInput& image) const {
    auto [frames,height,width]=image.grid_thw;
    auto patch=cfg.integer("vision_config.patch_size",16),pool=cfg.integer("vision_config.pooling_kernel_size",3);
    auto vh=cfg.integer("vision_config.hidden_size"),nh=cfg.integer("vision_config.num_attention_heads");
    auto nk=cfg.integer("vision_config.num_key_value_heads",nh),dim=cfg.integer("vision_config.head_dim",vh/(nh?nh:1));
    auto nlayer=cfg.integer("vision_config.num_hidden_layers");
    auto eps=cfg.real("vision_config.rms_norm_eps",1e-6);
    TORCH_CHECK(frames==1 && height>0 && width>0 && height%pool==0 && width%pool==0 && image.merge_size==pool &&
      image.pixels.sizes()==at::IntArrayRef({height*width,patch*patch*3}),"Invalid packed Gemma vision patches");
    TORCH_CHECK(vh>0 && nh>0 && nk>0 && nh%nk==0 && dim>0 && dim%4==0 && nlayer>0,"Invalid Gemma vision geometry");
    TORCH_CHECK(!cfg.boolean("vision_config.use_clipped_linears",false),"Clipped Gemma vision linears unsupported");
    auto root=weights.count("model.vision_tower.patch_embedder.input_proj.weight")?std::string("model."):std::string();
    auto vw=[&](const std::string& name)->const at::Tensor& { return weight(weights,root+name); };
    auto vl=[&](const at::Tensor& x,const std::string& name) {
      auto key=root+name+".weight";
      if(!weights.count(key)) key=root+name+".linear.weight";
      return at::linear(x,weight(weights,key));
    };
    auto vn=[&](const at::Tensor& x,const std::string& name) {
#ifdef TEI_TORCH_CUDA_KERNELS
      if(device.is_cuda()&&dtype==at::kBFloat16) {
        auto scale=name.empty()?norm_ones.at(x.size(-1)):weight(norm_scales,root+name+".weight");
        return gemma_norm_cuda(x.contiguous(),scale,eps,true);
      }
#endif
      auto f=x.to(at::kFloat)/at::sqrt(x.to(at::kFloat).square().mean(-1,true)+eps);
      if(!name.empty()) f=f*vw(name+".weight").to(at::kFloat);
      return f.to(x.scalar_type());
    };
    auto ix=at::arange(height*width,image.pixels.options().dtype(at::kLong));
    auto columns=ix.remainder(width),rows=at::floor_divide(ix,width);
    auto table=vw("vision_tower.patch_embedder.position_embedding_table");
    auto states=vl((image.pixels-.5)*2.,"vision_tower.patch_embedder.input_proj")+
      table.select(0,0).index_select(0,columns)+table.select(0,1).index_select(0,rows);
    auto col_angles=columns.to(at::kFloat).unsqueeze(1)*vision_frequency,row_angles=rows.to(at::kFloat).unsqueeze(1)*vision_frequency;
    auto cos=at::cat({col_angles.cos(),row_angles.cos()},-1).to(dtype).unsqueeze(1);
    auto sin=at::cat({col_angles.sin(),row_angles.sin()},-1).to(dtype).unsqueeze(1);
    auto rotate2d=[&](const at::Tensor& x) {
      std::vector<at::Tensor> parts;
      for(int axis=0;axis<2;++axis) {
        auto part=x.narrow(-1,axis*dim/2,dim/2),a=part.narrow(-1,0,dim/4),b=part.narrow(-1,dim/4,dim/4);
        auto c=cos.narrow(-1,axis*dim/4,dim/4),s=sin.narrow(-1,axis*dim/4,dim/4);
        parts.push_back(at::cat({a*c-b*s,b*c+a*s},-1));
      }
      return at::cat(parts,-1);
    };
    const int32_t offsets[]={0,static_cast<int32_t>(height*width)};
    auto cu=at::arange(2,ix.options().dtype(at::kInt))*(height*width);
    PackedInput input{ix,ix,ix,cu,offsets,1,height*width};
    for(int64_t i=0;i<nlayer;++i) {
      auto p="vision_tower.encoder.layers."+std::to_string(i)+".";
      auto normalized=vn(states,p+"input_layernorm");
      auto q=rotate2d(vn(vl(normalized,p+"self_attn.q_proj").view({height*width,nh,dim}),p+"self_attn.q_norm"));
      auto k=rotate2d(vn(vl(normalized,p+"self_attn.k_proj").view({height*width,nk,dim}),p+"self_attn.k_norm"));
      auto v=vn(vl(normalized,p+"self_attn.v_proj").view({height*width,nk,dim}),"");
      auto output=packed_attention(q,k,v,input,1.).reshape({height*width,nh*dim});
      states=states+vn(vl(output,p+"self_attn.o_proj"),p+"post_attention_layernorm");
      normalized=vn(states,p+"pre_feedforward_layernorm");
      auto gate=vl(normalized,p+"mlp.gate_proj");
      auto activation=cfg.string("vision_config.hidden_activation","gelu_pytorch_tanh");
#ifdef TEI_TORCH_CUDA_KERNELS
      if(device.is_cuda()&&dtype==at::kBFloat16&&activation!="silu"&&activation!="gelu")gate=gemma_activation_cuda(gate,true);
      else
#endif
      gate=activation=="silu"?at::silu(gate):at::gelu(gate,activation=="gelu"?"none":"tanh");
      auto mlp=vl(gate*vl(normalized,p+"mlp.up_proj"),p+"mlp.down_proj");
      states=states+vn(mlp,p+"post_feedforward_layernorm");
    }
    auto bins=at::floor_divide(columns,pool)+at::floor_divide(rows,pool)*(width/pool);
    auto pooled=at::zeros({height*width/(pool*pool),vh},states.options().dtype(at::kFloat));
    pooled.index_add_(0,bins,states.to(at::kFloat)/(pool*pool));
    pooled=pooled.to(dtype).to(at::kFloat)*std::sqrt(static_cast<double>(vh));
    if(cfg.boolean("vision_config.standardize",false)) pooled=(pooled-vw("vision_tower.std_bias").to(at::kFloat))*vw("vision_tower.std_scale").to(at::kFloat);
    pooled=pooled.to(dtype);
    auto f=pooled.to(at::kFloat);
    return vl((f/at::sqrt(f.square().mean(-1,true)+eps)).to(dtype),"embed_vision.embedding_projection");
  }
  bool has(const std::string& key) const { return weights.count(prefix + key); }
  const at::Tensor& w(const std::string& key) const {
    auto it = weights.find(prefix + key);
    TORCH_CHECK(it != weights.end(), "Missing Gemma weight: ", prefix, key);
    return it->second;
  }
  std::string key(const std::string& name) const {
    return cfg.values.count("text_config." + name) ? "text_config." + name : name;
  }
  int64_t integer(const std::string& name, int64_t fallback) const { return cfg.integer(key(name), fallback); }
  double real(const std::string& name, double fallback) const { return cfg.real(key(name), fallback); }
  bool boolean(const std::string& name, bool fallback) const { return cfg.boolean(key(name), fallback); }
  std::string string(const std::string& name, const std::string& fallback) const { return cfg.string(key(name), fallback); }
  at::Tensor linear(const at::Tensor& x, const std::string& name) const {
    return at::linear(x, w(name + ".weight"), has(name + ".bias") ? std::optional<at::Tensor>(w(name + ".bias")) : std::nullopt);
  }
  at::Tensor norm(const at::Tensor& x, const std::string& name, bool unit = false) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    if(device.is_cuda() && dtype==at::kBFloat16 && x.size(-1)<=8192) {
      const auto& scale=unit?norm_ones.at(x.size(-1)):weight(norm_scales,prefix+name+".weight");
      return gemma_norm_cuda(x,scale,epsilon,!legacy);
    }
#endif
    auto f = x.to(at::kFloat);
    auto variance=f.square().mean(-1,true)+epsilon;
    f = legacy ? f*at::rsqrt(variance) : f/at::sqrt(variance);
    if (!unit) {
      f = f * weight(norm_scales,prefix + name + ".weight");
    }
    return f.to(x.scalar_type());
  }
  at::Tensor activate(const at::Tensor& x) const {
    const auto act = string("hidden_activation", "gelu_pytorch_tanh");
    if(act=="gelu")return at::gelu(x,"none");
#ifdef TEI_TORCH_CUDA_KERNELS
    if(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&act!="relu")return gemma_activation_cuda(x,act!="silu"&&act!="swish");
#endif
    if (act == "silu" || act == "swish") return x/(at::ones({},x.options())+at::exp(-x));
    if (act == "relu") return at::relu(x);
    auto cube=(x*x)*x;
    auto alpha=x+at::scalar_tensor(.044715,x.options())*cube;
    auto value=at::tanh(at::scalar_tensor(M_2_SQRTPI*M_SQRT1_2,x.options())*alpha);
    return (at::scalar_tensor(.5,x.options())*x)*(at::ones({},x.options())+value);
  }
  at::Tensor rotate(const at::Tensor& x, const Layer& layer, const at::Tensor& positions) const {
    auto cos = layer.cos.index_select(0, positions).unsqueeze(1);
    auto sin = layer.sin.index_select(0, positions).unsqueeze(1);
    auto half = x.size(-1) / 2;
    auto a = x.slice(-1, 0, half), b = x.slice(-1, half);
    // Gemma4's composed reference rounds each BF16 product before addition;
    // retaining activation-dtype ATen operations preserves those stages.
    return at::cat({a * cos - b * sin, b * cos + a * sin}, -1).to(x.scalar_type());
  }
  at::Tensor attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                       const PackedInput& input, const Layer& layer, bool use_causal) const {
    const bool causal=use_causal;
    auto left = layer.sliding ? (causal ? layer.window - 1 : layer.window / 2) : -1;
    auto right = layer.sliding ? (causal ? 0 : layer.window / 2) : -1;
    auto scale = legacy ? 1.0 / std::sqrt(real("query_pre_attn_scalar", layer.dim)) : 1.0;
    if (device.is_cuda()) {
      if(layer.dim>256) {
        TORCH_CHECK(!layer.sliding,"Torch512-head packed sliding attention requires a specialized kernel");
        // cuDNN's long-sequence varlen path accepts head512, but mixed-length
        // parity is not established. CUTLASS handles all actual lengths here.
        auto repeat=heads/layer.kv;
        auto keys=repeat==1?k:at::repeat_interleave(k,repeat,1);
        auto values=repeat==1?v:at::repeat_interleave(v,repeat,1);
        return std::get<0>(at::_efficient_attention_forward(q.unsqueeze(0),keys.unsqueeze(0),values.unsqueeze(0),
          std::nullopt,input.cumulative,input.cumulative,input.max_sequence,input.max_sequence,0.,causal?1:0,false,scale)).squeeze(0);
      }
      auto output=std::get<0>(at::_flash_attention_forward(q, k, v, input.cumulative,
        input.cumulative, input.max_sequence, input.max_sequence, 0., causal,
        false, scale, left, right));
      if(causal && layer.sliding) {
        for(const auto& image:input.images) {
          auto key_start=std::max(image.sequence_start,image.token_start-(layer.window-1));
          auto key_len=image.token_start+image.token_count-key_start;
          auto cuq=at::arange(2,q.options().dtype(at::kInt))*image.token_count;
          auto cuk=at::arange(2,q.options().dtype(at::kInt))*key_len;
          auto correction=std::get<0>(at::_flash_attention_forward(q.narrow(0,image.token_start,image.token_count),
            k.narrow(0,key_start,key_len),v.narrow(0,key_start,key_len),cuq,cuk,image.token_count,key_len,
            0.,false,false,scale,layer.window-1,-1));
          output.narrow(0,image.token_start,image.token_count).copy_(correction);
        }
      }
      return output;
    }
    std::vector<at::Tensor> rows;
    for (int64_t i = 0; i < input.batch; ++i) {
      auto start = input.offsets[i], len = input.offsets[i + 1] - start;
      auto slice = [&](const at::Tensor& t) { return t.narrow(0, start, len).transpose(0, 1).unsqueeze(0); };
      std::optional<at::Tensor> mask;
      if (layer.sliding) {
        auto ix = at::arange(len, input.ids.options());
        auto delta = ix.unsqueeze(0) - ix.unsqueeze(1);
        mask = delta >= -left;
        if(!causal || input.images.empty()) mask=*mask & (delta<=right);
        if (causal) {
          auto allowed=delta<=0;
          for(const auto& image:input.images) {
            auto image_start=image.token_start-start,image_end=image_start+image.token_count;
            if(image_start>=0 && image_end<=len)
              allowed=allowed|((ix>=image_start)&(ix<image_end)).unsqueeze(1).logical_and(((ix>=image_start)&(ix<image_end)).unsqueeze(0));
          }
          mask=*mask & allowed;
        }
      }
      rows.push_back(at::scaled_dot_product_attention(slice(q), slice(k), slice(v), mask,
        0., causal && !mask.has_value(), scale, true).squeeze(0).transpose(0, 1));
    }
    return at::cat(rows, 0);
  }
  int64_t override_int(int64_t index, const std::string& name, int64_t fallback) const {
    // Accept both "05" (released checkpoints) and "5" layer keys.
    for (auto stem : {"per_layer_config." + std::to_string(index),
                      "per_layer_config." + (index < 10 ? std::string("0") : std::string()) + std::to_string(index)}) {
      if (cfg.values.count(key(stem + "." + name))) return integer(stem + "." + name, fallback);
    }
    return fallback;
  }
 public:
  void prepare_input(PackedInput& input) const override {
    for(auto& audio:input.audios) {
      auto valid=audio.validity.to(at::kBool);
      for(int block=0;block<2;++block) {
        auto stride=cfg.integer("audio_config.sscp_conv_stride_size."+std::to_string(block)+".0",2);
        TORCH_CHECK(stride>0&&valid.numel()>0,"Invalid Gemma audio subsampling geometry");
        auto length=(valid.numel()+stride-1)/stride;
        auto indices=at::arange(length,valid.options().dtype(at::kLong))*stride;
        valid=valid.index_select(0,indices.clamp_max(valid.numel()-1));
      }
      audio.selected_frames=at::nonzero(valid).squeeze(-1).contiguous();
      TORCH_CHECK(audio.selected_frames.numel()==audio.token_count,"Gemma selected audio frame count does not match token span");
    }
  }
  bool supports_images() const override { return !legacy && cfg.integer("vision_config.num_hidden_layers",0)>0; }
  bool supports_audio() const override { return embedding2 && cfg.integer("audio_config.hidden_size",0)>0; }
  int64_t output_width() const override { return embedding2 ? integer("embedding_dim", hidden) : hidden; }
  int64_t classification_width() const override {
    auto score=weights.find("score.weight");
    return legacy || embedding2 || score==weights.end()?0:score->second.size(0);
  }
  Gemma(const Options& options, Weights& tensors, c10::Device dev, at::ScalarType type)
    : cfg(options), weights(tensors), device(dev), dtype(type) {
    family = cfg.string("model_type", "");
    legacy = family == "gemma3" || family == "gemma3_text";
    embedding2 = family == "embedding_gemma2";
    hidden = integer("hidden_size", 0); heads = integer("num_attention_heads", 0);
    layers = integer("num_hidden_layers", 0); epsilon = real("rms_norm_eps", 1e-6);
    ple_width = integer("hidden_size_per_layer_input", 0);
    causal = legacy ? !boolean("use_bidirectional_attention", false) : false;
    for (const auto& p : {"model.language_model.", "language_model.", "model.", ""}) {
      if (weights.count(std::string(p) + "embed_tokens.weight")) { prefix = p; break; }
    }
  }
  void ready() override {
    TORCH_CHECK(hidden > 0 && heads > 0 && layers > 0, "Invalid Gemma architecture dimensions");
    if(boolean("enable_moe_block",false)) TORCH_CHECK(!legacy && !embedding2,"Gemma4 expert routing requires Gemma4 architecture");
    TORCH_CHECK(device.is_cpu() || device.is_cuda(), "Gemma packed attention supports CPU/CUDA only");
    if (device.is_cuda()) TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "Packed CUDA Gemma requires half/bfloat16");
    w("embed_tokens.weight"); w("norm.weight");
    norm_ones[hidden]=at::ones({hidden},at::TensorOptions().dtype(at::kFloat).device(device));
    if(cfg.integer("vision_config.num_hidden_layers",0)>0) {
      auto width=cfg.integer("vision_config.hidden_size",0),head_count=cfg.integer("vision_config.num_attention_heads",0);
      TORCH_CHECK(width>0&&head_count>0,"Invalid Gemma vision hidden/head widths");
      auto head_width=cfg.integer("vision_config.head_dim",width/head_count);
      TORCH_CHECK(head_width>0&&head_width%4==0,"Invalid Gemma vision rotary width");
      auto theta=cfg.real("vision_config.rope_parameters.rope_theta",100.);
      std::vector<float> inverse(head_width/4);
      for(int64_t i=0;i<head_width/4;++i)inverse[i]=1.f/static_cast<float>(std::pow(theta,4.*i/head_width));
      vision_frequency=at::from_blob(inverse.data(),{head_width/4},at::TensorOptions().dtype(at::kFloat)).clone().to(device);
      norm_ones[head_width]=at::ones({head_width},at::TensorOptions().dtype(at::kFloat).device(device));
    }
    for(const auto& [name,value]:weights) {
      if(name.ends_with(".weight") && name.find("norm")!=std::string::npos) {
        auto scale=value.to(at::kFloat);
        norm_scales[name]=legacy?scale+1.:scale;
      }
      if(name.ends_with(".input_min")) {
        auto base=name.substr(0,name.size()-10);
        audio_bounds[base]={value.item<double>(),weight(weights,base+".input_max").item<double>(),
          weight(weights,base+".output_min").item<double>(),weight(weights,base+".output_max").item<double>()};
      }
    }
    auto first_shared = layers - integer("num_kv_shared_layers", 0);
    TORCH_CHECK(first_shared >= 0, "Gemma shared KV layer count exceeds layer count");
    auto pattern = integer("_sliding_window_pattern", integer("sliding_window_pattern", 6));
    TORCH_CHECK(pattern > 0, "Gemma3 sliding pattern must be positive");
    for (int64_t i = 0; i < layers; ++i) {
      Layer layer;
      layer.p = "layers." + std::to_string(i) + ".";
      auto kind = string("layer_types." + std::to_string(i), "full_attention");
      layer.sliding = legacy ? (i + 1) % pattern != 0 : kind == "sliding_attention";
      TORCH_CHECK(legacy || kind == "sliding_attention" || kind == "full_attention", "Unsupported Gemma attention type ", kind);
      auto default_dim = integer("head_dim", hidden / heads);
      layer.dim = embedding2 ? override_int(i, "head_dim", default_dim) :
        (layer.sliding || legacy ? default_dim : integer("global_head_dim", default_dim));
      layer.k_equals_v = !legacy && !layer.sliding && boolean("attention_k_eq_v", false);
      layer.kv = override_int(i, "num_key_value_heads", layer.k_equals_v ?
        integer("num_global_key_value_heads", integer("num_key_value_heads", heads)) : integer("num_key_value_heads", heads));
      TORCH_CHECK(layer.dim > 0 && layer.dim % 2 == 0 && layer.kv > 0 && heads % layer.kv == 0, "Invalid Gemma attention geometry at layer ", i);
      if(!norm_ones.count(layer.dim))norm_ones[layer.dim]=at::ones({layer.dim},at::TensorOptions().dtype(at::kFloat).device(device));
      if (device.is_cuda()) TORCH_CHECK(layer.dim % 8 == 0,
        "Torch varlen FlashAttention requires head dimension divisible by 8; this Gemma layer requires ", layer.dim);
      layer.window = integer("sliding_window", 4096) * (embedding2 ? 2 : 1);
      TORCH_CHECK(layer.window > 0, "Gemma sliding window must be positive");
      layer.shared = !legacy && first_shared < layers && i >= first_shared;
      layer.store = false;
      if (!layer.shared && first_shared < layers) {
        layer.store = true;
        for (int64_t j = i + 1; j < first_shared; ++j)
          if (string("layer_types." + std::to_string(j), "full_attention") == kind) layer.store = false;
      }
      if (legacy) {
        layer.qkv = at::cat({w(layer.p + "self_attn.q_proj.weight"), w(layer.p + "self_attn.k_proj.weight"), w(layer.p + "self_attn.v_proj.weight")}, 0);
      } else {
        w(layer.p + "self_attn.q_proj.weight");
        if (!layer.shared) { w(layer.p + "self_attn.k_proj.weight"); if (!layer.k_equals_v) w(layer.p + "self_attn.v_proj.weight"); }
      }
      layer.gate_up = at::cat({w(layer.p + "mlp.gate_proj.weight"), w(layer.p + "mlp.up_proj.weight")}, 0);
      w(layer.p + "mlp.down_proj.weight");
      auto rope_kind = layer.sliding ? "sliding_attention" : "full_attention";
      auto theta = legacy ? real(layer.sliding ? "rope_local_base_freq" : "rope_theta", 10000.) :
        real(std::string("rope_parameters.") + rope_kind + ".rope_theta", 10000.);
      TORCH_CHECK(theta > 0, "Gemma RoPE theta must be positive");
      auto pairs = layer.dim / 2;
      // Match Candle's host FP32 powf frequency construction exactly.
      std::vector<float> inverse(pairs);
      for(int64_t j=0;j<pairs;++j)inverse[j]=1.f/std::pow(float(theta),float(2*j)/float(layer.dim));
      auto freq=at::from_blob(inverse.data(),{pairs},at::TensorOptions().dtype(at::kFloat)).clone().to(device);
      if (!legacy && !embedding2 && !layer.sliding) {
        auto factor = real("rope_parameters.full_attention.partial_rotary_factor", 1.);
        TORCH_CHECK(factor > 0. && factor <= 1., "Invalid Gemma partial rotary factor");
        freq = freq * (at::arange(pairs, freq.options()) < static_cast<int64_t>(factor * pairs));
      }
      bool reused=false;
      for(const auto& prior:stack) {
        if(prior.dim==layer.dim && prior.sliding==layer.sliding) {
          layer.cos=prior.cos;layer.sin=prior.sin;reused=true;break;
        }
      }
      if(!reused) {
        auto angles = at::arange(integer("max_position_embeddings", 8192), freq.options()).unsqueeze(1) * freq.unsqueeze(0);
        layer.cos = angles.cos().to(dtype); layer.sin = angles.sin().to(dtype);
      }
      stack.push_back(std::move(layer));
    }
  }
  at::Tensor forward(const PackedInput& input) const override { return forward_impl(input,causal); }
  at::Tensor predict(const PackedInput& input,bool tokens) const override {
    TORCH_CHECK(classification_width()>0,"Gemma classifier score head is missing");
    TORCH_CHECK(!tokens,"Gemma4 classifier supports sequence logits only");
    auto states=forward_impl(input,true);
    std::vector<int64_t> indices;
    for(int64_t i=1;i<=input.batch;++i)indices.push_back(input.offsets[i]-1);
    auto ids=at::from_blob(indices.data(),{input.batch},at::TensorOptions().dtype(at::kLong)).clone().to(device);
    auto logits=at::linear(states.index_select(0,ids),weight(weights,"score.weight"));
    auto cap=real("final_logit_softcapping",cfg.real("final_logit_softcapping",0));
    return cap>0?at::tanh(logits/cap)*cap:logits;
  }
  at::Tensor forward_impl(const PackedInput& input,bool is_causal) const {
    auto scale = at::scalar_tensor(std::sqrt(static_cast<double>(hidden)), w("embed_tokens.weight").options());
    auto states = at::embedding(w("embed_tokens.weight"), input.ids) * scale;
    at::Tensor ple;
    auto inject_images=[&]() {
      TORCH_CHECK(input.images.empty() || supports_images(),"Gemma vision tower unavailable");
      for(const auto& image:input.images) {
        auto features=vision(image);
        TORCH_CHECK(image.token_start>=0 && image.token_start+image.token_count<=states.size(0) && features.size(0)==image.token_count,"Gemma image token geometry mismatch");
        states.slice(0,image.token_start,image.token_start+image.token_count).copy_(features);
      }
    };
    if(embedding2) inject_images();
    TORCH_CHECK(input.audios.empty() || supports_audio(),"Gemma audio tower unavailable");
    for(const auto& media:input.audios) {
      auto features=audio(media);
      TORCH_CHECK(media.token_start>=0 && media.token_start+media.token_count<=states.size(0) && features.size(0)==media.token_count,"Gemma audio token geometry mismatch");
      states.narrow(0,media.token_start,media.token_count).copy_(features);
    }

    if (ple_width > 0) {
      auto stem = embedding2 ? "ple." : "";
      auto projected = linear(states, std::string(stem) + "per_layer_model_projection") *
        at::scalar_tensor(1./std::sqrt(static_cast<double>(hidden)),states.options());

      ple = norm(projected.view({input.ids.size(0), layers, ple_width}), std::string(stem) + "per_layer_projection_norm");
      if (!embedding2) {
        auto token_inputs = at::embedding(w("embed_tokens_per_layer.weight"), input.ids) *
          at::scalar_tensor(std::sqrt(static_cast<double>(ple_width)),states.options());
        ple = (token_inputs.view_as(ple) + ple) * at::scalar_tensor(1./std::sqrt(2.),states.options());
      }
    }
    if(!embedding2) inject_images();

    std::array<std::pair<at::Tensor,at::Tensor>,2> shared;
    for (int64_t i = 0; i < layers; ++i) {
      const auto& layer = stack[i]; auto p = layer.p;
      auto normalized = norm(states, p + "input_layernorm");

      at::Tensor q, k, v;
      if (legacy) {
        std::optional<at::Tensor> bias;
        if (has(p + "self_attn.q_proj.bias")) bias = at::cat({w(p + "self_attn.q_proj.bias"), w(p + "self_attn.k_proj.bias"), w(p + "self_attn.v_proj.bias")}, 0);
        auto qkv = at::linear(normalized, layer.qkv, bias).view({input.ids.size(0), heads + 2 * layer.kv, layer.dim});
        q = qkv.narrow(1, 0, heads); k = qkv.narrow(1, heads, layer.kv); v = qkv.narrow(1, heads + layer.kv, layer.kv);
      } else {
        q = linear(normalized, p + "self_attn.q_proj").view({input.ids.size(0), heads, layer.dim});
        if (!layer.shared) {
          k = linear(normalized, p + "self_attn.k_proj").view({input.ids.size(0), layer.kv, layer.dim});
          v = layer.k_equals_v ? k : linear(normalized, p + "self_attn.v_proj").view({input.ids.size(0), layer.kv, layer.dim});
        }
      }
      q = rotate(norm(q, p + "self_attn.q_norm"), layer, input.positions);
      if (layer.shared) {
        k = shared[layer.sliding].first; v = shared[layer.sliding].second;
        TORCH_CHECK(k.defined(), "Missing shared Gemma KV states at layer ", i);
      } else {
        k = rotate(norm(k, p + "self_attn.k_norm"), layer, input.positions);
        if (!legacy) v = norm(v, "", true);
        if (layer.store) shared[layer.sliding] = {k,v};
      }

      auto output = attention(q, k, v, input, layer,is_causal).reshape({input.ids.size(0), heads * layer.dim});

      output=linear(output,p+"self_attn.o_proj");
      states = states + norm(output,p+"post_attention_layernorm");

      normalized = norm(states, p + "pre_feedforward_layernorm");

      auto gu = at::linear(normalized, layer.gate_up);
      auto width = gu.size(-1) / 2;
      at::Tensor mlp;
#ifdef TEI_TORCH_CUDA_KERNELS
      auto act=string("hidden_activation","gelu_pytorch_tanh");
      if(device.is_cuda() && dtype==at::kBFloat16 && act!="relu" && act!="gelu")mlp=gemma_gated_cuda(gu,act!="silu" && act!="swish");
      else
#endif
      mlp=activate(gu.narrow(-1,0,width))*gu.narrow(-1,width,width);
      mlp=linear(mlp,p+"mlp.down_proj");

      if(boolean("enable_moe_block",false))mlp=moe(states,mlp,p);
      states = states + norm(mlp, p + "post_feedforward_layernorm");

      if (ple.defined()) {
        auto stem = p + (embedding2 ? "ple_block." : "");
        auto contribution = activate(linear(states, stem + "per_layer_input_gate")) * ple.select(1,i);
        states = states + norm(linear(contribution, stem + "per_layer_projection"), stem + "post_per_layer_input_norm");
      }
      if (!legacy) states = states * w(p + "layer_scalar");

    }
    auto output = norm(states, "norm");
    output=embedding2 ? linear(output, "embedding_projection") : output;
    return output;
  }
};
}
std::unique_ptr<Model> create_gemma(const Options& cfg, Weights& weights, c10::Device device, at::ScalarType dtype) {
  auto family = cfg.string("model_type", "");
  if (family != "gemma3" && family != "gemma3_text" && family != "gemma4" && family != "gemma4_text" && family != "embedding_gemma2") return nullptr;
  return std::make_unique<Gemma>(cfg, weights, device, dtype);
}
}

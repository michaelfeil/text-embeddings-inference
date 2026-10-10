// SPDX-License-Identifier: Apache-2.0
#include "decision_models.h"
#include <c10/core/InferenceMode.h>
#include <ATen/Parallel.h>
#include <iostream>
using namespace tei;
namespace {
void close(const at::Tensor&a,const at::Tensor&b,double tol=1e-5){TORCH_CHECK((a.to(at::kFloat)-b.to(at::kFloat)).abs().max().item<float>()<tol,"Decision expected result mismatch");}
void reject(const std::function<void()>&run){bool failed=false;try{run();}catch(const c10::Error&){failed=true;}TORCH_CHECK(failed,"Invalid decision request was accepted");}
void norm(Weights&w,const std::string&p,int64_t n){w[p+".weight"]=at::ones({n});w[p+".bias"]=at::zeros({n});}
void linear(Weights&w,const std::string&p,int64_t in,int64_t out,bool bias=true){w[p+".weight"]=at::zeros({out,in});if(bias)w[p+".bias"]=at::zeros({out});}
void attention(Weights&w,const std::string&p,int64_t n){w[p+".in_proj_weight"]=at::zeros({3*n,n});w[p+".in_proj_bias"]=at::zeros({3*n});linear(w,p+".out_proj",n,n);}
struct Input {
 std::vector<int32_t> offsets{0,3,5};PackedInput input;
 Input(){auto ids=at::tensor({1,2,3,4,5},at::TensorOptions().dtype(at::kLong));input={ids,at::zeros_like(ids),at::tensor({0,1,2,0,1},ids.options()),at::tensor({0,3,5},ids.options().dtype(at::kInt)),offsets.data(),2,3};}
};
void options_test(){
 Input in;auto hidden=at::arange(80).view({5,16}).to(at::kFloat)*.01;
 Weights backbone;backbone["lm_head.weight"]=at::arange(128).view({8,16}).to(at::kFloat)*.001;
 Options cfg;cfg.values={{"hidden_size","16"},{"model_type","qwen3_5"}};
 auto head=create_decision_head("option_tokens",{},cfg,{},backbone);backbone.clear();
 DecisionRequest first{DecisionRequest::Kind::OptionTokens},second{DecisionRequest::Kind::Warmup};first.token_ids={4,1};
 auto result=head->forward(hidden,in.input,{first,second});
 auto expectedweights=at::arange(128).view({8,16}).to(at::kFloat)*.001;
 close(result[0].logits,at::matmul(expectedweights.index_select(0,at::tensor({4,1},in.input.ids.options())),hidden[2]));
 close(result[1].logits,at::matmul(expectedweights[0],hidden[4]).view({1}));
 TORCH_CHECK(head->output_count(first)==2&&head->output_count(second)==1,"Incorrect option-token output count");
 close(result[0].action_probability,at::ones({}));first.token_ids={8};reject([&]{head->output_count(first);});
 cfg.values["model_type"]="gemma4";cfg.values["final_logit_softcapping"]=".5";cfg.values["tie_word_embeddings"]="false";cfg.values["text_config.tie_word_embeddings"]="true";
 backbone["model.embed_tokens.weight"]=expectedweights;auto gemma=create_decision_head("option_tokens",{},cfg,{},backbone);first.token_ids={4,1};
 auto scores=gemma->forward(hidden,in.input,{first,second});close(scores[0].logits,at::tanh(result[0].logits/.5)*.5);
 Weights readout;readout["weight"]=at::arange(255*16).view({255,16}).to(at::kBFloat16)*.001;
 auto pplx=create_decision_head("pplx",{},cfg,readout,{});first.token_ids={0,1};auto values=pplx->forward(hidden,in.input,{first,second});
 close(values[0].logits,at::linear(hidden[2].to(at::kBFloat16).unsqueeze(0),readout["weight"].narrow(0,0,2)).flatten().to(at::kFloat));
 first.token_ids={1,0};reject([&]{pplx->output_count(first);});Options bad;bad.values["pooling"]="mean";reject([&]{create_decision_head("pplx",bad,cfg,readout,{});});
 std::cout<<"Option tokens, tied embeddings, last-token selection, Gemma softcap and BF16 Pplx readout passed\n";
}
Weights laya_weights(){Weights w;w["type_emb.weight"]=at::zeros({3,16});norm(w,"scorer.0",16);linear(w,"scorer.1",16,16);linear(w,"scorer.3",16,1);linear(w,"act_head.0",20,256);linear(w,"act_head.2",256,2);
 w["act_head.0.weight"][0][16]=1.;w["act_head.0.weight"][0][18]=1.;w["act_head.2.weight"][0][0]=1.;
 auto p=std::string("head.layers.0");norm(w,p+".norm1",16);norm(w,p+".norm2",16);attention(w,p+".self_attn",16);linear(w,p+".linear1",16,64);linear(w,p+".linear2",64,16);return w;}
void laya_test(){Input in;Options cfg,backbone;cfg.values={{"head_layers","1"},{"max_len","32"},{"head_max_len","32"}};backbone.values["hidden_size"]="16";
 auto weights=laya_weights();auto head=create_decision_head("laya",cfg,backbone,weights,{});weights.clear();
 DecisionRequest first{DecisionRequest::Kind::Laya},second{DecisionRequest::Kind::Warmup};first.question_type=2;first.markers={0,2};
 auto hidden=at::sin(at::arange(80).view({5,16}).to(at::kFloat));auto out=head->forward(hidden,in.input,{first,second});close(out[0].logits,at::zeros({2}));close(out[1].logits,at::zeros({1}));
 close(out[0].action_probability,at::sigmoid(at::gelu(at::tensor(1.5),"none")));close(out[1].action_probability,at::sigmoid(at::gelu(at::tensor(1.),"none")));
 TORCH_CHECK(head->output_count(first)==2&&head->output_count(second)==1,"Wrong Laya output count");first.markers={3};reject([&]{head->forward(hidden,in.input,{first,second});});first.question_type=3;reject([&]{head->output_count(first);});
 std::cout<<"Laya head, learned action top1/entropy features, warmup and marker validation passed\n";
}
Weights clef_weights(){Weights w;for(auto p:{"hidden_norm","option_summary_norm","field_norm","option_norm"})norm(w,p,16);
 for(auto p:{"memory_projection","question_projection","option_question_projection","global_projection","option_context_projection","option_lexical_projection"})linear(w,p,16,16,false);
 w["option_lexical_projection.weight"]=at::eye(16);w["type_embedding.weight"]=at::zeros({3,16});linear(w,"residual_scorer.0",64,16);linear(w,"residual_scorer.3",16,1);
 w["prior_logit_scale"]=at::tensor(0.);w["joint_logit_scale"]=at::tensor(0.);w["residual_gate"]=at::tensor(-100.);
 auto p=std::string("evidence_layers.0");for(auto n:{"query_norm","memory_norm","feedforward_norm"})norm(w,p+"."+n,16);attention(w,p+".attention",16);linear(w,p+".feedforward.0",16,32);linear(w,p+".feedforward.3",32,16);
 p="layers.0";for(auto n:{"norm1","norm2","norm3"})norm(w,p+"."+n,16);attention(w,p+".self_attn",16);attention(w,p+".multihead_attn",16);linear(w,p+".linear1",16,32);linear(w,p+".linear2",32,16);return w;}
void clef_test(){Input in;Options cfg,backbone;cfg.values={{"hidden_size","16"},{"width","16"},{"heads","2"},{"feedforward","32"},{"routing_layers","1"},{"layers","1"}};backbone.values={{"hidden_size","16"},{"model_type","qwen3_5"}};
 auto weights=clef_weights();Weights bw;bw["lm_head.weight"]=at::cos(at::arange(128).view({8,16}).to(at::kFloat)*.13);auto lexical=bw["lm_head.weight"];
 auto head=create_decision_head("clef",cfg,backbone,weights,bw);weights.clear();bw.clear();
 DecisionRequest first{DecisionRequest::Kind::Clef},second{DecisionRequest::Kind::Warmup};first.fields={{0,{0,1},{{1,2},{2,3}}},{2,{1,2},{{0,2}}}};
 auto hidden=at::sin(at::arange(80).view({5,16}).to(at::kFloat)*.17);auto out=head->forward(hidden,in.input,{first,second});
 auto normalized=at::layer_norm(hidden.narrow(0,0,3),{16},at::ones({16}),at::zeros({16}),1e-5);std::vector<at::Tensor> expected;
 for(const auto&f:first.fields){auto anchor=normalized.narrow(0,f.question.first,f.question.second-f.question.first).mean(0)+normalized[2];anchor=anchor/anchor.square().sum().sqrt();for(auto[a,b]:f.options){auto word=lexical.index_select(0,in.input.ids.narrow(0,a,b-a)).mean(0);word=word/word.square().sum().sqrt();expected.push_back((word*anchor).sum());}}
 close(out[0].logits,at::stack(expected));TORCH_CHECK(head->output_count(first)==3&&head->output_count(second)==1,"Wrong Clef output counts");close(out[1].action_probability,at::ones({}));
 auto changed=hidden.clone();changed.narrow(0,3,2).fill_(17.);close(head->forward(changed,in.input,{first,second})[0].logits,out[0].logits);
 first.fields[0].options={{2,4}};reject([&]{head->forward(hidden,in.input,{first,second});});
 std::cout<<"Clef lexical prior/span means, routing/field decoder, variable option counts and sequence isolation passed\n";
}
}
int main(){c10::InferenceMode inference;at::set_num_threads(1);options_test();laya_test();clef_test();}

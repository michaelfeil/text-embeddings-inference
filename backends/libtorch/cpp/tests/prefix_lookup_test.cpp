#include "batching/prefix_lookup.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace tei;
static void check(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
static PrefixIdentity identity(int64_t n) {
  PrefixIdentity key;
  for(int64_t i=0;i<n;++i)key.tokens.push_back({{i+10,i,i%2,i/3,i/4,i/5}});
  return key;
}
static auto entry(PrefixIdentity key,int operation=0,bool ready=true) {
  auto value=std::make_shared<PrefixEntry>();prefix_fingerprints(key,key.tokens.size());
  value->identity=std::move(key);value->operation=operation;value->ready=[ready]{return ready;};return value;
}
// Original scalar token/media search, independent of the fingerprint ranking.
static int64_t reference(const PrefixIdentity& a,const PrefixIdentity& b,int64_t limit) {
  if(a.image_context!=b.image_context)return 0;
  int64_t n=0;
  while(n<limit&&n<int64_t(a.tokens.size())&&n<int64_t(b.tokens.size())&&a.tokens[n]==b.tokens[n])++n;
  for(const auto* key:{&a,&b})for(const auto& media:key->media)if(media.start<n) {
    const auto& other=key==&a?b:a;
    if(std::find(other.media.begin(),other.media.end(),media)==other.media.end()
      ||media.start+media.count>n)n=std::min(n,media.start);
  }
  return n;
}
static void compare(PrefixIdentity key,const std::vector<std::shared_ptr<PrefixEntry>>& entries,
  int operation,int64_t limit,bool checkpoint=false) {
  int64_t expected=0;
  for(const auto& e:entries)if(e->operation==operation&&e->ready()) {
    auto n=reference(key,e->identity,limit);
    if(!checkpoint||n==int64_t(e->identity.tokens.size()))expected=std::max(expected,n);
  }
  auto result=prefix_lookup(key,entries,operation,limit,checkpoint);
  check(result.length==expected,"Fingerprint lookup changed longest available prefix");
  check(bool(result.entry)==bool(expected),"Fingerprint lookup returned an invalid reader");
  if(result.entry)check(reference(key,result.entry->identity,limit)==expected,
    "Fingerprint lookup selected a non-matching identity");
}
int main() {
  auto key=identity(257);prefix_fingerprints(key,128);auto first=key.fingerprints;
  prefix_fingerprints(key,257);check(key.fingerprints.size()==4,"Incomplete block was hashed");
  check(std::equal(first.begin(),first.end(),key.fingerprints.begin()),"Extending fingerprints changed an earlier parent");
  prefix_fingerprints(key,65);check(key.fingerprints.size()==1,"Truncated identity retained later block digests");
  prefix_fingerprints(key,257);auto fresh=identity(257);prefix_fingerprints(fresh,257);
  check(key.fingerprints==fresh.fingerprints,"Rebuilt fingerprints differ from fresh prepared identity");

  std::mt19937_64 random(8943);
  for(int trial=0;trial<160;++trial) {
    auto query=identity(128+random()%130);query.image_context=trial%3==0;
    if(query.image_context) {
      PrefixMedia media;media.start=61;media.count=16;media.layout={1,4,4,2};
      media.hashed=trial%2;media.digest[0]=93;media.content={12,44,19};query.media.push_back(media);
    }
    std::vector<std::shared_ptr<PrefixEntry>> entries;
    for(int row=0;row<64;++row) {
      auto candidate=query;candidate.tokens.resize(1+random()%query.tokens.size());
      if(row%5)candidate.tokens[random()%candidate.tokens.size()].values[random()%6]^=31+row;
      if(!candidate.media.empty()&&row%4==0)candidate.media[0].digest[0]^=1;
      if(!candidate.media.empty()&&row%7==0)candidate.media[0].content.push_back(83);
      if(row%11==0)candidate.image_context=!candidate.image_context;
      entries.push_back(entry(std::move(candidate),row%6==0?1:0,row%9!=0));
    }
    entries.push_back(entry(query,0,true));
    for(auto limit:{int64_t(0),int64_t(63),int64_t(64),int64_t(65),int64_t(127),int64_t(query.tokens.size()-1)})
      for(bool checkpoint:{false,true})compare(query,entries,0,limit,checkpoint);
    compare(query,entries,1,query.tokens.size()-1);
  }
  // Deterministic late divergences exercise both complete-block selection and
  // partial-final-block refinement, including an unready maximum candidate.
  for(int64_t n:{128,191,192,257,2048}) {
    auto query=identity(n+1);std::vector<std::shared_ptr<PrefixEntry>> entries;
    for(int row=0;row<64;++row) {
      auto candidate=identity(n);candidate.tokens[n-64+row].values[0]+=900;
      entries.push_back(entry(std::move(candidate)));
    }
    entries.push_back(entry(identity(n),0,false));
    compare(query,entries,0,n);compare(query,entries,0,n-1);compare(query,entries,0,n,true);
  }
  // Simulate a collision across every digest. An overestimated long candidate
  // must not hide a valid shorter recurrent checkpoint or trigger false reuse.
  auto query=identity(192),shorter=identity(128),wrong=identity(192);wrong.tokens[3].values[4]^=1;
  prefix_fingerprints(query,192);auto good=entry(shorter),bad=entry(wrong);
  for(auto* hashes:{&query.fingerprints,&good->identity.fingerprints,&bad->identity.fingerprints})
    for(auto& hash:*hashes)hash={};
  auto result=prefix_lookup(query,{good,bad},0,191,false);
  check(result.collision_fallback&&result.entry==good&&result.length==128,"Collision did not fall back to the exact longest prefix");
  result=prefix_lookup(query,{good,bad},0,191,true);
  check(result.entry==good&&result.length==128,"Collision changed recurrent checkpoint eligibility");
  check(!prefix_same(query,bad->identity),"Collision treated distinct snapshots as identical");
  auto tied=identity(192);tied.tokens.back().values[0]^=1;
  auto first_tied=entry(tied),later=entry(identity(192));auto tie_key=identity(192);
  result=prefix_lookup(tie_key,{entry(identity(128)),first_tied,later},0,191,false);
  check(result.entry==first_tied&&result.length==191,"Complete-block shortcut changed the first valid tie");
  // Stale/missing advisory metadata cannot break ordinary exact lookup.
  good->identity.fingerprints.clear();compare(query,{good,bad},0,191);
  // Out-of-order/overlapping native media layouts retain the original oracle
  // semantics rather than relying on the ordered-image early bound.
  auto images=identity(192);images.image_context=true;
  PrefixMedia wide;wide.start=0;wide.count=100;wide.hashed=true;wide.digest[0]=29;
  images.media.push_back(wide);auto overlap=images;
  auto extra=wide;extra.start=50;extra.count=8;extra.digest[0]^=1;overlap.media.push_back(extra);
  compare(images,{entry(overlap)},0,191);
  std::reverse(overlap.media.begin(),overlap.media.end());compare(images,{entry(overlap)},0,191);
  auto unrelated=identity(192);unrelated.tokens[0].values[0]=-1;int probes=0;
  bad->ready=[&]{++probes;return true;};prefix_lookup(unrelated,{bad},0,191,false);
  check(probes==0,"Unrelated identity unnecessarily queried GPU readiness");
  std::cout<<"Prefix lookup: scalar equivalence, media boundaries, checkpoint/readiness, collisions and eviction-independent readers passed\n";
}

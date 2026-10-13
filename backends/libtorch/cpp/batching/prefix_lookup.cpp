#include "batching/prefix_lookup.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <type_traits>
#define XXH_INLINE_ALL
#define XXH_NAMESPACE tei_prefix_
#include "vendor/xxhash/xxhash.h"

namespace tei {
PrefixFingerprint prefix_admission_fingerprint(const PrefixIdentity& key,int operation) {
  // Collisions only permit a capture; retained identities still compare exactly.
  // Observe common causal stems rather than requiring an identical changing tail.
  auto count=std::min<size_t>(prefix_block_tokens,key.tokens.size());
  auto hash=XXH3_128bits_withSeed(key.tokens.data(),count*sizeof(PrefixToken),
    uint64_t(operation)*2+uint64_t(key.image_context));
  return {hash.low64,hash.high64};
}
void prefix_fingerprints(PrefixIdentity& key,int64_t tokens) {
  auto count=std::min<int64_t>(tokens,key.tokens.size())/prefix_block_tokens;
  if(int64_t(key.fingerprints.size())>count)key.fingerprints.resize(count);
  key.fingerprints.reserve(count);
  static_assert(std::is_trivially_copyable_v<PrefixToken> && sizeof(PrefixToken)==6*sizeof(int64_t));
  static_assert(sizeof(PrefixFingerprint)==2*sizeof(uint64_t));
  std::array<uint8_t,sizeof(PrefixFingerprint)+prefix_block_tokens*sizeof(PrefixToken)> bytes;
  auto parent=key.fingerprints.empty()?PrefixFingerprint{0x1122334455667788ULL,0x8877665544332211ULL}:key.fingerprints.back();
  for(int64_t block=key.fingerprints.size();block<count;++block) {
    std::memcpy(bytes.data(),&parent,sizeof(parent));
    std::memcpy(bytes.data()+sizeof(parent),key.tokens.data()+block*prefix_block_tokens,
      prefix_block_tokens*sizeof(PrefixToken));
    auto hash=XXH3_128bits(bytes.data(),bytes.size());parent={hash.low64,hash.high64};
    key.fingerprints.push_back(parent);
  }
}
static int64_t media_common(const PrefixIdentity& a,const PrefixIdentity& b,int64_t n) {
  for(const auto* key:{&a,&b})for(const auto& media:key->media)if(media.start<n) {
    const auto& other=key==&a?b:a;
    auto match=std::find(other.media.begin(),other.media.end(),media);
    if(match==other.media.end()||media.start+media.count>n)n=std::min(n,media.start);
  }
  return n;
}
static bool ordered_media(const PrefixIdentity& key) {
  int64_t end=0;
  for(const auto& media:key.media) {
    if(media.start<end||media.count<=0||media.count>std::numeric_limits<int64_t>::max()-media.start)return false;
    end=media.start+media.count;
  }
  return true;
}
static int64_t media_bound(const PrefixIdentity& a,const PrefixIdentity& b,int64_t end) {
  // Non-overlapping ordered images make the atomic media prefix bound
  // idempotent. Check it before comparing token rows that cannot be reused.
  // Preserve the original scalar ordering for other native caller layouts.
  return ordered_media(a)&&ordered_media(b)?media_common(a,b,end):end;
}
static int64_t token_common(const PrefixIdentity& a,const PrefixIdentity& b,int64_t n,int64_t end) {
  while(n<end) {
    auto count=std::min<int64_t>(prefix_block_tokens,end-n);
    if(std::memcmp(a.tokens.data()+n,b.tokens.data()+n,count*sizeof(PrefixToken))==0) {n+=count;continue;}
    while(n<end&&a.tokens[n]==b.tokens[n])++n;
    break;
  }
  return n;
}
int64_t prefix_common(const PrefixIdentity& a,const PrefixIdentity& b,int64_t limit) {
  if(a.image_context!=b.image_context)return 0;
  auto end=std::min<int64_t>({limit,int64_t(a.tokens.size()),int64_t(b.tokens.size())});
  end=media_bound(a,b,end);
  return media_common(a,b,token_common(a,b,0,end));
}
bool prefix_same(const PrefixIdentity& a,const PrefixIdentity& b) {
  if(a.image_context!=b.image_context||a.tokens.size()!=b.tokens.size())return false;
  auto blocks=a.tokens.size()/prefix_block_tokens;
  if(blocks&&a.fingerprints.size()>=blocks&&b.fingerprints.size()>=blocks
    &&a.fingerprints[blocks-1]!=b.fingerprints[blocks-1])return false;
  return prefix_common(a,b,a.tokens.size())==int64_t(a.tokens.size());
}
static int64_t estimate(const PrefixIdentity& a,const PrefixIdentity& b,int64_t end) {
  auto blocks=end/prefix_block_tokens;
  if(int64_t(a.fingerprints.size())<blocks||int64_t(b.fingerprints.size())<blocks)
    return prefix_common(a,b,end);
  // A differing parent-chained digest bounds the true prefix. Even a hash
  // collision that makes matches non-monotonic only overestimates this bound:
  // every genuinely equal prefix has equal digests at every earlier block.
  int64_t low=0,high=blocks;
  while(low<high) {
    auto mid=(low+high+1)/2;
    if(a.fingerprints[mid-1]==b.fingerprints[mid-1])low=mid;else high=mid-1;
  }
  auto start=low*prefix_block_tokens;
  return media_common(a,b,token_common(a,b,start,std::min(end,start+prefix_block_tokens)));
}
bool prefix_batch_compatible(const PrefixEntry& entry,const PackedInput& input) {
  return entry.batch_max_sequence==input.max_sequence
    &&entry.compact_tokens==(input.compact_ids.defined()?input.compact_ids.numel():0)
    &&entry.batch_offsets.size()==size_t(input.batch+1)
    &&std::equal(entry.batch_offsets.begin(),entry.batch_offsets.end(),input.offsets);
}
PrefixLookup prefix_lookup(PrefixIdentity& key,const std::vector<std::shared_ptr<PrefixEntry>>& entries,
  int operation,int64_t limit,bool checkpoint_only,const PackedInput* layout) {
  limit=std::min<int64_t>(limit,key.tokens.size());PrefixLookup best;
  bool hashed=false,advisory=false;
  for(const auto& entry:entries) {
    auto end=std::min<int64_t>(limit,entry->identity.tokens.size());
    if((layout&&!prefix_batch_compatible(*entry,*layout))
      ||entry->operation!=operation||entry->identity.image_context!=key.image_context
      ||end<=best.length||key.tokens[best.length]!=entry->identity.tokens[best.length])continue;
    end=media_bound(key,entry->identity,end);
    if(end<=best.length)continue;
    // Before fingerprint ranking is active, avoid a long exact comparison of
    // snapshots still pending on another stream. Metadata misses above still
    // avoid readiness queries entirely.
    const bool ready_checked=!hashed;
    if(ready_checked&&!entry->ready())continue;
    if(!hashed&&best.length>=prefix_block_tokens) {
      prefix_fingerprints(key,std::min<int64_t>(key.tokens.size(),limit+prefix_block_tokens-1));hashed=true;
      // An exact full-bound hit often exists among many late-diverging entries.
      // Try its complete-block digest before refining every competing tail.
      auto probe=(limit+prefix_block_tokens-1)/prefix_block_tokens;
      if(probe&&int64_t(key.fingerprints.size())>=probe)for(const auto& candidate:entries) {
        const auto& identity=candidate->identity;
        if((layout&&!prefix_batch_compatible(*candidate,*layout))
          ||candidate->operation!=operation||identity.image_context!=key.image_context
          ||int64_t(identity.tokens.size())<limit||int64_t(identity.fingerprints.size())<probe
          ||(checkpoint_only&&int64_t(identity.tokens.size())!=limit)
          ||identity.fingerprints[probe-1]!=key.fingerprints[probe-1]
          ||media_bound(key,identity,limit)<limit||!candidate->ready())continue;
        if(prefix_common(key,identity,limit)==limit) {
          // A ceil-block digest includes tokens beyond the reuse bound. An
          // earlier entry may match the entire bound while differing only in
          // those recomputed tokens. Preserve the original first-match tie.
          for(const auto& earlier:entries) {
            if(earlier==candidate)break;
            const auto& prior=earlier->identity;
            if((layout&&!prefix_batch_compatible(*earlier,*layout))
              ||earlier->operation!=operation||prior.image_context!=key.image_context
              ||int64_t(prior.tokens.size())<limit||(checkpoint_only&&int64_t(prior.tokens.size())!=limit))continue;
            if(estimate(key,prior,limit)==limit&&earlier->ready()
              &&prefix_common(key,prior,limit)==limit)return {earlier,limit,false};
          }
          return {candidate,limit,false};
        }
      }
    }
    auto n=hashed?estimate(key,entry->identity,end):prefix_common(key,entry->identity,end);
    if(n<=best.length||(checkpoint_only&&n!=int64_t(entry->identity.tokens.size()))
      ||(!ready_checked&&!entry->ready()))continue;
    best={entry,n,false};advisory=hashed;
    if(n==limit)break;
  }
  if(!advisory||prefix_common(key,best.entry->identity,best.length)==best.length)return best;
  // Do not let a colliding overestimate hide another valid candidate, including
  // a shorter recurrent checkpoint or a prefix with matching media.
  best={nullptr,0,true};
  for(const auto& entry:entries)if((!layout||prefix_batch_compatible(*entry,*layout))
    &&entry->operation==operation&&entry->ready()) {
    auto n=prefix_common(key,entry->identity,limit);
    if(n>best.length&&(!checkpoint_only||n==int64_t(entry->identity.tokens.size()))) {
      best.entry=entry;best.length=n;
    }
  }
  return best;
}
}

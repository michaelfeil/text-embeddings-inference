#pragma once
#include "batching/prefix_cache.h"

namespace tei {
constexpr int64_t prefix_block_tokens=64;
void prefix_fingerprints(PrefixIdentity&,int64_t tokens);
PrefixFingerprint prefix_admission_fingerprint(const PrefixIdentity&,int operation);
int64_t prefix_common(const PrefixIdentity&,const PrefixIdentity&,int64_t limit);
bool prefix_same(const PrefixIdentity&,const PrefixIdentity&);
struct PrefixLookup {
  std::shared_ptr<PrefixEntry> entry;
  int64_t length=0;
  bool collision_fallback=false;
};
bool prefix_batch_compatible(const PrefixEntry&,const PackedInput&);
// Fingerprints only rank candidates. A selected prefix is always verified
// exactly; a collision falls back to the complete exact scan.
PrefixLookup prefix_lookup(PrefixIdentity&,const std::vector<std::shared_ptr<PrefixEntry>>&,
  int operation,int64_t limit,bool checkpoint_only,const PackedInput* layout=nullptr);
}

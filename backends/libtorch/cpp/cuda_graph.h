#pragma once
#include "model.h"
namespace tei {
// The caller serializes model inference. Graphs cache exact packed shapes only;
// pooling and all transfers remain outside capture. Outputs are overwritten by
// the next replay of the same entry, so consume them before another inference.
class CudaGraphCache {
  struct Impl;
  std::unique_ptr<Impl> impl;
public:
  explicit CudaGraphCache(size_t entries = 4, int64_t max_tokens = 4096);
  ~CudaGraphCache();
  CudaGraphCache(const CudaGraphCache&) = delete;
  CudaGraphCache& operator=(const CudaGraphCache&) = delete;
  at::Tensor forward(const Model&, const PackedInput&);
  size_t size() const;
};
}

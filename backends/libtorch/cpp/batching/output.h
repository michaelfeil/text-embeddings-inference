// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace tei {
// Two reusable slots bound pinned memory and pending GPU outputs per replica.
// A lease stays alive through CPU result construction, including cancellation.
struct OutputSlot {
  bool busy = false;
  at::Tensor host, device_output;
  std::vector<at::Tensor> gpu;
  std::vector<at::Tensor> inputs;
};
struct OutputPool {
  std::mutex mutex;
  std::condition_variable available;
  std::array<OutputSlot, 2> slots;
  size_t next_slot = 0;
};
class OutputLease {
  std::shared_ptr<OutputPool> pool;
  OutputSlot* slot = nullptr;
  at::Device device;
  size_t input_cursor = 0;

#ifdef TEI_TORCH_CUDA_KERNELS
  c10::cuda::CUDAStream stream;
  at::cuda::CUDAEvent ready;
  bool recorded = false;
  std::atomic<bool> finished = false;
#endif
public:
  const size_t size;
  OutputLease(std::shared_ptr<OutputPool> owner, at::Device dev, size_t count)
      : pool(std::move(owner)), device(dev),
#ifdef TEI_TORCH_CUDA_KERNELS
        stream(c10::cuda::getCurrentCUDAStream(dev.index())),
#endif
        size(count) {
    std::unique_lock lock(pool->mutex);
    // One model producer and FIFO completion alternate A/B, waiting through
    // result packing before reusing the next slot.
    pool->available.wait(lock, [&] { return !pool->slots[pool->next_slot].busy; });
    slot = &pool->slots[pool->next_slot];
    slot->busy = true;
    pool->next_slot = (pool->next_slot + 1) % pool->slots.size();
  }
  ~OutputLease() {
#ifdef TEI_TORCH_CUDA_KERNELS
    // Also drain partially submitted work on errors, before releasing pins.
    try { if (!finished.load(std::memory_order_acquire)) { if (recorded) ready.synchronize(); else stream.synchronize(); } }
    catch (...) { /* CUDA failures are reported by submit/wait. */ }
#endif
    slot->gpu.clear();
    { std::lock_guard lock(pool->mutex); slot->busy = false; }
    pool->available.notify_one();
  }
  void allocate() {
    // Keep wait() non-null even when no output rows were requested.
    const auto capacity = static_cast<int64_t>(std::max(size, size_t{1}));
    if (!slot->host.defined() || static_cast<size_t>(slot->host.numel()) < size)
      slot->host = at::empty({capacity},
          at::TensorOptions().dtype(at::kFloat).pinned_memory(true));
    if (!slot->device_output.defined() || static_cast<size_t>(slot->device_output.numel()) < size)
      slot->device_output = at::empty({capacity},
          at::TensorOptions().dtype(at::kFloat).device(device));
  }
  at::Tensor upload(const at::Tensor& host) {
    TORCH_CHECK(host.device().is_cpu() && host.is_contiguous(), "Invalid staged input");
    if (input_cursor == slot->inputs.size()) slot->inputs.emplace_back();
    auto& pin = slot->inputs[input_cursor++];
    if (!pin.defined() || pin.scalar_type() != host.scalar_type() || pin.numel() < host.numel())
      pin = at::empty({host.numel()}, host.options().pinned_memory(true));
    auto view = pin.narrow(0, 0, host.numel()).view(host.sizes());
    std::memcpy(view.data_ptr(), host.data_ptr(), host.nbytes());
    auto result = view.to(device, /*non_blocking=*/true);
    slot->gpu.push_back(result);
    return result;
  }
  void copy(const at::Tensor& tensor, size_t offset) {
    auto source = tensor.contiguous().view({-1});
    TORCH_CHECK(offset <= size && static_cast<size_t>(source.numel()) <= size - offset,
                "Native output exceeds provided capacity");
    auto converted = slot->device_output.narrow(0, offset, source.numel());
    converted.copy_(source, true);
    slot->host.narrow(0, offset, source.numel()).copy_(converted, true);
    slot->gpu.push_back(std::move(source));
  }
  void record() {
#ifdef TEI_TORCH_CUDA_KERNELS
    // H2D, graph replay and D2H use this same stream. The next replay cannot
    // overwrite graph inputs/outputs before this submission has consumed them.
    ready.record(stream);
    recorded = true;
#endif
  }
  const float* wait() {
#ifdef TEI_TORCH_CUDA_KERNELS
    c10::cuda::CUDAGuard guard(stream.device());
    TORCH_CHECK(recorded, "Output event was not recorded");
    ready.synchronize();
    finished.store(true, std::memory_order_release);
#endif
    return slot->host.data_ptr<float>();
  }
};
}

// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
#include "helpers/admission_error.h"
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/core/CachingHostAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#endif

namespace tei {
struct PinnedAdmissionState {
  std::mutex mutex;
  size_t bytes = 0;
  size_t max_bytes = 512ULL << 20;
  PinnedAdmissionState() {
    if (const auto* value = std::getenv("TEI_TORCH_PINNED_STAGING_MAX_BYTES")) {
      const auto* end = value + std::strlen(value);
      size_t parsed = 0;
      const auto result = std::from_chars(value, end, parsed);
      TORCH_CHECK(result.ec == std::errc{} && result.ptr == end && parsed > 0,
                  "Invalid process native pinned staging memory budget");
      max_bytes = parsed;
    }
  }
};
inline std::shared_ptr<PinnedAdmissionState> pinned_admission_state() {
  static auto state = std::make_shared<PinnedAdmissionState>();
  return state;
}
inline size_t pinned_allocation_bytes(size_t bytes) {
  constexpr size_t page = 4096;
  TORCH_CHECK(bytes <= std::numeric_limits<size_t>::max() - (page - 1),
              "Native pinned staging capacity overflow");
  return (std::max(bytes, size_t{1}) + page - 1) & ~(page - 1);
}
#ifdef TEI_TORCH_CUDA_KERNELS
// This storage does not enter Torch's pinned size-class cache. Its credit stays
// charged through all Tensor views and is returned only after cudaFreeHost.
class PinnedAllocation {
  std::shared_ptr<PinnedAdmissionState> state = pinned_admission_state();
  size_t bytes;
  c10::DeviceIndex device_index;
public:
  void* data = nullptr;
  explicit PinnedAllocation(size_t requested) : bytes(pinned_allocation_bytes(requested)),
    device_index(c10::cuda::current_device()) {
    std::lock_guard lock(state->mutex);
    auto* allocator = at::getHostAllocator(at::kCUDA);
    auto cached = allocator->get_stats().allocated_bytes.current;
    TORCH_CHECK(cached >= 0, "Invalid Torch pinned allocator accounting");
    auto fits = [&] {
      return static_cast<uint64_t>(cached) <= state->max_bytes
        && state->bytes <= state->max_bytes - static_cast<size_t>(cached)
        && bytes <= state->max_bytes - static_cast<size_t>(cached) - state->bytes;
    };
    if (!fits()) {
      // Only completed, unused cache blocks are reclaimed; pending transfers
      // remain in allocated_bytes and continue to consume the shared budget.
      allocator->empty_cache();
      cached = allocator->get_stats().allocated_bytes.current;
      TORCH_CHECK(cached >= 0, "Invalid Torch pinned allocator accounting");
    }
    require_memory_admission(fits(), "Process native pinned staging memory budget is occupied");
    auto status = cudaHostAlloc(&data, bytes, cudaHostAllocPortable);
    if (status == cudaErrorMemoryAllocation) {
      cudaGetLastError();
      TORCH_CHECK_WITH(OutOfMemoryError, false, "Native pinned staging allocation failed");
    }
    C10_CUDA_CHECK(status);
    state->bytes += bytes;
  }
  PinnedAllocation(const PinnedAllocation&) = delete;
  PinnedAllocation& operator=(const PinnedAllocation&) = delete;
  ~PinnedAllocation() {
    // A failed free cannot prove reclamation. Keep its credit charged forever
    // rather than allowing later allocations to reuse unproven physical space.
    try {
      c10::cuda::CUDAGuard guard(device_index);
      const auto status = cudaFreeHost(data);
      if (status == cudaSuccess) {
        std::lock_guard lock(state->mutex);
        state->bytes -= bytes;
      } else {
        TORCH_WARN("Native pinned staging release failed; memory credit retained: ", cudaGetErrorString(status));
      }
    } catch (...) { /* Device/free failure retains the credit, including at shutdown. */ }
  }
};
#endif
inline at::Tensor allocate_pinned_staging(int64_t count, at::TensorOptions options) {
  TORCH_CHECK(options.device().is_cpu() && options.layout() == at::kStrided,
              "Native pinned staging requires a strided CPU tensor");
  const auto element_size = c10::elementSize(options.dtype().toScalarType());
  TORCH_CHECK(element_size > 0 && count >= 0 && static_cast<uint64_t>(count) <=
    std::numeric_limits<size_t>::max() / element_size,
    "Native pinned staging capacity overflow");
#ifdef TEI_TORCH_CUDA_KERNELS
  auto allocation = std::make_shared<PinnedAllocation>(
    static_cast<size_t>(count) * element_size);
  return at::from_blob(allocation->data, {count},
    [allocation](void*) {}, options.device(at::kCPU).pinned_memory(false));
#else
  return at::empty({count}, options.device(at::kCPU));
#endif
}
}

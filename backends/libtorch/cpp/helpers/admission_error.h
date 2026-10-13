// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdexcept>
#include <c10/util/Exception.h>
namespace tei {
class MemoryAdmissionError final : public std::runtime_error {
public:
  explicit MemoryAdmissionError(const char* message):std::runtime_error(message) {}
};
inline void require_memory_admission(bool accepted,const char* message) {
  if(!accepted)throw MemoryAdmissionError(message);
}
// This backend supports CPU and CUDA. CUDA resource APIs report allocation
// failure through AcceleratorError, whereas the caching allocator uses OOM.
inline int32_t accelerator_error_kind(const c10::AcceleratorError& error) {
  constexpr int32_t cuda_memory_allocation=2;
  return error.get_error_code()==cuda_memory_allocation?3:1;
}
}

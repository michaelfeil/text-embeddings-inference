// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdexcept>
namespace tei {
class MemoryAdmissionError final : public std::runtime_error {
public:
  explicit MemoryAdmissionError(const char* message):std::runtime_error(message) {}
};
inline void require_memory_admission(bool accepted,const char* message) {
  if(!accepted)throw MemoryAdmissionError(message);
}
}

// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
namespace tei {
// Only vendor calls that expose an actual allocation-failure status use this
// signal. Operation names have static storage; unwinding needs no message copy.
struct VendorAllocationError final {
  const char* operation;
  int32_t status;
};
}

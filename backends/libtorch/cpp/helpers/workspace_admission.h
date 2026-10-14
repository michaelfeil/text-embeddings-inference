// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_map>
namespace tei {
struct WorkspaceAdmissionState { std::mutex mutex;size_t limit=0,retained=0; };
inline std::shared_ptr<WorkspaceAdmissionState> workspace_admission_state(int index) {
  static std::mutex mutex;static std::unordered_map<int,std::shared_ptr<WorkspaceAdmissionState>> devices;
  std::lock_guard lock(mutex);auto& state=devices[index];if(!state)state=std::make_shared<WorkspaceAdmissionState>();return state;
}
inline void configure_workspace_admission(int index,size_t limit) {
  auto state=workspace_admission_state(index);std::lock_guard lock(state->mutex);
  // Models sharing a device respect the most restrictive positive policy.
  state->limit=state->limit?std::min(state->limit,limit):limit;
}
class WorkspaceMemoryPermit {
  std::shared_ptr<WorkspaceAdmissionState> state;size_t bytes;
public:
  WorkspaceMemoryPermit(std::shared_ptr<WorkspaceAdmissionState> owner,size_t count):state(std::move(owner)),bytes(count) {}
  WorkspaceMemoryPermit(const WorkspaceMemoryPermit&)=delete;
  ~WorkspaceMemoryPermit(){std::lock_guard lock(state->mutex);state->retained-=bytes;}
};
inline std::shared_ptr<WorkspaceMemoryPermit> reserve_workspace_memory(int index,size_t bytes) {
  auto state=workspace_admission_state(index);std::lock_guard lock(state->mutex);
  const auto limit=state->limit?state->limit:size_t(512)<<20;
  if(state->retained>limit||bytes>limit-state->retained)return {};
  auto permit=std::make_shared<WorkspaceMemoryPermit>(state,bytes);state->retained+=bytes;return permit;
}
}

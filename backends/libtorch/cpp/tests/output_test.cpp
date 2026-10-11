// SPDX-License-Identifier: Apache-2.0
#include "batching/output.h"
#include "batching/cuda_graph.h"
#include <c10/core/InferenceMode.h>
#include <future>
#include <iostream>

class Model final : public tei::Model {
public:
  void ready() override {}
  at::Tensor forward(const tei::PackedInput& input) const override {
    return (input.ids.to(at::kFloat) + input.positions.to(at::kFloat)).unsqueeze(1).repeat({1, 768});
  }
};
int main() {
  try {
    c10::InferenceMode inference;
    const at::Device device("cuda:0");
    c10::cuda::CUDAGuard guard(device);
    auto pool = std::make_shared<tei::OutputPool>();
    tei::CudaGraphCache cache(2, 4096);
    Model model;
    std::array<std::unique_ptr<tei::OutputLease>, 2> pending;
    std::array<int, 2> patterns{};
    std::array<int64_t, 2> rows{};
    const float* previous_host = nullptr;
    for (int step = 0; step < 12; ++step) {
      const int index = step % 2;
      if (pending[index]) {
        auto* lease = pending[index].get();
        // Complete on another CPU thread, as the production Rust worker does.
        auto actual = std::async(std::launch::async, [lease] { return lease->wait(); }).get();
        for (int64_t row = 0; row < rows[index]; ++row)
          for (int column = 0; column < 768; ++column)
            TORCH_CHECK(actual[row * 768 + column] == patterns[index] + row, "Stale graph/slot result");
        if (step == 2) previous_host = actual;
        pending[index].reset();
      }
      // Exact shapes change and evict cached graphs while another result is pending.
      const int64_t tokens = step < 4 ? 4096 : step % 3 == 0 ? 32 : step % 3 == 1 ? 128 : 4096;
      auto lease = std::make_unique<tei::OutputLease>(pool, device, tokens * 768);
      lease->allocate();
      const int pattern = step * 7;
      auto host = at::full({tokens}, pattern, at::TensorOptions().dtype(at::kLong));
      auto ids = lease->upload(host);
      host.fill_(-1); // The asynchronous upload must own its host data already.
      auto positions = lease->upload(at::arange(tokens, host.options()));
      std::array<int32_t, 2> offsets{0, static_cast<int32_t>(tokens)};
      auto cu = lease->upload(at::from_blob(offsets.data(), {2}, host.options().dtype(at::kInt)));
      tei::PackedInput input{ids, lease->upload(at::zeros_like(host)), positions, cu, offsets.data(), 1, tokens};
      lease->copy(cache.forward(model, input), 0);
      lease->record();
      patterns[index] = pattern;
      rows[index] = tokens;
      if (step == 2) TORCH_CHECK(lease->wait() == previous_host, "Pinned output was not reused");
      pending[index] = std::move(lease);
    }
    for (int index = 0; index < 2; ++index) {
      auto actual = pending[index]->wait();
      for (int64_t row = 0; row < rows[index]; ++row)
        for (int column = 0; column < 768; ++column)
          TORCH_CHECK(actual[row * 768 + column] == patterns[index] + row, "Final result mismatch");
      pending[index].reset();
    }
    // Dropping unfinished work must drain before reusing either input or output pins.
    { tei::OutputLease abandoned(pool, device, 4096 * 768); abandoned.allocate();
      abandoned.copy(at::ones({4096,768}, at::TensorOptions().device(device).dtype(at::kHalf)), 0);
      abandoned.record(); }
    tei::OutputLease reused(pool, device, 768); reused.allocate();
    reused.copy(at::full({1,768}, 42, at::TensorOptions().device(device)), 0); reused.record();
    TORCH_CHECK(reused.wait()[767] == 42, "Abandoned lease contaminated reuse");
    std::cout << "output staging, overlap, graph eviction, reuse and cancellation passed\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

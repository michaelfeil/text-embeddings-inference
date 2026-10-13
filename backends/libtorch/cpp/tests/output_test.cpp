// SPDX-License-Identifier: Apache-2.0
#include "batching/output.h"
#include "batching/cuda_graph.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/core/InferenceMode.h>
#include <chrono>
#include <future>
#include <iostream>
#include <utility>

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
    tei::CudaGraphCache cache(2, 4096,1);
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
    // GPU completion alone must not release A while CPU packing still owns it.
    // Even if B is free, C waits for A rather than selecting any free slot.
    auto alternating = std::make_shared<tei::OutputPool>();
    auto first = std::make_unique<tei::OutputLease>(alternating, device, 1);
    first->allocate(); first->copy(at::ones({1}, at::TensorOptions().device(device)), 0); first->record();
    const auto first_host = first->wait();
    { tei::OutputLease second(alternating, device, 1); second.allocate();
      second.copy(at::full({1}, 2, at::TensorOptions().device(device)), 0); second.record();
      TORCH_CHECK(second.wait() != first_host, "A and B shared an output buffer"); }
    std::promise<void> started;
    auto third = std::async(std::launch::async, [&] {
      c10::InferenceMode worker_inference;
      c10::cuda::CUDAGuard worker_guard(device);
      started.set_value();
      tei::OutputLease lease(alternating, device, 1); lease.allocate();
      lease.copy(at::full({1}, 3, at::TensorOptions().device(device)), 0); lease.record();
      const auto host = lease.wait();
      return std::make_pair(host, host[0]);
    });
    started.get_future().wait();
    const auto waiting = third.wait_for(std::chrono::milliseconds(20));
    first.reset(); // Release only after simulated result packing finishes.
    const auto [third_host, third_value] = third.get();
    TORCH_CHECK(waiting == std::future_status::timeout, "C reused free B while A was still owned");
    TORCH_CHECK(third_host == first_host && third_value == 3, "C did not reuse A safely");
    // Admission protects pending results; only completed idle buffers may be evicted.
    auto bounded = std::make_shared<tei::OutputPool>();
    bounded->max_bytes = 256;
    auto held = std::make_unique<tei::OutputLease>(bounded, device, 8);
    held->allocate(); held->copy(at::ones({8}, at::TensorOptions().device(device)), 0); held->record();
    bool rejected = false;
    try { tei::OutputLease oversized(bounded, device, 32); oversized.allocate(); }
    catch (const c10::Error&) { rejected = true; }
    TORCH_CHECK(rejected && bounded->retained_bytes() == 64,
                "Admission allocated over budget or freed a pending result");
    TORCH_CHECK(held->wait()[7] == 1, "Budget rejection corrupted pending output");
    held.reset();
    // Next submission uses the other slot and releases the completed idle cache.
    { tei::OutputLease small(bounded, device, 8); small.allocate(); small.record(); small.wait(); }
    { tei::OutputLease larger(bounded, device, 32); larger.allocate();
      TORCH_CHECK(bounded->retained_bytes() == 256, "Idle staging eviction did not respect the budget");
      rejected = false;
      try { larger.upload(at::ones({1}, at::TensorOptions().dtype(at::kFloat))); }
      catch (const c10::Error&) { rejected = true; }
      TORCH_CHECK(rejected && bounded->retained_bytes() == 256, "Input staging exceeded the budget");
      larger.copy(at::full({32}, 9, at::TensorOptions().device(device)), 0); larger.record();
      TORCH_CHECK(larger.wait()[31] == 9, "Staging did not recover after admission rejection"); }
    rejected = false;
    try { tei::OutputLease oversized(bounded, device, std::numeric_limits<size_t>::max()); oversized.allocate(); }
    catch (const c10::Error&) { rejected = true; }
    TORCH_CHECK(rejected, "Oversized capacity overflow bypassed staging admission");
    auto varied = std::make_shared<tei::OutputPool>(); varied->max_bytes = 4096;
    { tei::OutputLease many_inputs(varied, device, 8); many_inputs.allocate();
      many_inputs.upload(at::ones({256}, at::TensorOptions().dtype(at::kFloat)));
      many_inputs.upload(at::ones({256}, at::TensorOptions().dtype(at::kFloat)));
      many_inputs.record(); many_inputs.wait(); }
    { tei::OutputLease other(varied, device, 8); other.allocate(); other.record(); other.wait(); }
    { tei::OutputLease larger(varied, device, 500); larger.allocate();
      TORCH_CHECK(varied->slots[0].inputs.empty(), "Selected slot retained unused input pins");
      larger.record(); larger.wait(); }
    { tei::OutputLease other(varied, device, 8); other.allocate(); other.record(); other.wait(); }
    { tei::OutputLease smaller(varied, device, 8); smaller.allocate();
      smaller.upload(at::ones({32}, at::TensorOptions().dtype(at::kFloat)));
      TORCH_CHECK(varied->retained_bytes() <= varied->max_bytes,
                  "Selected slot's old output capacity blocked a smaller request");
      smaller.copy(at::full({8}, 5, at::TensorOptions().device(device)), 0); smaller.record();
      TORCH_CHECK(smaller.wait()[7] == 5, "Selected-slot reclamation corrupted output"); }
    // Simulate device allocation pressure without consuming the GPU's memory.
    // An allocation failure must release the slot and its partial host allocation.
    auto pressured = std::make_shared<tei::OutputPool>();
    const auto fraction = c10::cuda::CUDACachingAllocator::getMemoryFraction(0);
    c10::cuda::CUDACachingAllocator::emptyCache();
    c10::cuda::CUDACachingAllocator::setMemoryFraction(0, 0);
    bool allocation_failed = false;
    try { tei::OutputLease failed(pressured, device, 8 * 1024 * 1024); failed.allocate(); }
    catch (const c10::OutOfMemoryError&) { allocation_failed = true; }
    catch (...) { c10::cuda::CUDACachingAllocator::setMemoryFraction(fraction, 0); throw; }
    c10::cuda::CUDACachingAllocator::setMemoryFraction(fraction, 0);
    TORCH_CHECK(allocation_failed && pressured->retained_bytes() == 0,
                "Failed allocation retained partial staging buffers");
    { tei::OutputLease recovered(pressured, device, 8); recovered.allocate();
      recovered.copy(at::full({8}, 11, at::TensorOptions().device(device)), 0); recovered.record();
      TORCH_CHECK(recovered.wait()[7] == 11, "Device allocation failure poisoned subsequent submission"); }
    std::cout << "output staging, overlap, graph eviction, alternation, reuse and cancellation passed\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

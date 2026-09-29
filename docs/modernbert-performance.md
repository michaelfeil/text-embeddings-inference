# ModernBERT correctness and performance

This comparison isolates the ModernBERT changes in PR #54 against `main`
(`5cdaee06599f783c6e56d9e295bc4f5ae433670d`). The baseline **ignores the
attention mask** in dense CUDA attention. Its timing is a historical comparison,
not an acceptable alternative implementation.

## Correctness

- `gelu` evaluates the erf formula. `gelu_new` and `gelu_pytorch_tanh` evaluate
  the explicit tanh approximation. Existing relu, silu, swiglu, and tanh
  configuration values remain accepted.
- The mask is added to the attention scores before softmax. It is not passed
  as a cuBLASLt output buffer with implicit beta=0.
- Standalone CPU and H100 CUDA regressions cover exact GELU constants,
  global/local attention, lengths 133/160, unequal batch padding, solo-versus-batch
  agreement, and repeated reuse of the same unchanged mask. CUDA covers F32/BF16.
- Both normal CUDA math and `NVIDIA_TF32_OVERRIDE=0` pass. Main's pinned cudarc
  selects `CUBLAS_COMPUTE_32F_FAST_TF32` for F32 matmuls. Disabling TF32 makes the
  analytical F32 attention oracle pass at 2e-6; normal TF32 uses 5e-4 for these
  bounded outputs. BF16 uses 0.004. Mask preservation is checked exactly.
- Negative controls fail as intended: tanh-GELU produces -0.04540229 instead
  of -0.045500264; the original CUDA mask path produces 17.18672 instead of
  0.375 for a padded attention example.

Main's old CUDA slice clone copies the supplied output allocation, so the
regression there is an ignored mask. Other backend versions can reuse that
allocation. Keeping the immutable mask separate covers both behaviors.

## Measurement scope

The opt-in Rust harness runs the **complete dense ModernBERT encoder**, including
mask construction, transfers, rotary embeddings, all 28 encoder layers, final
normalization, and CLS pooling. It excludes tokenization, the decision head,
batching queue, and HTTP. These numbers are not SystemOne request latency.

- Hardware: one NVIDIA H100 80GB HBM3, GPU UUID
  `GPU-4c264e9c-7925-dfbc-02cf-fe1996e5de7e`.
- Same BF16 weights, GPU, release settings, input shapes, and harness for both runs.
  No Flash Attention. Default CUDA math settings; no TF32 override.
- `RAYON_NUM_THREADS=8`; request concurrency 1; 3 warmups and 20 timed forwards
  per case. Each forward is synchronized before stopping the timer.
- Deterministic synthetic token IDs, lengths 128/512/1024, batches 1/8/32.
  The mixed batch alternates lengths 512 and 1024 across 8 sequences.
- The harness calls `ModernBertModel` directly. This allows BF16 comparison even
  though main's older public backend dtype parser only accepts FP16/F32.
- Median uses the two middle samples; p95 is the 19th ordered sample of 20.
  Raw samples are retained; small differences should not be overinterpreted.

Checkpoint: `convaiinnovations/laya-typed-decisions`, revision
`1a793eb568e6718f15941d08f85432581df534e3`. Only the `encoder.*` weights are used;
this does not time the typed-decision head or evaluate decision accuracy.

SHA256:

- `model.safetensors`: `4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e`
- `encoder/config.json`: `5268d24ad3b77c8151de5dcb0762ba4391619aad9ab0bda33e36fb083cfeae6d`

The harness normalizes the checkpoint's newer nested `rope_parameters` keys to
main's `global_rope_theta`/`local_rope_theta` configuration fields without changing
the values.

## Reproduce

Obtain the two checkpoint files at the pinned revision. Set the directory
containing `model.safetensors` and `encoder/config.json` below. Run from the PR
checkout on an idle H100; choose the appropriate GPU index for your machine.

```sh
export MODERNBERT_BENCH_CHECKPOINT=/path/to/laya-typed-decisions
export MODERNBERT_BENCH_OUTPUT=/tmp/modernbert-after.json
export CUDA_VISIBLE_DEVICES=3 CUDA_COMPUTE_CAP=90 RAYON_NUM_THREADS=8
export LIBRARY_PATH=/usr/local/cuda/lib64 LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
cargo test --release -p text-embeddings-backend-candle -p text-embeddings-router \
  --no-default-features \
  --features text-embeddings-router/candle-cuda-volta,text-embeddings-router/dynamic-linking,text-embeddings-router/http \
  --lib modernbert_encoder_latency -- --ignored --nocapture
```

For the baseline, create a worktree at the exact main revision above, copy
`backends/candle/src/models/modernbert_benchmark.rs` from this PR, and append only
this test module declaration to its `modernbert.rs`:

```rust
#[cfg(all(test, feature = "cuda"))]
#[path = "modernbert_benchmark.rs"]
mod benchmark;
```

Run the identical command in that worktree with a different output path. No
baseline production code is changed. The harness is ignored by ordinary test
runs; remove `--ignored` and use filter `modernbert` to run the regressions.

## Current measurement status

The [corrected encoder run](benchmarks/modernbert-corrected-latency.json) is
complete. The baseline and mask-only comparison runs are still in progress;
no before/after speedup claim is made from the partial comparison. The main
lineage retains CPU construction and transfer of its dense local mask, and
these full-forward samples have substantial host-side timing variability.

Activation fusion is being investigated separately on the merged PR #53
lineage, which already constructs its mask on the GPU. That optimization is
not part of this independent correctness fix against main.

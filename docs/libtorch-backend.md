# Experimental native LibTorch backend

LibTorch 2.14.1 runs alongside Candle as an optional inference runtime. Rust owns
model configuration, safetensors loading, batch validation and TEI integration;
a small C ABI bridge runs the model with the PyTorch C++ tensor API. Python is
not required at build time or inference time when using a standalone LibTorch
distribution. There is no TorchScript export step.

## Packed inference

Token embeddings, projections, attention outputs and MLP activations use
`[total_tokens, hidden_size]`, with Q/K/V shaped
`[total_tokens, heads, head_dim]`. Sequence boundaries come directly from TEI's
cumulative lengths. No token padding or dense attention mask is allocated.

On CUDA, attention calls `aten::_flash_attention_forward` with device-resident
INT32 cumulative lengths and maximum sequence length. This is the Flash
Attention operator behind
[PyTorch's `varlen_attn`](https://docs.pytorch.org/docs/stable/nn.attention.varlen.html).
The C++ call uses zero dropout and full bidirectional attention for BERT. Kernel
errors propagate to Rust; there is no padded fallback.

CPU has an unpadded reference implementation: all projections/MLPs stay packed,
and attention uses SDPA on each sequence at its actual length. This is useful for
correctness testing; it is not a fused CPU varlen kernel. MPS and XPU are rejected
in this prototype because this Torch varlen operator is exposed through CUDA.

## Initial model support

- Encoder BERT (`model_type: bert`) with absolute position embeddings.
- Single-file or sharded safetensors; both bare and `bert.` weight prefixes.
- CLS, mean and last-token pooling, and raw token embeddings.
- FP32, FP16 and BF16 weights/activations. CUDA varlen requires FP16/BF16 and a
  head dimension divisible by eight, at most 256. Auto uses FP32 on CPU and FP16
  on CUDA. Mean pooling accumulates in FP32.
- GELU, tanh GELU and ReLU. Standard `gelu` uses Torch's exact GELU; Candle's BERT
  implementation uses approximate GELU, so small numerical differences are
  expected. Parity tests select `gelu_pytorch_tanh` in both runtimes.

Classifiers, decision models, multimodal models, SPLADE, radix folding, dynamic
FP8 and Sentence Transformers Dense/custom modules are rejected explicitly.
Other Hugging Face architectures require their own native model implementations.

## Build and run

Install the [LibTorch C++ distribution](https://docs.pytorch.org/cppdocs/installing.html)
for **2.14.1**, choosing CPU or a CUDA build compatible with your GPU driver.
Set `LIBTORCH` to its root (containing `include/`, `lib/` and `share/`). Building
requires CMake and a C++20 compiler. CUDA distributions may also require a CUDA
toolkit during CMake configuration.

```bash
export LIBTORCH=/opt/libtorch
export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"

# Compile both backends; Candle remains the default.
cargo build --release -p text-embeddings-router --features libtorch

target/release/text-embeddings-router \
  --backend libtorch --torch-device cuda --dtype float16 \
  --model-id /path/to/bert-model --pooling mean \
  --backend-device-ids 0,1
```

An A10 uses the CUDA path. Each selected GPU loads a complete model and uses
TEI's existing replica queue; this is data parallel serving. Use `--device-id 0`
for one GPU or omit device selection to use all visible CUDA devices.
`CUDA_VISIBLE_DEVICES` controls visible ordinals. Auto chooses CUDA when a device
is available, otherwise CPU. Other accelerator families are not implemented yet.

For CPU, use `--torch-device cpu --dtype float32` and omit replica selection.
To build without Candle, replace build features with
`--no-default-features --features libtorch,http`, and select `--backend libtorch`
when starting the server. Existing Candle builds do not require LibTorch.

CMake handles LibTorch's compiler/ABI flags. The bridge is a small shared library;
the server's build rpath points to its Cargo output directory. When moving the
server to another machine, ship `libtei_torch.so` from that build alongside the
executable and provide LibTorch's shared libraries via `LD_LIBRARY_PATH` (retain
their licenses/notices). The executable and bridge also search their own directory
on Linux. No `libtorch_python` library is linked by the bridge.

## Validation and performance

```bash
cargo test -p text-embeddings-backend-libtorch --lib

# Requires a working NVIDIA GPU and the CUDA LibTorch distribution.
cargo test -p text-embeddings-backend-libtorch \
  cuda_varlen_matches_cpu_for_ragged_outputs -- --ignored
```

CPU parity checks cover ragged pooled/raw output, all three pooling modes,
sequence isolation, sharded checkpoints and weight prefixes. Error tests cover
malformed batches and missing native weights. The GPU parity test exercises the
actual packed CUDA varlen operator and moving a model onto the inference worker.

CUDA varlen parity has been validated on an H100. A10 throughput still needs
hardware validation. Native C++
avoids Python dispatch, but eager tensor operations still launch GPU kernels;
this prototype does not use CUDA graphs. The initial
[H100 BERT comparison](benchmarks/libtorch-bert-h100.md) shows Candle ahead,
including when the GPU assignments are swapped.

## Comparing against Candle

The `compare_bert` example loads the same checkpoint into both runtimes on CUDA
devices 0 (LibTorch) and 1 (Candle), concurrently, with FP16 and CLS pooling. It covers single requests, uniform batches,
and ragged batches. Inputs contain deterministic synthetic WordPiece IDs bounded
to BERT's vocabulary, with CLS/SEP markers; tokenization and HTTP are excluded.
Both runtimes return host embeddings before the timer stops, including input
transfer, model execution, pooling and output transfer. Input cloning is excluded
equally. It checks embedding cosine agreement before measuring each workload.

Use an optimized build for both backends:

```bash
CUDA_COMPUTE_CAP=90 CARGO_PROFILE_RELEASE_LTO=false \
  CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
  cargo build --release -p text-embeddings-backend-libtorch \
    --features benchmark-cuda --example compare_bert

target/release/examples/compare_bert /path/to/bert-model 100 0 1 > bert-results.json
```

Set `CUDA_COMPUTE_CAP` for your GPU (90 for H100, 86 for A10). Each workload uses
20 warmups per runtime and a synchronized start for parallel measurements. GPU
ordinals are configurable; swap them to check hardware bias. The JSON includes P50/P95 latency, token throughput at P50 and output
agreement. These are synchronous backend measurements, not HTTP throughput or
multi-GPU scaling measurements. GPU clocks are not locked by the harness.

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
cumulative lengths. No token padding is allocated. Relative position bias paths use attention bias at each sequence's actual length.

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

Native routed MoE CUDA kernels reuse Candle’s retained MIT sources and the same
pinned CUTLASS revision `e406c186f510a15091cce01f782020ceb7ba8eb5`.
Set `CUTLASS_ROOT` for an offline checkout; otherwise CMake reuses the Candle
cache or downloads that pinned revision. Installation includes the retained
CUTLASS BSD notice under `share/licenses/tei_torch`. Kernel-level exact routing
tests do not establish trained checkpoint accuracy.

## Model support

Native implementations now cover BERT/RoBERTa, DistilBERT, GTE, ModernBERT,
Nomic, Jina/JinaCode, Llama/Mistral, Qwen2/Qwen3 (including Qwen3 MoE), and the
Gemma families. Model availability and numerical validation are tracked in the
[coverage ledger](libtorch-model-coverage.md); source implementation alone does
not establish production support or the latency target.

Single-file and sharded safetensors, CLS/mean/last-token pooling and raw token
embeddings remain supported. BERT supports tanh GELU, ReLU and SiLU; other
BERT activation names fail at startup instead of silently selecting GELU.
Classification and SPLADE heads are being checked
against their Candle counterparts. FP32, FP16 and BF16 are accepted; CUDA Flash
requires FP16/BF16. Wider Gemma attention heads use native packed efficient
attention. The installed Torch Flash build disables ALiBi, so Jina uses native
efficient varlen attention in one packed call per layer. A shared sequence-local
ALiBi tile uses overlapping read-only strides with bounded backing storage:
`O(T*M + H*M*M)` elements for total tokens `T`, maximum actual sequence length
`M`, and heads `H`. Q/K/V contain only actual tokens. Jina reuses the bias across
all layers within each request; no mutable cross-request cache is used.

Typed decision heads have native implementations and synthetic parity checks;
trained-checkpoint validation remains pending. Sentence Transformers Dense
chains support identity/tanh projections of pooled outputs, retaining raw token
widths. Dense modules require safetensors weights. Radix folding, dynamic FP8
and other custom modules remain unsupported and are rejected explicitly. Image and
audio towers have experimental implementations; trained-checkpoint numerical
validation remains incomplete, including a known Gemma audio accuracy gap.
BERT uses tanh GELU to match Candle's inference kernels. CUDA fuses bias and
GELU into the intermediate projection; CPU uses the matching approximation.

## Build and run

Install the [LibTorch C++ distribution](https://docs.pytorch.org/cppdocs/installing.html)
for **2.14.1**, choosing CPU or a CUDA build compatible with your GPU driver.
Set `LIBTORCH` to its root (containing `include/`, `lib/` and `share/`). Building
requires CMake and a C++20 compiler. CUDA distributions require a CUDA toolkit to compile the native packed operators.

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

Native model and kernel fixtures can also be built independently:

```bash
cmake -S backends/libtorch/cpp -B /tmp/tei-native-tests \
  -DCMAKE_PREFIX_PATH="$LIBTORCH" -DCMAKE_BUILD_TYPE=Release \
  -DTEI_BUILD_NATIVE_TESTS=ON -DTEI_RUN_CUDA_TESTS=ON
cmake --build /tmp/tei-native-tests --parallel
CUDA_VISIBLE_DEVICES=0 ctest --test-dir /tmp/tei-native-tests --output-on-failure
```

Omit `TEI_RUN_CUDA_TESTS` to register only CPU tests. Shared CUDA libraries must
be discoverable by the linker and runtime when using a CUDA distribution.

CUDA varlen parity has been validated on an H100. A10 throughput still needs
hardware validation. Native C++
avoids Python dispatch, but eager tensor operations still launch GPU kernels.
Set `TEI_TORCH_CUDA_GRAPHS=1` to enable an experimental four-shape replay cache
for eligible dense models. Cache keys include exact sequence boundaries; no
tokens are padded. Capture cost and cache misses matter for serving latency.
The default capture limit is 4096 tokens; `TEI_TORCH_CUDA_GRAPH_MAX_TOKENS` accepts
a positive override. Unsupported capture operations run eagerly. Media capture
requires the additional experimental flag described below.

`TEI_TORCH_CUDNN_VARLEN=1` enables experimental packed cuDNN attention for eligible
BERT and eligible decoder requests on supported GPUs and cuDNN versions.
Restrictive sliding windows retain Flash attention. This option remains opt-in.
`TEI_TORCH_FULL_PRECISION_GEMM=1` disables reduced-precision GEMM accumulation
and TF32 for this process; experiments have not established a general accuracy
or latency benefit.
ModernBERT CUDA uses the Candle normalization kernel adapter by default, together
with matching half-precision rotary arithmetic. The trained embedding checkpoint
passes both official-token and arbitrary-ID mean parity checks.
CUDA mean pooling uses Candle's exact model-dtype reduction tree in one packed
launch over selected sequence spans; raw token outputs retain their original widths.
`TEI_TORCH_EXACT_ENCODER_NORM=0` selects ATen normalization for diagnostics.
`TEI_TORCH_MEDIA_CUDA_GRAPHS=1`, together with `TEI_TORCH_CUDA_GRAPHS=1`, enables
experimental image/audio graph capture. The cache keys include exact media shapes
and descriptors and copy media values and prepared audio indices on replay.
Unsupported captures fall back to eager inference. Trained media accuracy and
performance remain under evaluation.

The [checkpoint comparison](benchmarks/libtorch-checkpoints-h100.md) records
Qwen3, GTE, BERT, RoBERTa, DistilBERT, ModernBERT, Nomic, Jina/JinaCode, Mistral and
Qwen vision workloads meeting the
5% P50 target with the recorded options. ModernBERT still has a separate
arbitrary-ID MLM parity failure; Gemma has trained-checkpoint accuracy gaps.
The Jina/JinaCode packed Torch bias implementation resolves the previously
recorded batched latency gap. The initial
[H100 BERT comparison](benchmarks/libtorch-bert-h100.md) shows Candle ahead,
including when the GPU assignments are swapped.

## Comparing against Candle

The `compare_bert` example loads the same checkpoint into both runtimes on CUDA
devices 0 (LibTorch) and 1 (Candle), concurrently, with FP16 and CLS pooling by
default. The optional final argument selects `cls`, `mean`, or `last_token`
pooling. It covers single requests, uniform batches and ragged batches. Inputs
contain deterministic synthetic IDs bounded to the checkpoint vocabulary, with
configured special tokens and the router position offset; tokenization and HTTP
are excluded.
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

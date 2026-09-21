# Candle CUDA and BF16

The workspace pins Candle 0.11.0 to
[`b14cda1466e915ce045ecfa4813b79ff4ea4a813`](https://github.com/michaelfeil/candle/commit/b14cda1466e915ce045ecfa4813b79ff4ea4a813),
based on upstream `ddf1b879dc3a1760cbcb3f3c4a7c6467850cec4a`.
The fork adds explicit static/dynamic CUDA linking features to Candle and its
kernel libraries. `cudarc` is pinned to **0.19.8**. Rust **1.93.1** is selected
by `rust-toolchain.toml`.

`--dtype bfloat16` selects BF16 for Candle. CUDA BF16 requires Ampere (sm80) or
newer; older GPUs receive a startup error. Float16 remains the default. Qwen3
decisions use the same RadixMLP path for both precisions. BF16 has lower mantissa
precision, so scores can differ between compact, expanded, and single-option
matrix shapes; use FP16 when tight agreement between those scores matters.

## Static CUDA toolkit build

Install a CUDA development toolkit with its static archives, a C++ compiler with
`libstdc++.a`, GNU BFD, and the usual router build dependencies. For an H100:

```bash
CUDA_COMPUTE_CAP=90 CUDA_ROOT=/usr/local/cuda \
  scripts/cargo-static-cuda.sh build --release -p text-embeddings-router \
  --no-default-features --features candle-cuda,http,static-linking
```

`Dockerfile-cuda` uses this wrapper for both dependency and application builds.
It provides rustc with the C++ archive search path and selects GNU BFD. CUDA 13.1
static NVRTC reported zero supported architectures with LLVM LLD in verification;
BFD correctly initialized NVRTC and compiled the GPU kernels.

“Static” means CUDA **toolkit** libraries are linked into the executable. The
NVIDIA driver (`libcuda.so.1`) and operating-system libraries remain dynamic.
Inspect the executable with `readelf -d target/release/text-embeddings-router`;
there should be no `NEEDED` entry for `libcudart`, `libcublas`, `libcublasLt`, or
`libnvrtc`. To also catch libraries opened at runtime, launch with `LD_DEBUG=libs`
and inspect the loader log after model warmup and a `/decide` request.
Do not combine `static-linking` with the default `dynamic-linking` feature; use
`--no-default-features` as above. Neither linking selector activates CUDA for a
CPU-only build.

For dynamic toolkit linking, use ordinary `cargo build` with
`--no-default-features --features candle-cuda,http,dynamic-linking` and supply
matching CUDA runtime libraries. Turing FP16 builds use `candle-cuda-turing`
instead of `candle-cuda`; BF16 is unavailable on Turing.

## Extension provenance

The five small crates in `extensions/` are ported from
[`candle-extensions` at e287dd09535ab5d7631c0f2ae0a93d7ec168cb86](https://github.com/huggingface/candle-extensions/tree/e287dd09535ab5d7631c0f2ae0a93d7ec168cb86).
They preserve their MIT/Apache licenses and the original fused CUDA kernels.
Standalone upstream benchmark/profiler programs are omitted.
The port updates Candle/cudarc APIs, passes Candle's actual stream to CUDA FFI,
and retains pointer guards until launches are submitted. BF16 gather copies the
16-bit representation directly, including NaNs and subnormals.

FlashAttention v1 fetches CUTLASS at the exact commit
`5e497243f7ad13a2aa842143f9b10bbb23d98292`; it does not vendor the full header tree.
One unused shared-memory transpose template was removed because it crashes
nvcc 13.1. The sm75 kernels for head dimensions 32, 64, and 128 compile with that
toolkit. This is compilation coverage, not a Turing hardware runtime test.

## Verification

The extension tests cover BF16 fused cuBLASLt bias/activation, RMSNorm and residual
fusion, rotary embedding, gather offsets/repeated rows/bit preservation, and
FlashAttention v1. CUDA tests use a nondefault stream. The model test compares
compact and expanded Qwen3 scoring against complete independent likelihoods for
FP16 and BF16, with tied/untied heads and logit chunk boundaries.

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_COMPUTE_CAP=90 CUDA_ROOT=/usr/local/cuda \
  scripts/cargo-static-cuda.sh test -p text-embeddings-backend-candle \
  -p text-embeddings-router --lib decision_scores --no-default-features \
  --features text-embeddings-router/candle-cuda,text-embeddings-router/http,text-embeddings-router/static-linking
```

See [the HTTP integration test](../../integration_tests/README.md#radixmlp-decisions-with-qwen3-4b-instruct-2507)
for actual Qwen3-4B-Instruct-2507 scoring and recorded results.

# Standalone Candle upgrade verification

PR #10 targets `mf/faster-tei` directly, with merge base `ac28ad03` and no commits
from PR #9. Native source tested: `26d1f2306bf1ea5b191344d05b8b0935bf4ed49c`.
Later result/documentation commits do not change the Docker application inputs.

Both unmodified-base and upgraded `Dockerfile-cuda` HTTP images built successfully
for H100 / SM90 with CUDA 12.9.0. Runtime tests used an H100 80GB and driver
580.126.20. Model revisions, image/binary digests, raw BGE logits, embedding value
hashes, server configuration, and individual assertions are in
[candle_precision_results.json](candle_precision_results.json). Reproduction
commands are in [README.md](README.md#candle-bf16-and-upgrade-regression-checks).

| Check | Result |
| --- | --- |
| Qwen3-0.6B FP16 before/after | Exactly equal embeddings on all 8 inputs |
| Qwen3-0.6B BF16 | 1,024 dimensions; finite through 8,449 tokens; reference cosine ≥ 0.99988481 |
| Voyage BF16 | 2,048 dimensions; finite through 31,687 tokens; ordinary-set reference cosine ≥ 0.99996560 |
| Voyage FP16 overflow control | Non-finite outputs for repeated code at 534, 1,062, and 2,118 tokens; also numeric and one long prose case |
| Same Voyage stress inputs in BF16 | All 17 inputs finite, both normalized and unnormalized; batch/single comparison passes |
| BGE FP16 before/after | All 28 raw logits exactly equal; all 7 complete rankings identical |
| BGE BF16 vs FP32 reference | All 7 top choices agree; maximum absolute logit error 0.13413239 (bound 0.25) |
| BGE labeled accuracy | 4/7 for base FP16, upgraded FP16/BF16, and Transformers FP32 |
| Voyage BF16 without FlashAttention | Three short cases, 2,048 dimensions, reference cosine ≥ 0.99995 |
| CUDA extension tests | 36 passed, 0 failed (static CUDA 13.1 build with compatibility driver) |
| Local checks | Workspace check, gRPC check, strict workspace Clippy, formatting, and file hooks pass |

BGE includes four pairs of 499–501 tokens. FP16 error against the FP32 reference
is at most 0.05283 (bound 0.1). Neither precision changes pairwise ordering where
the reference logits differ by more than 0.1. Near-tied irrelevant documents may
change order relative to FP32; all before/after FP16 rankings are exactly equal.
These small fixtures establish regression coverage, not general retrieval accuracy.

The unchanged base rejects `--dtype bfloat16`. It also ignores Voyage's trained
output projection and returns 1,024 dimensions; that failing dimension check is
retained as a separate control. The upgrade loads the projection and honors
bidirectional attention in both Qwen3 implementations. Voyage uses mean pooling
and does not use RadixMLP prefix sharing.

The upgraded executable has no dynamic CUDA toolkit or C++ runtime dependency.
Both ELF inspection and process mappings after BF16 embedding requests confirm
that CUDA toolkit libraries are absent; NVIDIA driver and OS/OpenSSL libraries
remain dynamic. The report lists the actual loaded shared libraries.

A broader `cargo check --workspace --all-targets` encounters existing errors in
the base's `candle-bench` harness: missing `Batch` fields and a missing backend
constructor argument. This unrelated benchmark harness is unchanged.

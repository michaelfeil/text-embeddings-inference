# BGE dynamic FP8 on H100

The complete integrated runtime passes all seven trained BGE workloads in both
GPU assignments, with 200 iterations per backend in eager and graph modes.
All 28 comparisons have cosine 1 and maximum absolute output difference 0.
The unchanged gates are cosine ≥0.999 and P50 no more than 5% slower than Candle.
Worst P50 is **+2.03% eager** and **44.59% faster with graphs**.

This measures `BAAI/bge-base-en-v1.5` revision
`a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, FP16 checkpoint tensors with
**dynamic MLP-only FP8 on both backends**, and CLS pooling on H100 GPUs 2/3,
then 3/2. Attention, residuals, embeddings and pooling retain FP16; normalization
inputs/outputs retain FP16 with the original FP32 statistics. Inputs contain
synthetic valid vocabulary IDs; all Q/K/V and MLP rows
are actual tokens, with no padding or Python inference runtime. Timings include
input/output transfers and pooling, excluding HTTP, tokenization, startup and
input cloning. These results do not qualify raw-token output, classifiers,
SPLADE, other checkpoints, other precisions or A10.

The table reports the slower relative result across the two GPU assignments
for each workload. Negative percentages mean lower native P50 latency.

| Workload | Actual tokens | Eager vs Candle | Graphs vs Candle |
|---|---:|---:|---:|
| 1x32 | 32 | +2.03% | -46.45% |
| 1x128 | 128 | -4.84% | -44.59% |
| 8x128 | 1024 | -30.62% | -46.58% |
| 32x128 | 4096 | -50.02% | -53.35% |
| 32 ragged 32..256 | 4608 | -50.36% | -52.91% |
| 8x512 | 4096 | -48.84% | -51.40% |
| 32x512 | 16384 | -54.96% | -55.31% |

[Eager 2/3](tei-bert-fp8-bias-gelu-integrated200-graphs0-2-3.json),
[eager 3/2](tei-bert-fp8-bias-gelu-integrated200-graphs0-3-2.json),
[graphs 2/3](tei-bert-fp8-bias-gelu-integrated200-graphs1-2-3.json),
[graphs 3/2](tei-bert-fp8-bias-gelu-integrated200-graphs1-3-2.json).
Eager reports contain zero captured graphs; enabled reports contain four.
Every process map identifies the complete native library below and the reports
record executable/environment evidence with **no `LD_PRELOAD`**.

Native library SHA256:
`a981b9b5043aa0bb2a338ee1de8729a483314484b18a95b9f2e75df00e96bc66`.
Comparator SHA256:
`2b17e4f7cc6a058fa7c42acb44a7fed2456e97665dc5f6cae22024aaf81f349c`.
Reference versions are Torch 2.14.1+cu130 and Candle fork revision
`036f39e44015e78ba18d716d73dfdc2e78730c44`.
The controlled environment sets `TORCH_BLAS_PREFER_CUBLASLT=1` and
`CUBLASLT_WORKSPACE_SIZE=32768`; native packed cuDNN is enabled where eligible.
The complete runtime passed 29 native checks (31.29s) and eight Rust integration
checks (0.97s).

The first FP8 pilot failed at cosine
[0.9812788464](tei-bert-trained-fp8-pilot-failure.json).
[Attention-only/local-norm probe](tei-bert-fp8-live-norm-probe-failure.json)
and [exact-norm with the previous attention threshold](tei-bert-fp8-exact-norm-pilot-2-3-failure.json)
failures remain retained. Matching the original unrounded embedding/residual
norm contracts and native Torch packed FMA attention fixes accuracy.
[Actual original compiled Candle reference](tei-bert-fp8-live-original-norm-contract-proof.json)
checks all 24 live learned residual norms: zero differences against the exact
adapter. The earlier [component proof](tei-bert-fp8-original-norm-contract-proof.json)
is historical evidence, not a whole-model qualification.

The first exact-contract eager prototype still missed latency at
[+7.19% on 1×32](tei-bert-fp8-exact-contracts-isolated200-graphs0-3-2.json).
Fusing bias addition with original unary GELU preserves the separate FP16 bias
rounding and removes 12 launches per forward (149→137), as recorded in the
[profiling evidence](tei-bert-fp8-bias-gelu-launch-profile.json).
The [isolated 2/3](tei-bert-fp8-bias-gelu-isolated200-graphs0-2-3.json) and
[isolated 3/2](tei-bert-fp8-bias-gelu-isolated200-graphs0-3-2.json) prototypes
passed before integration; their faster timings do not replace the complete
runtime measurements above. Profiling overhead is excluded from qualification.
The existing GELU test retains its exhaustive independent staged oracle and
adds byte-level bias-rounding/current-device checks; existing norm tests include
original learned golden residuals, with the checkpoint's MIT notice retained.

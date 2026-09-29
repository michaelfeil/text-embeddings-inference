# Laya verification and benchmark (2026-09-29)

Checkpoint: [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions/tree/1a793eb568e6718f15941d08f85432581df534e3), revision
`1a793eb568e6718f15941d08f85432581df534e3`. Local weights were SHA-256
verified against this revision's Hub LFS metadata:
`4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e`.

The historical exact-GELU measured build is `f20c9a5` plus the source changes recorded (with hashes) in
[the latency artifact](benchmarks/laya-latency.json). It includes the ModernBERT
CUDA mask/exact-GELU fixes and Laya ReLU/exact-GELU fixes. Earlier PR latency
numbers did not apply attention masks correctly on CUDA and are superseded.
The ModernBERT fixes are also proposed independently against `main`.

## Current activation policy and controlled ablation

ModernBERT intentionally uses the tanh approximation for `hidden_activation:
"gelu"`. Explicit `gelu_new` and `gelu_pytorch_tanh` aliases also remain
approximate. This applies to the shared ModernBERT encoder MLP, including dense
and Flash Attention paths. Eligible rank-two Flash Attention projections reuse
the existing approximate gated-activation kernel; dense rank-three projections
retain its ordinary-operation fallback. No new kernel or rank-three fusion is
introduced here. The CUDA attention-mask fix remains in place.
Laya's custom head still uses ReLU, with exact GELU in its scorer/action layers.

This encoder policy deliberately differs from upstream Laya's exact GELU.
The parity, BF16 accuracy, and HTTP latency artifacts below describe the earlier
**exact-GELU baseline**, not fresh verification of the approximate policy.
Historical artifacts are retained without rewriting their results.

The [controlled FP32 ablation](benchmarks/laya-correctness-ablation.json) uses
all 400 cases / 2,000 decisions from the same pinned checkpoint and test split:

| ModernBERT variant | Accuracy | ECE (10 bins) | Score MAE |
|---|---:|---:|---:|
| Before mask/GELU fixes | 0.6285 | 0.19586 | 0.45119 |
| Mask fix, approximate GELU (current policy) | 0.7670 | 0.21434 | 0.24242 |
| Mask fix, exact GELU (historical baseline) | 0.7660 | 0.21324 | 0.24240 |

The mask fix changed 617 decisions. The activation choice changed two decisions.
The 0.10 percentage-point accuracy difference does not establish general quality
superiority; approximate GELU can change probabilities, confidence, and decisions
near thresholds. Its current-policy row was measured with the equivalent
mask-only source variant identified in the artifact, not this follow-up commit.
No new BF16 accuracy or performance claim follows from this FP32 ablation.

## Historical exact-GELU implementation parity

Reference: [upstream Laya](https://github.com/NandhaKishorM/laya/tree/9d955671415fc19f069b9cc998928075c1f255ec),
PyTorch CPU FP32, torch 2.14.0+cpu, transformers 5.17.0. The 13 cases cover all
three decision types, mixed-length batches, option order, Unicode/literal masks,
custom binary labels, option-count temperature buckets, and truncation at
128/512/1024 tokens. Usage and discrete decisions match exactly in these cases.

- **Candle CUDA FP32:** maximum reported numeric difference 0.001; all cases
  within the original 0.01 tolerance. [Full requests/results](benchmarks/laya-parity-f32.json).
- **Candle CUDA BF16:** maximum difference 0.0249, on the eight-option case.
  The original 0.01 check failed. A second characterization run allowed 0.05;
  all 13 cases completed with matching discrete decisions. This is reduced
  precision drift, not FP32-level numerical parity. [Full results](benchmarks/laya-parity-bf16.json).

Matching these cases does not guarantee identical decisions near every threshold.
The action head frequently saturates on this checkpoint; its output is not a
validated safety or escalation guarantee.

The review identified an incorrect Laya head activation (GELU instead of ReLU)
and a CUDA ModernBERT attention mask passed as a writable cuBLASLt output with
beta=0, so masking was ignored. The extension clones the output buffer before
writing it; the caller's original mask was not overwritten. Both fixes remain. Exact GELU was used for
reference verification; the encoder now intentionally uses approximate GELU,
while the scorer/action layers retain exact GELU. Mixed-length batch versus
single-input regression coverage protects local and padding masks across layers.

## Historical exact-GELU labelled decision quality

All **400 test cases / 2,000 decisions** from `LocalLLaMA/typed-decisions`, config
`all`, revision `f2491dda413a9d94afcb30464123b429c857e079`, were evaluated through
`POST /v1/systemone`. Every case succeeded. Structured dataset states are
explicitly JSON-serialized into text using upstream's formatting.

| Measurement | CUDA FP32 | CUDA BF16 |
|---|---:|---:|
| Accuracy | 0.7660 | 0.7675 |
| ECE, 10 bins using maximum answer probability | 0.21324 | 0.21488 |
| Score expectation MAE | 0.24240 | 0.24249 |

[FP32 result](benchmarks/laya-accuracy-f32.json), [BF16 result](benchmarks/laya-accuracy-bf16.json).
BF16 changed 11 of 2,000 discrete decisions compared with FP32; the small
accuracy increase is not evidence that lower precision improves the model.
Choice/score accuracy uses argmax; binary accuracy uses `noul >= 0.5`. Score MAE
uses the expected zero-based level. Entropy-based `confidence` is **not** used as
ECE confidence. These metrics reproduce upstream's published fine-tuned result
(to reported precision); accuracy alone does not establish calibrated confidence.

## Comparison with other open-source implementations/models

Two comparisons must be distinguished:

1. **Implementation parity** above compares the same checkpoint with upstream
   Python Laya. No Python service is used in the Candle serving path.
2. **Model quality** below audits saved results from
   [`infinitylogesh/systemone`](https://github.com/infinitylogesh/systemone/blob/41a96e27c4a623b119f7c07aea703b0eb3e0f8e7/results/sweep-rtxpro6000-2026-09-21.json).
   GPT-OSS was not rerun in this environment.

| Model/setup | Typed accuracy | ECE | Evidence |
|---|---:|---:|---|
| Laya typed-decisions, Candle FP32, approximate encoder | 0.7670 | 0.2143 | Controlled ablation above |
| Laya typed-decisions, Candle FP32, exact encoder | 0.7660 | 0.2132 | Historical baseline |
| Base Laya, proxy | 0.3620 | 0.0261 | Published, additionally recalibrated |
| GPT-OSS-20B, zero-shot proxy | 0.5970 | 0.0547 | Published, additionally recalibrated |
| GPT-OSS-20B, think64 proxy | 0.6260 | 0.0775 | Published, additionally recalibrated |

The roughly 0.36 result is **base Laya**, not our checkpoint. Our checkpoint was
fine-tuned on this dataset's training split; GPT-OSS results are zero-shot. Their
recalibration uses separate fitted temperatures (raw GPT-OSS ECE was 0.1509 and
0.1347 respectively). Thus these rows support task-specific Laya integration,
but not a general superiority or calibration claim. The proxy uses different
prompting/scoring paths, and its RTX PRO 6000 latency is not comparable to our
H100 measurements. Its five-question case latency is not single-question latency.

Laya clamps temperatures to [0.5, 5], as does upstream. The checkpoint's
`choice:11+` value is 0.10058 and becomes 0.5. Treat confidence for that bucket as
unverified; older published calibration results may predate the clamp.

## Historical exact-GELU HTTP latency

One NVIDIA H100 80GB HBM3, Candle CUDA BF16 release build (LTO disabled,
16 codegen units), no Flash Attention, no RadixMLP. One client, one replica,
5 warmups + 50 measurements per case. Localhost HTTP time includes tokenization,
queueing, inference and response. Questions cycle choice/score/noul; a one-question
request is choice. Repeated state is truncated to the listed **tokens per question**.

| Questions/request | 128 tokens p50 ms | 512 tokens p50 ms | 1024 tokens p50 ms |
|---:|---:|---:|---:|
| 1 | 11.78 | 13.16 | 23.19 |
| 2 | 11.99 | 19.37 | 39.62 |
| 4 | 13.82 | 32.70 | 72.74 |
| 8 | 21.20 | 60.24 | 139.33 |
| 16 | 34.83 | 114.29 | 273.09 |
| 32 | 62.38 | 217.84 | 537.63 |

[All p95, queue, inference and batch measurements](benchmarks/laya-latency.json).
`max_batch_tokens=16384`; the 32×1024 case needs two backend batches, all other
cases one. `max_client_batch_size=32`, `max_concurrent_requests=16`, two tokenizer
workers and `RAYON_NUM_THREADS=8`. Latency is not a saturated-throughput benchmark.
ModernBERT encodes each question separately inside a shared tensor batch; shared
state does not imply a shared encoder pass across all question tokens. Dense
attention costs grow with length; the custom head currently runs per question.

## Reproduce

Build the branch with Candle CUDA and start the normal router. Record the build
SHA and activation policy when comparing new results with the historical exact
baseline; these commands now exercise the approximate encoder:

```sh
CUDA_COMPUTE_CAP=90 CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
  cargo build --release -p text-embeddings-router --no-default-features \
  --features candle,http,text-embeddings-backend/cuda,dynamic-linking
CUDA_VISIBLE_DEVICES=0 RAYON_NUM_THREADS=8 target/release/text-embeddings-router \
  --model-id /path/to/pinned/laya-typed-decisions --dtype bfloat16 \
  --hostname 127.0.0.1 --port 18084 --max-client-batch-size 32 \
  --max-batch-tokens 16384 --max-concurrent-requests 16 --tokenization-workers 2
python3 scripts/benchmark-laya.py --port 18084 --output latency.json \
  --server-description 'record build SHA, model revision, hardware, dtype and limits'
```

For FP32 verification, restart with `--dtype float32`. In a separate Python
environment install `datasets`, torch, transformers and upstream Laya from the
pinned Git checkout above. Wait for the server's `Ready` message, then:

```sh
python scripts/verify-laya-reference.py --checkpoint /path/to/pinned/laya-typed-decisions \
  --url http://127.0.0.1:18084 --output parity.json \
  --server-description 'record build SHA, model revision, hardware and dtype'
python scripts/evaluate-laya.py --url http://127.0.0.1:18084 --output accuracy.json \
  --server-description 'record build SHA, model revision, hardware and dtype'
```

The parity script validates the upstream Git commit and source cleanliness.
It compares against upstream exact GELU; approximate-encoder differences are
intentional and a failed numerical/discrete check must not be presented as
exact parity or hidden by increasing the tolerance.
The evaluation script pins the test dataset and fails on any HTTP error or
missing answer rather than silently excluding failed cases. Full evaluation
outputs include per-case answers; committed artifacts keep aggregate metrics.

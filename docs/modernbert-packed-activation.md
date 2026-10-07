# Packed activation for dense ModernBERT

## Scope

Contiguous rank-three FP16/BF16 projections now fold their leading dimensions
into rows (a storage view), invoke the existing packed GELU/SiLU gate kernel,
and restore the output shape. Rank-two callers retain the same kernel.
Unsupported dtypes, activations, dimensions, and non-contiguous inputs keep the
ordinary-operation fallback. No CUDA kernel, activation policy, attention mask,
or Laya custom-head math changes.

This makes dense ModernBERT eligible for the already available approximate GELU
fusion. It does not newly accelerate rank-two Flash Attention or Gemma/Qwen
callers that already used this helper. Only Laya was measured end to end here.

## Quality: full pinned test set

H100 80GB, CUDA, checkpoint `convaiinnovations/laya-typed-decisions` revision
`1a793eb568e6718f15941d08f85432581df534e3`, model SHA256
`4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e`.
Dataset `LocalLLaMA/typed-decisions`, config `all`, test revision
`f2491dda413a9d94afcb30464123b429c857e079`, fingerprint `be1933e22e6fbb7c`.
All runs completed all 400 cases / 2,000 decisions with no errors.

| Build | Precision | Accuracy | ECE (10 bins) | Score MAE |
| --- | --- | ---: | ---: | ---: |
| Exact `v1.8.13rc1` tag | BF16 | 0.7685 | 0.21622245 | 0.24257022 |
| This change | BF16 | 0.7685 | 0.21622245 | 0.24257022 |
| Exact `v1.8.13rc1` tag | FP32 | 0.7670 | 0.21433555 | 0.24242414 |

All 2,000 complete BF16 answer objects are identical before and after fusion,
including probabilities, scores, confidence and actions. BF16 clears the 0.76
accuracy requirement. Release BF16 and FP32 differ on 9 discrete decisions;
release FP32 matches the earlier mask-only approximate-GELU ablation.
This is task-finetuned benchmark performance, not a general reasoning score.

The annotated tag object `ba43dff49fcf6a89e3163f98b6ca9ca1aa93636d` resolves to
`a8c9426b8d866ff3ba3a536a92702c017db6e65e`. The baseline binary was compiled
from that clean source and copied before editing; SHA256
`e980c6787385b9614ca67a0581ab04e28d4ffbba4f991a76815070896bb62592`.
These are direct source builds, not Docker-image measurements.

## HTTP latency

Same H100, BF16, checkpoint and server settings for both binaries. One request
in flight; five warmups per variant, then 50 paired measurements with alternating
baseline/fused order. Inputs use the repository fixture and repeated state from
`scripts/benchmark-laya.py`; all answers and token counts were checked equal.
The host was shared with CPU compilation, so small single-request changes and
tails should not be overinterpreted. No other inference ran on this GPU.

Both servers used `--max-batch-tokens 16384 --max-client-batch-size 32
--max-concurrent-requests 16 --tokenization-workers 2`, with
`RAYON_NUM_THREADS=8`. At 32 questions × 1,024 tokens the scheduler necessarily
splits the request into multiple backend batches; question count is not always
one model batch.

Times are milliseconds. Paired improvement is `1 - median(fused / baseline)`.

| Tokens/question | Questions | Baseline p50 / p95 | Fused p50 / p95 | Paired improvement |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 14.40 / 22.61 | 13.38 / 15.56 | 7.0% |
| 128 | 8 | 24.68 / 26.57 | 21.27 / 25.94 | 14.0% |
| 128 | 32 | 66.67 / 75.18 | 56.41 / 60.71 | 15.7% |
| 512 | 1 | 15.46 / 16.75 | 15.70 / 16.72 | -1.5% |
| 512 | 8 | 61.21 / 63.15 | 54.09 / 56.74 | 11.9% |
| 512 | 32 | 219.40 / 222.20 | 188.52 / 194.05 | 14.2% |
| 1024 | 1 | 24.98 / 26.82 | 23.34 / 24.73 | 6.5% |
| 1024 | 8 | 139.37 / 141.37 | 123.80 / 125.52 | 11.2% |
| 1024 | 32 | 534.38 / 536.23 | 472.37 / 475.01 | 11.6% |

The 8/32-question cases improve by 11–16%; one 512-token question is about 1.5%
slower in this run. No universal latency improvement is claimed.

## Activation operation timing

CUDA events on the tensor stream, 10 warmups per variant, 40 alternating paired
samples, 10 invocations per sample; output width 2,624. Times below are
microseconds per invocation, including dispatch/allocation effects rather than
isolated device instructions. No native build or other inference ran on this GPU
during this timing. The shared host still causes visible p95 outliers; these are
reported without filtering. Each shape also passed exact finite-output comparison.

| Dtype | Batch × tokens | Unfused p50 / p95 (µs) | Fused p50 / p95 (µs) |
| --- | ---: | ---: | ---: |
| F16 | 1 × 128 | 21.923 / 22.714 | 5.800 / 7.242 |
| F16 | 8 × 512 | 321.530 / 840.605 | 35.990 / 339.654 |
| F16 | 32 × 1024 | 2433.778 / 2591.411 | 224.086 / 347.123 |
| BF16 | 1 × 128 | 29.821 / 39.229 | 13.514 / 18.301 |
| BF16 | 8 × 512 | 320.584 / 324.992 | 35.549 / 36.502 |
| BF16 | 32 × 1024 | 2436.787 / 2508.067 | 224.866 / 381.331 |

GPU numerical tests: **2 passed** (all encodings; layouts/widths).
Opt-in timing test: **1 passed**, six shapes/dtypes.

## Reproduce

Build each revision into a separate target directory, preserving both binaries:

```sh
CUDA_COMPUTE_CAP=90 CARGO_PROFILE_RELEASE_LTO=false \
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
cargo build --release -p text-embeddings-router --no-default-features \
  --features candle,http,text-embeddings-backend/cuda,dynamic-linking
```

Start both with the settings above, the same pinned local checkpoint,
`--dtype bfloat16`, and different ports. Run the full quality script against each:

```sh
python scripts/evaluate-laya.py --url http://127.0.0.1:18095 \
  --output baseline.json --server-description 'v1.8.13rc1 BF16 H100'
python scripts/evaluate-laya.py --url http://127.0.0.1:18096 \
  --output fused.json --server-description 'rank3 fusion BF16 H100'
```

For a standalone latency sweep, use `scripts/benchmark-laya.py --port PORT
--output result.json --server-description DESCRIPTION`. The table above used
its same payloads with 1/8/32 questions, interleaving the two variants per sample
rather than running the two sweeps sequentially.

Run the numerical and opt-in operation timing tests (one idle GPU):

```sh
CUDA_COMPUTE_CAP=90 CUDA_VISIBLE_DEVICES=0 \
LIBRARY_PATH=/usr/local/cuda/lib64 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
cargo test --release -p text-embeddings-backend-candle \
  --no-default-features --features cuda,cuda-dynamic-linking --lib packed_glu
# Same command with: -- --ignored --nocapture
```

The numerical tests cover every FP16/BF16 input encoding, GELU and SiLU,
rank-two and rank-three output shapes, nonzero/unaligned storage offsets, varied
widths and non-contiguous fallbacks. They compare finite results bitwise,
including signed zero, and require matching NaNs without requiring payload equality.

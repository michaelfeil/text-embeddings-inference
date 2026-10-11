# DistilRoBERTa eager qualification on H100

The corrected whole build passes the fixed cosine ≥0.999 and P50 ≤5% slower gates for `sentence-transformers/all-distilroberta-v1`, FP16 mean pooling, on both GPU assignments. Seven actual packed workloads run for 200 measured iterations per backend after 20 warmups, with Torch and Candle concurrently on separate H100 GPUs. The worst slowdown is **3.04%** on GPUs 2/3 and **3.02%** after swapping; minimum pooled cosine is **0.9999978932840201**.

| Actual packed workload | Torch2 / Candle3 | Torch3 / Candle2 |
|---|---:|---:|
| 1x32 | -11.63% | -4.51% |
| 1x128 | -11.04% | -4.83% |
| 8x128 | -10.13% | -2.25% |
| 32x128 | -0.23% | +0.62% |
| 32 ragged 32..256 | +1.18% | +1.80% |
| 8x512 | +3.04% | +3.02% |
| 32x512 | +1.97% | -0.37% |

Full measurements: [eager2/3](tei-roberta-eager-fused-embedding-whole200-2-3.json), [eager3/2](tei-roberta-eager-fused-embedding-whole200-3-2.json). Inputs use the accepted canonical CLS 0 / SEP 2 and router position offset 2; synthetic vocabulary IDs and exact sequence lengths match the earlier trained benchmark. No checkpoint configuration is altered and no token padding exists. Native inference, device transfers and model-dtype mean pooling are timed; tokenization, HTTP, model loading and benchmark input cloning are excluded.

The scoped change combines three embedding gathers with the existing register LayerNorm kernel, preserving each Half word+type+position addition and the reduction/affine arithmetic. The independent learned-operand comparison is bitwise equal at 1×32, 1×128, 8×512 and 32×512. Torch private packed Flash is selected for at most 4096 real tokens; larger totals retain the existing attention policy. Selection is limited to CUDA SM90, RoBERTa FP16 hidden 768 / 12 heads, GELU, mean embeddings without a classifier or dynamic FP8. Rust retains `_pooling=embedding` and conveys actual mean pooling with `_embedding_pool=mean`.

A three-forward native trace confirms one fused gather/norm kernel, zero ATen embedding calls and zero ATen Flash wrapper calls per forward: 50 GPU kernels rather than 53. The [bounded proof](tei-roberta-eager-selector-and-replay-proof.json) also records an ephemeral same-geometry regression against independent CPU exact-length attention: mixed mean/raw ordering, repeated requests, changed-token graph replay, sequence isolation and restored-input equality all pass. Python only orchestrates the native profile; it performs no model inference.

[Graph2/3](tei-roberta-fused-embedding-graph-regression20-2-3.json) and [graph3/2](tei-roberta-fused-embedding-graph-regression20-3-2.json) run 20 iterations as capture/replay correctness regressions. Both pass all seven numerical checks and retain four captured graph entries. These samples do not establish a new graph latency qualification; the earlier accepted graph evidence remains historical.

The whole runtime SHA256 is `e178c24ea440160800e379de84bd2df5af1fd6d6fd90bb40e103a65b410ff949`; the corrected Rust comparator SHA256 is `1a1fed06519b02e525907b9b09f8fbcd3ad8e60457503245865aa6be9eaec48e`. Every report records live process mappings, showing exactly this native library, and uses no preload. Checkpoint revision is `842eaed40bee4d61673a81c92d5689a8fed7a09f`; weight SHA256 is `3c739fad12f3f12428a677cfcc73391a5860a8bad0ff7a971fd67e6065201fc2`. The [whole validation manifest](tei-native-pooling-handoff-test-manifest.json) records 29 native checks, 10 Rust checks and strict combined-backend Clippy passing.

The earlier whole build failed because Rust emitted `_pooling=embedding` while the proposed native guard expected `mean`, leaving the optimization inactive. Those [eager2/3](tei-roberta-selector-inactive-whole-eager200-2-3.json), [eager3/2](tei-roberta-selector-inactive-whole-eager200-3-2.json) and [graph 2/3](tei-roberta-selector-inactive-whole-graph20-2-3.json), [graph 3/2](tei-roberta-selector-inactive-whole-graph20-3-2.json) reports are retained separately. Their worst eager slowdowns were 12.65% and 13.37%; isolated passing pilots were never treated as whole-build qualification. Other checkpoints, pooling modes, trained heads, FP8, BF16 and other accelerator architectures are outside this new trained eager scope.

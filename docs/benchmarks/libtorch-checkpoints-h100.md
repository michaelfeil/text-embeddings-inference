# Native LibTorch H100 checkpoint comparisons

Measured on 2026-10-10 with Torch 2.14.1. Each runtime loads the same checkpoint on a separate H100; runs repeat with GPU assignments swapped. Timings include transfers, model execution and pooling, and exclude HTTP and tokenization. No token padding is used. These are warmed backend measurements, not a serving latency guarantee.

## Qwen3-Embedding-0.6B

FP16 with last-token pooling, 200 measured iterations and 20 warmups per backend. These integrated regressions use the private-FMA accepted runtime, with native attention and RMS selectors unchanged for this configuration. CUDA graphs are enabled with a 4096-token cap and four captured shapes; larger batches run eagerly. BLAS settings are `TORCH_BLAS_PREFER_CUBLASLT=1` and `CUBLASLT_WORKSPACE_SIZE=32768` (KiB). Capture cost is excluded after warmup.

| Workload | Torch / Candle P50, ms | Swapped Torch / Candle P50, ms | Worst slowdown |
|---|---:|---:|---:|
| 1x32 | 1.383 / 2.695 | 1.380 / 2.329 | -40.75% |
| 1x128 | 1.619 / 2.693 | 1.614 / 2.516 | -35.85% |
| 8x128 | 2.889 / 3.989 | 2.868 / 3.926 | -26.96% |
| 32x128 | 9.026 / 9.778 | 9.079 / 9.764 | -7.02% |
| 32 ragged 32..256 | 11.144 / 12.173 | 11.238 / 12.160 | -7.58% |
| 8x512 | 9.445 / 11.375 | 9.497 / 11.335 | -16.22% |
| 32x512 | 36.219 / 41.827 | 36.289 / 41.862 | -13.31% |

Minimum embedding cosine across both assignments: 0.999992935. All seven workloads meet the 5% P50 target on both assignments. Reports record immutable binary and native-library identities.

Raw measurements: [6/7](tei-qwen3-integrated-gate8-6-7.json), [7/6](tei-qwen3-integrated-gate8-7-6.json). Earlier results: [0/1](tei-qwen3-v6-graphs-0-1.json), [1/0](tei-qwen3-v6-graphs-1-0.json).

## GTE-base-en-v1.5

FP16 and CLS pooling, with eager execution.

| Workload | Torch / Candle P50, ms | Swapped Torch / Candle P50, ms | Worst slowdown |
|---|---:|---:|---:|
| 1x32 | 1.118 / 1.182 | 1.309 / 1.388 | -5.5% |
| 1x128 | 1.104 / 1.176 | 1.123 / 1.141 | -1.6% |
| 8x128 | 1.285 / 1.987 | 1.291 / 1.976 | -34.7% |
| 32x128 | 3.424 / 6.014 | 3.403 / 6.029 | -43.1% |
| 32 ragged 32..256 | 3.895 / 6.823 | 3.894 / 6.822 | -42.9% |
| 8x512 | 3.627 / 6.141 | 3.616 / 6.159 | -40.9% |
| 32x512 | 12.891 / 22.688 | 12.749 / 22.724 | -43.2% |

Minimum embedding cosine across both runs: 0.999998. The 5% P50 target is met for all seven measured workloads. Other checkpoints and latency percentiles need separate validation.

Raw measurements: [tei-gte-benchmark-2-3.json](tei-gte-benchmark-2-3.json).

Raw measurements: [tei-gte-benchmark-3-2.json](tei-gte-benchmark-3-2.json).

## BGE-base-en-v1.5 (BERT)

FP16 and CLS pooling. These results use Candle-compatible fused tanh GELU, `TEI_TORCH_CUDA_GRAPHS=1`, `TEI_TORCH_CUDA_GRAPH_MAX_TOKENS=16384` and `TEI_TORCH_CUDNN_VARLEN=1`. Packed cuDNN runs when maximum sequence length is at least 512; shorter requests use packed Flash attention. Native libraries and binaries were copied into a stable snapshot before each paired benchmark. Capture cost is excluded after warmup.

| Workload | Torch / Candle P50, ms | Swapped Torch / Candle P50, ms | Worst slowdown |
|---|---:|---:|---:|
| 1x32 | 0.503 / 0.883 | 0.512 / 1.091 | -43.0% |
| 1x128 | 0.565 / 0.899 | 0.552 / 0.866 | -36.2% |
| 8x128 | 0.810 / 1.037 | 0.799 / 0.974 | -18.0% |
| 32x128 | 1.921 / 1.998 | 1.926 / 1.979 | -2.6% |
| 32 ragged 32..256 | 2.237 / 2.268 | 2.250 / 2.282 | -1.4% |
| 8x512 | 2.141 / 2.098 | 2.143 / 2.100 | +2.0% |
| 32x512 | 7.425 / 7.397 | 7.501 / 7.336 | +2.2% |

Worst P50 slowdown across both assignments: 2.24%. Minimum embedding cosine: 0.999998269. These seven workloads meet the 5% target with the settings above.

Raw measurements: [tei-bert-fused-final-graphs-0-1.json](tei-bert-fused-final-graphs-0-1.json).

Raw measurements: [tei-bert-fused-final-graphs-1-0.json](tei-bert-fused-final-graphs-1-0.json).

## E5-Mistral-7B-Instruct

FP16 with last-token pooling, 200 measured iterations and 20 warmups per backend. These integrated regressions use the private-FMA accepted runtime, with native attention and RMS selectors unchanged for this configuration. CUDA graphs are enabled with a 4096-token cap and four captured shapes; larger batches run eagerly. BLAS settings are `TORCH_BLAS_PREFER_CUBLASLT=1` and `CUBLASLT_WORKSPACE_SIZE=32768` (KiB). Capture cost is excluded after warmup.

| Workload | Torch / Candle P50, ms | Swapped Torch / Candle P50, ms | Worst slowdown |
|---|---:|---:|---:|
| 1x32 | 6.537 / 6.720 | 6.549 / 6.696 | -2.19% |
| 1x128 | 7.308 / 8.008 | 7.293 / 8.012 | -8.74% |
| 8x128 | 25.543 / 26.055 | 25.352 / 26.491 | -1.97% |
| 32x128 | 102.034 / 100.490 | 100.985 / 101.920 | +1.54% |
| 32 ragged 32..256 | 116.549 / 116.608 | 115.058 / 118.533 | -0.05% |
| 8x512 | 102.259 / 104.551 | 100.883 / 106.177 | -2.19% |
| 32x512 | 394.982 / 404.996 | 391.284 / 411.326 | -2.47% |

Minimum embedding cosine across both assignments: 0.999998900. All seven workloads meet the 5% P50 target on both assignments. Reports record immutable binary and native-library identities.

Raw measurements: [6/7](tei-mistral-integrated-gate8-6-7.json), [7/6](tei-mistral-integrated-gate8-7-6.json). Earlier results: [0/1](tei-mistral-v6-graphs-0-1.json), [1/0](tei-mistral-v6-graphs-1-0.json).

## Gemma4-26B-A4B-it text

The official trained BF16 checkpoint is pinned to revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`. Both bidirectional embedding (last-token pooling) and the actual causal option-token Decision loader pass the seven packed workloads in both assignments: 200 measured iterations and 20 warmups. Native library SHA256 is `41dbdbe796c6c735ec9ee648796195a9f447083073fe202ca16846631740da76`; reports include checkpoint, comparator and fixture identities. No preload or diagnostic factory is used. Graph runs use a 16384-token cap; eager runs request no graphs and capture none.

| Pipeline | Worst P50 slowdown, graph pair | Worst P50 slowdown, eager pair | Accuracy |
|---|---:|---:|---|
| Bidirectional embedding | +0.995% | +3.488% | All fourteen outputs bitwise equal in each mode |
| Causal option-token Decision | −0.600% | +3.253% | Minimum logit cosine 0.999992159; original gates pass |

Eager execution is approximately 11–12% faster at 8x128 on both pipelines. Small 1x32 requests remain approximately 2.7–3.5% slower. These measurements establish the 5% target on this checkpoint and fixture, rather than a claim that every eager shape is faster.

- Embedding graphs: [0/1](gemma4-moe-staged-rotary-integrated-graphs16384-200-0-1.json), [1/0](gemma4-moe-staged-rotary-integrated-graphs16384-200-1-0.json).
- Embedding eager: [0/1](gemma4-moe-staged-rotary-integrated-eager-200-0-1.json), [1/0](gemma4-moe-staged-rotary-integrated-eager-200-1-0.json).
- Decision graphs: [6/7](gemma4-option-token-causal512-staged-rotary-integrated-graphs16384-200-6-7.json), [7/6](gemma4-option-token-causal512-staged-rotary-integrated-graphs16384-200-7-6.json).
- Decision eager: [6/7](gemma4-option-token-causal512-staged-rotary-integrated-eager-200-6-7.json), [7/6](gemma4-option-token-causal512-staged-rotary-integrated-eager-200-7-6.json).

Decision inputs use synthetic valid IDs and trained vocabulary readout rows 100/200/300/400. This tests computational fidelity, not decision quality; the fixture's action probabilities saturate at 1.0 and do not establish nonconstant calibration. Image decisions remain unqualified: the [post-causal-correction pilot](gemma4-option-token-decision-media-causal512-eager-pilot-0-1-failed.json) fails all five image cases, with minimum cosine 0.950608368. The official checkpoint has no audio tower. Prior [initial accuracy failure](tei-gemma4-moe-v7-pilot-0-1-failed.json), [router-cache latency miss](gemma4-moe-router-cached-full-graphs16384-1-0.json), and smaller-cap misses remain available; source arithmetic fixes, rather than gate changes, produced the current result.

## Additional trained families and remaining gaps

These runs use exact packed mean pooling, except DistilBERT and ModernBERT MLM (CLS), and Qwen3 MoE (last-token). Graph-enabled model execution is warmed; capture and cache misses are excluded. Visual inputs are synthetic patches with valid geometry, so these results validate checkpoint consumption and backend parity rather than real-image retrieval quality.

| Checkpoint | Workloads per assignment | Worst P50 slowdown | Minimum cosine | Result |
|---|---:|---:|---:|---|
| DistilBERT base uncased | 7 | +3.82% | 0.999997784 | Meets measured target |
| ModernBERT embed base (private Torch FMA) | 7 | -9.99% | 1.0 (bitwise) | Meets measured target |
| ModernBERT base MLM CLS (private Torch FMA) | 7 | -11.10% | 1.0 (bitwise) | Meets measured target |
| EmbeddingGemma2 text, image and audio | 9 | +2.43% | 1.0 (bitwise) | Meets measured target |
| EmbeddingGemma 300M plus both Dense heads | 7 | +1.20% | 1.0 (bitwise) | Meets measured target |
| Nemotron Llama 1B BF16 mean | 7 | +4.79% | 1.0 (bitwise, including arbitrary IDs) | Meets measured target |
| Qwen3-30B-A3B BF16 MoE | 7 | -2.27% | 1.0 (bitwise) | Meets measured target |
| Nomic embed text v2 MoE | 7 | -62.84% | 0.999239679 (including arbitrary IDs) | Meets measured target |
| Nomic embed text v1.5 | 7 | -44.68% | 0.999998367 | Meets measured target |
| Qwen3-VL embedding 2B | 9 | +1.49% | 0.999994471 | Meets measured target |
| Qwen3.5 0.8B | 9 | -24.68% | 0.999961894 | Meets measured target |
| DeBERTa v3 base NLI (CLS) | 7 | −8.58% | 0.999995064 | Meets measured target |
| DistilRoBERTa v1 | 7 | +4.75% | 0.999997893 | Meets measured target |
| Jina v2 base en | 7 | -38.09% | 0.999996964 | Meets measured target |
| Jina v2 base code | 7 | -42.61% | 0.999987606 | Meets measured target |

ModernBERT now matches Candle bitwise for both its embedding and MLM checkpoints, including the formerly failing arbitrary-ID CLS regression. Private forward kernels use the exact Flash source pinned by Torch, compiled with fused softmax arithmetic. Gemma3 uses the same private Torch arithmetic and in-place contracted rotary; its full Transformer, mean pooling and both Dense heads match bitwise. Gemma3 reports use 40 measured iterations and 20 warmups, and record both Dense paths and complete checkpoint identities. Qwen3-VL uses FP16 last-token pooling; Qwen3.5 uses BF16 mean pooling. Media graph capture was verified in isolated image and ragged-image runs. Jina now preserves ALiBi in a single packed Torch efficient-attention call per layer. Its shared local bias uses bounded overlapping read-only strides; Q/K/V retain actual token counts. Independent FP16/BF16 attention, sequence-isolation and NaN-unused-storage tests pass. Both checkpoints retain the same trained parity as the earlier per-sequence implementation. The previous 32–117% batched slowdowns are retained as historical reports below.

Raw paired measurements:

- DistilBERT base uncased: [tei-distilbert-packedalibi-final-graphs-benchmark-2-3.json](tei-distilbert-packedalibi-final-graphs-benchmark-2-3.json), [tei-distilbert-fused-graphs-benchmark-3-2.json](tei-distilbert-fused-graphs-benchmark-3-2.json).
- ModernBERT embedding: [private Torch FMA 2/3](tei-modernbert-embedding-mean-private-torch-fma-2-3.json), [private Torch FMA 3/2](tei-modernbert-embedding-mean-private-torch-fma-3-2.json).
- ModernBERT MLM CLS: [private Torch FMA 2/3](tei-modernbert-mlm-cls-private-torch-fma-2-3.json), [private Torch FMA 3/2](tei-modernbert-mlm-cls-private-torch-fma-3-2.json), [official and arbitrary-ID correctness](tei-modernbert-private-torch-fma-correctness.json).
- Gemma3 embedding plus both Dense heads: [private Torch FMA 4/5](gemma3-private-torch-fma-full-graphs-4-5.json), [private Torch FMA 5/4](gemma3-private-torch-fma-full-graphs-5-4.json).
- Nomic embed text v2 MoE: [integrated 2/3](tei-nomic-v2-moe-integrated-mean-2-3.json), [integrated 3/2](tei-nomic-v2-moe-integrated-mean-3-2.json), [official and arbitrary-ID accuracy](tei-nomic-v2-moe-integrated-correctness.json), [exhaustive actual Candle unary oracle](tei-nomic-expert-gelu-candle-unary-oracle.json). All eight accuracy cases and both paired latency comparisons pass; graphs remain disabled for the data-dependent routing path.
- Nomic embed text v1.5: [tei-nomic-mean-packedpool-graphs-benchmark-2-3.json](tei-nomic-mean-packedpool-graphs-benchmark-2-3.json), [tei-nomic-mean-packedpool-graphs-benchmark-3-2.json](tei-nomic-mean-packedpool-graphs-benchmark-3-2.json).
- Qwen3-VL embedding 2B: [tei-qwen3vl-full-graphs-4-5-metadata.json](tei-qwen3vl-full-graphs-4-5-metadata.json), [tei-qwen3vl-full-graphs-5-4-metadata.json](tei-qwen3vl-full-graphs-5-4-metadata.json).
- Qwen3.5 0.8B: [tei-qwen35-full-graphs-fixed-4-5-metadata.json](tei-qwen35-full-graphs-fixed-4-5-metadata.json), [tei-qwen35-full-graphs-fixed-5-4-metadata.json](tei-qwen35-full-graphs-fixed-5-4-metadata.json).
- DistilRoBERTa v1: [tei-roberta-cudnn-inference-4-5.json](tei-roberta-cudnn-inference-4-5.json), [tei-roberta-cudnn-inference-5-4.json](tei-roberta-cudnn-inference-5-4.json). These 400-iteration runs disable unused cuDNN backward statistics and preserve the same outputs. Historical packed-pooling runs [6/7](tei-roberta-packed-pool-6-7.json) and [7/6](tei-roberta-packed-pool-7-6.json) include the earlier +5.06% miss.
- Jina v2 base en: [tei-jina-mean-packedalibi-final-graphs-benchmark-2-3.json](tei-jina-mean-packedalibi-final-graphs-benchmark-2-3.json), [tei-jina-mean-packedalibi-final-graphs-benchmark-3-2.json](tei-jina-mean-packedalibi-final-graphs-benchmark-3-2.json).
- Jina v2 base code: [tei-jina-code-mean-packedalibi-final-graphs-benchmark-2-3.json](tei-jina-code-mean-packedalibi-final-graphs-benchmark-2-3.json), [tei-jina-code-mean-packedalibi-final-graphs-benchmark-3-2.json](tei-jina-code-mean-packedalibi-final-graphs-benchmark-3-2.json).

EmbeddingGemma2 now passes the complete measured text/image/audio set: all nine workloads match Candle bitwise and meet the P50 target in both GPU assignments (worst +2.43%, 40 iterations and 20 warmups). The production audio path matches three proven Candle contracts: im2col/FP32 GEMM convolution, shared-tree reduction with runtime powf, and BF16 matmul rounding before bias addition. Audio measures approximately 7.2–7.3ms versus Candle 330–335ms on this fixture; this does not establish quality on arbitrary real audio. Integrated immutable-library reports: [4/5](embeddinggemma2-private-torch-fma-audio-full-graphs-4-5.json), [5/4](embeddinggemma2-private-torch-fma-audio-full-graphs-5-4.json). Historical [vision-only qualification](embeddinggemma2-private-torch-fma-vision-full-graphs-4-5.json) and [earlier failures](tei-gemma-exact-pool-full-graphs-4-5.json) remain available.

Historical Jina per-sequence Torch attention reports: [Jina 2/3](tei-jina-mean-packedpool-graphs-benchmark-2-3.json), [Jina 3/2](tei-jina-mean-packedpool-graphs-benchmark-3-2.json), [JinaCode 2/3](tei-jina-code-mean-packedpool-graphs-benchmark-2-3.json), [JinaCode 3/2](tei-jina-code-mean-packedpool-graphs-benchmark-3-2.json). Updated Jina/JinaCode artifacts use accepted native library SHA256 `3a97963f4ca9b50757be1ecf7db08f24f6f5ad8293812c3b3bfede271c928793`, FP16 mean pooling, 100 iterations, 20 warmups, graphs enabled and a 16384-token graph cap.

Earlier decoder failures are retained separately: trained BF16 mean-pooled
[Nemotron Llama 1B](tei-llama-nemotron-lt32-v5-graphs-6-7.json) passes the
cosine gate but misses latency by up to 8.59% in a single GPU assignment.
The historical [Qwen3-30B-A3B](tei-qwen3-moe-v5-failed-0-1.json) run stopped at the unchanged
accuracy gate on the first 1x32 workload (cosine 0.998809860); that run does
not establish latency across the remaining workloads. Correct routing-kernel
oracles alone do not qualify these complete models.

DeBERTa’s [2/3](tei-deberta-trained-cls-packed-graphs-2-3.json) and
[3/2](tei-deberta-trained-cls-packed-graphs-3-2.json) runs use distinct
sequence-local relative bias tiles and one packed Torch attention call per layer.
Both report four cached CUDA graphs; the historical
[eager 2/3](tei-deberta-trained-cls-eager-baseline-2-3.json) and
[eager 3/2](tei-deberta-trained-cls-eager-baseline-3-2.json) runs requested graphs
but captured none, and had substantial medium-batch latency gaps. The new path
preserves [independent trained Transformers parity](tei-deberta-trained-packed-hf-cuda-parity.json),
including classifier logits, without padded Q/K/V tokens.

## Qwen2 and routed decoder qualification in progress

GTE Qwen2 1.5B now passes accuracy and the 5% P50 latency target for all seven workloads in both GPU assignments: 500 measured iterations per backend, minimum cosine 0.999965304, worst slowdown +3.88%. Aligned eight-lane FP16 loads improve gated MLP memory traffic while retaining the scalar arithmetic; narrower and unaligned inputs retain bounded fallbacks. Paired rotary preserves the original model-dtype rounding. Integrated immutable-library reports: [6/7](tei-qwen2-trained-integrated-gate8-6-7.json), [7/6](tei-qwen2-trained-integrated-gate8-7-6.json). Earlier paired-operator reports retain the swapped 32x128 +11.20% miss: [6/7](tei-qwen2-trained-paired-operators-6-7.json), [7/6](tei-qwen2-trained-paired-operators-7-6.json).

Qwen3 MoE originally failed the trained accuracy gate. An independent full 48-layer trace isolates two numerical causes: unfused Torch softmax arithmetic and a different residual RMS reduction. Correcting both matches all hidden states and expert routing bitwise; correcting either alone does not. The [numerical trace evidence](tei-qwen3-moe-numerical-trace-evidence.json) is distinct from full integrated accuracy and latency qualification, which now passes: all seven workloads in both GPU assignments match bitwise and meet the 5% P50 target. The completed 200-iteration graph-enabled comparisons have four resident graphs and are at least 2.27% faster on all seven workloads in both assignments: [0/1](tei-qwen3-moe-v7-graphs-0-1.json), [1/0](tei-qwen3-moe-v7-graphs-1-0.json). Startup allocator mapping warnings recovered; the largest eager shape completed with those graphs resident. Memory headroom remains a qualification limit on these 80GB GPUs. Historical eager results: [0/1](tei-qwen3-moe-private-torch-fma-eager-0-1.json), [1/0](tei-qwen3-moe-private-torch-fma-eager-1-0.json).

Nemotron Llama 1B BF16 mean pooling now matches Candle bitwise and meets the P50 target in both corrected 200-iteration comparisons (worst +4.79%): [0/1](tei-llama-nemotron-v8-corrected-0-1.json), [1/0](tei-llama-nemotron-v8-corrected-1-0.json). All seven [arbitrary valid boundary-ID workloads](tei-llama-arbitrary-ids-v8-corrected-0-1.json) also match bitwise. Learned layer traces isolate the attention arithmetic and later residual RMS differences; both corrections are required. The exact RMS adapter currently selects BF16 width 2048 bidirectional Llama; other widths and causal paths retain existing kernels. The [earlier arbitrary-ID failure](tei-llama-nemotron-v7-arbitrary-ids-failed.json) and [configured-token-only results](tei-llama-nemotron-v7-graphs-0-1.json) remain historical evidence.

## Ministral3 trained text tower

Ministral3 3B BF16 with last-token pooling meets the unchanged cosine and P50 latency gates on all seven workloads in both GPU assignments (200 iterations, 20 warmups, minimum cosine 0.999153403, at least 6.97% faster): [6/7](tei-ministral3-trained-integrated-6-7.json), [7/6](tei-ministral3-trained-integrated-7-6.json). Separate valid [boundary IDs 101/102](tei-ministral3-trained-arbitrary-6-7.json) also pass accuracy. Candle now resolves the checkpoint’s authoritative nested rope theta; independent YaRN tests cover scaling beyond the original context.

These runs use a derived text-only checkpoint from the pinned public composite model: all 236 text tensors are copied byte-for-byte with their prefix renamed, and the original text config is preserved. The [derivation manifest](tei-ministral3-text-checkpoint-derivation.json) records source/output SHA256 identities and each tensor mapping. No tensor arithmetic or dtype conversion occurs during extraction. The source Pixtral vision tower and LM readout are outside this embedding qualification; original checkpoint files remain unchanged.

## Eager versus CUDA graphs: short-input comparison

The integrated `41dbdbe7` native library was measured with FP16, CLS pooling,
200 iterations and 20 warmups on GPUs 2/3, then 3/2. Both runtimes execute
in parallel on separate H100s. All seven pooled-output accuracy gates pass
for both models and both execution modes. This Qwen3 run checks pooled CLS
outputs; it does not establish last-token or raw-token accuracy for causal
attention. The separate Qwen3 last-token qualification above remains relevant.

| Model | Worst eager P50 slowdown, either assignment | Worst graph-enabled P50 slowdown, either assignment | Scope |
|---|---:|---:|---|
| BGE-base-en-v1.5 | +10.12% | +2.17% | 32–512 tokens per sequence |
| Qwen3-Embedding-0.6B | +2.47% | -6.47% | 32–512 tokens per sequence |

BERT's eager 8×512 workload misses the 5% target in both assignments.
Graph-enabled runs meet it; graph capture and startup costs are excluded by
warmup. No 1K/2K result is inferred from these short-input measurements.
BGE's checkpoint declares only 512 positions.

Raw reports: [BERT eager 2/3](tei-bert-eager-vs-graphs-200-0-2-3.json),
[BERT eager 3/2](tei-bert-eager-vs-graphs-200-0-3-2.json),
[BERT graphs 2/3](tei-bert-eager-vs-graphs-200-1-2-3.json),
[BERT graphs 3/2](tei-bert-eager-vs-graphs-200-1-3-2.json),
[Qwen3 eager 2/3](tei-qwen3-eager-vs-graphs-200-0-2-3.json),
[Qwen3 eager 3/2](tei-qwen3-eager-vs-graphs-200-0-3-2.json),
[Qwen3 graphs 2/3](tei-qwen3-eager-vs-graphs-200-1-2-3.json),
[Qwen3 graphs 3/2](tei-qwen3-eager-vs-graphs-200-1-3-2.json).

`compare_decoders` accepts `TEI_BENCHMARK_LONG_CONTEXT=only` to measure
1×1024, 1×2048, 8×1024, 8×2048 and ragged 1024/1536/2048/1280 inputs.
The original checkpoint must declare at least 2048 positions. `=1` appends
these shapes to the normal suite. These inputs use actual sequence lengths
and unchanged positional configuration, without token padding.

## Actual 1K/2K-token inputs: eager and CUDA graphs

Qwen3-Embedding-0.6B uses FP16 last-token pooling; ModernBERT embedding
uses FP16 mean pooling. Both unchanged checkpoints declare sufficient
positional capacity (32768 and 8192 respectively). Each runtime receives
actual 1024/2048-token sequences, including eight-sequence and ragged
1024/1536/2048/1280 batches. There is no token padding. These are synthetic
valid vocabulary IDs, measuring backend accuracy and execution rather than
retrieval quality.

The immutable comparator `c22f2b16` and native library `d62642be` are identical
in all reports. Each backend has 20 warmups and 200 measured iterations,
running in parallel on GPUs 0/1 and then 1/0. All five unchanged pooled
accuracy gates pass in both assignments and both modes. Graph-enabled
runs request a 16384-token cap; recorded graph counts confirm capture.
BLAS settings are cuBLASLt preferred and 32768 KiB workspace. Timing includes
input/output transfers and pooling, excluding cloning, HTTP, tokenization,
startup and graph capture. Throughput is tokens divided by synchronous
backend P50; it does not measure concurrent serving throughput.

| Model / mode | Worst P50 slowdown, either assignment | 8×2048 Torch / Candle P50, ms (0/1) | 8×2048 Torch / Candle P50, ms (1/0) | Torch 8×2048 tokens/s, lower assignment |
|---|---:|---:|---:|---:|
| Qwen3 eager | -13.47% | 41.251 / 49.382 | 42.008 / 49.004 | 390,018 |
| Qwen3 graphs | -14.69% | 41.189 / 49.492 | 41.829 / 49.032 | 391,692 |
| ModernBERT eager | -13.61% | 15.369 / 18.226 | 15.643 / 18.108 | 1,047,375 |
| ModernBERT graphs | -14.11% | 15.228 / 18.216 | 15.532 / 18.084 | 1,054,836 |

ModernBERT pooled outputs match bitwise on every measured shape. Qwen3
minimum last-token cosine is 0.999994894. Both meet the 5% target in eager
and graph modes; these results do not establish raw-token parity, other
precisions, all long-context checkpoints or A10 performance.

Qwen3: [eager 0-1](tei-qwen3-long-context-200-graphs0-0-1.json), [eager 1-0](tei-qwen3-long-context-200-graphs0-1-0.json), [graphs 0-1](tei-qwen3-long-context-200-graphs1-0-1.json), [graphs 1-0](tei-qwen3-long-context-200-graphs1-1-0.json).

ModernBERT: [eager 0-1](tei-modernbert-long-context-200-graphs0-0-1.json), [eager 1-0](tei-modernbert-long-context-200-graphs0-1-0.json), [graphs 0-1](tei-modernbert-long-context-200-graphs1-0-1.json), [graphs 1-0](tei-modernbert-long-context-200-graphs1-1-0.json).

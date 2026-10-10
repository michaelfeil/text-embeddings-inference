# Native LibTorch H100 checkpoint comparisons

Measured on 2026-10-10 with Torch 2.14.1. Each runtime loads the same checkpoint on a separate H100; runs repeat with GPU assignments swapped. Timings include transfers, model execution and pooling, and exclude HTTP and tokenization. No token padding is used. These are warmed backend measurements, not a serving latency guarantee.

## Qwen3-Embedding-0.6B

FP16 with last-token pooling and CUDA graphs enabled. Capture cost is excluded after warmup. The cache matches exact sequence boundaries and holds four shapes; variable production batches can miss it.

| Workload | Torch / Candle P50, ms | Swapped Torch / Candle P50, ms | Worst slowdown |
|---|---:|---:|---:|
| 1x32 | 1.332 / 2.203 | 1.346 / 2.115 | -36.4% |
| 1x128 | 1.610 / 2.494 | 1.517 / 2.418 | -35.5% |
| 8x128 | 3.047 / 3.964 | 3.047 / 3.955 | -22.9% |
| 32x128 | 9.883 / 9.783 | 9.868 / 9.750 | +1.2% |
| 32 ragged 32..256 | 12.100 / 12.160 | 12.126 / 12.130 | -0.0% |
| 8x512 | 11.017 / 11.437 | 11.040 / 11.403 | -3.2% |
| 32x512 | 42.467 / 41.656 | 42.293 / 41.658 | +1.9% |

Minimum embedding cosine across both runs: 0.999992093. Inputs use the checkpoint-configured BOS/EOS token IDs. The 5% P50 target is met for all seven measured workloads. Other checkpoints and latency percentiles need separate validation.

Raw measurements: [tei-qwen3-graphs-6-7.json](tei-qwen3-graphs-6-7.json).

Raw measurements: [tei-qwen3-graphs-7-6.json](tei-qwen3-graphs-7-6.json).

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

FP16 and last-token pooling, 200 iterations per backend. These paired H100 results require `TEI_TORCH_CUDA_GRAPHS=1`, eligible packed `TEI_TORCH_CUDNN_VARLEN=1`, `TORCH_BLAS_PREFER_CUBLASLT=1` and `CUBLASLT_WORKSPACE_SIZE=32768` (KiB). The graph cap is 4096 tokens; larger batches execute eagerly. Packed cuDNN is bypassed when a configured sliding window restricts attention.

| Workload | Torch / Candle P50, ms | Swapped Torch / Candle P50, ms | Worst slowdown |
|---|---:|---:|---:|
| 1x32 | 6.616 / 6.660 | 6.561 / 6.728 | -0.7% |
| 1x128 | 7.372 / 7.934 | 7.360 / 7.970 | -7.1% |
| 8x128 | 25.970 / 26.177 | 26.409 / 25.834 | +2.2% |
| 32x128 | 103.373 / 101.395 | 104.763 / 100.309 | +4.4% |
| 32 ragged 32..256 | 117.831 / 117.583 | 119.946 / 115.791 | +3.6% |
| 8x512 | 103.123 / 105.482 | 105.131 / 104.115 | +1.0% |
| 32x512 | 400.851 / 410.000 | 406.746 / 402.375 | +1.1% |

Minimum embedding cosine across both runs: 0.999998713. All measured workloads meet the 5% P50 target with these settings.

Raw measurements: [tei-mistral-lt32-v2-graphs-0-1.json](tei-mistral-lt32-v2-graphs-0-1.json), [tei-mistral-lt32-v2-graphs-1-0.json](tei-mistral-lt32-v2-graphs-1-0.json).

## Additional trained families and remaining gaps

These runs use exact packed mean pooling, except DistilBERT (CLS). Graph-enabled model execution is warmed; capture and cache misses are excluded. Visual inputs are synthetic patches with valid geometry, so these results validate checkpoint consumption and backend parity rather than real-image retrieval quality.

| Checkpoint | Workloads per assignment | Worst P50 slowdown | Minimum cosine | Result |
|---|---:|---:|---:|---|
| DistilBERT base uncased | 7 | +3.82% | 0.999997784 | Meets measured target |
| ModernBERT embed base | 7 | -8.82% | 0.999893990 | Meets measured target |
| Nomic embed text v1.5 | 7 | -44.68% | 0.999998367 | Meets measured target |
| Qwen3-VL embedding 2B | 9 | +1.49% | 0.999994471 | Meets measured target |
| Qwen3.5 0.8B | 9 | -24.68% | 0.999961894 | Meets measured target |
| DistilRoBERTa v1 | 7 | +4.75% | 0.999997893 | Meets measured target |
| Jina v2 base en | 7 | -38.09% | 0.999996964 | Meets measured target |
| Jina v2 base code | 7 | -42.61% | 0.999987606 | Meets measured target |

ModernBERT still has a separate arbitrary-ID MLM checkpoint parity failure; its embedding checkpoint passes the retained arbitrary-ID mean regression (cosine >=0.999552693). Qwen3-VL uses FP16 last-token pooling; Qwen3.5 uses BF16 mean pooling. Media graph capture was verified in isolated image and ragged-image runs. Jina now preserves ALiBi in a single packed Torch efficient-attention call per layer. Its shared local bias uses bounded overlapping read-only strides; Q/K/V retain actual token counts. Independent FP16/BF16 attention, sequence-isolation and NaN-unused-storage tests pass. Both checkpoints retain the same trained parity as the earlier per-sequence implementation. The previous 32–117% batched slowdowns are retained as historical reports below.

Raw paired measurements:

- DistilBERT base uncased: [tei-distilbert-packedalibi-final-graphs-benchmark-2-3.json](tei-distilbert-packedalibi-final-graphs-benchmark-2-3.json), [tei-distilbert-fused-graphs-benchmark-3-2.json](tei-distilbert-fused-graphs-benchmark-3-2.json).
- ModernBERT embed base: [tei-modernbert-embed-mean-packedpool-graphs-benchmark-2-3.json](tei-modernbert-embed-mean-packedpool-graphs-benchmark-2-3.json), [tei-modernbert-embed-mean-packedpool-graphs-benchmark-3-2.json](tei-modernbert-embed-mean-packedpool-graphs-benchmark-3-2.json).
- Nomic embed text v1.5: [tei-nomic-mean-packedpool-graphs-benchmark-2-3.json](tei-nomic-mean-packedpool-graphs-benchmark-2-3.json), [tei-nomic-mean-packedpool-graphs-benchmark-3-2.json](tei-nomic-mean-packedpool-graphs-benchmark-3-2.json).
- Qwen3-VL embedding 2B: [tei-qwen3vl-full-graphs-4-5-metadata.json](tei-qwen3vl-full-graphs-4-5-metadata.json), [tei-qwen3vl-full-graphs-5-4-metadata.json](tei-qwen3vl-full-graphs-5-4-metadata.json).
- Qwen3.5 0.8B: [tei-qwen35-full-graphs-fixed-4-5-metadata.json](tei-qwen35-full-graphs-fixed-4-5-metadata.json), [tei-qwen35-full-graphs-fixed-5-4-metadata.json](tei-qwen35-full-graphs-fixed-5-4-metadata.json).
- DistilRoBERTa v1: [tei-roberta-cudnn-inference-4-5.json](tei-roberta-cudnn-inference-4-5.json), [tei-roberta-cudnn-inference-5-4.json](tei-roberta-cudnn-inference-5-4.json). These 400-iteration runs disable unused cuDNN backward statistics and preserve the same outputs. Historical packed-pooling runs [6/7](tei-roberta-packed-pool-6-7.json) and [7/6](tei-roberta-packed-pool-7-6.json) include the earlier +5.06% miss.
- Jina v2 base en: [tei-jina-mean-packedalibi-final-graphs-benchmark-2-3.json](tei-jina-mean-packedalibi-final-graphs-benchmark-2-3.json), [tei-jina-mean-packedalibi-final-graphs-benchmark-3-2.json](tei-jina-mean-packedalibi-final-graphs-benchmark-3-2.json).
- Jina v2 base code: [tei-jina-code-mean-packedalibi-final-graphs-benchmark-2-3.json](tei-jina-code-mean-packedalibi-final-graphs-benchmark-2-3.json), [tei-jina-code-mean-packedalibi-final-graphs-benchmark-3-2.json](tei-jina-code-mean-packedalibi-final-graphs-benchmark-3-2.json).

EmbeddingGemma2 remains unqualified. Its latest exact-pooling run has minimum text cosine 0.995607, image cosine 0.993262 and audio cosine 0.998332. The image P50 is 4.877 ms versus Candle 4.603 ms. These failures are retained in [the full Gemma run](tei-gemma-exact-pool-full-graphs-4-5.json).

Historical Jina per-sequence Torch attention reports: [Jina 2/3](tei-jina-mean-packedpool-graphs-benchmark-2-3.json), [Jina 3/2](tei-jina-mean-packedpool-graphs-benchmark-3-2.json), [JinaCode 2/3](tei-jina-code-mean-packedpool-graphs-benchmark-2-3.json), [JinaCode 3/2](tei-jina-code-mean-packedpool-graphs-benchmark-3-2.json). Updated Jina/JinaCode artifacts use accepted native library SHA256 `3a97963f4ca9b50757be1ecf7db08f24f6f5ad8293812c3b3bfede271c928793`, FP16 mean pooling, 100 iterations, 20 warmups, graphs enabled and a 16384-token graph cap.

Current decoder failures are retained separately: trained BF16 mean-pooled
[Nemotron Llama 1B](tei-llama-nemotron-lt32-v5-graphs-6-7.json) passes the
cosine gate but misses latency by up to 8.59% in a single GPU assignment.
[Qwen3-30B-A3B](tei-qwen3-moe-v5-failed-0-1.json) stops at the unchanged
accuracy gate on the first 1x32 workload (cosine 0.998809860); that run does
not establish latency across the remaining workloads. Correct routing-kernel
oracles alone do not qualify these complete models.

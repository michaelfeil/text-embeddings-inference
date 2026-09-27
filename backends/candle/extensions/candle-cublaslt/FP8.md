# Experimental dynamic FP8 MLPs

Build the router with `--no-default-features --features http,experimental-fp8,dynamic-linking`
using CUDA toolkit 12.9 or newer. `Dockerfile-cuda` includes this build feature
for Hopper (compute capability 9.0). Enable with `--enable-fp8-dynamic` or
`ENABLE_FP8_DYNAMIC=true`, and select `--dtype float16`.

The runtime option defaults to false. Initial support is the CUDA flash-attention
implementation of Qwen2, Qwen3, Llama and Mistral. Hopper (compute capability 9.0)
is required; validation so far uses H100. cuBLAS outer-vector scaling is
[Hopper-only in CUDA 12.9 and 13.1](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html#narrow-precision-data-types-usage),
even though other FP8 scaling modes support additional GPU architectures. Other architectures, CPU/Metal, BF16,
and builds without `experimental-fp8` reject the option. BERT and ModernBERT
remain research work: their bias/activation and accuracy tradeoffs need separate
validation. An explicitly requested FP8 configuration does not silently fall
back to a different backend or precision.

This is calibration-free quantization, with no calibration dataset or new model
artifact. Load the ordinary full-precision checkpoint. MLP gate/up/down weights
are converted once to E4M3 with an FP32 scale per output channel. Activations are
converted at inference with an FP32 scale per token. Each scale is the row's
maximum absolute value (floored at 1e-12) divided by 448. Accumulation is FP32,
with FP16 outputs. cuBLASLt fast accumulation is used when a tactic exists;
otherwise the same FP8 inputs use full accumulation (needed for some short,
wide projections on CUDA 12.9). This may change last-bit rounding between shapes. Attention, normalization,
residuals, activation arithmetic, embedding tables and output heads retain their
existing precision. Gate/up share one activation conversion through the existing
combined projection. Only finite FP16 inputs are supported by the quantizer.

The API's `/info` reports `enable_fp8_dynamic`. Startup logs identify the recipe.
Each backend execution thread lazily owns one 32 MiB GEMM workspace and at most
16 cached shape plans, shared across layers. Model weights may move from the
loading thread to the inference thread; mutable cuBLASLt handles do not.

## Accuracy and performance status

FP8 changes model outputs. The research Qwen3-Embedding-8B recipe lost an average
0.173 NDCG@10 points across five English retrieval tasks; SCIDOCS showed a
statistically detectable decline. Qwen3-Embedding-0.6B lost 0.874 points on
SciFact. These are research implementation results, not qualification of the
native serving path or a guarantee for other checkpoints or languages.

Research whole-transformer speedups were approximately 27% for Qwen8B and
20–22% for Llama3B at the tested large batches, versus only 4% for Qwen0.6B.
Native linear GPU timings show that input conversion can outweigh GEMM savings
on small shapes. They exclude serving overhead. Measure latency, throughput and
representative task accuracy for the actual model before enabling this option.

## Native H100 serving validation

Qwen3-Embedding-8B, CUDA 12.9, one H100, FP16 baseline versus dynamic FP8 MLPs.
Each shape uses five warmups and 20 serial HTTP requests; values are medians,
including response transport. These are fixed-batch measurements, not sustained
concurrent-load or P99 guarantees.

| Sequences × tokens | FP16 latency | FP8 latency | Throughput change |
| --- | ---: | ---: | ---: |
| 8 × 512 | 106.8 ms | 76.9 ms | +38.9% |
| 32 × 512 | 418.4 ms | 306.2 ms | +36.6% |
| 2 × 8192 | 507.8 ms | 396.0 ms | +28.2% |

Native SciFact evaluation (300 queries, 5,183 documents) gives NDCG@10
78.639 → 78.355 points. The paired bootstrap 95% interval for the difference is
[-0.712, +0.121] points. An interval crossing zero does not prove equivalent
accuracy. Keep this experimental option disabled unless the application's own
quality evaluation accepts the tradeoff.

Five native Rust tests pass under Compute Sanitizer with zero reported errors.
Independent conversion and linear fixtures agree bitwise with their reference,
including both accumulation modes. This verifies the quantization implementation;
it does not imply FP8 model outputs match FP16. Qwen2, Qwen3-0.6B and Llama3B
also pass functional loading/inference checks, without broad quality qualification.

## Fused SwiGLU conversion

With dynamic FP8 enabled, supported packed FP16 SwiGLU inputs combine the
activation and row conversion in one kernel. Intermediate widths 3072, 8192,
and 12288 are supported; other layouts, widths, and activations retain separate
operations. There is no additional runtime flag. FP16 inference is unchanged.

The kernel preserves FP16 intermediate rounding before FP8 conversion. Tests
compare quantized bytes and scale bits with the separate Candle operations.
Qwen3 0.6B and 8B serving fixtures were bitwise equal to the unfused FP8 path,
with approximately 1–3% throughput gains on the measured larger batches.
These checks establish no additional drift in those fixtures, not equivalence
between FP8 and FP16 or accuracy qualification for other models.

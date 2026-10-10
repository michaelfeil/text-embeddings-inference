# Native LibTorch vs Candle: BERT on H100

Candle is faster than this initial LibTorch prototype in every measured workload.

Model: `BAAI/bge-base-en-v1.5`, revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`.
Both runtimes use FP16, CLS pooling, and packed token batches without padding.
LibTorch 2.14.1 and Candle run concurrently on separate H100 80GB GPUs; the second run swaps GPU assignments. Each workload has 20 warmups and 200 measured calls per backend. GPU clocks are not locked.

| Packed workload | LibTorch P50, ms | Candle P50, ms | Candle speedup |
|---|---:|---:|---:|
| 1x32 | 1.36–1.50 | 0.93–0.94 | 1.44–1.62× |
| 1x128 | 1.39–1.50 | 0.92–0.93 | 1.50–1.62× |
| 8x128 | 1.35–1.56 | 0.92–0.93 | 1.45–1.69× |
| 32x128 | 2.44–2.45 | 1.97–1.99 | 1.23–1.24× |
| 32 ragged 32..256 | 2.77–2.78 | 2.27–2.28 | 1.22–1.22× |
| 8x512 | 2.65–2.65 | 2.10–2.11 | 1.26–1.26× |
| 32x512 | 8.89–9.05 | 7.32–7.42 | 1.20–1.24× |

Ranges span the two GPU assignments; they are not confidence intervals. Measurements include input transfer, forward pass, pooling and output transfer to the host. Tokenization, HTTP and input cloning are excluded. Inputs are synthetic token IDs, not a semantic evaluation corpus.

The minimum embedding cosine agreement is above 0.99995 across both runs. Standard GELU is exact in this Torch implementation and approximate in Candle; FP16 and kernel differences also contribute to output differences.

Large batches put Torch about 20–26% behind Candle in latency. Small batches have a larger gap. The prototype uses separate Q/K/V projections, residual adds and layer normalization; Candle combines Q/K/V and fuses residual-add/layernorm. Matching those fusions is a concrete optimization target. A profiler is needed to attribute the measured gap, and CUDA graphs would need a separate experiment.

These results apply to this BERT implementation and H100 hardware. They do not establish Qwen3 performance, A10 performance, HTTP throughput or multi-GPU scaling.

[Raw results and environment](libtorch-bert-h100.json) · [Build and benchmark instructions](../libtorch-backend.md#comparing-against-candle)

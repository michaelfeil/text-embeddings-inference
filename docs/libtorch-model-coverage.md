# Native LibTorch coverage ledger

Status on 2026-10-10. Work is in progress. The target is at most 5% additional
latency relative to Candle on the same checkpoint, precision, pooling, input
lengths and hardware. It has **not** been achieved across these models.

All inference preserves actual token counts. CUDA paths use packed Torch
attention or native packed kernels, including exact-length biased varlen
attention where the installed Flash build lacks bias support.

| Candle family / aliases | Native implementation | Evidence / remaining work |
|---|---|---|
| BERT | Packed encoder, fused QKV/GELU, pooling, classifier and SPLADE heads | CPU embedding/classifier/SPLADE tests pass. Trained BGE meets the 5% P50 target in both GPU assignments (worst +2.24%, cosine >=0.99999827), with graphs and eligible packed cuDNN enabled. |
| RoBERTa, XLM-RoBERTa, CamemBERT | BERT path with checkpoint prefixes and matching heads | CPU Candle and independent classifier references pass all three aliases, four pooling modes, sequence/token heads and mixed pooled/raw outputs. Trained RoBERTa parity passes, but larger mean-pooled batches initially trail by 11–17%; packed pooling optimization is under evaluation. Input positions already include the router offset. |
| DistilBERT | Packed encoder, SPLADE projection | CPU/CUDA Candle parity fixtures pass; trained DistilBERT cosine >0.999998. Fused GELU and graphs meet the 5% latency target on seven workloads in one GPU assignment; swapped confirmation pending. |
| GTE / New | Packed rotary encoder, gated MLP and classifier | CPU/CUDA Candle parity fixtures pass. Trained GTE meets the 5% P50 target in both GPU assignments. |
| ModernBERT | Packed local/global rotary attention, gated MLP and heads | Synthetic parity passes. Trained modernbert-embed-base passes official-token and arbitrary-ID mean parity with default CUDA normalization and matching half-precision rotary arithmetic, and meets the 5% P50 target in both GPU assignments with graphs. The MLM checkpoint still fails arbitrary-ID CLS (cosine 0.996147). |
| Nomic BERT | Packed rotary encoder, dense/MoE MLP | Dense/gated/MoE Candle parity fixtures pass. Trained Nomic v1.5 mean pooling meets the target in both GPU assignments with graphs, approximately 1.8–2.1x faster; trained MoE performance pending. |
| Jina BERT / JinaCode | Packed encoder, exact-length native efficient varlen ALiBi | Trained mean parity passes both checkpoints. Installed Torch Flash rejects ALiBi; per-sequence efficient attention remains 32–117% slower on measured batched workloads. Structural optimization remains open. |
| Llama / Llama bidirectional | Packed GQA, RoPE, RMSNorm, fused projections | CUDA Candle parity fixtures, including Llama3/NTK scaling and bidirectional attention. Real-checkpoint benchmarking pending. |
| Mistral / Ministral3 | Packed GQA, sliding windows, YaRN | CUDA Candle parity fixtures, including sliding windows and Ministral3 scaling. Trained E5 Mistral meets the 5% P50 target in both GPU assignments with graphs, eligible cuDNN and 32 MiB cuBLASLt workspace (worst +4.44%, minimum cosine 0.99999871). Ministral3 trained performance pending. |
| Qwen2 | Packed GQA with projection biases | CUDA Candle parity fixtures; performance pending. |
| Qwen3 / Qwen3 MoE | Packed Q/K norm, GQA, dense/routed MLP and optional projection | CUDA Candle parity over dense and fused/unfused MoE layouts. BF16 grouped routing tested. Trained dense Qwen3 meets the 5% P50 target on seven workloads with graphs in both GPU assignments; trained MoE performance pending. |
| Gemma3 text | Packed local/global attention, Gemma norm semantics | Native CPU packed/single isolation; CUDA Candle parity pending. |
| EmbeddingGemma2 | Packed text model, shared KV, PLE, embedding projection | Native CPU/CUDA fixtures pass. Latest exact-pooling text graphs meet the latency target but minimum cosine 0.995607 fails the trained parity gate. Audio cosine 0.998332 and image cosine 0.993262 remain below the gate; image graph latency also misses the target. |
| Gemma4 / Gemma4 unified | Packed text plus image tower implementation | Native text/image/audio implementations and functional MoE prototype. Wide-head packed attention works; trained image/audio accuracy and MoE performance remain unqualified. |
| Qwen3-VL | Native vision tower, mRoPE, patch/deepstack mergers and decoder injection | Synthetic CPU/CUDA image consumption and sequence isolation pass. Trained 2B embedding checkpoint meets parity and the 5% latency target for seven text and two image workloads in both GPU assignments, with media graphs enabled. |
| Qwen3.5 dense/MoE/text aliases | Packed full attention, DeltaNet, MoE/shared experts and vision tower | Native dense DeltaNet CPU/CUDA parity and MoE/vision isolation fixtures pass. Trained dense 0.8B meets parity and the latency target for seven text and two image workloads in both GPU assignments. Aligned BF16 CUDA uses grouped dispatch; other paths retain a synchronizing reference. Trained MoE qualification pending. |
| MPNet | Native exact-length relative-biased attention | Independent Transformers CPU/CUDA references pass, including logarithmic bucket distances. Candle rejects packed MPNet; trained performance pending. |
| DeBERTa-v2/v3 (experimental) | Native disentangled relative attention, convolution and classifier | Independent Transformers CPU/CUDA references pass over c2p/p2c, bucket and convolution variants; trained performance pending. |

Classifier and SPLADE coverage is architecture-specific. Typed Laya, Clef,
Qwen3.5 Pplx and option-token decisions now have explicit native contracts and
sidecar loading. Synthetic CPU startup and CUDA Candle comparisons pass,
including learned Laya action probabilities and Pplx causal/noncausal modes;
trained checkpoint qualification remains pending. Sentence Transformers Dense
chains support identity/tanh projections and retain raw token widths; CPU Candle
parity passes for all three pooling modes with mixed pooled/raw outputs.
CUDA Dense mixed-width outputs also pass native GPU versus CPU comparisons.
Radix folding and dynamic FP8 remain open work.

The initial BGE benchmark is recorded in
[the H100 report](benchmarks/libtorch-bert-h100.md). Updated Qwen3 and GTE results are in the
[checkpoint report](benchmarks/libtorch-checkpoints-h100.md). BERT fusion, graph
replay and packed cuDNN meet the target for the measured BGE workloads; other
checkpoints and uncached shapes need their own measurements. These figures are
backend timings with transfers, excluding HTTP and tokenization; no A10 has been
measured.

# Native LibTorch coverage ledger

Status on 2026-10-10. Work is in progress. The target is at most 5% additional
latency relative to Candle on the same checkpoint, precision, pooling, input
lengths and hardware. It has **not** been achieved across these models.

All inference preserves actual token counts. CUDA paths use packed Torch
attention or native packed kernels, including exact-length biased varlen
attention where the installed Flash build lacks bias support.

| Candle family / aliases | Native implementation | Evidence / remaining work |
|---|---|---|
| BERT | Packed encoder, fused QKV/GELU, pooling, classifier and SPLADE heads | CPU embedding/classifier/SPLADE tests pass. Trained BGE FP16 CLS meets the 5% P50 target in both H100 assignments with eager and graph execution: worst +4.896% eager / +1.865% graphs, cosine >=0.99999499. The private Torch packed-attention selector is scoped to H768/head64 embeddings at <=4096 actual tokens; classifier/SPLADE paths retain their separate qualification. |
| RoBERTa, XLM-RoBERTa, CamemBERT | BERT path with checkpoint prefixes and matching heads | CPU Candle and independent classifier references pass all three aliases, four pooling modes, sequence/token heads and mixed pooled/raw outputs. Trained DistilRoBERTa mean pooling meets the 5% P50 target on seven workloads in both GPU assignments (worst +4.75%, minimum cosine 0.999997893), using packed pooling, graphs and eligible cuDNN without backward statistics. Historical larger-batch latency misses are retained in the benchmark report. Input positions already include the router offset. |
| DistilBERT | Packed encoder, SPLADE projection | CPU/CUDA Candle parity fixtures pass; trained DistilBERT cosine >0.999997. Fused GELU and graphs meet the 5% P50 target on seven workloads in both GPU assignments (worst +3.82%, minimum cosine 0.999997784). |
| GTE / New | Packed rotary encoder, gated MLP and classifier | CPU/CUDA Candle parity fixtures pass. Trained GTE meets the 5% P50 target in both GPU assignments. |
| ModernBERT | Packed local/global rotary attention, gated MLP and heads | Private Torch-pinned fused-softmax kernels match Candle bitwise for both trained embedding mean and MLM CLS checkpoints, including official and arbitrary IDs. All seven workloads pass both GPU assignments with graphs (at least 9.99% faster). The earlier MLM arbitrary-ID CLS failure is retained as historical evidence. |
| Nomic BERT | Packed rotary encoder, dense/MoE MLP | Dense/gated/MoE Candle parity fixtures pass. Trained Nomic v1.5 mean pooling meets the target in both GPU assignments with graphs, approximately 1.8–2.1x faster; Nomic v2 MoE now passes official and arbitrary-ID accuracy (minimum cosine 0.999239679) and all seven paired latency workloads on both assignments (at least 2.69x faster). Dedicated model-dtype GELU matches all finite FP16/BF16 values against the actual Candle unary kernel; exact beta LayerNorm preserves unrounded residual statistics. The earlier arbitrary-ID failure is retained as historical evidence. |
| Jina BERT / JinaCode | Packed encoder, one Torch efficient varlen ALiBi call per layer | Trained mean parity passes both checkpoints, with cosine >=0.999996964 / >=0.999987606. Shared local bias uses bounded overlapping read-only strides, with actual packed Q/K/V and no total-token-squared allocation. Both checkpoints meet the 5% P50 target on seven workloads in both GPU assignments (at least 1.61x / 1.74x faster). Independent attention, sequence isolation and NaN-unused-storage CUDA tests pass for FP16/BF16. |
| Llama / Llama bidirectional | Packed GQA, RoPE, RMSNorm, fused projections | CUDA Candle fixtures include Llama3/NTK scaling and bidirectional attention. Trained Nemotron Llama 1B BF16 mean pooling now matches Candle bitwise across all seven workloads in both 200-iteration GPU assignments and meets the P50 target (worst +4.79%). Separate valid boundary IDs 101/102 also match bitwise across all seven workloads. Learned layer traces prove both fused Torch attention arithmetic and the original Welford residual RMS are needed; the exact RMS adapter covers the registered BF16 width 2048 geometry for causal and bidirectional Llama. Trained causal TinyLlama now matches Candle bitwise; its default eager and FP8 graph qualifications are described below. [Paired evidence](benchmarks/tei-llama-nemotron-v8-corrected-0-1.json), [swapped evidence](benchmarks/tei-llama-nemotron-v8-corrected-1-0.json), [arbitrary IDs](benchmarks/tei-llama-arbitrary-ids-v8-corrected-0-1.json). The prior arbitrary-ID accuracy failure and earlier latency misses are retained. |
| Mistral / Ministral3 | Packed GQA, sliding windows, YaRN | CUDA Candle parity fixtures, including sliding windows and Ministral3 scaling. Trained E5 Mistral meets the 5% P50 target in both GPU assignments with graphs, eligible cuDNN and 32 MiB cuBLASLt workspace (worst +1.54%, minimum cosine 0.999998900; [integrated result](benchmarks/tei-mistral-integrated-gate8-6-7.json), [swapped result](benchmarks/tei-mistral-integrated-gate8-7-6.json)). Derived trained Ministral3 3B text-only BF16 last-token pooling passes all seven workloads on both GPU assignments (200 iterations, minimum cosine 0.999153403, at least 6.97% faster); separate arbitrary boundary-ID cases pass too. [Paired evidence](benchmarks/tei-ministral3-trained-integrated-6-7.json), [swapped evidence](benchmarks/tei-ministral3-trained-integrated-7-6.json), [byte-exact extraction scope](benchmarks/tei-ministral3-text-checkpoint-derivation.json). This qualifies the Ministral3 text tower, excluding the source composite checkpoint’s Pixtral vision tower. |
| Qwen2 | Packed GQA with projection biases | Trained GTE Qwen2 1.5B meets accuracy and latency on all seven workloads in both GPU assignments (500 iterations, minimum cosine 0.999965304, worst P50 slowdown +3.88%). Paired rotary and aligned eight-lane gated loads retain exact scalar arithmetic and actual token counts. The earlier swapped 32x128 latency miss is retained as historical evidence. |
| Qwen3 / Qwen3 MoE | Packed Q/K norm, GQA, dense/routed MLP and optional projection | CUDA Candle parity over dense and fused/unfused MoE layouts. BF16 grouped routing tested. Trained dense Qwen3 meets the 5% P50 target on seven workloads with graphs in both GPU assignments; the earlier trained Qwen3-30B-A3B run failed accuracy (cosine 0.998809860). Private Torch fused-softmax attention plus exact Welford residual RMS now matches every hidden state and routing decision in an independent 48-layer trace. Integrated trained Qwen3-30B-A3B is bitwise identical on all seven workloads in both GPU assignments and meets the latency target with four resident graphs (at least 2.27% faster; 200 iterations; [results](benchmarks/tei-qwen3-moe-v7-graphs-0-1.json), [swapped results](benchmarks/tei-qwen3-moe-v7-graphs-1-0.json)). Recovered allocator mapping warnings occurred at startup; the largest eager workload still completed with graphs resident. These measurements do not establish a general memory guarantee on smaller GPUs. |
| Gemma3 text | Packed local/global attention, Gemma norm semantics | Trained embeddinggemma-300m, including both Dense heads, is bitwise identical to Candle on seven workloads in both GPU assignments. Private Torch fused-softmax and in-place contracted rotary meet the latency target (worst +1.20%; 40 iterations per backend, 20 warmups). Native CPU/CUDA fixtures also pass. |
| EmbeddingGemma2 | Packed text model, shared KV, PLE, embedding projection and image/audio towers | Native CPU/CUDA fixtures pass. All seven trained text workloads plus image and audio now match Candle bitwise in both GPU assignments and meet the P50 target (worst +2.43%, 40 iterations, 20 warmups). Audio preserves the actual Candle im2col/FP32 GEMM convolution, shared reduction-tree/runtime-pow norm, and separate BF16 projection-bias rounding; audio is approximately 46x faster on this workload. [Full evidence](benchmarks/embeddinggemma2-private-torch-fma-audio-full-graphs-4-5.json), [swapped evidence](benchmarks/embeddinggemma2-private-torch-fma-audio-full-graphs-5-4.json). Historical failures are retained. |
| Gemma4 / Gemma4 unified | Packed text plus image tower implementation | Trained Gemma4-26B-A4B-it BF16 bidirectional embedding text and actual causal option-token Decision text pass all seven workloads in both 200-iteration GPU assignments, with and without CUDA graphs. Embedding outputs are bitwise equal; Decision minimum logit cosine is 0.999992159 and all original scalar/vector/action/argmax gates pass. Worst P50 slowdown: embedding +0.995% with graphs (16384-token cap), +3.488% eager; Decision −0.600% with graphs, +3.253% eager. Source-proven staged BF16 rotary, reference RMS arithmetic and private Torch causal/noncausal varlen attention preserve actual token/head dimensions. See [embedding eager 0/1](benchmarks/gemma4-moe-staged-rotary-integrated-eager-200-0-1.json), [1/0](benchmarks/gemma4-moe-staged-rotary-integrated-eager-200-1-0.json), [Decision eager 6/7](benchmarks/gemma4-option-token-causal512-staged-rotary-integrated-eager-200-6-7.json), [7/6](benchmarks/gemma4-option-token-causal512-staged-rotary-integrated-eager-200-7-6.json), and [paired graph reports](benchmarks/libtorch-checkpoints-h100.md#gemma4-26b-a4b-it-text). Image Decision pilots still fail accuracy after the causal correction (minimum cosine 0.950608368); vision/audio decisions remain unqualified. [Failed media evidence](benchmarks/gemma4-option-token-decision-media-causal512-eager-pilot-0-1-failed.json) and the prior [router-cache latency miss](benchmarks/gemma4-moe-router-cached-full-graphs16384-1-0.json) and [initial accuracy failure](benchmarks/tei-gemma4-moe-v7-pilot-0-1-failed.json) are retained. |
| Qwen3-VL | Native vision tower, mRoPE, patch/deepstack mergers and decoder injection | Synthetic CPU/CUDA image consumption and sequence isolation pass. Trained 2B embedding checkpoint meets parity and the 5% latency target for seven text and two image workloads in both GPU assignments, with media graphs enabled. |
| Qwen3.5 dense/MoE/text aliases | Packed full attention, DeltaNet, MoE/shared experts and vision tower | Native dense DeltaNet CPU/CUDA parity and MoE/vision isolation fixtures pass. Trained dense 0.8B meets parity and the latency target for seven text and two image workloads in both GPU assignments. Aligned BF16 CUDA uses grouped dispatch; other paths retain a synchronizing reference. Trained BF16 35B-A3B MoE text meets bitwise pooled parity and the latency target on seven eager workloads in both GPU assignments. |
| MPNet | Native exact-length relative-biased attention | Independent Transformers CPU/CUDA references pass, including logarithmic bucket distances. Trained all-mpnet-base-v2 CUDA passes four packed workloads against independent exact-length Transformers FP16 calls (minimum mean/CLS/token cosine 0.999998838/0.999997727/0.999993200). Candle rejects packed MPNet, so paired Candle latency is unavailable; see [trained evidence](benchmarks/tei-mpnet-trained-hf-cuda-parity.json). |
| DeBERTa-v2/v3 (experimental) | Native disentangled relative attention, convolution and classifier | Independent Transformers CPU/CUDA references pass over c2p/p2c, bucket and convolution variants. Trained nli-deberta-v3-base CUDA passes four workloads against independent Transformers FP16 calls (minimum CLS/token cosine 0.999996700, classifier-logit cosine 0.999969728); see [trained evidence](benchmarks/tei-deberta-trained-hf-cuda-parity.json). The packed relative-bias Torch path with verified CUDA graphs beats Candle on all seven workloads in both GPU assignments (at least 8.58% faster; minimum CLS cosine 0.999995064 and classifier cosine 0.999970544). Historical per-sequence eager latency misses are retained in the checkpoint report. |

Classifier and SPLADE coverage is architecture-specific. Typed decision contracts
and sidecar loading have synthetic CPU/CUDA coverage, including nonconstant
learned Laya actions and Pplx causal/noncausal modes. Trained BF16 text decisions
now pass the strict cosine/action/argmax gates and the 5% P50 latency target in
both H100 assignments for Laya, Clef, Pplx and Qwen3.5 option-token decisions:

- Laya `convaiinnovations/laya-typed-decisions` revision `e929ae5cf69bc34259cd2f95c9e91145b818b1f0`: eight workloads per assignment, including the published three-question fixture and valid extended states through 256 tokens. Minimum logit cosine 0.999996787; worst P50 slowdown +2.75%. The combined trained Q/K/V projection preserves all tested native outputs bitwise. The independent upstream CPU fixture passes its original tolerances. Actual tested action probabilities saturate at 1.0, so these trained runs establish agreement on saturated actions; nonconstant actions remain covered by the synthetic fixture. [Correctness and scope](benchmarks/tei-laya-trained-decisions-correctness.json).
- Clef `Cloudflare/clef-flash` revision `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3`: real joint choice/noul/three-level-score schemas, 320–325 actual tokens, batches 1/3/8/32. Minimum individual-field cosine 0.999885575; every field's selected option agrees; both assignments are at least 18.30% faster. [2/3](benchmarks/tei-clef-trained-decisions-graphs-2-3.json), [3/2](benchmarks/tei-clef-trained-decisions-graphs-3-2.json). These categorical score fields have three logits. A separate one-level scalar score passes the fixed magnitude gate `absolute error <= 0.006 + 2*BF16 epsilon*abs(reference)`: native −0.20703125 versus Candle −0.2109375, absolute error 0.00390625. Scalar cosine is excluded from this fidelity gate; both GPU assignments reproduce the same scalar error. [Scalar 2/3](benchmarks/tei-clef-trained-scalar-decisions-2-3.json), [scalar 3/2](benchmarks/tei-clef-trained-scalar-decisions-3-2.json).
- Option-token Qwen3.5 `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17`: real choice prompts with actual vocabulary IDs 32/33/34, 108–113 tokens, batches 1/3/8/32. Minimum cosine 0.999997970; every selected option agrees; both assignments are at least 24.72% faster. [2/3](benchmarks/tei-option-token-trained-decisions-graphs-2-3.json), [3/2](benchmarks/tei-option-token-trained-decisions-graphs-3-2.json).

- Pplx `perplexity-ai/pplx-decider-v1.1-27b` revision `6195aa55a72ae21e503c8a60198f7b390d42adb5`: real three-option choice prompts, 108–113 tokens per prompt, batches 1/3/8/32, trained external BF16 readout rows 0/1/2. Minimum logit cosine 0.999995742; every selected option agrees; all eight paired latency cases are at least 15.27% faster. This covers the published noncausal-full-attention text decision contract; all 255 readout rows and image inference are not qualified by these runs. [2/3](benchmarks/tei-pplx-trained-decisions-prefix-2-3.json), [3/2](benchmarks/tei-pplx-trained-decisions-prefix-3-2.json).

Laya uses `TEI_TORCH_CUDA_GRAPHS=true`. Clef, Pplx and option-token runs also enable
`TEI_TORCH_MEDIA_CUDA_GRAPHS=true`; actual graph counts progress from 1 to 4.
All use native Torch varlen attention without padded token activations or a
Python inference runtime. Historical eager/separate-QKV latency misses are
retained. Pplx's legitimate 5120-wide hidden states and 24 attention heads of
explicit dimension 256 have a narrowly scoped Qwen3.5 startup guard with CPU
and independent Candle CUDA regressions. Genuine `visual.*` checkpoint weights
are normalized to the native tower namespace, preserving the original FP16
vision conversion. Full text/image prefix fixtures remain bitwise equal;
incomplete and ambiguous towers are rejected. The prior startup failures are
preserved in [startup evidence](benchmarks/tei-pplx-trained-decisions-startup.json).
Qwen3.5 now owns its weight map and releases source FFN tensors after caching
concatenated projections. On the same trained Pplx checkpoint, live allocation
after initialization falls from 75.034 GB to 52.215 GB (22.819 GB lower).
Reserved allocator memory initially remains high; normal reclamation succeeds
on the 32x512 workload without startup or inference cache trims. Both assignments
pass all six actual-token decision workloads, including 8x512 and 32x512, at
least 14.37% faster than Candle. The original four native outputs remain bitwise
unchanged. Allocation warnings recovered on both runs; peak live allocation is
55.264 GB, and these runs do not guarantee headroom on smaller GPUs.
[Allocator evidence](benchmarks/tei-pplx-memory-ownership.json),
[2/3 extended cases](benchmarks/tei-pplx-trained-decisions-memory-owned-2-3.json),
[3/2 extended cases](benchmarks/tei-pplx-trained-decisions-memory-owned-3-2.json).
Sentence Transformers Dense
chains support identity/tanh projections and retain raw token widths; CPU Candle
parity passes for all three pooling modes with mixed pooled/raw outputs.
CUDA Dense mixed-width outputs also pass native GPU versus CPU comparisons.
Packed radix folding supports causal Llama/Mistral/Qwen2/Qwen3/Qwen3 MoE text
models, with native CPU fixtures for all five and compact CUDA graph replay
coverage. Rust strips unused alignment rows rather than executing padded tokens;
host validation proves every alias has the same causal past. Bidirectional,
multimodal and typed-decision folding remain unsupported. Trained
Qwen3-Embedding-0.6B FP16 passes all five shared-prefix workloads in both
200-iteration assignments: every raw token, pooled outputs, full/fold parity,
changed-map graph replay, compact-count recapture, and suffix isolation pass.
All measured P50 latencies are at least 10.95% lower than Candle's folded path.
[6/7 evidence](benchmarks/tei-qwen3-trained-radix-4096-6-7.json),
[7/6 evidence](benchmarks/tei-qwen3-trained-radix-4096-7-6.json).
Derived trained Ministral BF16 now passes the stricter shared-prefix raw-token
qualification on all five workloads in both 200-iteration assignments. Full
native/Candle, folded native/Candle, full/fold raw and pooled comparisons are
bitwise equal; changed-map replay, compact-count recapture and suffix isolation
also pass. Every measured last-token P50 is at least 8.88% faster than Candle.
The scoped correction preserves original BF16 width3072 causal Mistral RMS,
rotary and attention arithmetic; unsupported geometries retain native fallbacks.
[6/7 evidence](benchmarks/tei-ministral3-trained-radix-corrected-4096-6-7.json),
[7/6 evidence](benchmarks/tei-ministral3-trained-radix-corrected-4096-7-6.json),
[learned-operand math proof](benchmarks/tei-ministral3-radix-math-contract-proof.json).
The earlier 32x128 raw-token cosine failure (0.998972242), present with and
without folding, remains in [historical evidence](benchmarks/tei-ministral3-trained-radix-4096-6-7-failed.json).
Trained SPLADE (`naver/splade-cocondenser-ensembledistil`) now passes all eight
workloads in both assignments, including support-set and relative-error gates,
at least 14.24% faster than Candle. Requested real token spans are projected in
one packed operation, preserving mixed raw/pooled output order.
[2/3 evidence](benchmarks/tei-splade-trained-heads-optimized-2-3.json),
[3/2 evidence](benchmarks/tei-splade-trained-heads-optimized-3-2.json).
Trained MiniLM reranking passes the fixed scalar magnitude gate and the 5% P50
latency target on all eight workloads in both 200-iteration assignments after
source-proven normalization correction (worst +3.99% and +3.56%). Earlier
32x512 latency misses remain in the historical reports.
[2/3 evidence](benchmarks/tei-minilm-trained-heads-integrated-norm-2-3.json),
[3/2 evidence](benchmarks/tei-minilm-trained-heads-integrated-norm-3-2.json).
Authentic trained TinyLlama causal shared-prefix inference now matches Candle
bitwise across all five workloads in both 200-iteration assignments, with
CUDA graphs disabled. Every measured P50 latency meets the unchanged +5%
target (at least 0.26% faster). The registered BF16 width2048 RMS and native
Torch fused-softmax contracts remove the earlier cross-backend cosine gap.
[6/7 default evidence](benchmarks/tei-tinyllama-causal-trained-default-eager-radix-200-4096-6-7.json),
[7/6 default evidence](benchmarks/tei-tinyllama-causal-trained-default-eager-radix-200-4096-7-6.json),
[source proof](benchmarks/tei-tinyllama-causal-fp8-source-math-proof.json),
[checkpoint verification](benchmarks/tei-tinyllama-causal-checkpoint-verification.json).
Earlier graph-enabled comparisons with minimum cosine 0.999391257 remain in
[historical 6/7 evidence](benchmarks/tei-tinyllama-causal-trained-radix-200-4096-6-7.json)
and [historical 7/6 evidence](benchmarks/tei-tinyllama-causal-trained-radix-200-4096-7-6.json).
Authentic `Qwen/Qwen3.5-35B-A3B` revision
`59d61f3ce65a6d9863b86d2e96597125219dc754` now passes all seven BF16
last-token workloads in both 200-iteration H100 assignments. The integrated
library produces bitwise-identical pooled outputs with CUDA graphs disabled,
including ragged batches and 32×512 actual tokens; P50 is 15.40–26.37% faster
than Candle. The routed-BF16 path preserves the original no-FMA DeltaNet,
norm reduction tree, model-dtype sigmoid and private native Torch packed
attention contracts. Dense Qwen3.5 keeps its existing path. All 28 native checks
and eight Rust integration checks pass. These measurements establish text
pooled-output accuracy, without qualifying this MoE checkpoint's image path,
raw-token outputs, other precisions or A10 performance.
[2/3 evidence](benchmarks/tei-qwen35-moe-integrated-eager-2-3.json),
[3/2 evidence](benchmarks/tei-qwen35-moe-integrated-eager-3-2.json).
The original accuracy failure (cosine 0.997000259) and unsuccessful individual
fixes remain documented.
[Original failure](benchmarks/tei-qwen35-moe-trained-first-workload-failure.json),
[combined-operation proof](benchmarks/tei-qwen35-moe-exact-contracts-candidate-2-3.json).

The same Qwen3.5 BF16 last-token checkpoint also matches Candle bitwise
on actual 1024/2048-token and ragged inputs in both 200-iteration eager
H100 assignments. All five workloads meet the target, with 20.12–26.04%
lower P50 latency. The 8x2048 batch takes 585.06/586.86 ms versus Candle's
791.01/789.74 ms; no input padding is used. Graph-requested runs retain eager execution for this MoE geometry;
CUDA graph performance is unqualified.
[Graph-requested eager-fallback pilot](benchmarks/tei-qwen35-moe-long-context-20-graphs1-0-1.json).
[0/1 eager evidence](benchmarks/tei-qwen35-moe-long-context-200-graphs0-0-1.json),
[1/0 eager evidence](benchmarks/tei-qwen35-moe-long-context-200-graphs0-1-0.json).

Actual 1024/2048-token Qwen3 FP16 last-token and ModernBERT FP16 mean
workloads meet their unchanged pooled accuracy gates and the P50 target in
both 200-iteration H100 assignments, with graphs disabled and enabled.
ModernBERT outputs are bitwise identical; Qwen3 minimum cosine is 0.999994894.
These results cover the measured checkpoints, precision and pooling; they do
not establish raw-token accuracy or A10 performance.
[Long-context scope and reports](benchmarks/libtorch-checkpoints-h100.md#actual-1k2k-token-inputs-eager-and-cuda-graphs).
Experimental dynamic FP8 dense-MLP paths are implemented for
BERT/RoBERTa/ModernBERT and dense Llama/Mistral/Qwen2/Qwen3. Native dense
decoder fixtures and ragged BERT Rust FP16/BF16 integration tests pass.
Authentic trained TinyLlama BF16 causal shared-prefix inference matches Candle
bitwise for every full/raw/folded/last-token pooled output in both 200-iteration
H100 assignments. With CUDA graphs enabled (four resident graphs, 4096-token
cap), all five P50 workloads meet the target and are at least 6.11% faster,
with gains up to 33.44%.
[6/7 FP8 graph evidence](benchmarks/tei-tinyllama-causal-trained-fp8-graphs-radix-200-4096-6-7.json),
[7/6 FP8 graph evidence](benchmarks/tei-tinyllama-causal-trained-fp8-graphs-radix-200-4096-7-6.json).
Eager FP8 outputs are also bitwise in both assignments, but the swapped
8x128 prefix96 case misses the latency target by +8.88%; eager paired
performance remains unqualified.
[6/7 eager evidence](benchmarks/tei-tinyllama-causal-trained-fp8-eager-radix-200-4096-6-7.json),
[7/6 eager latency failure](benchmarks/tei-tinyllama-causal-trained-fp8-eager-radix-200-4096-7-6.json).
Direct original Candle norm and attention entrypoints on identical learned
operands prove the upstream arithmetic correction. The native FP8 GEMMs
match the independently translated original cuBLASLt recipe bitwise on live
trained-size inputs; tiny norm/attention differences crossed FP8 quantization
boundaries before that correction. The prior
[TinyLlama accuracy failure](benchmarks/tei-tinyllama-causal-trained-fp8-pilot-failure.json)
is retained. This qualification covers hidden-state inference and last-token
pooling on synthetic valid IDs, without token padding or a Python runtime;
it does not establish text-generation quality or A10 performance.
Trained FP8 encoder qualification remains pending: the first BGE pilot fails
1x32 accuracy at cosine 0.9812788464430842.
[BGE failure evidence](benchmarks/tei-bert-trained-fp8-pilot-failure.json).
Other decoder checkpoints and precisions still require trained FP8 validation.
[Component evidence](benchmarks/tei-native-fp8-component-validation.json).

The BERT eager private-attention hybrid candidate meets the +5% target in
both isolated 200-iteration assignments, but the first integrated selector
did not activate for canonical GELU checkpoints. Those integrated eager
failures remain retained
([2/3](benchmarks/tei-bert-inactive-gelu-selector-200-graphs0-2-3.json),
[3/2](benchmarks/tei-bert-inactive-gelu-selector-200-graphs0-3-2.json)).
The corrected selector accepts both canonical and approximate GELU and
retains the existing larger-batch path above 4096 actual tokens. Its complete
runtime passes 29 native and eight Rust component tests, including an
independent packed-attention oracle and changed-input graph replay.
The corrected complete runtime passes all seven trained BGE FP16 CLS workloads
in both H100 assignments, with 200 iterations per backend in eager and graph
modes (28 comparisons). Minimum cosine is 0.9999949947; the worst P50 slowdown
is +4.8956445% eager, a narrow margin below the unchanged +5% gate, and
+1.8647306% with graphs. Eager reports contain zero captured graphs; graph
reports contain four. This scope covers synthetic valid vocabulary IDs and
includes transfers and pooling, with no token padding or Python inference.
[Eager 2/3](benchmarks/tei-bert-canonical-private-flash-200-graphs0-2-3.json),
[eager 3/2](benchmarks/tei-bert-canonical-private-flash-200-graphs0-3-2.json),
[graphs 2/3](benchmarks/tei-bert-canonical-private-flash-200-graphs1-2-3.json),
[graphs 3/2](benchmarks/tei-bert-canonical-private-flash-200-graphs1-3-2.json).
[Integration evidence](benchmarks/tei-native-bert-canonical-gelu-integration-validation.json).

The initial BGE benchmark is recorded in
[the H100 report](benchmarks/libtorch-bert-h100.md). Updated Qwen3 and GTE results are in the
[checkpoint report](benchmarks/libtorch-checkpoints-h100.md). BERT fusion and the scoped packed-attention hybrid meet the target for
the measured BGE workloads in eager and graph modes; other
checkpoints and uncached shapes need their own measurements. These figures are
backend timings with transfers, excluding HTTP and tokenization; no A10 has been
measured.

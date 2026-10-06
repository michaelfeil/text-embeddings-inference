# Choosing an attention backend

This is the mf-TEI selection policy based on the September 2026 H100 experiments.
It separates current availability from candidates for automatic selection.
**This document does not enable a new runtime dispatcher.**

## Decision

Keep FA2 as the general CUDA baseline. Prioritize the validated FA3 implementation
for Hopper ModernBERT, then the tested Qwen3-8B configuration. Keep FA4 explicit
opt-in until it offers a repeatable serving benefit and passes model-quality
qualification. Do not select a backend merely because its version number is newer.

| Workload or device | Use now | Candidate to promote | Reason |
|---|---|---|---|
| H100, FP16 BERT / BGE-large, global d64 MHA | FA2 | None from these results | Serving throughput was effectively tied across FA2, FA3 and FA4. |
| H100, FP16 ModernBERT, global and local d64 MHA | FA2 in existing builds; validated FA3 in research builds | FA3 with unused varlen scratch initialization removed | About 29% more throughput than FA2 at the measured large HTTP batch; exact FP16 output parity on the fixtures. |
| H100, FP16 Qwen3-8B, causal d128, Q/KV ratio 4 | FA2 in existing builds; validated FA3 in research builds | FA3, subject to repeatability confirmation | About 5% more throughput than FA2; FA4 brought no clear additional gain. |
| H100, BF16 versions of these models | Current validated backend, normally FA2 | Qualify FA3 separately | BF16 had only one performance round. FP16 timings are not a BF16 policy. |
| Other Qwen/Llama sizes, GQA ratios or head dimensions | Current supported backend, normally FA2 | Model/geometry-specific qualification | Results for Qwen3-8B do not qualify every causal model. Qwen3-0.6B has ratio 2 and falls back from the current FA4 integration. |
| Ampere / Ada | Existing FA2 path in a compatible build | No FA3/FA4 promotion from H100 data | No matching full-model measurements in this experiment. |
| Blackwell, including B200 / RTX targets | Existing supported path in the matching image | Separate FA4 qualification | Upstream FA4 targets Blackwell, but the tested mf-TEI native bundle is SM90-only. |
| Turing | Existing model/image default | FA1 only where explicitly supported and validated | Preserve the documented Turing precision restrictions; do not globally turn on FA1. |
| CPU / Metal / other unsupported combinations | Existing supported non-flash path, or an explicit unsupported error | None from this experiment | Do not silently move CUDA requests to CPU to satisfy a selection preference. |

“Validated FA3” means the finite-window masking fix and the tested wrapper/kernel
revision. An arbitrary older FA3 package is not interchangeable with that build.
FA3 scratch optimization is an implementation of FA3, not a separate user-facing
backend. Keep plain FA3 as a comparison/control while validating that optimization.

## What is actually available

- The `mf/faster-tei` base examined here is `a264b5c`. Its CUDA attention routing
  uses the existing FA1/FA2 paths; it does not ship the research FA3 selector.
- [PR #26](https://github.com/michaelfeil/text-embeddings-inference/pull/26) adds the
  shared FA4 wrapper as an experimental option. It is a draft, not a released
  automatic-selection feature. Its build feature is `experimental-fa4`; runtime
  selection is `TEI_PERF_FA4=1`. A separately built native bundle is required.
- The tested FA4 wrapper is pinned to `a0ec524` in
  [candle-flash-attn-v4](https://github.com/michaelfeil/candle-flash-attn-v4).
  It supports the exported FP16/BF16 SM90 families: noncausal d64 MHA with global
  or finite two-sided local masks, and causal d128 GQA4. The TEI integration
  registers batches for BERT, ModernBERT and Qwen3. ALiBi is excluded from this
  FA4 route. Valid geometry alone does not establish model-quality equivalence.
- [Upstream FlashAttention](https://github.com/Dao-AILab/flash-attention) describes
  FA3 as Hopper-oriented and FA4 as targeting Hopper and Blackwell. Those upstream
  capabilities do not establish coverage in our pinned Rust/AOT wrappers.

## Rules for the future dispatcher

Apply these checks in order, per model and device:

1. **Capability:** check the installed backend, native ABI, GPU architecture,
   dtype, Q/K/V dimensions and strides, MHA/GQA ratio, mask and window semantics,
   sequence layout, and required features such as ALiBi. Preserve causal alignment
   and local windows exactly. Reject an invalid request rather than changing its
   meaning to fit a kernel.
2. **Qualification:** restrict automatic selection to a tested combination of
   implementation revision, architecture, model family, geometry and dtype.
   Capability support and accuracy qualification are different checks.
3. **Performance:** prefer the qualified candidate with a repeatable serving gain.
   As a promotion rule, require at least 5% improvement on a representative
   workload, beyond measured variation, with no material latency regression on
   other important shapes. This is a proposed acceptance threshold, not a claim
   that a 5% point estimate alone proves a win. Keep the existing backend on ties.
4. **Selection:** choose the model/device backend at load time. Restrict subsequent
   per-batch changes to qualified dispatch rules. Avoid online autotuning on live
   user requests, which introduces latency and changes numerical behavior.
5. **Observability:** log the selected backend, implementation revision and reason
   once per model/device. Expose aggregate dispatch/fallback counts for auditing.
   Do not log every layer invocation.

Use one explicit preference (`auto`, `fa2`, `fa3`, `fa4`) when implementing that
configuration, rather than independent booleans that can conflict. These names
are a proposed interface, **not CLI options available today**. For a future strict
explicit selection, fail clearly if it cannot be honored; `auto` may fall back
for a known unsupported capability. A launch failure or illegal memory access
must propagate and make the replica unhealthy, never trigger an in-process retry
on another attention implementation. PR #26's current experimental flag has
capability fallback; a strict preference would be a subsequent API change.

A fallback must preserve the same mathematical operation. For example, ALiBi
cannot be dropped, a finite window cannot become global attention, and a causal
model cannot switch to a noncausal kernel. If no installed backend supports the
operation, return an error.

## Shape-dependent optimizations

**FA3 tile selection:** worth implementing and qualifying next. For packed FP16
ModernBERT-shaped attention (12 heads, d64), the best measured tiles were:

| Fixture | Original FA3 | Tuned FA3 | FA4 |
|---|---:|---:|---:|
| 1 x 128, global | 4.77 us | 3.47 us | 5.84 us |
| 8 x 512, global | 28.84 us | 22.47 us | 28.33 us |
| 8 x 512, local +/-64 | 26.88 us | 20.63 us | 24.11 us |
| 8 x 2048, local +/-64 | 84.92 us | 83.73 us | 77.87 us |
| 1 x 8192, local +/-64 | 46.32 us | 46.29 us | 42.18 us |

Smaller FA3 tiles won at the short/smaller shapes. FA4 won some larger local
shapes. These are attention-only CUDA-graph timings. “Tuned” chooses the best
measured tile after the experiment; it is not an implemented production policy.

Do not turn these few points into a blanket `sequence_length >= 2048 => FA4`
rule. A dispatch rule needs actual GPU batch metadata: batch size, total Q/KV
tokens, maximum and distribution of lengths, padding versus packing, heads, dtype,
mask and window. HTTP batch size is not necessarily GPU batch size. Validate
ragged and interleaved layouts and both dtypes before expanding a rule. Require
full-model quality tests of any hybrid that mixes attention implementations across
layers or batches; standalone layer checks do not qualify the hybrid.

**cuDNN and other attention methods:** retain as profiling candidates. There is
not enough broad model/shape evidence here to recommend automatic selection.
Unsupported masks must not be emulated by changing the requested operation.

**FP8:** decide separately from the attention implementation. An upstream kernel's
FP8 support does not mean the Candle wrapper, model scales, or embedding accuracy
are qualified. Re-benchmark attention choices after changing MLP precision or
fusion, because the fraction of time spent in attention changes.

## Serving and quality evidence

One H100, FP16, 32 sequences per HTTP request, 16,320 actual input tokens:

| Model | FA2 tokens/s | FA3 tokens/s | FA3 scratch optimization tokens/s | FA4 tokens/s |
|---|---:|---:|---:|---:|
| BGE-large | 525,694 | 520,750 | 520,799 | 522,794 |
| ModernBERT | 678,048 | 838,587 | 876,679 | 849,328 |
| Qwen3-8B | 40,631 | 42,754 | 42,670 | 42,568 |

These are request-level tokens/s including queueing, tokenization and HTTP
serialization, not isolated GPU peak throughput. The FA2/FA3/FA4 comparison used
the same research binary with two reversed-order rounds, five warmups and 25
samples per shape. Scratch-optimized FA3 was measured separately in two rounds;
small differences should not be ranked as reliable wins. No GPU clocks were
changed. The exact models were `BAAI/bge-large-en-v1.5`,
`nomic-ai/modernbert-embed-base`, and the local Qwen3-Embedding-8B checkpoint.

FA3 scratch optimization preserved bitwise FP16 outputs on the measured shapes
and all 1,500 STS-B validation pairs. FA4 passed the independent attention
references and sanitizer checks but changed embeddings. ModernBERT FP16 retrieval
on the main-based PR build gave:

| Dataset | FA4 minus FA2 NDCG@10 points | Paired bootstrap 95% interval |
|---|---:|---|
| NFCorpus (323 queries, 3,633 documents) | -0.02854 | [-0.08086, +0.00025] |
| SciFact (300 queries, 5,183 documents) | -0.11707 | [-0.36907, +0.01787] |

Intervals containing zero do not establish equivalence. Promotion requires
independent numerical references, sanitizer checks, representative model-level
retrieval/semantic tests, and an agreed noninferiority criterion selected before
evaluating the candidate. Preserve exact parity where the optimization allows it.
Do not pick a tolerance after seeing a loss merely to approve the new backend.

Implementation and validation links:

- [FA4 integration and serving results, PR #26](https://github.com/michaelfeil/text-embeddings-inference/pull/26).
- [FA4 nonpositive-scale correctness fix, PR #5](https://github.com/michaelfeil/candle-flash-attn-v4/pull/5).

## Implementation order

1. Package the fixed FA3 implementation and scratch optimization with FA2 fallback;
   qualify the targeted Hopper model/dtype combinations and record artifact hashes.
2. Implement a single explicit backend preference and an auditable qualification
   table. Keep the BERT/general baseline on FA2; promote ModernBERT first and confirm
   Qwen3-8B's smaller gain. Do not generalize by model name alone.
3. Qualify bounded FA3 tile rules on actual serving batches and repeat end-to-end
   measurements with sufficient warmup, alternating controls, latency percentiles,
   and both single-request and concurrent load.
4. Keep FA4 in PR #26 as opt-in research until a specific workload clears both
   performance and quality gates. Qualify Blackwell independently with a native
   bundle for that architecture.

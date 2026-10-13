# Experimental LibTorch prefix cache

The native backend can retain transformer prefix state between requests. It is disabled by default; enable it after numerical and performance controls for your deployment. The implementation covers the currently supported causal decoder families; support follows the loaded operation's actual attention policy.

## Configuration

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `TEI_TORCH_PREFIX_CACHE_BYTES` | `0` | Charged resident cache budget in bytes; zero disables the manager. |
| `TEI_TORCH_PREFIX_CACHE_MAX_TOKENS` | `8192` | Maximum retained prefix length per entry. |
| `TEI_TORCH_PREFIX_CACHE_ENTRIES` | `64` | Maximum published entries per loaded engine. |

The byte budget must be nonnegative. Token and entry bounds must be positive. Startup reports whether the loaded model and task can use prefix state. Each engine owns its cache; replicas do not share GPU storage.

## Model and operation coverage

| Loaded causal path | Saved state |
| --- | --- |
| Llama, Mistral/Ministral3, Qwen2, Qwen3 dense and Qwen3 MoE | Per-layer keys and values, plus final hidden rows. |
| Gemma3 | Keys and values for full and sliding attention, plus final hidden rows. |
| Gemma4 dense/MoE option-token decisions and sequence classification | Physical KV producers, including shared consumers and local/global layers, plus final hidden rows. |
| Qwen3.5 dense/MoE and text aliases | Full-attention KV, FP32 DeltaNet checkpoints, projected convolution history and final hidden rows. |
| Qwen3-VL | Text KV and final hidden rows, with image layout/content and every rotary axis included in identity. |

Bidirectional encoders and bidirectional decoder modes continue through their existing execution paths. Ordinary Gemma4 embeddings, EmbeddingGemma2 and Qwen3.5's `noncausal_full_attention` decision mode cannot reuse arbitrary prefix states. Existing unsupported task heads remain unsupported; enabling a cache does not change model eligibility.

## Identity, ownership and output behavior

Identity uses prepared token IDs, token types, positions, multimodal rotary axes and image content/layout. The native bridge supplies borrowed host token views for lookup, avoiding a device-to-host token transfer. Rust supplies SHA-256 image identities; native callers without a digest use exact pixel bytes. Retained keys, values, recurrent state and output rows are independently owned snapshots.

Image-batch execution context is also part of identity for text-only rows. Qwen3-VL's deepstack residual stages affect the whole batch's arithmetic. When all visual rows are already cached, empty feature metadata preserves those stages on real suffix rows without repeating image processing.

Entries are scoped to the loaded model instance and operation. Changing the instance or arithmetic settings, including BLAS workspace limits and SDPA backend/reduction/priority settings, invalidates them. Failed cached work clears the registry. CUDA snapshots can be reused immediately on their producing stream because writes are queued before publication; other streams require their completion event to be ready. Source allocations are retained on the consuming stream. Charged bytes include tensor payloads and saved input metadata. Admission reserves a complete entry before capturing any layers, including recurrent checkpoints. Entries larger than the budget bypass capture. CUDA allocator reservations and temporary execution buffers are separate from this resident budget.

Startup warmup and model health probes carry an explicit probe operation. LibTorch temporarily bypasses the user prefix cache for these calls and restores the previous mode on success or error. Token values do not determine probe intent, and an existing user entry remains available afterward.

The manager preserves original logical offsets. Pooling, repeated/mixed output selectors, raw token outputs and trained heads receive restored full hidden rows. It recomputes at least the final query token on an exact input hit. Qwen3.5 admits sparse, aligned recurrent endpoints; a token match alone cannot create a missing checkpoint.

The cache uses the original logical GEMM geometry to choose supported MLP output-projection recipes while executing only real suffix rows. This removed measured Qwen3 rounding differences caused by suffix geometry. Plans use a bounded LRU, honor reduction and SM settings, and keep scratch workspace owned by ATen's current stream. CPU execution and unsupported native recipes retain the existing operator path. This does not guarantee identical reduction order across every family and BLAS provider; learned numerical qualification covers each family, dtype and operation separately.

Planning uses the original provider's workspace allowance, which can differ from default cuBLASLt scratch. Valid suffix recipes exceeding that default use temporary ATen storage bounded by the original provider allowance. The measured Gemma2048 router needed only231 bytes beyond the default1MiB; falling back at that boundary changed its reductions. Temporary scratch remains separate from charged resident entries and follows CUDA stream lifetime.

Cold capture keeps the original input buffers and sequence-length bound. A real hit packs only the suffix; a single sequence uses token/position/type views and returns its restored output without another concatenation. Publication events are created only when admitting snapshots.

Ordinary decoder output and gate/up projections use the planner for reused suffixes. Gemma and Qwen3.5 also preserve ordinary text/input/gate/router projection geometry while retaining existing Radix recipes. The registered Gemma4 FP32 expert router uses the shared planner. Cold capture and existing dynamic FP8 providers remain unchanged.

## Combining persistence with RadixMLP

The manager derives uncached compact/unfold maps from the original full-context Radix mapping. It does not rebuild a tree from isolated suffix tokens. Equal suffixes with different parent histories remain distinct. Prefix hits skip layer computation for cached tokens; RadixMLP can still share token-local work on uncached branches.

Suffix attention uses independent query/key cumulative arrays and a causal mask that includes the prefix offset. Prepared logical positions and API token usage retain their original meaning.

For default Torch global attention with head64 or head128, a single remaining query uses a separate native specialization with Torch's unfused softmax arithmetic and no single-query GQA layout swap. Longer and windowed suffixes retain their existing attention provider. Matched-operand controls compare the final query against original full-context execution. Across96 reproducible head128 inputs, BF16 matches bitwise; five FP16 cases differ by one representable output step. The FP16 regression includes that rounding-boundary input and bounds the difference to one step rather than claiming universal bitwise equivalence. Random-input controls found rounding differences for broader uses of this specialization, so they are excluded from its dispatch contract.

## Current execution limits

[Native cache design references](libtorch-cache-references.md) compare mistral.rs, SGLang, llama.cpp and FlashInfer and describe the C++ ownership boundary and next integration steps.

The first implementation assembles contiguous prefix/suffix KV rather than using paged attention. Sliding-window layers currently retain the whole admitted prefix and apply the original window mask. Requests routed through the manager use eager execution, including misses; existing graph dispatch remains for operations that bypass the manager. Future paged and graph integration must preserve the same identity and ownership contracts.

Optional cuDNN execution keeps the original packed query geometry and causal provider. Cached query rows use discarded zero placeholders; only real suffix rows are returned. This repeats full attention work, while QKV and MLP computation still skip the prefix. Dense suffix attention with an explicit bias was mathematically equivalent but changed trained logits, so it is not used. Matched-operand single/mixed-batch controls and learned Qwen2 classification/last/mean controls match exactly at the tested lengths. The replay buffers are temporary execution storage, separate from the resident cache budget.

Gemma4 image support continues to follow the existing vision implementation. Checkpoints requiring clipped vision linears remain unsupported before cache lookup; enabling persistence does not override their model configuration.

Functional CPU/CUDA controls are necessary but do not establish learned-model equivalence. Qualification also compares cache-off, cold fill and warm/partial reuse on real checkpoints, including full vectors and trained logits. Timing evidence must distinguish backend inference from HTTP/tokenization, and short-prefix or churn regressions require separate attention. Production deployments should qualify their actual model, provider and request distribution before enabling the cache.

Current H100/Torch 2.14.1 controls match exactly for the tested Qwen3 BF16, FP16 and dynamic MLP FP8 checkpoint at128/512/2048, its mixed/raw/mean/changing-tail requests, and the tested Skywork scalar classifier. Qwen2 BF16 classifier logits and complete last-token/mean vectors match at128/512/2048 on cold fill and exact-input reuse. Additional learned controls match for TinyLlama BF16, E5-Mistral FP16, Qwen3 MoE BF16 and Qwen3.5 dense/MoE BF16 at128/512/2048, and the Gemma4 MoE classifier at32/128/512/2048. These are numerical controls on specific checkpoints, not a complete family/dtype/task or end-to-end performance qualification.

Qwen3-VL and Qwen3.5 dense learned-image controls match cache-off results exactly for cold fill, exact-input warm reuse and changed text tails/pixels/rotary axes at128/512/2048 tokens. A Qwen3-VL512 layer trace matches all447 captured stages after preserving default Torch head128 attention arithmetic. Ministral3 BF16 text controls also match cold/warm outputs at128/512/2048 using a documented byte-preserving extraction of its original learned text submodule; this does not add support for the original Mistral3 vision model.

The tested causal Gemma3 BF16 checkpoint also matches complete cold/warm vectors at128/512/2048, including contexts exceeding its512-token sliding window. The checkpoint is the public Unsloth mirror of Gemma3-1B-IT, with a recorded immutable revision and file hashes.

The learned Qwen2 BF16 classifier also matches all20 tested512-token requests with changing64-token tails on default Torch attention. This extends its exact-input controls without claiming every partial span or dtype is qualified.

Gemma4 E2B dense and Qwen3.5 dense option-token controls match complete hidden vectors and logits at128/512/2048, including changed choices/tails and a warmup probe between hits. Their extracted text-config aliases also match.

Qwen3, Qwen3.5 dense and Gemma4 E2B also match cache-off results for the tested growing requests, mixed batches, mean pooling and complete raw token outputs, both with and without RadixMLP. Gemma4's cached dense Radix projection fallback uses the same logical projection planner as its ordinary cached path; a513-token continuation trace matches all536 captured stages. These controls establish the tested combinations rather than every supported shape or dtype.

Three alternating off/on prepared-token H100/Qwen3 BF16 eager runs measured p50 ranges of7.13-7.18ms without caching versus4.27-4.33ms for a repeated2048-token input. Short8/32/128-token requests measured about1.99-2.12ms off versus2.26-2.38ms on; unique128-token churn measured1.98-2.00ms off versus2.55-2.66ms on. The512-token on results varied2.68-3.77ms versus2.74-2.78ms off, so that shape does not establish a stable gain. Complete cold/warm/churn outputs match exactly. Each hot run has100 samples; churn has128 measured requests after32 discarded calls. These are shared-machine backend measurements; HTTP, tokenization and model loading are excluded.

Qualification is specific to the recorded checkpoints, providers, dtypes and shapes. Other combinations still need deployment-specific numerical and performance controls.

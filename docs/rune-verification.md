# Rune text-decision verification

## Scope and sources

- Model: `michaelfeil/rune-26b-a4b` at `d9507c3d09de24e948f69aaa53c7bcb4a271effb`, BF16 Safetensors mirror of Rune v3 from `surogate/rune-26b-a4b-GGUF` at `c6b360d47895bb77bdf3805a13ee5a5557ab1921`.
- Trained prompt/readout: [Surogate decisions v1](https://github.com/invergent-ai/surogate/blob/d500563399c27c8ce425bcd1214194ce5dda87bb/docs/inference/decisions.md). This adapter accepts text states, string instructions/descriptions, and the existing SystemOne request envelope. It does not implement the upstream server's native object state, thinking, images, or order averaging.
- Hardware: one H100 80GB, BF16, FlashAttention 2, dynamic CUDA linking. Optimized release build with LTO disabled and 16 codegen units; one replica, 8192 batch tokens, 32 client questions, 8 admission slots, 2 tokenizer workers, Rayon 8. Server device allocation after warmup was 53,435 MiB (52.2 GiB).

The MoE router projection retains the full logical batch shape to keep its FP32 reduction order unchanged by Radix folding; dense and expert MLPs still reuse compact rows. The expert kernels and model precision are unchanged. The new head gathers only the requested vocabulary rows, projects the final causal hidden state, and applies Gemma4's configured logit softcap. The HTTP layer computes the selected-option softmax in double precision at temperature 1 and applies Rune's confidence formulas.

## Correctness and numerical limits

The generated mixed choice/noul/score fixture has exact prompt bytes, input IDs, and continuation option IDs against the checkpoint's Transformers chat template with thinking disabled. A 30-option live request also successfully selects the requested option, exercising the extended codebook. Single-level scores return score 0 and confidence 1. Native messages and over-budget prompts return 422.

The CUDA checkpoint regression compares individual requests, a mixed-length batch, and a manually folded shared prefix with an independent Transformers 5.17.0 / Torch 2.11.0 BF16 SDPA reference:

- All three fixture decisions agree.
- Maximum observed absolute logit difference from Transformers: **0.65625**.
- Maximum selected-option probability difference: **0.02345**.
- Transformers comparison bounds are 1.0 for logits and 0.03 for probabilities, plus identical fixture decisions. These are approximate numerical checks; Radix versus the same unfurled batch separately requires exact logit equality.

**Radix on/off now has exact probability agreement on all 2000 labelled decisions.** Before the fix, 40 decisions changed and the maximum probability difference was 0.42796. Fresh runs with and without folding now have zero changed decisions and maximum/mean probability difference zero. All 129 synthetic latency-case answers also exactly match the unfurled baseline, including long states and 32-question requests. The CUDA regression requires exact selected-logit equality between its folded and unfurled batches; Transformers and single-versus-batch comparisons retain their separate approximate checks.

The cause was isolated on `agent_trace_observability_000017`, a real changed five-question case (1217 tokens). Repeated unfurled execution was bit-identical at every layer. Folded embeddings, first attention, dense MLP, and normalized MoE routing inputs were also identical. The first difference occurred in the FP32 router matrix multiplication: changing its row count changed 122599/155776 logits by at most 1.907e-6. First-layer top-eight expert IDs were unchanged, but changed weights affected 529 expert-output elements; the differences then propagated through the model. Unfolding only that small projection restored exact equality at all 30 layer outputs and final logits. This preserves the shared dense/expert computation.

The 2048-token Hopper expert-kernel selection was separately considered: 19 of the original changed answers crossed that boundary and 21 did not. The complete rerun resolves both groups without changing that selection. This is measured parity for these inputs, dtype, and hardware, not a guarantee of bitwise invariance across every GPU or batch shape. The pre-existing Transformers and single-versus-batch numerical differences remain.

## Labelled evaluation

`LocalLLaMA/typed-decisions`, `all`, full `test` split, revision `f2491dda413a9d94afcb30464123b429c857e079`, fingerprint `be1933e22e6fbb7c`: 400 cases, 2000 decisions, zero HTTP errors. States are serialized as text before submission; Rune then JSON-quotes that string as required by its text-state protocol. This is not a measurement of the upstream native object-state endpoint.

| Radix threshold | Accuracy | Correct / 2000 | ECE (10 bins) | Score MAE |
|---|---:|---:|---:|---:|
| 0 | 72.70% | 1454 | 0.14948 | 0.38699 |
| 0.92 | 72.70% | 1454 | 0.14948 | 0.38699 |

Accuracy uses choice/score argmax and noul threshold 0.5. ECE uses maximum option probability, rather than Rune's rescaled confidence. These results do not establish a general quality ranking against Laya or other decision models.

## HTTP latency

One client, three warmups and 20 samples per case. Questions have distinct instructions and mix the three decision types. Shared state is repeated 1, 32, or 128 times, yielding approximately 127, 593, or 2033 tokens per question. Latency includes request preparation on the server, queueing and inference. Larger requests split at the 8192-token batch budget; Radix only shares work within a batch.

| Approx. tokens/question | Questions | Radix off p50 (ms) | Radix on p50 (ms) | Radix on p95 (ms) |
|---:|---:|---:|---:|---:|
| 127 | 1 | 39.91 | 39.72 | 41.04 |
| 128 | 2 | 48.98 | 45.67 | 61.50 |
| 127 | 8 | 105.42 | 63.78 | 64.56 |
| 128 | 32 | 361.65 | 144.73 | 151.79 |
| 592 | 1 | 76.46 | 75.79 | 76.52 |
| 592 | 2 | 121.36 | 83.68 | 85.47 |
| 592 | 8 | 414.59 | 120.56 | 124.51 |
| 593 | 32 | 1623.96 | 405.32 | 414.98 |
| 2032 | 1 | 204.82 | 205.93 | 210.56 |
| 2032 | 2 | 367.22 | 217.37 | 219.16 |
| 2032 | 8 | 1400.12 | 509.20 | 532.49 |
| 2033 | 32 | 5601.58 | 2033.22 | 2085.35 |

## Reproduce

1. Set `RUNE_CHECKPOINT_DIR` to the local checkpoint and `RUNE_FIXTURE_OUT` to a JSON output path. Run the router unit test `http::systemone::rune::tests::checkpoint_prompt_fixture` with `candle,http` features.
2. Run `scripts/verify-rune-reference.py --checkpoint "$RUNE_CHECKPOINT_DIR" --fixture "$RUNE_FIXTURE_OUT" --output reference.json` in an environment with Torch, Transformers and Accelerate. This generates independent reference logits on CUDA.
3. Set `RUNE_FIXTURE` and `RUNE_REFERENCE` to those two JSON files, and run the ignored Candle integration test `test_rune` in a release CUDA/FlashAttention build. It requires the checkpoint on a free GPU.
4. Start the router using the README's Rune command. Run `scripts/benchmark-rune.py --fixture "$RUNE_FIXTURE_OUT" --output latency.json --server-description "<checkpoint, build, GPU, dtype, limits, Radix setting>"`. Repeat with thresholds 0 and 0.92 on the same idle GPU.
5. The existing `scripts/evaluate-laya.py` evaluator also accepts Rune's response schema: pass `--url`, `--output`, and `--server-description` for each server configuration. It evaluates every labelled case and retains individual answers for agreement checks.

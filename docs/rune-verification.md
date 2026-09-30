# Rune text-decision verification

## Scope and sources

- Model: `michaelfeil/rune-26b-a4b` at `d9507c3d09de24e948f69aaa53c7bcb4a271effb`, BF16 Safetensors mirror of Rune v3 from `surogate/rune-26b-a4b-GGUF` at `c6b360d47895bb77bdf3805a13ee5a5557ab1921`.
- Trained prompt/readout: [Surogate decisions v1](https://github.com/invergent-ai/surogate/blob/d500563399c27c8ce425bcd1214194ce5dda87bb/docs/inference/decisions.md). This adapter accepts text states, string instructions/descriptions, and the existing SystemOne request envelope. It does not implement the upstream server's native object state, thinking, images, or order averaging.
- Hardware: one H100 80GB, BF16, FlashAttention 2, dynamic CUDA linking. Optimized release build with LTO disabled and 16 codegen units; one replica, 8192 batch tokens, 32 client questions, 8 admission slots, 2 tokenizer workers, Rayon 8. Server device allocation after warmup was 53,435 MiB (52.2 GiB).

The existing Gemma4 forward, MoE kernels and RadixMLP math are unchanged. The new head gathers only the requested vocabulary rows, projects the final causal hidden state, and applies Gemma4's configured logit softcap. The HTTP layer computes the selected-option softmax in double precision at temperature 1 and applies Rune's confidence formulas.

## Correctness and numerical limits

The generated mixed choice/noul/score fixture has exact prompt bytes, input IDs, and continuation option IDs against the checkpoint's Transformers chat template with thinking disabled. A 30-option live request also successfully selects the requested option, exercising the extended codebook. Single-level scores return score 0 and confidence 1. Native messages and over-budget prompts return 422.

The CUDA checkpoint regression compares individual requests, a mixed-length batch, and a manually folded shared prefix with an independent Transformers 5.17.0 / Torch 2.11.0 BF16 SDPA reference:

- All three fixture decisions agree.
- Maximum observed absolute logit difference from Transformers: **0.65625**.
- Maximum selected-option probability difference: **0.02345**.
- Test bounds are 1.0 for logits and 0.03 for probabilities, plus identical fixture decisions. These are approximate numerical checks, not bitwise equivalence.

**Radix on/off is not decision-identical.** On the complete labelled test set, 40 of 2000 decisions change: 16 choice, 7 noul, 17 score argmaxes. Nineteen become correct and seventeen become incorrect; four change between incorrect answers. Agreement is 98%. The mean per-question maximum probability difference is 0.01935; its maximum is 0.42796. The synthetic latency cases also change 11 of 129 answers, with maximum probability difference 0.22557.

Changing packed token shapes changes floating-point execution. The existing backbone uses FP32 router projection and a different scaling order from Transformers' BF16 router. It also chooses a Hopper expert kernel at 2048 physical tokens; folding can cross that boundary. However, 21 of the 40 test-set changes occur without crossing it, so kernel selection alone does not explain the differences. No model-math change or claim of exact parity is made here. Use `--radix-mlp-threshold 0` to measure or deploy without folding; ordinary batch-shape variation still exists.

## Labelled evaluation

`LocalLLaMA/typed-decisions`, `all`, full `test` split, revision `f2491dda413a9d94afcb30464123b429c857e079`, fingerprint `be1933e22e6fbb7c`: 400 cases, 2000 decisions, zero HTTP errors. States are serialized as text before submission; Rune then JSON-quotes that string as required by its text-state protocol. This is not a measurement of the upstream native object-state endpoint.

| Radix threshold | Accuracy | Correct / 2000 | ECE (10 bins) | Score MAE |
|---|---:|---:|---:|---:|
| 0 | 72.70% | 1454 | 0.14948 | 0.38699 |
| 0.92 | 72.80% | 1456 | 0.14801 | 0.38658 |

Accuracy uses choice/score argmax and noul threshold 0.5. ECE uses maximum option probability, rather than Rune's rescaled confidence. These results do not establish a general quality ranking against Laya or other decision models.

## HTTP latency

One client, three warmups and 20 samples per case. Questions have distinct instructions and mix the three decision types. Shared state is repeated 1, 32, or 128 times, yielding approximately 127, 593, or 2033 tokens per question. Latency includes request preparation on the server, queueing and inference. Larger requests split at the 8192-token batch budget; Radix only shares work within a batch.

| Approx. tokens/question | Questions | Radix off p50 (ms) | Radix on p50 (ms) | Radix on p95 (ms) |
|---:|---:|---:|---:|---:|
| 127 | 1 | 41.94 | 43.95 | 51.50 |
| 128 | 2 | 50.67 | 48.41 | 50.00 |
| 127 | 8 | 107.10 | 65.14 | 68.58 |
| 128 | 32 | 362.31 | 146.13 | 156.30 |
| 592 | 1 | 77.57 | 77.85 | 79.19 |
| 592 | 2 | 121.94 | 84.58 | 86.40 |
| 592 | 8 | 417.14 | 120.94 | 132.76 |
| 593 | 32 | 1623.94 | 400.92 | 405.65 |
| 2032 | 1 | 204.98 | 207.23 | 210.79 |
| 2032 | 2 | 366.94 | 215.99 | 217.57 |
| 2032 | 8 | 1406.63 | 500.20 | 512.71 |
| 2033 | 32 | 5571.65 | 1991.92 | 2029.40 |

## Reproduce

1. Set `RUNE_CHECKPOINT_DIR` to the local checkpoint and `RUNE_FIXTURE_OUT` to a JSON output path. Run the router unit test `http::systemone::rune::tests::checkpoint_prompt_fixture` with `candle,http` features.
2. Run `scripts/verify-rune-reference.py --checkpoint "$RUNE_CHECKPOINT_DIR" --fixture "$RUNE_FIXTURE_OUT" --output reference.json` in an environment with Torch, Transformers and Accelerate. This generates independent reference logits on CUDA.
3. Set `RUNE_FIXTURE` and `RUNE_REFERENCE` to those two JSON files, and run the ignored Candle integration test `test_rune` in a release CUDA/FlashAttention build. It requires the checkpoint on a free GPU.
4. Start the router using the README's Rune command. Run `scripts/benchmark-rune.py --fixture "$RUNE_FIXTURE_OUT" --output latency.json --server-description "<checkpoint, build, GPU, dtype, limits, Radix setting>"`. Repeat with thresholds 0 and 0.92 on the same idle GPU.
5. The existing `scripts/evaluate-laya.py` evaluator also accepts Rune's response schema: pass `--url`, `--output`, and `--server-description` for each server configuration. It evaluates every labelled case and retains individual answers for agreement checks.

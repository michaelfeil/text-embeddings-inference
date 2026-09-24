# Exhaustive grouped decisions with RadixMLP

`POST /decide` accepts shared `context` and named `questions`. Each question is a
finite JSON Schema with an optional `group` marker. Omitted markers belong to
`default`; explicit `"group": "default"` joins that same group.

```json
{
  "context": "A customer requests a refund after 45 days. Policy allows 30 days.",
  "questions": {
    "action": {
      "description": "Choose the action under the refund policy.",
      "enum": ["approve", "reject", "escalate"]
    },
    "urgent": {
      "description": "Does this request need immediate attention?",
      "type": "boolean"
    },
    "language": {
      "group": "language",
      "description": "Identify the customer's language.",
      "enum": ["en", "de", "other"]
    }
  }
}
```

All combinations within a group are evaluated jointly. Different groups are
independent: their schemas and answers are not included in one another's prompts.
This request evaluates **6 + 3 = 9 candidates**. Remove the language group marker
to evaluate all **18** joint combinations instead. Group names never express
execution order or dependencies. There is no adaptive mode or `depends_on`.

The server removes group metadata and renders the context and each group's
schema, including descriptions, into the supported model's chat format (Qwen3
ChatML or Gemma4 text turns). Callers do not supply chat-template tokens. Each
candidate is a canonical JSON object
keyed by question name. Supported schemas are finite `enum`, `const`, booleans,
and nested objects with all properties required and `additionalProperties: false`.
Other constraints are rejected; there is no XGrammar dependency.

All groups are submitted as **one atomic batch and one transformer forward pass**.
The endpoint bypasses the embedding batching scheduler but uses the request
concurrency limit and shared backend channel with backpressure. RadixMLP shares
identical causal prefixes for projections and MLPs; attention uses expanded
sequences. Vocabulary scoring is chunked after the forward pass to bound memory.

Only the server's `--max-batch-tokens` controls the budget. The total is
`sum(group_prompt_tokens + candidate_tokens)` over every candidate of every
group. Schema expansion is consumed incrementally; exceeding that budget or any
sequence's context limit rejects the entire request before backend submission.
Groups are never split across calls, truncated, pruned, or partially returned.

Responses contain `groups`, keyed by group name, plus total `expanded_tokens` and
`compact_tokens`. Each group contains its selected `decision` object, `index`, all
serialized `options`, `log_scores`, and `probabilities`. Scores include candidate
tokens and the assistant turn-ending token, exclude prompt tokens, and use
full-vocabulary normalization.
Probabilities are normalized **within each group**, not across groups, and are
not calibrated correctness confidence. No length normalization is applied.

The implementation supports causal Qwen3 and dense Gemma4 text models with an
LM head (or tied word embeddings) on Candle CUDA/FlashAttention. Gemma4 uses
variable-length FlashAttention on GPU; its bidirectional embedding path does
not fold prefixes, while causal classifier and decision paths can. Gemma4 MoE
checkpoints are not supported. For Qwen3-4B-Instruct-2507:

```sh
CUDA_VISIBLE_DEVICES=0 text-embeddings-router \
  --model-id Qwen/Qwen3-4B-Instruct-2507 --dtype bfloat16 --pooling last-token \
  --max-batch-tokens 16384 --auto-truncate --port 18091 --prometheus-port 19091
python3 integration_tests/decision_smoke.py --output /tmp/grouped-decisions.json
```

The smoke test compares grouped requests against independently submitted groups,
checks joint versus independent expansion, and checks server-budget rejection.
The CUDA unit test additionally compares variable prompt boundaries against
independent complete-sequence likelihoods, including EOS and tied/untied heads.
For a local dense Gemma4 checkpoint, set `GEMMA4_MODEL_ROOT` and run the ignored
`test_gemma4_decision` CUDA test to compare folded and unfolded candidate scores.

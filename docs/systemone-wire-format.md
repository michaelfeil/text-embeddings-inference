# SystemOne wire format

`POST /v1/systemone` uses the Jev request and answer envelope, with
backend-specific extensions. It is wire compatibility, not a promise that models
share prompts, calibration, limits, or supported modalities.

## Model selection

`model` is optional when exactly one model is loaded. Existing advisory model
names and aliases continue to select that model. With multiple loaded models,
`model` must name a served model; a missing or unknown name is a validation error.
There is no implicit first-model fallback in a multi-model deployment.
The response identifies the model used. This differs intentionally from the
[current Jev API](https://docs.typesafe.ai/api), which requires `model`.

## State and questions

`state` accepts text, a JSON object, or a JSON array. Ordinary structured data is
rendered as JSON text for the model, preserving nested values and insertion order.
Text strings keep their existing representation. This does not execute instructions
or fetch media embedded in ordinary JSON data.

For the common adapters, an object containing `messages` is reserved for Infinity's
native conversation extension. It must pass the existing message validation;
malformed conversations are not silently converted to text. To evaluate a record
whose own field is named `messages`, nest it under another key such as `record`,
or send its serialized JSON as a string. Native messages and media remain subject
to the selected adapter's capabilities. D3 retains its separate JSON-state and
`images`/`videos` contract.

`questions` is a map keyed by caller-supplied IDs. Types are `choice`, `score`,
and `noul`, subject to the loaded adapter's support. Laya now accepts `instructions`
as text, an object, or an array, using the same JSON rendering as structured state.
Other adapters retain their existing instruction and criteria validation.

```json
{
  "state": {"ticket": {"message": "My payout failed", "attempts": 3}},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": {"question": "Which team should handle this ticket?"},
      "criteria": {"billing": "Payments and refunds", "technical": "Software issues"}
    }
  }
}
```

Choice uses an option-to-description map; score uses an ordered rubric array;
noul optionally describes `false` and `true`. Existing adapter extensions, such as
Laya's choice arrays, noul labels and token budgets, remain supported. Limits are
model-specific and are not widened merely to match another provider's limits.

## Answers and extensions

Responses retain `model`, `answers` keyed by the original question IDs, and
`usage`. Existing answer fields, including `action` and `answer_confidence` where
provided, remain intact. Clients should tolerate additional response fields and
preserve them when forwarding responses.

Do not substitute `answer_confidence` for `confidence`: backend-specific scoring
and calibration still apply. This change does not alter probabilities, confidence
formulas, usage accounting, or response serialization. Unknown execution controls
in requests remain errors rather than being silently ignored.

# System One input design

Status: typed wire schema implemented; native message inference awaits a model adapter.

Keep `POST /v1/systemone` in the existing TEI HTTP server, with the existing
question types and batching infrastructure. The canonical input forms are:

## Plain text

```json
{
  "state": "The customer requests a refund.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Is a refund requested?"}
  }
}
```

## Native messages

```json
{
  "state": {
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "The delivery arrived like this."},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
      }
    ]
  },
  "questions": {
    "damage": {"type": "noul", "instructions": "Is the package damaged?"}
  }
}
```

This uses the explicit native-chat state envelope from
[openjev-sglang](https://github.com/ekzhang/openjev-sglang). Our contract accepts
only plain text or this envelope; bare arrays and arbitrary objects are rejected.
There is no second top-level messages field or separate media collection.

The shared Rust contract in `text_embeddings_core::input` is
`ModelInput::Text(String)` or `ModelInput::Messages(MessageInput)`.
`SystemOneInput` remains a compatibility alias. Message content is either a string or
ordered `ContentPart` values: Text, ImageUrl, InputAudio, VideoUrl. Initial roles
are system, developer, user, assistant. Tool messages require a future explicit
schema for call identifiers and metadata; they are currently rejected.

## Shared embedding API

The embedding endpoints use `EmbeddingInput`, sharing `Message`, `MessageRole`,
`MessageContent`, and `ContentPart` with the decision API. The outer envelope
remains endpoint-specific:

| Input | Meaning |
| --- | --- |
| `"hello"` | One text input |
| `["hello", "world"]` | Two independent text inputs |
| `[{"role":"user","content":"hello"},{"role":"assistant","content":"world"}]` | One conversation input |

Use `inputs` for `/embed`, `/embed_sparse`, and `/embed_all`; use `input` for
`/v1/embeddings`. `/embed` and `/v1/embeddings` return one embedding per conversation,
not one per turn. `/embed_all` retains its per-token output format. A batch of
conversations, mixed string/message arrays, arbitrary objects, and numeric token-ID
arrays are rejected. An empty array is an empty text batch and returns 400.
Removing numeric token-ID arrays is a breaking HTTP embedding API change, including
on `/v1/embeddings`; clients must send text. `/decode` retains its token-ID input.
The decision API continues to use `state: {"messages": [...]}`.

An embedding conversation can contain ordered text and image parts:

```json
{
  "inputs": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this picture."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }
  ]
}
```

**Current capability:** this establishes the request contract and preserves the
conversation until preprocessing. The current embedding text processor returns
422 for every conversation, including text-only conversations. The example above
requires a future model processor; this change does not enable image inference or
remote downloads. Plain strings and string batches keep their existing behavior.
System/developer roles and audio/video parts remain representable in the shared
schema, with actual support owned by each model processor.

Remote image resolution belongs before final tokenization and queue admission of
prepared inputs. A follow-on implementation must enforce configured HTTPS storage
host/bucket allowlists, destination IP checks, byte/pixel/time limits, and redaction
of signed URL queries. Decoding, template rendering, visual token expansion, and
positions belong to the model processor. No remote URL is fetched by deserialization
or by the current text processor.

## Semantics

- Parse messages into typed roles and ordered content blocks. Preserve their
  structure until the model adapter applies its native template and processor.
- Each question independently evaluates the same context. The adapter defines
  question placement according to the checkpoint's training/reference format.
- Text and image/audio/video blocks remain in their original order. Unsupported
  roles, content types, or input modes return validation errors; never silently
  stringify or discard native message content.
- Keep the existing question and answer contract. Model-specific action scores
  are optional; probability calibration belongs to the adapter.
- The HTTP contract uses familiar chat content blocks: text, image_url,
  input_audio, and (as an explicit extension) video_url. Actual support is
  capability-dependent, not implied by accepting the request schema.
- Resolve messages and media before queueing model-ready inputs. Reuse the TEI
  queue, admission, cancellation, and replicas. Carry media alignment and resource
  costs with prepared inputs; token-only accounting is insufficient for media.
- Decode shared media once per request where possible. Do not promise shared
  transformer computation across questions.

## Compatibility and implementation boundary

This deliberately narrows the earlier PR's arbitrary JSON state contract. Clients
sending records must serialize them explicitly as text. Native message envelopes
are recognized structurally and never fall back to JSON text. The Laya adapter
returns 422 for native messages because it has no native chat processor or
multimodal encoder. Existing plain-text tokenization and output remain unchanged.

Jev-Omni's reference uses one user turn per question. First validate exact
single-turn processing and scores, then evaluate multi-turn question placement.
An ordinary Qwen checkpoint additionally needs a defined decision scoring method.
Neither model support nor multimodal Candle execution follows from a schema change.

Sources:

- https://docs.typesafe.ai/api
- https://github.com/ekzhang/openjev-sglang
- https://huggingface.co/docs/transformers/main/en/chat_templating_multimodal
- https://huggingface.co/akhilaaa3/Jev-Omni/blob/main/jev_omni.py
- https://docs.vllm.ai/en/latest/features/multimodal_inputs/

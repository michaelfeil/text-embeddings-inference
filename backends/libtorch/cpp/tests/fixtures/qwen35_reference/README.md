This fixture contains 32 actual tokens from one DeltaNet value head in the
original Candle forward of `Qwen/Qwen3.5-35B-A3B`, revision
`59d61f3ce65a6d9863b86d2e96597125219dc754`. It retains the checkpoint's Apache-2.0
license in LICENSE. The original checkpoint is available at
https://huggingface.co/Qwen/Qwen3.5-35B-A3B/tree/59d61f3ce65a6d9863b86d2e96597125219dc754.

`learned_head.hex` is a small tensor-only extraction, not a modified model.
`provenance.json` records its binary layout, source head, exact checksum and
reference-output origin. It includes only the relevant convolution/norm
parameters and already-computed projection operands, followed by the original
Candle output. Python was used only to extract and serialize these tensors;
no Python model inference was performed.

The original no-FMA DeltaNet contract produces exactly the saved BF16 values.
Restoring default fused multiply-add changes a value at token 15, column 21,
which makes this an arithmetic regression discriminator. Packed duplicated
and poisoned subsequent sequences also verify state isolation without padding.

The checked-in representation is readable hexadecimal text, 32 original bytes
per line. The existing test decodes the byte pairs before reading the unchanged
tensor layout. Decoded bytes have SHA-256
`fbf5f2fb808ab3980d3175850a8e4389119ac5c427e83a6d009b7f8c9c2b36f4`; both representation and decoded-byte
checksums are recorded in `provenance.json`.

The combined runtime passes a bounded regression of Gemma4 Decision media on both GPU assignments, with and without CUDA graphs. This supplements the separate [200-iteration cache qualification](libtorch-gemma-media-cache.md); these measurements contain **40 iterations and 20 warmups** per case/backend.

| Mode | Physical Torch/Candle GPUs | Worst P50 difference | Report |
|---|---|---:|---|
| Eager | 6/7 | +4.6226% | [Six cases](benchmarks/tei-gemma4-media-cache-integrated-eager-40-6-7.json) |
| Eager | 7/6 | +4.0152% | [Six cases](benchmarks/tei-gemma4-media-cache-integrated-eager-40-7-6.json) |
| CUDA graphs | 6/7 | −4.9834% | [Six cases](benchmarks/tei-gemma4-media-cache-integrated-graphs-40-6-7.json) |
| CUDA graphs | 7/6 | −5.4311% | [Six cases](benchmarks/tei-gemma4-media-cache-integrated-graphs-40-7-6.json) |

All four runs pass every unchanged Decision accuracy and +5% P50 gate on the [original six-case fixture](benchmarks/tei-gemma4-trained-decision-media-fixture.json). Minimum logit cosine is 0.9999945054. Torch and Candle output arrays exactly match their corresponding earlier `dfbb` reports. Graph runs enable both graph flags with the 16,384-token cap and report counts `[1,2,3,4,4,4]`; eager counts remain zero.

These results use immutable native SHA **`afd1fcdc287255584504e18baed6fb32ea952c7bc15c8431302a92966dbeeea3`** and freshly rebuilt comparator SHA **`df4d5bba67c0978f7f7fc705c90eb5cfad812b3786df3468ddc7e97d9e2b3122`**. Each process’s `/proc` mapping verifies the pinned native path. The [compact regression/source manifest](benchmarks/tei-gemma4-integrated-cache-regression-manifest.json) includes matching source identities and the baseline’s 29/29 native and 10/10 Rust checks, including GPU tests.

Scope remains warm repeated images, BF16, official Gemma4 checkpoint revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, matching 120-byte media descriptors and no explicit token/head padding or preload. Cold execution, misses, eviction throughput and changing-image latency remain unqualified. The cache still limits owned feature storage to 64 MiB/32 entries; in-flight references and graph clones require additional storage. Historical failures and the earlier `dfbb` 200-iteration evidence remain unchanged.

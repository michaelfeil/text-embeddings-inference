# Third-party notices

This index identifies existing notices for bundled source components. The
referenced texts and file headers remain authoritative for their respective
components. This is not a complete audit of Cargo dependencies, Python packages,
CUDA libraries, model weights, or container base images.

| Component | Retained license and attribution locations |
| --- | --- |
| Hugging Face Text Embeddings Inference | [LICENSE](LICENSE), [NOTICE](NOTICE), and inherited file headers. |
| Candle CUDA activation kernel | [MIT text](backends/candle/src/kernels/LICENSE.candle-MIT) and [kernel notices](backends/candle/src/kernels/gated_activation.cu). |
| Candle cuBLASLt extension | [Apache text](backends/candle/extensions/candle-cublaslt/LICENSE-APACHE) and [MIT text](backends/candle/extensions/candle-cublaslt/LICENSE-MIT). |
| Candle FlashAttention v1 extension | [Apache text](backends/candle/extensions/candle-flash-attn-v1/LICENSE-APACHE), [MIT text](backends/candle/extensions/candle-flash-attn-v1/LICENSE-MIT), and kernel file notices. |
| Candle index-select extension | [Apache text](backends/candle/extensions/candle-index-select-cu/LICENSE-APACHE), [MIT text](backends/candle/extensions/candle-index-select-cu/LICENSE-MIT), and file headers. |
| Candle layer-norm extension | [Apache text](backends/candle/extensions/candle-layer-norm/LICENSE-APACHE), [MIT text](backends/candle/extensions/candle-layer-norm/LICENSE-MIT), and [BSD text](backends/candle/extensions/candle-layer-norm/LICENSE). |
| Candle rotary extension | [Apache text](backends/candle/extensions/candle-rotary/LICENSE-APACHE) and [MIT text](backends/candle/extensions/candle-rotary/LICENSE-MIT). |
| RadixMLP-derived layers | MIT attribution in [index-select](backends/candle/src/layers/index_select.rs) and [RadixMLP](backends/candle/src/layers/radix_mlp.rs) source headers. |
| CUTLASS-derived Gemma4 MoE kernels | [BSD-3-Clause text](backends/candle/extensions/candle-gemma4-moe/LICENSE.cutlass). |
| ALiBi implementation | [Source notices](backends/candle/src/alibi.rs) for Google AI Language Team, Hugging Face, and Jina AI. |

Basetenkenizer is a Cargo dependency rather than vendored source. Its published
package carries Apache-2.0 licensing and additional component notices; consult
its [license audit](https://github.com/basetenlabs/basetenkenizer/blob/main/THIRD_PARTY_LICENSES.md)
for the version used by the release. Other dependencies must likewise be audited
from their actual pinned versions when assembling distributed artifacts.

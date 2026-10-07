# Integration Tests

This directory contains integration tests for the project. This starts the TEI server and run an /embed request to it while checking the output is as expected.

## Running the tests for HPU

First you have to build the docker image.
```bash
platform="hpu"

docker build . -f Dockerfile-intel --build-arg PLATFORM=$platform -t tei_hpu
```

Then you can run the tests.
```bash
uv run pytest --durations=0 -sv .
```

## EmbeddingGemma 2 and Rune

Install PyTorch, Transformers with `embedding_gemma2` support, Pillow, NumPy,
and safetensors in a Python environment. FFmpeg must be available. Generate
independent BF16 references from the released checkpoint, then run the native
CUDA and CPU processor comparisons:

```bash
python integration_tests/embeddinggemma2_reference.py /path/to/embeddinggemma-2 /tmp/eg2-reference
export EMBEDDINGGEMMA2_MODEL_ROOT=/path/to/embeddinggemma-2
export EMBEDDINGGEMMA2_FIXTURE_DIR=/tmp/eg2-reference
cargo test -p text-embeddings-backend-candle --features flash-attn --test test_embedding_gemma2 -- --ignored --nocapture
cargo test -p text-embeddings-core --lib multimodal::embedding_gemma2 -- --ignored
python integration_tests/embeddinggemma2_http.py http://localhost:8080 /tmp/eg2-reference
```

The fixtures cover ragged text batches, a sequence crossing the local attention
window, single/multiple images, video with uniform sampling down to the 32-frame
cap, unequal-length batched audio, multiple
audio clips, the 30-second audio limit, and mixed inputs. HTTP checks also
cover raw output lengths, task prompts, dimensions, normalization, the OpenAI
endpoint, concurrent requests, the 8,192-token text boundary, invalid media,
and audio duration boundaries. Run the server with
`--backend-device-ids 0,1` to exercise replica routing.

Pooled cosine thresholds are 0.999 for text and 0.995 for media. Raw checks use
full-output cosine and relative L2 error: low-norm media tokens change direction
substantially between Transformers' own BF16 eager and SDPA implementations.
Generate controls with `--attention eager` and `--audio-convolution fp32` to
inspect that numerical variation. The latter preserves Transformers' causal
convolution padding and casts convolution outputs back to BF16. At 30 seconds,
this Transformers control produces 0.974 raw cosine and 0.227 relative L2 against
its BF16 convolution implementation, while pooled cosine remains 0.9993. Native
inference produces 0.978 raw cosine, 0.211 relative L2, and 0.9994 pooled cosine.
The maximum-length audio case therefore uses raw bounds of 0.97 cosine and 0.24
relative L2; shorter media use 0.99 and 0.15, and text uses 0.999 and 0.04.
Frontend tests separately check exact media/token placement and log-mel features.

Rune's original text/image decision outputs are protected by an exact BF16
logit regression captured from `mf/faster-tei` at `50ad482`:

```bash
python integration_tests/rune_regression_inputs.py /path/to/rune-26b-a4b /tmp/rune-inputs
export RUNE_CHECKPOINT_DIR=/path/to/rune-26b-a4b
export RUNE_IMAGE_FIXTURE_DIR=/tmp/rune-inputs
cargo test -p text-embeddings-backend-candle --features flash-attn --test test_rune_regression -- --ignored
```

The independent references were generated with PyTorch `2.14.1+cu130` and
Transformers `5.19.0.dev0` at `e598fbad926d80bc356c0c4d96532030e3dfdb62`.
On H100, the full EmbeddingGemma 2 backend and two-replica HTTP checks pass,
including PNG/JPEG/WebP, WAV/MP3, and MP4/WebM. Rune's original and updated text
and image logits are bit-for-bit equal, including each mode's original Radix
outputs. Two older ignored Rune checks fail with fresh fixtures: the vision-only
Transformers comparison misses its 0.995 cosine threshold, and Radix folded
logits differ slightly from unfolded logits. Both discrepancies also occur in
the unmodified branch implementation. Those tests and Rune's execution behavior
are unchanged; the new regression checks each mode against its original output.

Both local CUDA Docker image variants were built and checked on H100 with the
complete HTTP matrix, including the 32-frame video cap, 30-second audio limit,
and 8,192-token text boundary, plus Rune text/image requests. A fresh-cache startup using the public model ID also
passes the complete HTTP matrix (checkpoint revision
`914f7f89142e33e77833254d9c9b90c3cef7303b`). To build the two CUDA image variants and repeat the checks:

```bash
docker build -f Dockerfile-cuda --build-arg CUDA_COMPUTE_CAP=90 -t tei:embeddinggemma2-hopper .
docker build -f Dockerfile-cuda-all -t tei:embeddinggemma2-cuda-all .
docker run --rm --gpus all -p 8080:80 -v /path/to/embeddinggemma-2:/model:ro \
  tei:embeddinggemma2-hopper --model-id /model --device-id 0
# In another terminal:
python integration_tests/embeddinggemma2_http.py http://localhost:8080 /tmp/eg2-reference
```

The verified local images are `tei:embeddinggemma2-hopper` (`d9d4994cd1b7`) and
`tei:embeddinggemma2-cuda-all` (`6e47b39b7a40`). The CUDA-all entrypoint selects
its SM90 binary on H100; the same fixture command checks that image. Its SM75
and SM80 binaries also build successfully; their runtime execution was not
tested on the corresponding hardware. Both runtime images include FFmpeg and ffprobe. The released
upstream `hopper-latest` image (2026-10-06, version 1.9.4) fails to load this
checkpoint's new pooling configuration; the local images use the updated parser.

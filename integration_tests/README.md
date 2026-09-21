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

## Candle BF16 and upgrade regression checks

`candle_precision.py` uses pinned revisions of `Qwen/Qwen3-0.6B`,
`voyageai/voyage-4-nano`, and `BAAI/bge-reranker-base`. It checks eight embedding
inputs (including contexts above 500, 1,000, 2,000, 4,000, and 8,000 tokens),
normalized and unnormalized finiteness, output dimensions, and batch/single
agreement. Qwen uses last-token pooling; Voyage uses mean pooling and its trained
2,048-dimensional projection. The BGE probe compares 28 query/document logits,
including near-512-token pairs, and reports labeled top-1 accuracy on seven small
handwritten groups. These are regression fixtures, not a retrieval benchmark.

Build both images from separate clean worktrees, using the same compute capability:

```bash
# Run in a worktree at the base revision ac28ad03.
docker build -f Dockerfile-cuda --build-arg CUDA_COMPUTE_CAP=90 \
  --build-arg CARGO_BUILD_JOBS=8 --build-arg RAYON_NUM_THREADS=8 \
  -t tei-candle-before:pr10 .
# Run in the PR worktree.
docker build -f Dockerfile-cuda --build-arg CUDA_COMPUTE_CAP=90 \
  --build-arg CARGO_BUILD_JOBS=8 --build-arg RAYON_NUM_THREADS=8 \
  -t tei-candle-after:pr10 .
```

Download each model at the revision in `candle_precision.py` and use its local
snapshot directory for both implementations. To generate an independent reference,
use a CUDA PyTorch environment with `transformers==4.57.6`. The Voyage reference
loads the checkpoint's custom Python model (`trust_remote_code=True`); inspect
that pinned code before running it. Reference embeddings use BF16 and FP32 mean
pooling/normalization; BGE reference logits use FP32. The HTTP probe itself uses
only the Python standard library.

```bash
python integration_tests/candle_precision.py reference --model voyage \
  --path /absolute/model/snapshot --output voyage-reference.json

docker run --rm --gpus device=0 -p 18910:80 \
  -v /absolute/model/cache:/models:ro tei-candle-after:pr10 \
  --model-id /models/snapshot --dtype bfloat16 --max-batch-tokens 40960 \
  --max-client-batch-size 16

python integration_tests/candle_precision.py http --model voyage \
  --reference voyage-reference.json --output voyage-after-bf16.json
```

Use `--model qwen` and add `--pooling last-token` to its server command. Use
`--model bge` and no pooling override for the reranker. Preserve the Hugging Face
cache directory structure when mounting snapshots that contain symlinks.

Run the base image in `float16`, then the upgraded image in both `float16` and
`bfloat16`. Pass the base HTTP result as `--baseline before.json` to compare
before/after results. Embedding comparisons require cosine similarity at least
0.99; BGE comparisons require identical top-1 choices and at most 0.1 absolute
logit error against the FP32 reference (use `--logit-tolerance 0.25` for BF16).
The before/after FP16 BGE bound is 0.05. The script records all metrics and checks
in JSON and exits nonzero on failures, including null/non-finite embeddings.
If FP16 Voyage produces non-finite outputs, keep that result as a control; do not
relax finiteness checks. Finite results on these inputs do not establish that FP16
is safe for every input.

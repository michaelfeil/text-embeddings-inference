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


## Candle precision regression test

`candle_precision.py` compares a running server with pinned Transformers references
for Qwen3-0.6B, Voyage 4 Nano, and BGE reranker. HTTP mode uses only the Python
standard library; reference mode needs CUDA PyTorch and `transformers==4.57.6`.
Voyage's reference loads custom code from the pinned checkpoint.

```bash
python integration_tests/candle_precision.py reference --model voyage \
  --path /absolute/pinned/model/snapshot --output /tmp/voyage-reference.json
python integration_tests/candle_precision.py http --model voyage \
  --reference /tmp/voyage-reference.json --output /tmp/voyage-result.json
```

Serve Voyage with `--dtype bfloat16 --max-batch-tokens 40960`. Qwen also needs
`--pooling last-token`. Use `--model bge` for the reranker, and `--baseline` to
compare against a previous HTTP result. `--stress` adds embedding overflow cases,
including 534-token repeated code that fails in Voyage FP16. Reference and HTTP
runs must use the same stress setting. Failures produce a nonzero exit status.

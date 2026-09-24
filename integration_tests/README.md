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

## RadixMLP grouped decisions

See [the decision API](../DECISIONS.md) for the `context` / `questions` format and
Qwen3-4B-Instruct-2507 startup command. Run the standard-library HTTP smoke test:

```bash
python3 integration_tests/decision_smoke.py --output /tmp/grouped-decisions.json
```

It verifies 6 + 3 independent candidates versus 18 joint candidates, per-group
probability normalization, grouped versus standalone scores, and atomic rejection
when the expanded tokens exceed the server budget. Score comparisons allow absolute
log-score differences of 0.25 for BF16 and 0.1 otherwise because batch shapes can
change reduced-precision arithmetic; winners must still agree. Reports stay outside
the repo.

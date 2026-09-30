#!/usr/bin/env python3
"""Benchmark distinct questions over shared text; compare Radix on/off result files."""
import argparse
import copy
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://127.0.0.1:18094")
parser.add_argument("--fixture", required=True, help="Rune checkpoint_prompt_fixture output")
parser.add_argument("--output", required=True)
parser.add_argument("--server-description", required=True)
parser.add_argument("--samples", type=int, default=20)
args = parser.parse_args()
fixture = json.loads(Path(args.fixture).read_text())["request"]
templates = list(fixture["questions"].values())
results = []
for repeats in (1, 32, 128):
    for count in (1, 2, 8, 32):
        questions = {}
        for i in range(count):
            question = copy.deepcopy(templates[i % len(templates)])
            question["instructions"] += f" (Question {i + 1}.)"
            questions[f"q{i}"] = question
        body = json.dumps({"state": (fixture["state"] + "\n") * repeats, "questions": questions}).encode()
        samples = []
        for sample in range(args.samples + 3):
            request = urllib.request.Request(args.url + "/v1/systemone", data=body,
                                             headers={"Content-Type": "application/json"})
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.load(response)
            elapsed = (time.perf_counter() - started) * 1000
            assert len(result["answers"]) == count
            assert result["usage"]["output_tokens"] == count
            if sample >= 3:
                samples.append(elapsed)
        row = {"state_repeats": repeats, "questions": count,
               "input_tokens": result["usage"]["input_tokens"],
               "p50_ms": statistics.median(samples),
               "p95_ms": sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
               "answers": result["answers"], "samples_ms": samples}
        results.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("answers", "samples_ms")}), flush=True)
        Path(args.output).write_text(json.dumps({"server": args.server_description, "results": results}, indent=2) + "\n")

#!/usr/bin/env python3
"""Single-client HTTP latency sweep; run against an otherwise idle single-replica server."""

import argparse, copy, http.client, json, math, statistics, time, subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=18084)
p.add_argument("--output", required=True)
p.add_argument(
    "--server-description",
    required=True,
    help="Running build, checkpoint revision, hardware, dtype and server limits",
)
a = p.parse_args()
PORT = a.port
fixture = json.load(
    open(
        Path(__file__).resolve().parents[1]
        / "router/tests/fixtures/laya-systemone.json"
    )
)
question_templates = list(fixture["request"]["questions"].values())
state = "The customer was billed twice and requests a refund. " * 120


def request(method, path, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=180)
    payload = None if body is None else json.dumps(body)
    start = time.perf_counter()
    connection.request(method, path, payload, {"Content-Type": "application/json"})
    response = connection.getresponse()
    raw = response.read()
    elapsed = (time.perf_counter() - start) * 1000
    assert response.status == 200, (response.status, raw)
    headers = dict(response.getheaders())
    connection.close()
    return elapsed, raw, headers


def batches():
    _, raw, _ = request("GET", "/metrics")
    for line in raw.decode().splitlines():
        if line.startswith('te_replica_batches{replica="0"}'):
            return int(float(line.split()[-1]))
    raise RuntimeError("Replica 0 batch metric missing; use a single-replica server")


results = []
for tokens in (128, 512, 1024):
    for count in (1, 2, 4, 8, 16, 32):
        body = {
            "state": state,
            "max_len": tokens,
            "questions": {
                f"q{i}": copy.deepcopy(question_templates[i % 3]) for i in range(count)
            },
        }
        for _ in range(5):
            request("POST", "/v1/systemone", body)
        start_batches = batches()
        samples = []
        native = []
        queue = []
        for _ in range(50):
            elapsed, raw, headers = request("POST", "/v1/systemone", body)
            result = json.loads(raw)
            assert len(result["answers"]) == count
            assert result["usage"]["input_tokens"] == tokens * count, result["usage"]
            samples.append(elapsed)
            native.append(float(headers["x-inference-time"]))
            queue.append(float(headers["x-queue-time"]))
        measured_batches = batches() - start_batches
        ordered = sorted(samples)
        median = statistics.median(samples)
        row = {
            "tokens_per_question": tokens,
            "questions": count,
            "input_tokens": count * tokens,
            "samples": 50,
            "p50_ms": round(median, 2),
            "p95_ms": round(ordered[math.ceil(0.95 * len(ordered)) - 1], 2),
            "ms_per_question": round(median / count, 2),
            "questions_per_second": round(1000 * count / statistics.mean(samples), 1),
            "backend_batches_per_request": measured_batches / 50,
            "mean_inference_header_ms": round(statistics.mean(native), 2),
            "mean_queue_header_ms": round(statistics.mean(queue), 2),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        json.dump(
            {
                "server": a.server_description,
                "client_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "client_dirty": bool(
                    subprocess.check_output(
                        ["git", "status", "--porcelain"], text=True
                    ).strip()
                ),
                "concurrency": 1,
                "warmups_per_case": 5,
                "results": results,
            },
            open(a.output, "w"),
            indent=2,
        )

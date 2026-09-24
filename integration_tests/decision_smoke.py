#!/usr/bin/env python3
"""Exercise /decide against a running Qwen3-Instruct server (standard library only)."""

import argparse
import json
import math
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request(base_url, path, payload=None):
    encoded = None if payload is None else json.dumps(payload).encode()
    req = Request(base_url.rstrip("/") + path, data=encoded,
                  headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urlopen(req, timeout=300) as response:
            return response.status, json.load(response), time.perf_counter() - start
    except HTTPError as error:
        return error.code, json.load(error), time.perf_counter() - start


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run(base_url, tolerance):
    status, info, _ = request(base_url, "/info")
    require(status == 200, f"/info failed: {info}")
    if tolerance is None:
        tolerance = 0.25 if info["model_dtype"] == "bfloat16" else 0.1
    payload = {
        "context": 'Refund policy: approve within 30 days, reject after 30 days. '
                   'Customer message: "I request a refund after 45 days. This is urgent."',
        "questions": {
            "action": {"description": "Choose the action under the refund policy.",
                       "enum": ["approve", "reject", "escalate"]},
            "urgent": {"description": "Does the customer explicitly say this is urgent?", "type": "boolean"},
            "language": {"group": "language", "description": "Identify the language of the customer message.",
                         "enum": ["en", "de", "other"]},
        },
    }
    status, result, elapsed = request(base_url, "/decide", payload)
    require(status == 200, f"Grouped inference failed: {status}, {result}")
    expected = {"default": {"action": "reject", "urgent": True}, "language": {"language": "en"}}
    report = {"server_info": info, "score_absolute_tolerance": tolerance, "grouped_seconds": elapsed,
              "expanded_tokens": result["expanded_tokens"], "compact_tokens": result["compact_tokens"],
              "groups": {}, "guards": {}}
    require(set(result["groups"]) == set(expected), "Incorrect group names")
    require(result["compact_tokens"] < result["expanded_tokens"], "No prefix sharing")
    for name, count in [("default", 6), ("language", 3)]:
        group = result["groups"][name]
        require(len(group["options"]) == count, f"Incorrect candidate count for {name}")
        require(len(group["log_scores"]) == len(group["probabilities"]) == count, "Score dimensions mismatch")
        require(all(math.isfinite(s) for s in group["log_scores"]), "Nonfinite scores")
        require(abs(sum(group["probabilities"]) - 1) < 1e-5, "Group probabilities must sum to one")
        require(group["decision"] == expected[name], f"Wrong answer: {group}")
        require(json.loads(group["options"][group["index"]]) == group["decision"], "Decision/index mismatch")
        require(group["log_scores"][group["index"]] == max(group["log_scores"]), "Winner is not maximal")
        standalone = {"context": payload["context"], "questions": {
            key: value for key, value in payload["questions"].items() if value.get("group", "default") == name
        }}
        code, independent, _ = request(base_url, "/decide", standalone)
        require(code == 200, f"Standalone group failed: {independent}")
        reference = independent["groups"][name]
        require(group["options"] == reference["options"], "Group membership changed options")
        error = max(abs(a-b) for a, b in zip(group["log_scores"], reference["log_scores"]))
        require(error <= tolerance, f"Grouped vs standalone score error {error} exceeds {tolerance}")
        require(group["decision"] == reference["decision"], "Group membership changed winner")
        report["groups"][name] = {"decision": group["decision"], "options": count, "max_score_error": error}
    joint = json.loads(json.dumps(payload))
    del joint["questions"]["language"]["group"]
    code, all_joint, _ = request(base_url, "/decide", joint)
    require(code == 200, f"Joint inference failed: {all_joint}")
    require(list(all_joint["groups"]) == ["default"], "Missing group markers must form one group")
    require(len(all_joint["groups"]["default"]["options"]) == 18, "Expected full Cartesian product")
    # Each copy fits individually; all independent groups together exceed capacity.
    copies = info["max_batch_tokens"] // result["expanded_tokens"] + 1
    oversized = {"context": payload["context"], "questions": {}}
    for i in range(copies * 2):
        for key, schema in payload["questions"].items():
            oversized["questions"][f"{key}_{i}"] = dict(schema, group=f"{schema.get('group', 'default')}_{i}")
    code, rejected, _ = request(base_url, "/decide", oversized)
    require(code == 413, f"Combined groups must respect server budget: {code}, {rejected}")
    report["guards"]["all_groups_over_budget"] = code
    for bad in [
        {"context": "test", "questions": {}},
        {"context": "test", "questions": {"x": {"type": "string"}}},
        {"context": "test", "questions": {"x": {"enum": [True], "group": 1}}},
    ]:
        code, rejected, _ = request(base_url, "/decide", bad)
        require(code == 413, f"Invalid question schema must fail: {code}, {rejected}")
    report["passed"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18091")
    parser.add_argument("--score-tolerance", type=float,
                        help="Absolute sequence-log-score tolerance; defaults to 0.25 for BF16, 0.1 otherwise")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.base_url, args.score_tolerance)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"All decision checks passed; report: {args.output}")


if __name__ == "__main__":
    main()

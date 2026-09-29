#!/usr/bin/env python3
"""Compare a running Candle SystemOne endpoint with pinned upstream Laya.

Run in a Python environment containing upstream laya and torch. No GPU or hosted
API is needed for the reference. See docs/laya-verification.md for commands.
This is implementation parity, not a labelled accuracy or latency benchmark.
"""

import argparse
import hashlib
import json
import math
import subprocess
import urllib.request
from pathlib import Path


def cases():
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "router/tests/fixtures/laya-systemone.json"
        ).read_text()
    )
    yield "mixed_types", fixture["request"]
    yield "empty_questions", {"state": "", "questions": {}}
    yield (
        "unicode_mask",
        {
            "state": "Café: payé deux fois [MASK]. 请退款。",
            "questions": {
                "refund": {
                    "type": "noul",
                    "instructions": "Is a refund [MASK] requested?",
                },
                "team": {
                    "type": "choice",
                    "instructions": "Select the team.",
                    "criteria": {
                        "billing": {"purpose": "paiements", "flags": [False, 0]},
                        "support": None,
                        "sales": "",
                    },
                },
            },
        },
    )
    yield (
        "custom_noul",
        {
            "state": "The service is healthy.",
            "questions": {
                "healthy": {
                    "type": "noul",
                    "instructions": "Is the service healthy?",
                    "labels": {"false": " N ", "true": " Y "},
                    "criteria": {"false": {"status": "down"}, "true": {"status": "up"}},
                }
            },
        },
    )
    yield (
        "score_levels",
        {
            "state": "The outage blocks all payments.",
            "questions": {
                "severity": {
                    "type": "score",
                    "instructions": "How severe is the outage?",
                    "criteria": ["none", "minor", "moderate", "major", "critical"],
                }
            },
        },
    )
    for count in (2, 5, 8, 12):
        yield (
            f"choice_{count}",
            {
                "state": "Route this request about topic 3.",
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Which topic matches?",
                        "criteria": {
                            f"topic_{i}": f"Requests about topic {i}"
                            for i in range(count)
                        },
                    }
                },
            },
        )
    for length in (128, 512, 1024):
        yield (
            f"truncated_{length}",
            {
                "state": "The customer requests a refund. " * 200,
                "max_len": length,
                "head_max_len": 64,
                "questions": {
                    "refund": {
                        "type": "noul",
                        "instructions": "Does the customer request a refund?",
                    }
                },
            },
        )
    yield (
        "reordered_options",
        {
            "state": "I need a refund.",
            "questions": {
                "a": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": "refunds", "tech": "bugs"},
                },
                "b": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"tech": "bugs", "billing": "refunds"},
                },
            },
        },
    )


def compare(actual, expected, tolerance, path=""):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), (path, actual.keys(), expected.keys())
        return max(
            (
                compare(actual[k], v, tolerance, f"{path}.{k}")
                for k, v in expected.items()
            ),
            default=0.0,
        )
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        assert math.isfinite(actual), (path, actual)
        delta = abs(actual - expected)
        assert delta <= tolerance, (path, actual, expected, delta, tolerance)
        return delta
    assert actual == expected, (path, actual, expected)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18085")
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-description", required=True)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument(
        "--reference-commit", default="9d955671415fc19f069b9cc998928075c1f255ec"
    )
    args = parser.parse_args()
    import torch
    import transformers
    import laya
    from laya.agent import Agent

    reference_root = Path(laya.__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "-C", str(reference_root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == args.reference_commit, (commit, args.reference_commit)
    # Restrict to model source: benchmark/result files can legitimately be local.
    source_changes = subprocess.check_output(
        ["git", "-C", str(reference_root), "status", "--porcelain", "--", "laya"],
        text=True,
    )
    assert not source_changes.strip(), source_changes
    torch.set_num_threads(8)
    agent = Agent(args.checkpoint, device="cpu")
    report = {
        "server": args.server_description,
        "reference_commit": commit,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "reference_device": "cpu",
        "reference_dtype": str(agent.dtype),
        "checkpoint_config_sha256": hashlib.sha256(
            (Path(args.checkpoint) / "rl_agent_config.json").read_bytes()
        ).hexdigest(),
        "tolerance": args.tolerance,
        "cases": [],
    }
    for name, request in cases():
        expected = agent.system_one(
            request["state"],
            request["questions"],
            **{k: request[k] for k in ("max_len", "head_max_len") if k in request},
        )
        req = urllib.request.Request(
            args.url.rstrip("/") + "/v1/systemone",
            data=json.dumps(request, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as response:
            actual = json.load(response)
        assert actual["usage"] == expected["usage"], (
            name,
            actual["usage"],
            expected["usage"],
        )
        delta = compare(actual["answers"], expected["answers"], args.tolerance, name)
        decisions = {}
        for key, answer in expected["answers"].items():
            other = actual["answers"][key]
            if answer["type"] == "choice":
                decisions[key] = answer[
                    "choice"
                ]  # compare() already requires exact choice.
            elif answer["type"] == "noul":
                assert (answer["noul"] >= 0.5) == (other["noul"] >= 0.5), (
                    name,
                    key,
                    answer,
                    other,
                )
                decisions[key] = answer["noul"] >= 0.5
            else:
                target = max(answer["probabilities"], key=answer["probabilities"].get)
                assert target == max(
                    other["probabilities"], key=other["probabilities"].get
                ), (name, key)
                decisions[key] = target
        row = {
            "name": name,
            "max_numeric_delta": delta,
            "decisions": decisions,
            "usage": actual["usage"],
            "request": request,
            "reference_answers": expected["answers"],
            "candle_answers": actual["answers"],
        }
        report["cases"].append(row)
        Path(args.output).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"{name}: PASS; max numeric delta={delta:.6f}", flush=True)
    print(f"PASS: {len(report['cases'])} cases", flush=True)


if __name__ == "__main__":
    main()

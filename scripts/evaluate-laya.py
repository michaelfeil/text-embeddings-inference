#!/usr/bin/env python3
"""Evaluate every labelled typed-decisions test case through the HTTP API.

Requires `datasets`. States are explicitly serialized as text, exactly as in
upstream Laya's serialize_state; this does not imply native JSON-state support.
Metrics follow the published typed-decisions argmax/threshold convention.
"""

import argparse
import json
import statistics
import urllib.request
from pathlib import Path


def metrics(records):
    correct, confidence, errors = [], [], []
    for record in records:
        for key, question in record["questions"].items():
            answer, gold = record["answers"][key], record["gold"][key]
            kind = question["type"]
            if kind == "noul":
                p = answer["noul"]
                predicted = "true" if p >= 0.5 else "false"
                target = str(gold["label"]).lower()
                confidence.append(max(p, 1 - p))
            else:
                labels = (
                    list(question["criteria"])
                    if kind == "choice"
                    else list(map(str, range(len(question["criteria"]))))
                )
                probabilities = [answer["probabilities"][label] for label in labels]
                predicted = labels[
                    max(range(len(labels)), key=probabilities.__getitem__)
                ]
                target = str(gold["label"])
                confidence.append(max(probabilities))
                if kind == "score":
                    errors.append(
                        abs(
                            sum(i * p for i, p in enumerate(probabilities))
                            - gold["score"]
                        )
                    )
            correct.append(float(predicted == target))
    bins = [[] for _ in range(10)]
    for c, k in zip(confidence, correct):
        bins[min(9, int(c * 10))].append((c, k))
    ece = sum(
        len(b) / len(correct) * abs(statistics.mean(c - k for c, k in b))
        for b in bins
        if b
    )
    return {
        "decisions": len(correct),
        "accuracy": statistics.mean(correct),
        "ece_10_bins": ece,
        "score_mae": statistics.mean(errors),
        "mean_max_probability": statistics.mean(confidence),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18086")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dataset-revision", default="f2491dda413a9d94afcb30464123b429c857e079"
    )
    parser.add_argument(
        "--server-description",
        required=True,
        help="Checkpoint revision, build commit and dtype of the running server",
    )
    args = parser.parse_args()
    from datasets import load_dataset

    dataset = load_dataset(
        "LocalLLaMA/typed-decisions",
        "all",
        split="test",
        revision=args.dataset_revision,
    )
    records = []
    for row in dataset:
        questions = json.loads(row["questions"])
        state = json.dumps(json.loads(row["state"]), ensure_ascii=False)
        request = urllib.request.Request(
            args.url.rstrip("/") + "/v1/systemone",
            data=json.dumps({"state": state, "questions": questions}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
        assert result["answers"].keys() == questions.keys(), row["id"]
        records.append(
            {
                "id": row["id"],
                "workflow": row["workflow"],
                "questions": questions,
                "gold": json.loads(row["gold"]),
                "answers": result["answers"],
            }
        )
        if len(records) % 25 == 0:
            print(f"{len(records)}/{len(dataset)} cases", flush=True)
    report = {
        "dataset": "LocalLLaMA/typed-decisions",
        "config": "all",
        "split": "test",
        "dataset_revision": args.dataset_revision,
        "dataset_fingerprint": dataset._fingerprint,
        "server": args.server_description,
        "cases": len(records),
        "errors": 0,
        "metrics": metrics(records),
        "records": records,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["metrics"]), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pinned-model embedding/reranker regression probe (stdlib for HTTP mode)."""
import argparse
import json
import math
from pathlib import Path
import urllib.request

MODELS = {
    "qwen": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "voyage": ("voyageai/voyage-4-nano", "67fabc9bef010dabc5f6024aa1b1b6b93410426f"),
    "bge": ("BAAI/bge-reranker-base", "2cfc18c9415c912f9d8155881c133215df768a70"),
}
GROUPS = [
    ("What is the capital of France?", ["Paris is the capital of France.", "Berlin is the capital of Germany.", "Whales are marine mammals.", "A database index speeds up queries."]),
    ("How do plants convert sunlight into energy?", ["Photosynthesis converts sunlight, water and carbon dioxide into sugars.", "Plants can grow in pots or in garden soil.", "Solar panels generate electricity using photovoltaic cells.", "Mountains form when tectonic plates collide."]),
    ("Why use an index in a database?", ["Database indexes speed up searches by avoiding full table scans.", "Databases store information in tables.", "The index at the end of a book lists topics.", "Regular exercise improves cardiovascular health."]),
    ("How can bread dough rise?", ["Yeast produces carbon dioxide during fermentation, causing dough to rise.", "Bread is often made from flour and water.", "Baking powder can be stored in a dry cupboard.", "The Moon orbits Earth approximately once a month."]),
    ("What keeps an astronaut in orbit?", ["Gravity provides the centripetal acceleration that keeps an astronaut in orbit.", "Astronauts wear spacesuits to survive outside their spacecraft.", "Satellites can transmit television signals.", "Rice grows well in flooded fields."]),
    ("How can I prevent a Python KeyError?", ["Use dict.get or check whether a key exists before accessing a Python dictionary.", "Python dictionaries map keys to values.", "A SyntaxError means the Python source cannot be parsed.", "A balanced diet includes vegetables and grains."]),
]
PASSAGE = "The research team measured water quality in the river. Samples collected upstream contained less sediment than samples near the bridge. Temperature, rainfall, and seasonal changes were recorded alongside each observation. The report recommends repeated measurements before drawing conclusions. "


def inputs(model, stress=False):
    prefix = "Represent the document for retrieval: " if model == "voyage" else ""
    texts = [prefix + x for x in [
        "Paris is the capital of France.",
        "A database index speeds up searches through a large table.",
        "Yeast makes bread dough rise during fermentation.",
        *[PASSAGE * n for n in (12, 24, 48, 96, 192)],
    ]]
    if stress:
        texts += [prefix + text for text in [
            PASSAGE * 384,
            PASSAGE * 720,
            *["def index_records(rows):\n    return {row['id']: row for row in rows if row['active']}\n" * n for n in (24, 48, 96)],
            "研究团队测量了河流水质，并比较了不同季节的温度和降雨量。 " * 96,
            "Les chercheurs analysent la qualité de l’eau. Die Ergebnisse werden sorgfältig verglichen. " * 96,
            " ".join(str(i) for i in range(2000)),
            "{} [] () => :: ; , . ! ? " * 192,
        ]]
    return texts


def pairs():
    groups = [(q, list(ds)) for q, ds in GROUPS]
    # Longer, still untruncated pairs exercise padding and BERT position handling.
    groups.append((GROUPS[0][0], [d + " " + PASSAGE * 8 for d in GROUPS[0][1]]))
    return groups


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def request(url, route, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + route, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def reference(args):
    import torch
    import transformers
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.path)
    dtype = torch.bfloat16 if args.model != "bge" else torch.float32
    cls = AutoModelForSequenceClassification if args.model == "bge" else AutoModel
    model = cls.from_pretrained(args.path, dtype=dtype, trust_remote_code=args.model == "voyage", attn_implementation="sdpa").cuda().eval()
    result = {"torch": torch.__version__, "transformers": transformers.__version__, "dtype": str(dtype), "token_counts": [], "values": []}
    with torch.inference_mode():
        if args.model == "bge":
            for q, docs in pairs():
                values = []
                for doc in docs:
                    encoded = tokenizer(q, doc, return_tensors="pt", truncation=False).to("cuda")
                    result["token_counts"].append(encoded.input_ids.shape[1])
                    assert encoded.input_ids.shape[1] <= 512
                    values.append(model(**encoded).logits.float().item())
                result["values"].append(values)
        else:
            for text in inputs(args.model, args.stress):
                encoded = tokenizer(text, return_tensors="pt", truncation=False).to("cuda")
                result["token_counts"].append(encoded.input_ids.shape[1])
                hidden = model(**encoded).last_hidden_state.float()
                pooled = hidden.mean(dim=1) if args.model == "voyage" else hidden[:, -1]
                result["values"].append(torch.nn.functional.normalize(pooled, dim=-1)[0].cpu().tolist())
    return result


def http(args):
    result = {"info": request(args.url, "/info"), "token_counts": [], "values": [], "single_values": []}
    if args.model == "bge":
        for query, docs in pairs():
            rows = request(args.url, "/rerank", {"query": query, "texts": docs, "truncate": False, "raw_scores": True})
            result["values"].append([row["score"] for row in sorted(rows, key=lambda r: r["index"])])
            result["single_values"].append([request(args.url, "/rerank", {"query": query, "texts": [doc], "truncate": False, "raw_scores": True})[0]["score"] for doc in docs])
    else:
        texts = inputs(args.model, args.stress)
        result["token_counts"] = [len(request(args.url, "/tokenize", {"inputs": text})[0]) for text in texts]
        result["input_previews"] = [text[:120] for text in texts]
        for start in range(0, len(texts), 16):
            result["values"].extend(request(args.url, "/embed", {"inputs": texts[start:start + 16], "normalize": True, "truncate": False}))
        result["single_values"] = [request(args.url, "/embed", {"inputs": text, "normalize": True, "truncate": False})[0] for text in texts]
        result["unnormalized"] = [request(args.url, "/embed", {"inputs": text, "normalize": False, "truncate": False})[0] for text in texts]
    return result


def validate(args, result):
    checks = {}
    values = result["values"]
    result["finite_by_input"] = [all(isinstance(v, (float, int)) and math.isfinite(v) for v in row) for row in values]
    checks["finite"] = all(result["finite_by_input"])
    if "unnormalized" in result:
        result["unnormalized_finite_by_input"] = [all(isinstance(v, (float, int)) and math.isfinite(v) for v in row) for row in result["unnormalized"]]
    if not checks["finite"]:
        return checks
    if args.model == "bge":
        result["top1_correct"] = sum(max(range(len(row)), key=row.__getitem__) == 0 for row in values)
        result["top1_total"] = len(pairs())
    else:
        checks["dimensions"] = all(len(row) == (2048 if args.model == "voyage" else 1024) for row in values)
        result["max_norm_error"] = max(abs(math.sqrt(sum(v * v for v in row)) - 1) for row in values)
        checks["unit_norm"] = result["max_norm_error"] < 1e-4
        checks["long_context"] = max(result["token_counts"]) > 8000
        if "unnormalized" in result:
            checks["unnormalized_finite"] = all(isinstance(v, (int, float)) and math.isfinite(v) for row in result["unnormalized"] for v in row)
    comparisons = {}
    targets = {}
    if "single_values" in result:
        targets["batch_vs_single"] = result["single_values"]
    if args.reference:
        ref = json.loads(Path(args.reference).read_text())
        targets["reference"] = ref["values"]
        if args.model != "bge":
            checks["same_tokens"] = result["token_counts"] == ref["token_counts"]
    if args.baseline:
        targets["before_upgrade"] = json.loads(Path(args.baseline).read_text())["values"]
    for name, expected in targets.items():
        shape_ok = len(values) == len(expected) and all(len(a) == len(b) for a, b in zip(values, expected))
        checks[name + "_shape"] = shape_ok
        if not shape_ok:
            continue
        error = max(abs(a - b) for row, refrow in zip(values, expected) for a, b in zip(row, refrow))
        comparison = {"max_absolute_error": error}
        if args.model == "bge":
            tolerance = args.logit_tolerance if name != "before_upgrade" else 0.05
            comparison["tolerance"] = tolerance
            comparison["top1_agreement"] = all(max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__) for a, b in zip(values, expected))
            checks[name] = error <= tolerance and comparison["top1_agreement"]
        else:
            comparison["min_cosine"] = min(cosine(a, b) for a, b in zip(values, expected))
            checks[name] = comparison["min_cosine"] >= 0.99
        comparisons[name] = comparison
    result["comparisons"] = comparisons
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["reference", "http"])
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--path")
    parser.add_argument("--stress", action="store_true", help="Add long, code, multilingual, number, and punctuation embedding inputs")
    parser.add_argument("--url", default="http://localhost:18910")
    parser.add_argument("--reference")
    parser.add_argument("--baseline")
    parser.add_argument("--logit-tolerance", type=float, default=0.1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.stress and args.model == "bge":
        parser.error("--stress applies to embedding models")
    result = {"stress": args.stress, "model": MODELS[args.model][0], "revision": MODELS[args.model][1], "mode": args.mode}
    try:
        result.update(reference(args) if args.mode == "reference" else http(args))
        result["checks"] = validate(args, result)
        result["passed"] = all(result["checks"].values())
    except Exception as exc:
        result.update(passed=False, error=f"{type(exc).__name__}: {exc}")
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"values", "single_values", "unnormalized"}}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

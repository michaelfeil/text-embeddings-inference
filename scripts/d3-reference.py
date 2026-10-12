#!/usr/bin/env python3
"""Generate native D3 text/image/video fixtures for the Rust checkpoint tests.

Requires the downloaded vllm-sr/d3 repository and its published Python dependencies
(including accelerate and opencv-python-headless). Large pixel arrays stay in the
output directory, outside the source tree.
"""
import argparse
import base64
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(checkpoint))
    import torch
    from d3_runtime import D3

    route = {"type": "choice", "instructions": "Which team should handle this request?",
             "criteria": {"returns": "Refunds, replacements and damaged deliveries",
                          "billing": "Payments, invoices and charges", "technical": "Product setup and faults"}}
    direction = {"type": "choice", "instructions": "Which way does the square move?",
                 "criteria": {"right": "From left to right", "left": "From right to left", "still": "It does not move"}}
    receipt = {"type": "noul", "instructions": "Does the receipt list a blender?"}
    image = str(checkpoint / "assets/example-receipt.png")
    video = str(checkpoint / "assets/example-video.mp4")
    cases = {
        "text": {"state": "The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
                 "questions": {"route": route, "receipt": {"type": "noul", "instructions": "Does the customer have a receipt?"},
                               "urgency": {"type": "score", "instructions": "How urgent is this request?", "criteria": ["Routine", "Soon", "Today"]}}},
        "image": {"state": "The customer says the blender arrived cracked and attached the receipt.", "images": [image],
                  "questions": {"receipt": receipt, "payment": {"type": "choice", "instructions": "How was the order paid?",
                                                              "criteria": {"card": None, "cash": None, "gift card": None}}}},
        "video": {"state": "A short clip from a test camera.", "videos": [video],
                  "questions": {"direction": direction, "color": {"type": "noul", "instructions": "Does the square change color?"}}},
        "mixed": {"state": "The attached receipt and a short clip from a test camera.", "images": [image], "videos": [video],
                  "questions": {"receipt": receipt, "direction": direction}},
    }
    model = D3(checkpoint, device=args.device, batch_size=1)
    for name, request in cases.items():
        prepared = model.prepare(**request)
        metadata = {}
        encoded = None
        if prepared.videos:
            image_texts = [model.processor.replace_image_token(prepared.media, image_idx=i) for i in range(len(prepared.images))]
            expanded = [model.processor.get_text_with_replacements([text], image_texts, prepared.video_texts)[0][0]
                        for text in prepared.texts.values()]
            encoded = model.processor.tokenizer([expanded[0]], add_special_tokens=False, return_tensors="pt")
            pixels = prepared.media["pixel_values_videos"]
            if prepared.images:
                pixels = torch.cat([prepared.media["pixel_values"], pixels])
            image_grid = prepared.media.get("image_grid_thw")
            video_grid = prepared.media["video_grid_thw"]
        elif prepared.images:
            encoded = model.processor(text=[next(iter(prepared.texts.values()))], images=prepared.images,
                                      return_tensors="pt", add_special_tokens=False)
            pixels = encoded["pixel_values"]
            image_grid, video_grid = encoded["image_grid_thw"], None
        if encoded is not None:
            pixels.float().cpu().numpy().tofile(output / f"{name}-pixels.f32")
            types = torch.as_tensor(model.processor.create_mm_token_type_ids(encoded["input_ids"].tolist()))
            positions, _ = model.backbone.get_rope_index(encoded["input_ids"], image_grid_thw=image_grid,
                                                        video_grid_thw=video_grid, attention_mask=encoded["attention_mask"],
                                                        mm_token_type_ids=types)
            metadata = {"input_ids": encoded["input_ids"][0].tolist(), "position_ids": positions[:, 0].tolist()}
        (output / f"{name}-preprocessing.json").write_text(json.dumps(metadata))
        result = model.respond(prepared, *model.run(prepared))
        wire = dict(request)
        for field, mime in [("images", "image/png"), ("videos", "video/mp4")]:
            if field in wire:
                wire[field] = [f"data:{mime};base64," + base64.b64encode(Path(p).read_bytes()).decode() for p in wire[field]]
        (output / f"{name}-request.json").write_text(json.dumps(wire))
        (output / f"{name}-response.json").write_text(json.dumps(result, indent=2))
        print(name, json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

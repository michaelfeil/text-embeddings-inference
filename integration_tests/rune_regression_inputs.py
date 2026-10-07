"""Generate processor inputs for the unchanged Rune decision regression.

python integration_tests/rune_regression_inputs.py /path/to/rune-26b-a4b /tmp/rune-inputs
RUNE_CHECKPOINT_DIR=... RUNE_IMAGE_FIXTURE_DIR=... cargo test \
  -p text-embeddings-backend-candle --features flash-attn --test test_rune_regression -- --ignored

Requires the Rune-compatible Transformers version and safetensors. Only the
processor is loaded; the Rust test runs the full checkpoint and compares exact
BF16 text/image decision logits captured from branch head 50ad482.
"""
import argparse
import json
import pathlib

from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model", type=pathlib.Path)
parser.add_argument("output", type=pathlib.Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
processor = AutoProcessor.from_pretrained(args.model)
inputs = processor(
    images=[Image.new("RGB", (64, 48), (255, 0, 0))],
    text=["<|image|>"],
    return_tensors="pt",
)
save_file(
    {key: inputs[key].contiguous() for key in ["pixel_values", "image_position_ids"]},
    str(args.output / "red.inputs.safetensors"),
)
tokenizer = processor.tokenizer
options = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in ["A", "B"]]
rows = []
for question in ["Is Paris in France?", "Is the moon made of cheese?"]:
    prompt = ("<bos><|turn>system\nChoose A for false or B for true. Reply with one letter.<turn|>\n"
              f"<|turn>user\n{question}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
    rows.append({"ids": tokenizer.encode(prompt, add_special_tokens=False), "option_token_ids": options})
(args.output / "token_fixture.json").write_text(json.dumps({"sequences": rows}))

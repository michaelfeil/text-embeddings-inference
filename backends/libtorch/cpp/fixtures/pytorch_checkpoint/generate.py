# SPDX-License-Identifier: Apache-2.0
"""Generate synthetic torch.save fixtures; Python is not needed by tests or inference."""
from collections import OrderedDict
from pathlib import Path
import struct
import torch


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def tensor(values, shape):
    data = bytearray(b"".join(struct.pack("<f", v) for v in values))
    result = torch.frombuffer(data, dtype=torch.float32).reshape(shape).clone()
    # Retain independent serialized storage layouts while preserving logical values.
    if len(shape) == 2:
        return result.t().contiguous().t()
    storage = torch.cat([torch.zeros(3), result.flatten(), torch.ones(2)])
    return storage[3:-2].reshape(shape)


root = Path(__file__).parent
weights = OrderedDict()


def add(name, shape):
    count = 1
    for size in shape:
        count *= size
    salt = len(weights)
    values = [f32((((i * 17 + salt * 13) % 101) - 50) / 200) for i in range(count)]
    if name.endswith("LayerNorm.weight"):
        values = [f32(1 + v) for v in values]
    weights[name] = tensor(values, shape)


add("embeddings.word_embeddings.weight", [16, 16])
add("embeddings.position_embeddings.weight", [16, 16])
add("embeddings.token_type_embeddings.weight", [2, 16])
add("embeddings.LayerNorm.weight", [16])
add("embeddings.LayerNorm.bias", [16])
for layer in range(2):
    prefix = f"encoder.layer.{layer}."
    for name in ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"]:
        add(prefix + name + ".weight", [16, 16])
        add(prefix + name + ".bias", [16])
    for name in ["attention.output.LayerNorm", "output.LayerNorm"]:
        add(prefix + name + ".weight", [16])
        add(prefix + name + ".bias", [16])
    add(prefix + "intermediate.dense.weight", [24, 16])
    add(prefix + "intermediate.dense.bias", [24])
    add(prefix + "output.dense.weight", [16, 24])
    add(prefix + "output.dense.bias", [16])
for name, shape in [
    ("classifier.weight", [3, 16]), ("classifier.bias", [3]),
    ("cls.predictions.transform.dense.weight", [16, 16]),
    ("cls.predictions.transform.dense.bias", [16]),
    ("cls.predictions.transform.LayerNorm.weight", [16]),
    ("cls.predictions.transform.LayerNorm.bias", [16]),
    ("cls.predictions.decoder.weight", [16, 16]), ("cls.predictions.bias", [16]),
]:
    add(name, shape)
weights["embeddings.position_ids"] = torch.arange(16).reshape(1, 16)
weights._metadata = OrderedDict([("", {"version": 1})])
torch.save(weights, root / "bert.bin")
for index, (inputs, outputs) in enumerate([(16, 7), (7, 3)]):
    state = OrderedDict()
    state["linear.weight"] = tensor([f32((i % 11 - 5) * f32(.04)) for i in range(inputs * outputs)], [outputs, inputs])
    if index == 0:
        state["linear.bias"] = tensor([f32(i * f32(.01)) for i in range(outputs)], [outputs])
    state._metadata = OrderedDict([("", {"version": 1})])
    torch.save(state, root / f"dense{index}.bin")

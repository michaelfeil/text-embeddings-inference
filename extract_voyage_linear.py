#!/usr/bin/env python3
"""
Extract linear output projection layer from Voyage AI models
and create dense modules in sentence-transformers format.

Usage:
    python extract_voyage_linear.py --model-id voyageai/voyage-4-nemo --output-dir ./voyage_dense
"""

import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import safetensors.torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoConfig, AutoTokenizer


def extract_linear_layer(
    model_id: str, output_dir: Path, device: str = "cpu"
) -> Tuple[Path, Path]:
    """
    Extract linear output projection layer from Voyage model.

    Args:
        model_id: HuggingFace model ID
        output_dir: Output directory for dense module
        device: Device to load model on

    Returns:
        Tuple of (config_path, safetensors_path)
    """
    print(f"📥 Downloading model: {model_id}")

    # Download model files
    model_dir = snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.safetensors", "config.json", "tokenizer.json"],
    )

    print(f"📂 Model downloaded to: {model_dir}")

    # Load config to get model details
    config_path = Path(model_dir) / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Find safetensors file
    safetensors_files = list(Path(model_dir).glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError("No safetensors file found in model directory")

    safetensors_file = safetensors_files[0]
    print(f"🔧 Loading safetensors from: {safetensors_file}")

    # Load all tensors
    tensors = {}
    with safetensors.safe_open(safetensors_file, framework="pt", device=device) as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    print(f"📊 Found {len(tensors)} tensors")

    # Find linear output projection layer
    linear_keys = [
        k for k in tensors.keys() if "linear" in k.lower() and "weight" in k.lower()
    ]

    if not linear_keys:
        # Try common Voyage naming patterns
        voyage_patterns = [
            "linear.weight",
            "output_projection.weight",
            "dense.weight",
            "projection.weight",
            "head.weight",
        ]

        for pattern in voyage_patterns:
            if pattern in tensors:
                linear_keys = [pattern]
                break

    if not linear_keys:
        print("❌ No linear output projection layer found!")
        print("Available tensors:")
        for key in sorted(tensors.keys()):
            shape = tensors[key].shape
            print(f"  {key}: {shape}")
        raise ValueError("Linear output projection layer not found")

    linear_key = linear_keys[0]
    print(f"✅ Found linear layer: {linear_key}")
    print(f"   Shape: {tensors[linear_key].shape}")
    print(f"   Dtype: {tensors[linear_key].dtype}")

    # Extract linear layer tensors
    linear_weight = tensors[linear_key]
    in_features, out_features = linear_weight.shape

    # Look for bias
    bias_key = linear_key.replace("weight", "bias")
    linear_bias = tensors.get(bias_key, None)

    if linear_bias is not None:
        print(f"✅ Found bias: {bias_key}")
        print(f"   Shape: {linear_bias.shape}")
    else:
        print("⚠️  No bias found (this is normal for some models)")

    # Create dense module directory structure
    dense_dir = output_dir / "1_Dense"
    dense_dir.mkdir(parents=True, exist_ok=True)

    # Create dense config (sentence-transformers format)
    dense_config = {
        "in_features": in_features,
        "out_features": out_features,
        "bias": linear_bias is not None,
        "activation_function": "torch.nn.modules.linear.Identity",  # Default to Identity
    }

    dense_config_path = dense_dir / "config.json"
    with open(dense_config_path, "w") as f:
        json.dump(dense_config, f, indent=2)

    print(f"📝 Created dense config: {dense_config_path}")

    # Create dense safetensors
    dense_tensors = {"linear.weight": linear_weight}

    if linear_bias is not None:
        dense_tensors["linear.bias"] = linear_bias

    dense_safetensors_path = dense_dir / "model.safetensors"
    safetensors.torch.save_file(dense_tensors, dense_safetensors_path)

    print(f"💾 Created dense safetensors: {dense_safetensors_path}")

    # Create modules.json for the main model
    modules_config = {
        "modules": [
            {
                "name": "1_Dense",
                "type": "sentence_transformers.models.Dense",
                "path": "1_Dense/",
            }
        ]
    }

    modules_config_path = output_dir / "modules.json"
    with open(modules_config_path, "w") as f:
        json.dump(modules_config, f, indent=2)

    print(f"📝 Created modules.json: {modules_config_path}")

    # Copy main model config (without linear projection info)
    main_config_path = output_dir / "config.json"

    # Modify config to remove linear projection settings if they exist
    main_config = config.copy()
    linear_config_keys = [
        "use_linear_output_projection",
        "linear_output_size",
        "linear_output_bias",
    ]

    for key in linear_config_keys:
        main_config.pop(key, None)

    with open(main_config_path, "w") as f:
        json.dump(main_config, f, indent=2)

    print(f"📝 Created main config: {main_config_path}")

    return dense_config_path, dense_safetensors_path


def create_readme(output_dir: Path, model_id: str, dense_config: Dict):
    """Create README with usage instructions."""

    readme_content = f"""# Dense Module for {model_id}

This directory contains a dense module extracted from `{model_id}` using the sentence-transformers ecosystem format.

## Structure

```
{output_dir.name}/
├── config.json          # Main model config
├── modules.json         # Points to dense module
├── 1_Dense/
│   ├── config.json      # Dense layer config
│   └── model.safetensors # Dense layer weights
└── README.md           # This file
```

## Dense Layer Configuration

- **Input Features**: {dense_config["in_features"]}
- **Output Features**: {dense_config["out_features"]}
- **Bias**: {dense_config["bias"]}
- **Activation**: {dense_config["activation_function"]}

## Usage with Text-Embeddings-Inference

The dense module will be automatically loaded when using the model with text-embeddings-inference:

```bash
text-embeddings-inference --model-id {output_dir.as_posix()}
```

## Manual Loading

You can also manually load the dense module:

```python
from text_embeddings_backend_candle import CandleBackend
from text_embeddings_backend_core import ModelType, Pool

backend = CandleBackend::new(
    &model_path,
    "float32".to_string(),
    ModelType::Embedding(Pool::LastToken),
    None,  # dense_paths will be auto-detected from modules.json
    0,
)
```

## Extraction Details

- **Source Model**: {model_id}
- **Extraction Date**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Framework**: PyTorch + Sentence Transformers format
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)

    print(f"📖 Created README: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract linear layer from Voyage models"
    )
    parser.add_argument(
        "--model-id", default="voyageai/voyage-4-nemo", help="HuggingFace model ID"
    )
    parser.add_argument(
        "--output-dir",
        default="./voyage_dense",
        help="Output directory for dense module",
    )
    parser.add_argument(
        "--device", default="cpu", help="Device to load model on (cpu/cuda)"
    )

    args = parser.parse_args()

    # Convert to Path object
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Extracting linear output projection from Voyage model")
    print(f"   Model: {args.model_id}")
    print(f"   Output: {output_dir}")
    print()

    try:
        # Extract linear layer
        dense_config_path, dense_safetensors_path = extract_linear_layer(
            args.model_id, output_dir, args.device
        )

        # Load dense config for README
        with open(dense_config_path, "r") as f:
            dense_config = json.load(f)

        # Create README
        create_readme(output_dir, args.model_id, dense_config)

        print()
        print("✅ Extraction completed successfully!")
        print()
        print("📁 Output structure:")
        print(f"   {output_dir}/")
        print(f"   ├── config.json")
        print(f"   ├── modules.json")
        print(f"   ├── 1_Dense/")
        print(f"   │   ├── config.json")
        print(f"   │   └── model.safetensors")
        print(f"   └── README.md")
        print()
        print(f"🎯 Ready to use with text-embeddings-inference!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

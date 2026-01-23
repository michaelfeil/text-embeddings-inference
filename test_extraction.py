#!/usr/bin/env python3
"""
Test script to verify the linear layer extraction works correctly.
"""

import json
from pathlib import Path


def test_extraction():
    """Test that the extracted files have the correct structure."""

    output_dir = Path("./voyage_dense")

    if not output_dir.exists():
        print("❌ Output directory not found. Run extraction first.")
        return False

    # Check required files
    required_files = [
        "config.json",
        "modules.json",
        "1_Dense/config.json",
        "1_Dense/model.safetensors",
        "README.md",
    ]

    missing_files = []
    for file_path in required_files:
        full_path = output_dir / file_path
        if not full_path.exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    # Check modules.json structure
    modules_path = output_dir / "modules.json"
    with open(modules_path, "r") as f:
        modules_config = json.load(f)

    if "modules" not in modules_config:
        print("❌ modules.json missing 'modules' key")
        return False

    if len(modules_config["modules"]) != 1:
        print("❌ modules.json should have exactly one module")
        return False

    module = modules_config["modules"][0]
    if module.get("name") != "1_Dense":
        print("❌ Module name should be '1_Dense'")
        return False

    # Check dense config structure
    dense_config_path = output_dir / "1_Dense/config.json"
    with open(dense_config_path, "r") as f:
        dense_config = json.load(f)

    required_dense_keys = ["in_features", "out_features", "bias"]
    for key in required_dense_keys:
        if key not in dense_config:
            print(f"❌ Dense config missing '{key}'")
            return False

    print("✅ All files have correct structure!")
    print(f"   Input features: {dense_config['in_features']}")
    print(f"   Output features: {dense_config['out_features']}")
    print(f"   Has bias: {dense_config['bias']}")

    return True


if __name__ == "__main__":
    success = test_extraction()
    exit(0 if success else 1)

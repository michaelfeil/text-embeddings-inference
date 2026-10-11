These synthetic `torch.save` state dictionaries match the existing Rust BERT
and Dense fixtures. They contain OrderedDict metadata, transposed matrix
storage, offset vectors and an ignored integer position buffer. The regression
compares native outputs with the same weights loaded from safetensors, including
mixed pooled/raw outputs and both base-model and Dense-sidecar fallback.

`generate.py` records the original generation using Torch 2.14.1+cu130. Python
is used only to regenerate these checked-in files; tests and inference read
them directly in Rust. These tensors are original synthetic data under the
project license, not downloaded model weights.

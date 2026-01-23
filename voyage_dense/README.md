# Dense Module for voyageai/voyage-4-nano

This directory contains a dense module extracted from `voyageai/voyage-4-nano` using the sentence-transformers ecosystem format.

## Structure

```
voyage_dense/
├── config.json          # Main model config
├── modules.json         # Points to dense module
├── 1_Dense/
│   ├── config.json      # Dense layer config
│   └── model.safetensors # Dense layer weights
└── README.md           # This file
```

## Dense Layer Configuration

- **Input Features**: 1024
- **Output Features**: 2048
- **Bias**: false
- **Activation**: torch.nn.modules.linear.Identity

## Usage with Text-Embeddings-Inference

The dense module will be automatically loaded when using the model with text-embeddings-inference:

```bash
text-embeddings-inference --model-id ./voyage_dense
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

- **Source Model**: voyageai/voyage-4-nano
- **Extraction Date**: 2025-01-23 15:30:00
- **Framework**: PyTorch + Sentence Transformers format
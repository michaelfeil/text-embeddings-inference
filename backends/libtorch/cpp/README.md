# Native C++ backend layout

- `models/`: model interface, model families, embedding and decision heads.
- `helpers/`: configuration and weights, shared attention composition, FP8 linear wrapper, image-feature cache.
- `batching/`: packed input and media descriptors, radix folding, CUDA graph lifecycle, pooling interface.
- `kernels/`: CUDA operators, device arithmetic and their native entry points.
- `tests/`: existing native regression fixtures and reference data.
- `vendor/`: pinned third-party sources and their licenses.
- `bridge.cpp` and `bridge.h`: C ABI, model construction and request dispatch.

Includes use paths relative to this directory, so dependencies between categories
are explicit. `models/model.h` defines the model contract; packed request data
belongs to `batching/packed_input.h`, options and checkpoint weights to
`helpers/options.h`, and shared attention composition to `helpers/attention.h`.

The CMake target and test names remain stable. CUDA sources keep their original
compile flags and licensed Candle includes. BERT's existing implementation is
in `models/bert_model.h`; its public configuration remains `TeiConfig` from the
C ABI.

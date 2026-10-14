<div align="center">

# infinity_v3

<a href="docs/openapi.json">
  <img alt="Swagger API documentation" src="https://img.shields.io/badge/API-Swagger-informational">
</a>

A blazing fast inference solution for text embeddings models.

</div>

`infinity_v3` is a fork of [huggingface/text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference).

Covered fork contributions are offered under your choice of
**[PolyForm Noncommercial License 1.0.0](LICENSES/PolyForm-Noncommercial-1.0.0.txt)**
or **[Infinity License 1.0 for Commercial Use](LICENSES/Infinity-License-1.0-for-Commercial-Use.txt)**.
Researchers can use the unmodified standard PolyForm option on any platform
for uses it permits, without Infinity-specific restrictions. The commercial
alternative allows enterprise use on owned hardware or permitted providers
and prohibits deployment on the listed platforms. These are source-available
licenses with use restrictions. See the [adoption scope and full terms](LICENSE)
and [licensing guide](LICENSING.md).

Earlier Apache-2.0 and INFOIL grants and third-party licenses remain in
effect for the material and rights they cover. Retained attribution is in
[NOTICE](NOTICE) and [third-party notices](THIRD_PARTY_NOTICES.md).

The server command is `infinity_v3`; the LibTorch Rust package is
`infinity-backend-libtorch`. Package metadata credits
`infinity authors`; inherited attribution is recorded in [NOTICE](NOTICE).
LibTorch's native symbols retain their existing names.

Prometheus metrics use the `infinity_` prefix, including `infinity_embed_duration`
and `infinity_queue_size`. Update dashboards, alerts, and recording rules that
query the former `te_` names; those names are no longer emitted. Histogram buckets
and metric labels retain their existing behavior.

Project settings previously named `TEI_*` now use `INFINITY_*`, including
LibTorch, HTTP memory budgets, test fixtures, and benchmarks. Update deployment
and test configurations when upgrading; the old environment names are no longer
read. The queue's early-dispatch override is `INFINITY_EARLY_DISPATCH_TOKENS`.

## Table of contents

- [Get Started](#get-started)
    - [Supported Models](#supported-models)
    - [Docker](#docker)
    - [Docker Images](#docker-images)
    - [API Documentation](#api-documentation)
    - [Using a private or gated model](#using-a-private-or-gated-model)
    - [Air gapped deployment](#air-gapped-deployment)
    - [Using Re-rankers models](#using-re-rankers-models)
    - [Using Sequence Classification models](#using-sequence-classification-models)
    - [Using SPLADE pooling](#using-splade-pooling)
    - [Distributed Tracing](#distributed-tracing)
- [Local Install](#local-install)
- [Docker Build](#docker-build)
    - [Apple M1/M2 Arm](#apple-m1m2-arm64-architectures)
- [Examples](#examples)

This fork serves inference through HTTP using the deprecated Candle backend by
default, with an optional
[experimental native LibTorch 2.14.1 backend](docs/libtorch-backend.md). The Python
backend, Intel IPEX/Gaudi images, and public gRPC API have been removed.
OpenTelemetry export over OTLP/gRPC remains supported.

**The Candle backend is deprecated.** Use `--backend libtorch` (or
`INFINITY_BACKEND=libtorch`) for models supported by the
[native LibTorch backend](docs/libtorch-backend.md), with a build that includes
the `libtorch` feature. Candle remains the default for compatibility and emits
a deprecation warning when initialized.

infinity_v3 is a toolkit for deploying and serving open source text embeddings and sequence
classification models. infinity_v3 enables high-performance extraction for the most popular models, including FlagEmbedding,
Ember, GTE and E5. infinity_v3 implements many features such as:

* No model graph compilation step
* Small docker images and fast boot times. Get ready for true serverless!
* Token based dynamic batching
* Optimized transformers code for inference using [Flash Attention](https://github.com/HazyResearch/flash-attention),
  [Candle](https://github.com/huggingface/candle)
  and [cuBLASLt](https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api)
* [Safetensors](https://github.com/huggingface/safetensors) weight loading
* Production ready (distributed tracing with Open Telemetry, Prometheus metrics)

## Multiple resident models

Serve multiple models behind the same HTTP listener by repeating `--model-id`.
Each model owns its tokenizer pool, batching queue, admission limits, and backend
replicas. All models initialize and warm up before the listener starts; a startup
failure stops the deployment and shuts down models already initialized.

```bash
infinity_v3 \
  --model-id BAAI/bge-small-en-v1.5 \
  --model-id mixedbread-ai/mxbai-rerank-xsmall-v1 \
  --served-model-name embed --served-model-name rerank \
  --backend libtorch --dtype float16 \
  --backend-device-ids 0 --backend-device-ids 1
```

The same configuration can be supplied directly through Kubernetes or Knative
container environment variables:

```yaml
env:
  - name: INFINITY_MODEL_ID
    value: "BAAI/bge-small-en-v1.5;mixedbread-ai/mxbai-rerank-xsmall-v1"
  - name: INFINITY_SERVED_MODEL_NAME
    value: "embed;rerank"
  - name: INFINITY_BACKEND
    value: "libtorch"
  - name: INFINITY_DTYPE
    value: "float16"
  - name: INFINITY_BACKEND_DEVICE_IDS
    value: "0;1"
  - name: INFINITY_MAX_BATCH_TOKENS
    value: "8192;4096"
```

Model-specific options accept one value for all models or exactly one value per
model, in model-ID order. Semicolons separate model entries; commas within a
`backend-device-ids` entry select replicas (`"0,1;2,3"`). Optional settings accept
`none`, for example `--pooling 'mean;none'` or
`INFINITY_DECISION_PROTOCOL='none;pplx'`. Boolean model options accept aligned
`true;false` values; their existing bare CLI flags still mean `true`.
Mismatched option counts, duplicate served names, conflicting per-model settings,
and invalid device assignments are rejected before downloading weights.

Prompt text preserves literal semicolons. Repeat `--default-prompt` for per-model
prompts; `INFINITY_DEFAULT_PROMPT` accepts a single literal prompt or a JSON array,
for example `'["Represent this; retrieve it", null]'`. Use `null` for no prompt.
CLI prompt values always remain literal, including text starting with `[`.

Every CLI option has an `INFINITY_*` environment variable. Legacy unprefixed
variables remain accepted when the corresponding prefixed variable is absent.
An explicit CLI option replaces that option's ENV value as a whole.
Listener, authentication, CORS, payload limits, Hub credentials/cache, allowed
image hosts, and telemetry settings are shared across models. Tokenizer worker
and image-memory budgets remain per model. Runtime features must be compiled for
each selected backend. Native precision environment settings remain process-wide.

CUDA deployments with multiple models require explicit device assignments.
Models can share a GPU when their combined resident weights and startup batch
allocations fit; independent queues do not guarantee GPU scheduling fairness.

```bash
curl http://localhost:3000/v1/embeddings \
  -H 'Content-Type: application/json' -H 'Accept: application/json' \
  -d '{"model":"embed","input":"hello world"}'
```

Use `GET /v1/models` (or `/models`) for served names and capabilities and
`GET /info?model=embed` for one model's settings. With multiple models, inference,
`/tokenize`, `/decode`, `/vertex`, and `/info` require a valid model name. Vertex
selects one model for the whole request through its top-level `model` field.
Unknown/missing names and unsupported endpoints return 400. A failed model
returns 503 on inference; aggregate `/health`, `/ping`, and GET `/` return 500
if any model or replica is unhealthy. A sole model retains advisory model-name
compatibility. Explicit names are reported in model-bearing responses.
POST `/` and `/invocations` select the task of the requested model.

The focused mixed-backend integration check uses a Candle CPU embedding model
and a LibTorch CPU reranker. Set `MULTI_MODEL_EMBEDDING_DIR` and
`MULTI_MODEL_RERANK_DIR` to local CPU checkpoints; the check uses a 128-token batch budget,
then run `resident_models_route_isolate_and_shutdown` with `--ignored` in a
`candle,libtorch,http` router test build.

## GPU replicas

Candle CUDA uses all visible GPUs by default, loading a complete model on each GPU
in parallel. Each replica consumes work from one shared request queue and one shared
prepared-batch slot. Faster replicas naturally accept more batches. Any backend
failure makes the whole service unhealthy.

Use `CUDA_VISIBLE_DEVICES=0,1` to expose two GPUs, `--device-id 0` to select just one,
or `--backend-device-ids 0,1` to select visible CUDA ordinals explicitly. Each GPU
must have enough memory for the complete model and its batch. CPU retains
one backend.

With multiple GPUs, batches have a soft early-dispatch target of 5,000 tokens while
the queued backlog is below 20,000 tokens per replica. Larger backlogs use the normal
batch limits. Set `INFINITY_EARLY_DISPATCH_TOKENS` to override the target, or `0` to disable
it. Single-GPU batching keeps its existing behavior unless this variable is set.
Per-replica batch, token, and inference-duration metrics are exposed on `/metrics`.
Each replica logs inference throughput every 100 batches.

### BPE tokenization

Embedding text inputs automatically use `basetenkenizer` 0.2.9 when the tokenizer configuration is supported. No opt-in flag or environment variable is required. WordPiece and Unigram models, paired or token-ID inputs, classification/NER, `/tokenize`, and `/decode` keep using Hugging Face Tokenizers. Truncation and special-token processing also remain with Hugging Face. Short BPE inputs run directly on the tokenizer worker; larger inputs use Basetenkenizer's own shared pool.

Tokenization request workers are capped at 16, including explicit overrides.
Basetenkenizer 0.2.9 separately caps its shared BPE pool at eight threads;
`--tokenization-workers` controls request workers, not that library pool.
The waiting-job queue
has a separate default capacity of 1,024; set `--tokenization-queue-capacity` or
`TOKENIZATION_QUEUE_CAPACITY` to change it. Capacity counts individual inputs,
not HTTP requests, and must be positive. Producers wait when the queue is full.
The effective worker count and queue capacity are reported by `/info`.

## Get Started

### Supported Models

#### Text Embeddings

infinity_v3 currently supports Nomic, BERT, CamemBERT, XLM-RoBERTa models with absolute positions, JinaBERT
model with Alibi positions and Mistral, Alibaba GTE, Qwen2 models with Rope positions, ModernBERT, Qwen3, Gemma3, dense Gemma4 text models, and multimodal EmbeddingGemma 2.

Below are some examples of the currently supported models:

| MTEB Rank | Model Size             | Model Type     | Model ID                                                                                         |
|-----------|------------------------|----------------|--------------------------------------------------------------------------------------------------|
| 2         | 7.57B (Very Expensive) | Qwen3          | [Qwen/Qwen3-Embedding-8B](https://hf.co/Qwen/Qwen3-Embedding-8B)                                 |
| 3         | 4.02B (Very Expensive) | Qwen3          | [Qwen/Qwen3-Embedding-4B](https://hf.co/Qwen/Qwen3-Embedding-4B)                                 |
| 4         | 509M                   | Qwen3          | [Qwen/Qwen3-Embedding-0.6B](https://hf.co/Qwen/Qwen3-Embedding-0.6B)                             |
| 6         | 7.61B (Very Expensive) | Qwen2          | [Alibaba-NLP/gte-Qwen2-7B-instruct](https://hf.co/Alibaba-NLP/gte-Qwen2-7B-instruct)             |
| 7         | 560M                   | XLM-RoBERTa    | [intfloat/multilingual-e5-large-instruct](https://hf.co/intfloat/multilingual-e5-large-instruct) |
| 8         | 308M                   | Gemma3         | [google/embeddinggemma-300m](https://hf.co/google/embeddinggemma-300m) (gated)                   |
| N/A       | 744M                   | EmbeddingGemma2 | [google/embeddinggemma-2](https://hf.co/google/embeddinggemma-2)                              |
| 15        | 1.78B (Expensive)      | Qwen2          | [Alibaba-NLP/gte-Qwen2-1.5B-instruct](https://hf.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct)         |
| 18        | 7.11B (Very Expensive) | Mistral        | [Salesforce/SFR-Embedding-2_R](https://hf.co/Salesforce/SFR-Embedding-2_R)                       |
| 35        | 568M                   | XLM-RoBERTa    | [Snowflake/snowflake-arctic-embed-l-v2.0](https://hf.co/Snowflake/snowflake-arctic-embed-l-v2.0) |
| 41        | 305M                   | Alibaba GTE    | [Snowflake/snowflake-arctic-embed-m-v2.0](https://hf.co/Snowflake/snowflake-arctic-embed-m-v2.0) |
| 52        | 335M                   | BERT           | [WhereIsAI/UAE-Large-V1](https://hf.co/WhereIsAI/UAE-Large-V1)                                   |
| 58        | 137M                   | NomicBERT      | [nomic-ai/nomic-embed-text-v1](https://hf.co/nomic-ai/nomic-embed-text-v1)                       |
| 79        | 137M                   | NomicBERT      | [nomic-ai/nomic-embed-text-v1.5](https://hf.co/nomic-ai/nomic-embed-text-v1.5)                   |
| N/A       | 475M-A305M             | NomicBERT      | [nomic-ai/nomic-embed-text-v2-moe](https://hf.co/nomic-ai/nomic-embed-text-v2-moe)               |
| N/A       | 434M                   | Alibaba GTE    | [Alibaba-NLP/gte-large-en-v1.5](https://hf.co/Alibaba-NLP/gte-large-en-v1.5)                     |
| N/A       | 396M                   | ModernBERT     | [answerdotai/ModernBERT-large](https://hf.co/answerdotai/ModernBERT-large)                       |
| N/A       | 137M                   | JinaBERT       | [jinaai/jina-embeddings-v2-base-en](https://hf.co/jinaai/jina-embeddings-v2-base-en)             |
| N/A       | 137M                   | JinaBERT       | [jinaai/jina-embeddings-v2-base-code](https://hf.co/jinaai/jina-embeddings-v2-base-code)         |

To explore the list of best performing text embeddings models, visit the
[Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

#### Sequence Classification and Re-Ranking

infinity_v3 currently supports CamemBERT, and XLM-RoBERTa Sequence Classification models with absolute positions.

Below are some examples of the currently supported models:

| Task               | Model Type  | Model ID                                                                                                        |
|--------------------|-------------|-----------------------------------------------------------------------------------------------------------------|
| Re-Ranking         | XLM-RoBERTa | [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)                                       |
| Re-Ranking         | XLM-RoBERTa | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)                                         |
| Re-Ranking         | GTE         | [Alibaba-NLP/gte-multilingual-reranker-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base) |
| Re-Ranking         | ModernBert  | [Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) |
| Sentiment Analysis | RoBERTa     | [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)                     |

### Docker

Build this fork before running it. The examples use a local `infinity_v3:latest`
image; upstream TEI images do not contain this fork's changes. Set the compute
capability for your GPU (80 below is for A100):

```shell
docker build -f Dockerfile-cuda --build-arg CUDA_COMPUTE_CAP=80 -t infinity_v3:latest .
```

```shell
model=Qwen/Qwen3-Embedding-0.6B
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data infinity_v3:latest --model-id $model
```

And then you can make requests like

```bash
curl 127.0.0.1:8080/embed \
    -X POST \
    -d '{"inputs":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'
```

**Note:** To use GPUs, you need to install
the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
NVIDIA drivers on your machine need to be compatible with CUDA version 12.2 or higher.

To see all options to serve your models:

```console
$ infinity_v3 --help
infinity_v3 inference server

Usage: infinity_v3 [OPTIONS] --model-id <MODEL_ID>

Options:
      --model-id <MODEL_ID>
          The Hugging Face model ID, can be any model listed on <https://huggingface.co/models> with the `text-embeddings-inference` tag (meaning it's compatible with Text Embeddings Inference)

          Alternatively, the specified ID can also be a path to a local directory containing the necessary model files saved by the `save_pretrained(...)` methods of either Transformers or Sentence Transformers. The aliases `pplx-decider-v1.1-27b` and `d3` select their native decision protocols.

          [env: INFINITY_MODEL_ID=]

      --served-model-name <SERVED_MODEL_NAME>
          Public model names. Repeat per model; defaults to the checkpoint ID

          [env: INFINITY_SERVED_MODEL_NAME=]

      --revision <REVISION>
          The actual revision of the model if you're referring to a model on the hub. You can use a specific commit id or a branch like `refs/pr/2`

          [env: INFINITY_REVISION=]

      --tokenization-workers <TOKENIZATION_WORKERS>
          Optionally control the number of tokenizer workers used for payload tokenization, validation and truncation. Defaults to CPU cores minus one. Values are clamped to 1–16

          [env: INFINITY_TOKENIZATION_WORKERS=]

      --tokenization-queue-capacity <TOKENIZATION_QUEUE_CAPACITY>
          Maximum waiting tokenizer jobs, independent of worker count. Counts individual inputs, not HTTP requests. When full, producers wait for space

          [env: INFINITY_TOKENIZATION_QUEUE_CAPACITY=]
          [default: 1024]

      --dtype <DTYPE>
          Model dtype. Auto selects bfloat16 from model config when supported, otherwise the backend default

          [env: INFINITY_DTYPE=]
          [default: auto]
          [possible values: auto, float16, float32, bfloat16]

      --backend <BACKEND>
          Inference runtime (requires the matching Cargo feature). Candle is deprecated; use libtorch for supported models

          [env: INFINITY_BACKEND=]
          [default: candle]
          [possible values: candle, libtorch]

      --torch-device <TORCH_DEVICE>
          LibTorch accelerator family. Auto selects CUDA when available, otherwise CPU

          [env: INFINITY_TORCH_DEVICE=]
          [default: auto]
          [possible values: auto, cpu, cuda]

      --pooling <POOLING>
          Optionally control the pooling method for embedding models.

          If `pooling` is not set, the pooling configuration will be parsed from the model `1_Pooling/config.json` configuration.

          If `pooling` is set, it will override the model pooling configuration

          [env: INFINITY_POOLING=]

      --decision-protocol <DECISION_PROTOCOL>
          Serve a decision model with its trained prompt on /v1/systemone

          [env: INFINITY_DECISION_PROTOCOL=]

      --image-allowed-hosts <IMAGE_ALLOWED_HOSTS>
          Exact HTTPS hostnames allowed for image downloads. Empty permits inline data URLs only

          [env: INFINITY_IMAGE_ALLOWED_HOSTS=]

      --image-memory-budget-mib <IMAGE_MEMORY_BUDGET_MIB>
          Memory budget for encoded images, CPU processing and prepared image patches, in MiB

          [env: INFINITY_IMAGE_MEMORY_BUDGET_MIB=]
          [default: 512]

      --max-concurrent-requests <MAX_CONCURRENT_REQUESTS>
          The maximum amount of concurrent requests for this particular deployment. Having a low limit will refuse clients requests instead of having them wait for too long and is usually good to handle backpressure correctly

          [env: INFINITY_MAX_CONCURRENT_REQUESTS=]
          [default: 512]

      --max-batch-tokens <MAX_BATCH_TOKENS>
          **IMPORTANT** This is one critical control to allow maximum usage of the available hardware.

          This represents the total amount of potential tokens within a batch.

          For `max_batch_tokens=1000`, you could fit `10` queries of `total_tokens=100` or a single query of `1000` tokens.

          Overall this number should be the largest possible until the model is compute bound. Since the actual memory overhead depends on the model implementation, infinity_v3 cannot infer this number automatically.

          [env: INFINITY_MAX_BATCH_TOKENS=]
          [default: 16384]

      --max-batch-requests <MAX_BATCH_REQUESTS>
          Optionally control the maximum number of individual requests in a batch

          [env: INFINITY_MAX_BATCH_REQUESTS=]

      --enable-fp8-dynamic [<ENABLE_FP8_DYNAMIC>]
          Experimental calibration-free FP8 MLPs (Ada/Hopper/Blackwell FP16/BF16 dense decoder and BERT/ModernBERT encoders). Bundled in official SM89/90/100/120 images; disabled by default. May change embedding, reranking, and decision accuracy

          [env: INFINITY_ENABLE_FP8_DYNAMIC=]
          [default: false]
          [possible values: true, false]

      --radix-mlp-threshold <RADIX_MLP_THRESHOLD>
          RadixMLP threshold.

          Set the threshold for RadixMLP. If the compression ratio is lower than the threshold, RadixMLP will be used. The default is 0.92 for most models, and 0.0 (force disabled) for bidirectional models.

          [env: INFINITY_RADIX_MLP_THRESHOLD=]
          [default: 0.92]

      --max-client-batch-size <MAX_CLIENT_BATCH_SIZE>
          Control the maximum number of inputs that a client can send in a single request

          [env: INFINITY_MAX_CLIENT_BATCH_SIZE=]
          [default: 32]

      --max-decision-questions <MAX_DECISION_QUESTIONS>
          Maximum number of questions in one /v1/systemone request. Independent of max-client-batch-size; model option budgets still apply

          [env: INFINITY_MAX_DECISION_QUESTIONS=]
          [default: 512]

      --auto-truncate [<AUTO_TRUNCATE>]
          Automatically truncate inputs that are longer than the maximum supported size

          [env: INFINITY_AUTO_TRUNCATE=]
          [default: false]
          [possible values: true, false]

      --default-prompt-name <DEFAULT_PROMPT_NAME>
          The name of the prompt that should be used by default for encoding. If not set, no prompt will be applied.

          Must be a key in the `sentence-transformers` configuration `prompts` dictionary.

          For example if ``default_prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?" because the prompt text will be prepended before any text to encode.

          The argument '--default-prompt-name <DEFAULT_PROMPT_NAME>' cannot be used with '--default-prompt <DEFAULT_PROMPT>`

          [env: INFINITY_DEFAULT_PROMPT_NAME=]

      --default-prompt <DEFAULT_PROMPT>
          The prompt that should be used by default for encoding. If not set, no prompt will be applied.

          For example if ``default_prompt`` is "query: " then the sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?" because the prompt text will be prepended before any text to encode.

          The argument '--default-prompt <DEFAULT_PROMPT>' cannot be used with '--default-prompt-name <DEFAULT_PROMPT_NAME>`

          [env: INFINITY_DEFAULT_PROMPT=]

      --dense-path <DENSE_PATH>
          Optionally, define the path to the Dense module required for some embedding models.

          Some embedding models require an extra `Dense` module which contains a single Linear layer and an activation function. By default, those `Dense` modules are stored under the `2_Dense` directory, but there might be cases where different `Dense` modules are provided, to convert the pooled embeddings into different dimensions, available as `2_Dense_<dims>` e.g. https://huggingface.co/NovaSearch/stella_en_400M_v5.

          Note that this argument is optional, only required to be set if there is no `modules.json` file or when you want to override a single Dense module path, only when running with the `candle` backend.

          [env: INFINITY_DENSE_PATH=]

      --device-id <DEVICE_ID>
          Use one explicit accelerator device. CUDA otherwise uses all visible devices

          [env: INFINITY_DEVICE_ID=]

      --backend-device-ids <BACKEND_DEVICE_IDS>
          Visible CUDA ordinals for data-parallel replicas, e.g. 0,1, or auto

          [env: INFINITY_BACKEND_DEVICE_IDS=]

      --hf-token <HF_TOKEN>
          Your Hugging Face Hub token

          [env: INFINITY_HF_TOKEN=]

      --hostname <HOSTNAME>
          The IP address to listen on

          [env: HOSTNAME=michaelfeil-dev-pod-h100-0]
          [default: 0.0.0.0]

  -p, --port <PORT>
          The port to listen on

          [env: INFINITY_PORT=]
          [default: 3000]

      --huggingface-hub-cache <HUGGINGFACE_HUB_CACHE>
          The location of the huggingface hub cache. Used to override the location if you want to provide a mounted disk for instance

          [env: INFINITY_HUGGINGFACE_HUB_CACHE=]

      --payload-limit <PAYLOAD_LIMIT>
          Payload size limit in bytes

          Default is 2MB

          [env: INFINITY_PAYLOAD_LIMIT=]
          [default: 2000000]

      --api-key <API_KEY>
          Set an api key for request authorization.

          By default the server responds to every request. With an api key set, the requests must have the Authorization header set with the api key as Bearer token.

          [env: INFINITY_API_KEY=]

      --json-output
          Outputs the logs in JSON format (useful for telemetry)

          [env: INFINITY_JSON_OUTPUT=]

      --disable-spans
          [env: INFINITY_DISABLE_SPANS=]

      --otlp-endpoint <OTLP_ENDPOINT>
          The grpc endpoint for opentelemetry. Telemetry is sent to this endpoint as OTLP over gRPC. e.g. `http://localhost:4317`

          [env: INFINITY_OTLP_ENDPOINT=]

      --otlp-service-name <OTLP_SERVICE_NAME>
          The service name for opentelemetry. e.g. `infinity_v3.server`

          [env: INFINITY_OTLP_SERVICE_NAME=]
          [default: infinity_v3.server]

      --prometheus-port <PROMETHEUS_PORT>
          The Prometheus port to listen on

          [env: INFINITY_PROMETHEUS_PORT=]
          [default: 9000]

      --cors-allow-origin <CORS_ALLOW_ORIGIN>
          [env: INFINITY_CORS_ALLOW_ORIGIN=]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

### Docker Images

Use these local tags when building the corresponding Docker variants. These
names are build examples, not a claim that images have been published. Use
`Dockerfile` for CPU or `Dockerfile-cuda` with the appropriate
`CUDA_COMPUTE_CAP` for GPU, and set `-t` to the tag below:

| Architecture                        | Image                                                                   |
|-------------------------------------|-------------------------------------------------------------------------|
| CPU                                 | infinity_v3:cpu-latest                   |
| Volta                               | NOT SUPPORTED                                                           |
| Turing (T4, RTX 2000 series, ...)   | infinity_v3:turing-latest (experimental) |
| Ampere 80 (A100, A30)               | infinity_v3:latest                       |
| Ampere 86 (A10, A40, ...)           | infinity_v3:86-latest                    |
| Ada Lovelace (RTX 4000 series, ...) | infinity_v3:89-latest                    |
| Hopper (H100)                       | infinity_v3:hopper-latest (experimental) |

Turing uses packed FlashAttention v1 with float16. Models requiring sliding-window
attention or ALiBi require FlashAttention v2 on Ampere or newer GPUs.

### API documentation

You can consult the OpenAPI documentation of the `infinity_v3` REST API using the `/docs` route.
A checked-in [OpenAPI specification](docs/openapi.json) and [Swagger UI](docs/index.html) are also included.

### Using a private or gated model

You have the option to utilize the `HF_TOKEN` environment variable for configuring the token employed by
`infinity_v3`. This allows you to gain access to protected resources.

For example:

1. Go to https://huggingface.co/settings/tokens
2. Copy your CLI READ token
3. Export `HF_TOKEN=<your CLI READ token>`

or with Docker:

```shell
model=<your private model>
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
token=<your CLI READ token>

docker run --gpus all -e HF_TOKEN=$token -p 8080:80 -v $volume:/data infinity_v3:latest --model-id $model
```

### Air gapped deployment

To deploy infinity_v3 in an air-gapped environment, first download the weights and then mount them inside
the container using a volume.

For example:

```shell
# (Optional) create a `models` directory
mkdir models
cd models

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

# Set the models directory as the volume path
volume=$PWD

# Mount the models directory inside the container with a volume and set the model ID
docker run --gpus all -p 8080:80 -v $volume:/data infinity_v3:latest --model-id /data/Qwen3-Embedding-0.6B
```

### Using Re-rankers models

Upstream TEI v0.4.0 added support for CamemBERT, RoBERTa, XLM-RoBERTa, and GTE Sequence Classification models.
Re-rankers models are Sequence Classification cross-encoders models with a single class that scores the similarity
between a query and a text.

See [this blogpost](https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83) by
the LlamaIndex team to understand how you can use re-rankers models in your RAG pipeline to improve
downstream performance.

```shell
model=BAAI/bge-reranker-large
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data infinity_v3:latest --model-id $model
```

And then you can rank the similarity between a query and a list of texts with:

```bash
curl 127.0.0.1:8080/rerank \
    -X POST \
    -d '{"query": "What is Deep Learning?", "texts": ["Deep Learning is not...", "Deep learning is..."]}' \
    -H 'Content-Type: application/json'
```

### Using Sequence Classification models

You can also use classic Sequence Classification models like `SamLowe/roberta-base-go_emotions`:

```shell
model=SamLowe/roberta-base-go_emotions
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data infinity_v3:latest --model-id $model
```

Once you have deployed the model you can use the `predict` endpoint to get the emotions most associated with an input:

```bash
curl 127.0.0.1:8080/predict \
    -X POST \
    -d '{"inputs":"I like you."}' \
    -H 'Content-Type: application/json'
```

### Laya typed decisions (Jev API)

Serve `convaiinnovations/laya-typed-decisions` with the HTTP build using LibTorch
or Candle. For LibTorch CUDA inference:

```shell
infinity_v3 --model-id convaiinnovations/laya-typed-decisions \
  --backend libtorch --torch-device cuda:0 --dtype bfloat16
```

For LibTorch CPU inference, use `--torch-device cpu --dtype float32`.

The existing API server exposes `POST /v1/systemone`. Questions use infinity_v3's shared
batch queue, backend replicas, concurrency limits, authentication, and metrics.
ModernBERT encodes the batch once; Laya's custom head scores each question's
options. This is bidirectional inference; RadixMLP is disabled for this model.

`MAX_DECISION_QUESTIONS` (or `--max-decision-questions`) caps questions per
request (default 512), independently of `MAX_CLIENT_BATCH_SIZE`. Excess returns
HTTP 422 before inference. Usage and token headers count the full formatted input
for each question, including repeated state.

```shell
curl http://localhost:3000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "I was billed twice. Please refund the extra charge.",
    "questions": {
      "team": {"type": "choice", "instructions": "Which team?",
               "criteria": {"billing": "billing and payments", "support": "technical support"}},
      "urgency": {"type": "score", "instructions": "How urgent is this request?",
                  "criteria": ["low", "medium", "high"]},
      "refund": {"type": "noul", "instructions": "The customer requests a refund."}
    }
  }'
```

Answers contain the selected choice, an expected zero-based score, or a `noul`
probability of true, plus temperature-scaled `answer_confidence` and the action head's
`act_probability`. Choice/score answers also include option probabilities and
entropy-based `confidence`. Usage reports input tokens and zero output tokens.
Option order follows the request JSON. `state` is either a plain text string or
an explicit `{"messages": [...]}` envelope. Messages preserve roles and ordered
text/image/audio/video content blocks for a model's native processor. The current
Laya adapter supports **plain text only** and rejects native messages with 422;
the message schema does not imply multimodal backend support. Plain text truncates
from the right. Arbitrary JSON objects and bare arrays are no longer accepted as
state; serialize structured records explicitly when supplying text to Laya.
See [the input design](docs/systemone-input-design.md) for the message contract.

Optional `max_len` and `head_max_len` override the checkpoint's token budgets,
up to the server's maximum input length. Requests that cannot retain all options,
or make options identical after token truncation, return 422. The server accepts
Jev's `model` field as an alias and always uses its configured checkpoint.

Model limits: 100 choice options per question, 32 score levels, 512 total options
per request, and 50,000 state characters.
Use the existing `--max-batch-tokens`, `--max-batch-requests`,
`--max-concurrent-requests`, and replica options to control serving capacity.
For a local checkpoint, retain `rl_agent_config.json`, `encoder/config.json`,
`tokenizer/tokenizer.json`, and `model.safetensors` in their original layout.
Laya requires the HTTP API. The LibTorch CUDA head processes packed questions
together with variable-length attention and computes action features on the GPU;
CPU execution retains the original per-question head.

Unknown request/question fields are rejected. Execution extensions such as
`think`, `mode`, `depends_on`, `ask_if`, and `alone` are not implemented.
Temperatures are clamped to [0.5, 5], matching upstream Laya. The published
checkpoint's `choice:11+` temperature is outside this range; confidence for that
bucket is not verified as calibrated. ModernBERT uses approximate (tanh) GELU;
the custom head retains ReLU and exact scorer/action GELU.

### JSON and MessagePack responses

All HTTP routes negotiate response formats through `Accept`. API responses default
to JSON; request bodies remain JSON. To receive MessagePack, send
`Accept: application/msgpack` (`application/x-msgpack` is also accepted):

```sh
curl http://localhost:3000/v1/classify \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/msgpack' \
  -d '{"input":"I love this!"}' --output classification.msgpack
```

MessagePack uses the same field names and shapes as JSON. Typed API results are
serialized directly, retaining native integer and floating-point types. Token
usage, status codes and response metadata headers are preserved. MessagePack is
sent whenever `Accept` allows it, including `application/*` and `*/*`, regardless
of header order or positive quality weights. `q=0` exclusions are respected;
unacceptable media types return 406. Responses include `Vary: Accept`.
Swagger/OpenAPI lists both response formats with the same schemas.

Metrics and docs keep their native text/HTML formats when `Accept` is absent or
only allows their native format. JSON or MessagePack responses encode text as
strings; empty health responses encode as null. HEAD, 204 and 304 responses remain
bodyless. Framework errors also support the selected format.

### Chat inputs for embeddings and classification

`/v1/embeddings` accepts a top-level `messages` array as an alternative to `input`:

```json
{"messages":[{"role":"system","content":"Represent the conversation for retrieval."},{"role":"user","content":"How do I reset my password?"}]}
```

Provide exactly one of `input` or `messages`. A conversation produces one embedding;
`encoding_format` and `dimensions` work as they do for ordinary embedding inputs.
The existing `input` message-array format remains supported.

`/v1/classify` accepts a conversation under `messages`:

```json
{"messages":[{"role":"system","content":"Classify the user's sentiment."},{"role":"user","content":"I love this!"}]}
```

For plain text classification, send `{"input": "I love this!"}`. An `input` array
of strings describes independent inputs, so `["first", "second"]` is a batch,
rather than a text pair. Final token IDs and batches of token IDs are also accepted.
Provide exactly one of `input` or `messages`.

The SGLang-compatible response contains class probabilities in class-ID order,
the predicted label, and token usage:

```json
{
  "id": "classify-9bf17f2847b046c7b2d5495f4b4f9682",
  "object": "list",
  "created": 1791590400,
  "model": "your-deployed-model",
  "data": [{"index": 0, "label": "positive", "probs": [0.12, 0.88], "num_classes": 2}],
  "usage": {"prompt_tokens": 12, "completion_tokens": 0, "total_tokens": 12, "prompt_tokens_details": null}
}
```

`prompt_tokens` sums the actual input tokens across the batch, including chat
formatting and special tokens, after truncation. `total_tokens` equals
`prompt_tokens`; classification generates no completion tokens. Cached-token
billing details are unavailable and `prompt_tokens_details` is null. Pricing is
applied by your serving platform. Probabilities use softmax, including `[1.0]`
for a one-class model, matching SGLang. Use `/rerank` for reranking scores.

Optional `model`, `user` and `rid` fields are accepted as metadata; the response
identifies the deployed model and generates its own request ID. A batch of `rid`
values must match the input batch length. Nonzero `priority` is rejected because
priority scheduling is unsupported. `truncate` and `truncation_direction` are
infinity_v3 extensions. Errors use an `object`, `message`, `type`, `param` and `code` envelope.

`/predict` is deprecated; new clients should use `/v1/classify`. It remains
available with its existing text/pair/token-ID inputs, conversation objects and
label/score response arrays. Legacy classifier aliases `/` and `/invocations`
retain the same behavior. These legacy routes include `Deprecation` and a `Link`
to `/v1/classify` on responses. No removal date is scheduled. To migrate, rename
`inputs` to `input` for strings and token IDs, send chat conversations as
`messages`, and read `data[].probs` and `usage` from the new response. A legacy
text pair cannot be migrated as a two-string array: that array is a batch in the
new endpoint. Raw-score clients should continue to use `/predict` until they can
consume probabilities, or use `/rerank` when appropriate.

Text conversations use the deployed model's `chat_template.jinja` or the template
in `tokenizer_config.json`. System, developer, user and assistant roles are passed
to that template; the template determines which roles it accepts. Text content
blocks are joined in order. Classification conversations support text only.
Missing templates and unsupported media are rejected. Rendering uses
`add_generation_prompt=false`, adds no extra tokenizer special tokens and skips
plain-text default prompts. Request-level template overrides and template keyword
arguments are not supported. This expands request compatibility; the Candle
backend must still support the deployed model's architecture.

### Pretokenized embeddings

`/v1/embeddings` accepts one final token-ID sequence as `input: [101, 42, 102]`,
or independent sequences as `input: [[101, 42, 102], [101, 43, 102]]`.
Use `inputs` for `/embed`. Strings, string batches and native messages remain supported.

IDs must come from the deployed model's tokenizer and include its required special
tokens and instructions. They are passed directly to inference without decoding,
retokenizing or applying the server's default prompt. `prompt_name` is rejected.
Empty sequences and IDs outside the model's embedding vocabulary are rejected.
Length limits and explicit truncation behave as for text; position IDs use the
model's offset and segment IDs are zero. Character usage is zero because the
original text is unavailable; token usage counts the tokens actually processed.
Pretokenized Qwen3-VL inputs are unsupported because IDs alone lack media metadata.

`/predict` also accepts `inputs: [101, 42, 102]` and token-ID batches, with the
same validation and direct-inference behavior. Its existing string/pair formats,
prediction response arrays and token-usage headers are preserved. Numeric batches
contain independent complete sequences; segment IDs are zero, so models requiring
nonzero segment IDs for prepared pairs need a future structured input format.

### Qwen3-VL image embeddings

With a Candle CUDA/FlashAttention build, serve `Qwen/Qwen3-VL-Embedding-2B`:

```shell
infinity_v3 --model-id Qwen/Qwen3-VL-Embedding-2B \
  --dtype float16 --max-batch-tokens 8192 --auto-truncate \
  --image-allowed-hosts bucket.s3.us-east-1.amazonaws.com
```

Send one conversation as `input` to `/v1/embeddings` (or as `inputs` to `/embed`):

```json
{"input":[{"role":"user","content":[
  {"type":"image_url","image_url":{"url":"https://bucket.s3.us-east-1.amazonaws.com/image.png?SIGNED_QUERY"}},
  {"type":"text","text":"Represent the product in this image."}
]}]}
```

A string or list of independent strings remains valid. Ordered user/assistant messages
produce one embedding. Images may be base64 PNG/JPEG/WebP data URLs; remote URLs require
an exact configured HTTPS hostname. Redirects and private network addresses are rejected.
The initial limits are four images per conversation, 20 MiB per image, 16 megapixels per
image and a 30-second preprocessing timeout. `--image-memory-budget-mib` defaults to 512.
Only images decoded to eight-bit samples are accepted; WebP metadata is capped at 64 KiB
per chunk. Oversized image token spans are rejected even with truncation enabled.

This model currently requires FP16 and returns pooled embeddings only. RadixMLP is disabled
for Qwen3-VL because token identity alone does not identify image content. The text-only
Qwen3 embedding models retain their existing behavior and do not accept images.

### Rune text and image decisions

Use the existing `/v1/systemone` endpoint with a Candle CUDA/FlashAttention build:

```shell
infinity_v3 --model-id michaelfeil/rune-26b-a4b \
  --decision-protocol rune --dtype bfloat16 --radix-mlp-threshold 0.92
```

Supports text states and native `state.messages` with user/assistant text and user
images. Questions support `choice`, `noul`, and `score` with string instructions and
descriptions. Rune uses first-option-token probabilities and its own confidence formulas.

```shell
curl http://localhost:8080/v1/systemone \
  -H 'Content-Type: application/json' -d '{
    "state": {"messages": [{"role": "user", "content": [
      {"type": "text", "text": "Inspect the parcel."},
      {"type": "image_url", "image_url": {"url": "https://images.example.com/parcel.png"}}
    ]}]},
    "questions": {
      "damage": {"type": "noul", "instructions": "Is the parcel visibly damaged?",
        "criteria": {"false": "No visible damage", "true": "Visible damage"}},
      "condition": {"type": "choice", "instructions": "Describe the parcel condition.",
        "criteria": {"intact": "Intact", "torn": "Torn packaging", "crushed": "Crushed packaging"}}
    }
  }'
```

Add `--image-allowed-hosts images.example.com` to allow that exact remote host, or
supply inline `data:image/png;base64,...` URLs. Images use the same bounded resolver
as Qwen3-VL: PNG, JPEG, and WebP; at most four images per request, 20 MiB encoded per
image, 16 megapixels decoded, and a shared 512 MiB host processing budget by default
(`--image-memory-budget-mib`). Large inline images also need an appropriate
`--payload-limit` for their JSON/base64 body (default: 2,000,000 bytes).
JPEG decoding can differ numerically from Pillow.
The checkpoint's image processor determines the patch budget (280 soft tokens by
default for `michaelfeil/rune-26b-a4b`). Images precede the decision prompt; their
ordered `[image N]` references remain in the serialized state, with URLs removed.

Images are prepared once per request and shared across its questions. RadixMLP
remains enabled for text and for batches whose questions share identical prepared
images and complete image prefixes. Different images never share states based on
placeholder token IDs. Image features are encoded once per unique prepared image
within a batch. Image blocks use Gemma4's bidirectional local attention; text and
global layers retain causal attention.

Over-budget image/text prompts are rejected without truncation. Audio, video,
assistant images, non-auto image detail, and `head_max_len` are unsupported. Rune
images currently target the 26B-A4B vision configuration with no per-layer inputs.
BF16 decision probabilities can vary with batch shape, particularly for ambiguous
questions; Radix on/off equivalence is checked separately in the reference script.

### OneJev decisions

Qwen3.5-based OneJev checkpoints use the same `/v1/systemone` request and response
format. Start with `--model-id OmniJev/OneJev-0.8B --decision-protocol onejev
--dtype bfloat16`. The checkpoint's native chat template renders the trained
`qev-labels-v2` prompt with thinking disabled. RadixMLP remains enabled.
Choice, noul, and score are supported. Native `state.messages` accepts user
`image_url` parts using the same format, host allowlist and limits as Rune above.
Images are processed once per request and their GPU features are shared across
batched questions. The current vision tower allows up to 1024 image tokens per
image (4096 patches); larger images return 422 to bound dense vision attention.
With BF16 text, the Qwen3-VL vision tower uses FP16 projections for accuracy.
Explicit FP32 loading is preserved. The tower feeds Qwen3.5's multimodal
positions and hybrid attention backbone; image prefixes can use RadixMLP when their content
and positions match. Audio and video are unsupported.
Prompts exceeding the token limit return 422 without truncation. Limits: 255
choice options, 10 score levels, and 512 total options per request.

### Using SPLADE pooling

You can choose to activate SPLADE pooling for Bert and Distilbert MaskedLM architectures:

```shell
model=naver/efficient-splade-VI-BT-large-query
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data infinity_v3:latest --model-id $model --pooling splade
```

Once you have deployed the model you can use the `/embed_sparse` endpoint to get the sparse embedding:

```bash
curl 127.0.0.1:8080/embed_sparse \
    -X POST \
    -d '{"inputs":"I like you."}' \
    -H 'Content-Type: application/json'
```

### Distributed Tracing

`infinity_v3` is instrumented with distributed tracing using OpenTelemetry. You can use this feature
by setting the address to an OTLP collector with the `--otlp-endpoint` argument.

## Local install

### CPU

Candle CPU inference uses packed (ragged) attention by default for float32 and
float16 BERT/RoBERTa, DistilBERT, Jina, GTE, Nomic, ModernBERT, Qwen2/Qwen3,
and Llama/Mistral models. Tokens remain unpadded throughout these model paths;
attention respects each sequence's boundaries. Padded and ONNX backends are removed.
MPNet, DistilBERT classification, CPU BF16, and Metal are unsupported.
Gemma3 and Gemma4 require CUDA BF16 with FlashAttention v2. Packed Llama requires
bias-free projections and the standard head dimension. The CPU batch-size cap
remains four sequences.


You can also opt to install `infinity_v3` locally.

First [install Rust](https://rustup.rs/):

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then run:

```shell
# On x86 with Candle
cargo install --path router -F candle
# On x86 with Candle and Intel MKL
cargo install --path router -F mkl
# On M1 or M2
cargo install --path router -F metal
```

You can now launch infinity_v3 on CPU with:

```shell
model=Qwen/Qwen3-Embedding-0.6B

infinity_v3 --model-id $model --port 8080
```

**Note:** on some machines, you may also need the OpenSSL libraries and gcc. On Linux machines, run:

```shell
sudo apt-get install libssl-dev gcc -y
```

### CUDA

GPUs with CUDA compute capabilities < 7.5 are not supported (V100, Titan V, GTX 1000 series, ...).

Make sure you have CUDA and the nvidia drivers installed. NVIDIA drivers on your device need to be compatible with CUDA
version 12.2 or higher.
You also need to add the nvidia binaries to your path:

```shell
export PATH=$PATH:/usr/local/cuda/bin
```

Then run:

```shell
# This can take a while as we need to compile a lot of CUDA kernels

# On Turing GPUs (T4, RTX 2000 series ... )
cargo install --path router -F candle-cuda-turing

# On Ampere and Hopper
cargo install --path router -F candle-cuda
```

You can now launch infinity_v3 on GPU with:

```shell
model=Qwen/Qwen3-Embedding-0.6B

infinity_v3 --model-id $model --port 8080
```

## Docker build

You can build the CPU container with:

```shell
docker build -t infinity_v3:cpu-latest .
```

To build the CUDA containers, you need to know the compute cap of the GPU you will be using
at runtime.

Then you can build the container with:

```shell
# Get submodule dependencies
git submodule update --init

# Example for Turing (T4, RTX 2000 series, ...)
runtime_compute_cap=75

# Example for A100
runtime_compute_cap=80

# Example for A10
runtime_compute_cap=86

# Example for Ada Lovelace (RTX 4000 series, ...)
runtime_compute_cap=89

# Example for H100
runtime_compute_cap=90

docker build . -f Dockerfile-cuda --build-arg CUDA_COMPUTE_CAP=$runtime_compute_cap -t infinity_v3:latest
```

### Apple M1/M2 arm64 architectures

#### DISCLAIMER

As explained here [MPS-Ready, ARM64 Docker Image](https://github.com/pytorch/pytorch/issues/81224), Metal / MPS is not
supported via Docker. As such inference will be CPU bound and most likely pretty slow when using this docker image on an
M1/M2 ARM CPU.

```
docker build . -f Dockerfile --platform=linux/arm64 -t infinity_v3:cpu-latest
```

## Examples

These upstream TEI examples may need adaptation for this fork:

- [Set up an Inference Endpoint with TEI](https://huggingface.co/learn/cookbook/automatic_embedding_tei_inference_endpoints)
- [RAG containers with TEI](https://github.com/plaggy/rag-containers)

### Automatic attention selection

CUDA images for SM80, SM86, SM89, SM90 and SM120 build and include an architecture-specific FA4 native bundle.
`ATTN_BACKEND=auto` is the default. On a device matching the linked bundle, builds containing FA4
select it for validated FP16/BF16 packed attention shapes: head dimension 64
with equal query/KV heads and global or bidirectional window masks; dimension
128 with causal 4:1 GQA, or global bidirectional 2:1 GQA in BF16 only. Unsupported shapes,
ALiBi, mismatched devices, and builds without FA4 retain the existing attention backend.
EmbeddingGemma's dimension 256 remains on FA2. Set `ATTN_BACKEND=fa2` to disable
FA4, or `ATTN_BACKEND=fa4` to explicitly request the same supported FA4 paths
(with fallback for unsupported shapes). The environment selection is cached
on first use; restart the process to change it. FP8 is independently opt-in.

When FA4 is enabled, models using the shared flash-attention dispatcher register
their variable-length boundaries once per batch. The current native bundles support d64 MHA global
and two-sided local attention, d128 causal GQA with a 4:1 query/KV head ratio,
and d128 global GQA with a 2:1 ratio (including Voyage-4-nano). Use BF16 for
Voyage to avoid FP16 non-finite outputs at long context.
Unsupported devices, masks (including ALiBi), shapes, and layouts use the existing
backend. FA4 execution errors propagate. DeBERTa is built for SM90 and SM120;
SM120 DeBERTa exports are verified by offline compilation and linking, with
end-to-end correctness checked on SM90. Dynamic FP8 row scaling remains
Hopper-only; SM100/110 FA4 bundles are not packaged.

Source builds use `--features fa4` and `FA4_NATIVE_LIB_DIR` pointing to the native
bundle; include its shared libraries in `LD_LIBRARY_PATH`. `experimental-fa4`
remains an alias. `scripts/build-fa4-native.sh` builds the pinned bundle without
a GPU; pass the compute capability as its third argument (default 90). Python
dependencies stay in the build environment.

A10G (SM86), L4 (SM89), and RTX Pro 6000 Blackwell (SM120) passed direct
FA2/FA4 embedding comparisons for BGE-large, ModernBERT embed base and Qwen3
Embedding 4B in FP16/BF16. SM80 is build-verified only. These comparisons do
not qualify every model or attention shape.

**Quality qualification:** the pinned bundle aligns FA4's softmax denominator
reduction order and causal d128 key-tile boundaries with FA2. This fixes the observed ModernBERT discrepancy: raw and
pooled outputs match FA2 bitwise on the tested FP16/BF16 cases, and the measured
STS-B, SciFact and NFCorpus score differences disappear. The causal tile change
also removes the tested Qwen3-8B long-input differences in both precisions.
Auto selects only qualified shapes and dtypes;
these checks do not establish equivalence for every model, shape or architecture.

### Cloudflare Clef typed decisions

Clef and Clef-flash use the dense Qwen3.5 Candle backend and their joint schema
head on `/v1/systemone`. All questions in a request share one backbone pass.
For text inputs, launch with `--model-id Cloudflare/clef-flash
--decision-protocol clef --dtype bfloat16`. Standard sharded safetensors and
`joint_head.safetensors` are loaded directly. Images/video are not supported yet;
over-budget prompts return 422 rather than silently truncating the state.

### Perplexity typed decisions

For text decisions, launch `perplexity-ai/pplx-decider-v1-27b` or
`perplexity-ai/pplx-decider-v1.1-27b` with
`--decision-protocol pplx --dtype bfloat16` and use `/v1/systemone`.
The shorthand alias selects the v1.1 checkpoint and Pplx protocol automatically:

```bash
infinity_v3 --model-id pplx-decider-v1.1-27b --dtype bfloat16
```

The Qwen3.5 Candle backbone loads the checkpoint's `readout.safetensors` and
applies its saved calibration temperature. Choice, noul and score questions use
the upstream prompt and answer formulas, with up to 255 options per question.
Each question has a separate branch in the shared batch. The saved
`decision_config.json` attention mode is respected: v1.1 uses noncausal full
attention while linear-attention layers retain their native behavior. RadixMLP
prefix folding is disabled for this mode; it remains available for causal v1
checkpoints, which omit the attention-mode field. Unknown attention modes and
pooling modes other than `last` are rejected.
Prompts exceeding 8192 tokens (or a smaller `max_len`) return 422.
Image/video inputs are not supported by this protocol yet.

### D3 typed multimodal decisions

`vllm-sr/d3` serves choice, noul and score questions on `/v1/systemone` using the
LibTorch backend in BF16. Both `--model-id d3` and `--model-id vllm-sr/d3`
select the native D3 protocol automatically. For a local checkpoint, pass
`--decision-protocol d3` explicitly:

```bash
infinity_v3 --backend libtorch --torch-device cuda \
  --model-id vllm-sr/d3 --dtype bfloat16 --device-id 0
```

D3 uses its checkpoint chat template, noncausal full-attention layers, native
causal Gated DeltaNet layers, and an FP32 255-row readout. Prefix folding is
disabled for noncausal inference. Each question runs over the complete state
and options; probabilities normalize only over that question's valid options,
with checkpoint temperature applied once. Choices return entropy confidence;
score questions accept 2–10 distinct levels and return the expected zero-based
score, option probabilities, entropy confidence, and a legend. The adapter
matches the default upstream inference mode without permutation averaging.

The state can be text or any JSON value. Top-level `images` and `videos` are
ordered arrays of inline data URLs or HTTPS URLs; all questions see the same
media. Images precede videos, followed by the state and question text. PNG,
JPEG and WebP images are resized to 65,536–1,638,400 pixels. EXIF orientation is
ignored, matching the published D3 loader. Video uses FFmpeg:
2 fps, 4–32 uniformly selected source frames (or every frame of shorter clips),
paired in time with timestamps and at most 200,704 pixels per frame. Videos are
limited to 300 seconds, 32 MB each, and 16,384 visual tokens in total. The
server's configured media count, byte, pixel, and memory limits also apply.
Remote downloads require `--image-allowed-hosts`; local filesystem paths and
raw frame arrays are not accepted by the HTTP API.

```bash
curl http://localhost:3000/v1/systemone \
  -H 'Content-Type: application/json' -H 'Accept: application/json' -d '{
  "state": {"request": "My order arrived damaged", "has_receipt": true},
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this?",
      "criteria": {"returns": "Damaged deliveries", "billing": "Payments"}},
    "receipt": {"type": "noul", "instructions": "Does the customer have a receipt?"}
  }
}'
```

Add `"images": ["data:image/png;base64,..."]` or
`"videos": ["data:video/mp4;base64,..."]` to the same request for media.
Prompts exceeding `max_len` or the server's token budget are rejected with 422;
text and media are never silently truncated. D3 reports zero output tokens.

D3 qualification fixtures can be generated using the checkpoint's pinned Python
environment:

```bash
python scripts/d3-reference.py --checkpoint /path/to/d3 \
  --output /tmp/d3-reference --device cuda:0
```
Set `D3_REFERENCE_DIR` to the output directory and `D3_HTTP_URL` to a local D3
`/v1/systemone` endpoint. Run the single ignored
`deployed_checkpoint_matches_upstream_answers` router test with `--ignored` to
compare text, image, video, and mixed-media answers. This smoke check permits
small BF16 kernel differences; it does not establish bitwise parity or
benchmark-wide accuracy. HTTP callers requesting JSON should send
`Accept: application/json`.

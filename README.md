<div align="center">

# Text Embeddings Inference

<a href="https://github.com/huggingface/text-embeddings-inference">
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/huggingface/text-embeddings-inference?style=social">
</a>
<a href="https://huggingface.github.io/text-embeddings-inference">
  <img alt="Swagger API documentation" src="https://img.shields.io/badge/API-Swagger-informational">
</a>

A blazing fast inference solution for text embeddings models.

Benchmark for [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) on an NVIDIA A10 with a sequence
length of 512 tokens:

<p>
  <img src="assets/bs1-lat.png" width="400" />
  <img src="assets/bs1-tp.png" width="400" />
</p>
<p>
  <img src="assets/bs32-lat.png" width="400" />
  <img src="assets/bs32-tp.png" width="400" />
</p>

</div>

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
    - [gRPC](#grpc)
- [Local Install](#local-install)
- [Docker Build](#docker-build)
    - [Apple M1/M2 Arm](#apple-m1m2-arm64-architectures)
- [Examples](#examples)

Text Embeddings Inference (TEI) is a toolkit for deploying and serving open source text embeddings and sequence
classification models. TEI enables high-performance extraction for the most popular models, including FlagEmbedding,
Ember, GTE and E5. TEI implements many features such as:

* No model graph compilation step
* Small docker images and fast boot times. Get ready for true serverless!
* Token based dynamic batching
* Optimized transformers code for inference using [Flash Attention](https://github.com/HazyResearch/flash-attention),
  [Candle](https://github.com/huggingface/candle)
  and [cuBLASLt](https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api)
* [Safetensors](https://github.com/huggingface/safetensors) weight loading
* Production ready (distributed tracing with Open Telemetry, Prometheus metrics)

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
batch limits. Set `TEI_EARLY_DISPATCH_TOKENS` to override the target, or `0` to disable
it. Single-GPU batching keeps its existing behavior unless this variable is set.
Per-replica batch, token, and inference-duration metrics are exposed on `/metrics`.
Each replica logs inference throughput every 100 batches.

### BPE tokenization

Embedding text inputs automatically use `fastokens-b10` when the tokenizer configuration is supported. No opt-in flag or environment variable is required. WordPiece and Unigram models, paired or token-ID inputs, classification/NER, `/tokenize`, and `/decode` keep using Hugging Face Tokenizers. Truncation and special-token processing also remain with Hugging Face. Fast encoding uses a shared CPU pool bounded by `--tokenization-workers`.

## Get Started

### Supported Models

#### Text Embeddings

Text Embeddings Inference currently supports Nomic, BERT, CamemBERT, XLM-RoBERTa models with absolute positions, JinaBERT
model with Alibi positions and Mistral, Alibaba GTE, Qwen2 models with Rope positions, ModernBERT, Qwen3, Gemma3, and dense Gemma4 text models.

Below are some examples of the currently supported models:

| MTEB Rank | Model Size             | Model Type     | Model ID                                                                                         |
|-----------|------------------------|----------------|--------------------------------------------------------------------------------------------------|
| 2         | 7.57B (Very Expensive) | Qwen3          | [Qwen/Qwen3-Embedding-8B](https://hf.co/Qwen/Qwen3-Embedding-8B)                                 |
| 3         | 4.02B (Very Expensive) | Qwen3          | [Qwen/Qwen3-Embedding-4B](https://hf.co/Qwen/Qwen3-Embedding-4B)                                 |
| 4         | 509M                   | Qwen3          | [Qwen/Qwen3-Embedding-0.6B](https://hf.co/Qwen/Qwen3-Embedding-0.6B)                             |
| 6         | 7.61B (Very Expensive) | Qwen2          | [Alibaba-NLP/gte-Qwen2-7B-instruct](https://hf.co/Alibaba-NLP/gte-Qwen2-7B-instruct)             |
| 7         | 560M                   | XLM-RoBERTa    | [intfloat/multilingual-e5-large-instruct](https://hf.co/intfloat/multilingual-e5-large-instruct) |
| 8         | 308M                   | Gemma3         | [google/embeddinggemma-300m](https://hf.co/google/embeddinggemma-300m) (gated)                   |
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

Text Embeddings Inference currently supports CamemBERT, and XLM-RoBERTa Sequence Classification models with absolute positions.

Below are some examples of the currently supported models:

| Task               | Model Type  | Model ID                                                                                                        |
|--------------------|-------------|-----------------------------------------------------------------------------------------------------------------|
| Re-Ranking         | XLM-RoBERTa | [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)                                       |
| Re-Ranking         | XLM-RoBERTa | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)                                         |
| Re-Ranking         | GTE         | [Alibaba-NLP/gte-multilingual-reranker-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base) |
| Re-Ranking         | ModernBert  | [Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) |
| Sentiment Analysis | RoBERTa     | [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)                     |

### Docker

```shell
model=Qwen/Qwen3-Embedding-0.6B
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.8 --model-id $model
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
$ text-embeddings-router --help
Text Embedding Webserver

Usage: text-embeddings-router [OPTIONS]

Options:
      --model-id <MODEL_ID>
          The name of the model to load. Can be a MODEL_ID as listed on <https://hf.co/models> like `BAAI/bge-large-en-v1.5`. Or it can be a local directory containing the necessary files as saved by `save_pretrained(...)` methods of transformers

          [env: MODEL_ID=]
          [default: BAAI/bge-large-en-v1.5]

      --revision <REVISION>
          The actual revision of the model if you're referring to a model on the hub. You can use a specific commit id or a branch like `refs/pr/2`

          [env: REVISION=]

      --tokenization-workers <TOKENIZATION_WORKERS>
          Optionally control the number of tokenizer workers used for payload tokenization, validation and truncation. Default to the number of CPU cores on the machine

          [env: TOKENIZATION_WORKERS=]

      --dtype <DTYPE>
          Model dtype. Auto selects bfloat16 from model config when supported, otherwise the backend default

          [env: DTYPE=]
          [default: auto]
          [possible values: auto, float16, float32, bfloat16]

          Auto prefers config.json's dtype over torch_dtype and selects bfloat16 when configured.
          Otherwise it keeps the backend default (float16 for Candle CUDA). MKL, Accelerate,
          and gemma3_text retain their float32 defaults. Explicit dtype values override auto.

      --pooling <POOLING>
          Optionally control the pooling method for embedding models.

          If `pooling` is not set, the pooling configuration will be parsed from the model `1_Pooling/config.json` configuration.

          If `pooling` is set, it will override the model pooling configuration

          [env: POOLING=]

          Possible values:
          - cls:        Select the CLS token as embedding
          - mean:       Apply Mean pooling to the model embeddings
          - splade:     Apply SPLADE (Sparse Lexical and Expansion) to the model embeddings. This option is only available if the loaded model is a `ForMaskedLM` Transformer model
          - last-token: Select the last token as embedding

      --max-concurrent-requests <MAX_CONCURRENT_REQUESTS>
          The maximum amount of concurrent requests for this particular deployment. Having a low limit will refuse clients requests instead of having them wait for too long and is usually good to handle backpressure correctly

          [env: MAX_CONCURRENT_REQUESTS=]
          [default: 512]

      --max-batch-tokens <MAX_BATCH_TOKENS>
          **IMPORTANT** This is one critical control to allow maximum usage of the available hardware.

          This represents the total amount of potential tokens within a batch.

          For `max_batch_tokens=1000`, you could fit `10` queries of `total_tokens=100` or a single query of `1000` tokens.

          Overall this number should be the largest possible until the model is compute bound. Since the actual memory overhead depends on the model implementation, text-embeddings-inference cannot infer this number automatically.

          [env: MAX_BATCH_TOKENS=]
          [default: 16384]

      --max-batch-requests <MAX_BATCH_REQUESTS>
          Optionally control the maximum number of individual requests in a batch

          [env: MAX_BATCH_REQUESTS=]

      --max-client-batch-size <MAX_CLIENT_BATCH_SIZE>
          Control the maximum number of inputs that a client can send in a single request

          [env: MAX_CLIENT_BATCH_SIZE=]
          [default: 32]

      --max-decision-questions <MAX_DECISION_QUESTIONS>
          Maximum number of questions in one /v1/systemone request

          Independent of max-client-batch-size; model option budgets still apply

          [env: MAX_DECISION_QUESTIONS=]
          [default: 512]

      --auto-truncate
          Automatically truncate inputs that are longer than the maximum supported size

          Unused for gRPC servers

          [env: AUTO_TRUNCATE=]

      --default-prompt-name <DEFAULT_PROMPT_NAME>
          The name of the prompt that should be used by default for encoding. If not set, no prompt will be applied.

          Must be a key in the `sentence-transformers` configuration `prompts` dictionary.

          For example if ``default_prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?" because the prompt text will be prepended before any text to encode.

          The argument '--default-prompt-name <DEFAULT_PROMPT_NAME>' cannot be used with '--default-prompt <DEFAULT_PROMPT>`

          [env: DEFAULT_PROMPT_NAME=]

      --default-prompt <DEFAULT_PROMPT>
          The prompt that should be used by default for encoding. If not set, no prompt will be applied.

          For example if ``default_prompt`` is "query: " then the sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?" because the prompt text will be prepended before any text to encode.

          The argument '--default-prompt <DEFAULT_PROMPT>' cannot be used with '--default-prompt-name <DEFAULT_PROMPT_NAME>`

          [env: DEFAULT_PROMPT=]

      --dense-path <DENSE_PATH>
          Optionally, define the path to the Dense module required for some embedding models.

          Some embedding models require an extra `Dense` module which contains a single Linear layer and an activation function. By default, those `Dense` modules are stored under the `2_Dense` directory, but there might be cases where different `Dense` modules are provided, to convert the pooled embeddings into different dimensions, available as `2_Dense_<dims>` e.g. https://huggingface.co/NovaSearch/stella_en_400M_v5.

          Note that this argument is optional, only required to be set if the path to the `Dense` module is other than `2_Dense`. And it also applies when leveraging the `candle` backend.

          [env: DENSE_PATH=]
          [default: 2_Dense]

      --hf-token <HF_TOKEN>
          Your Hugging Face Hub token

          [env: HF_TOKEN=]

      --hostname <HOSTNAME>
          The IP address to listen on

          [env: HOSTNAME=]
          [default: 0.0.0.0]

      -p, --port <PORT>
          The port to listen on

          [env: PORT=]
          [default: 3000]

      --uds-path <UDS_PATH>
          The name of the unix socket some text-embeddings-inference backends will use as they communicate internally with gRPC

          [env: UDS_PATH=]
          [default: /tmp/text-embeddings-inference-server]

      --huggingface-hub-cache <HUGGINGFACE_HUB_CACHE>
          The location of the huggingface hub cache. Used to override the location if you want to provide a mounted disk for instance

          [env: HUGGINGFACE_HUB_CACHE=]

      --payload-limit <PAYLOAD_LIMIT>
          Payload size limit in bytes

          Default is 2MB

          [env: PAYLOAD_LIMIT=]
          [default: 2000000]

      --api-key <API_KEY>
          Set an api key for request authorization.

          By default the server responds to every request. With an api key set, the requests must have the Authorization header set with the api key as Bearer token.

          [env: API_KEY=]

      --json-output
          Outputs the logs in JSON format (useful for telemetry)

          [env: JSON_OUTPUT=]

      --disable-spans
          [env: DISABLE_SPANS=]

      --otlp-endpoint <OTLP_ENDPOINT>
          The grpc endpoint for opentelemetry. Telemetry is sent to this endpoint as OTLP over gRPC. e.g. `http://localhost:4317`

          [env: OTLP_ENDPOINT=]

      --otlp-service-name <OTLP_SERVICE_NAME>
          The service name for opentelemetry. e.g. `text-embeddings-inference.server`

          [env: OTLP_SERVICE_NAME=]
          [default: text-embeddings-inference.server]

      --prometheus-port <PROMETHEUS_PORT>
          The Prometheus port to listen on

          [env: PROMETHEUS_PORT=]
          [default: 9000]

      --cors-allow-origin <CORS_ALLOW_ORIGIN>
          Unused for gRPC servers

          [env: CORS_ALLOW_ORIGIN=]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

### Docker Images

Text Embeddings Inference ships with multiple Docker images that you can use to target a specific backend:

| Architecture                        | Image                                                                   |
|-------------------------------------|-------------------------------------------------------------------------|
| CPU                                 | ghcr.io/huggingface/text-embeddings-inference:cpu-1.8                   |
| Volta                               | NOT SUPPORTED                                                           |
| Turing (T4, RTX 2000 series, ...)   | ghcr.io/huggingface/text-embeddings-inference:turing-1.8 (experimental) |
| Ampere 80 (A100, A30)               | ghcr.io/huggingface/text-embeddings-inference:1.8                       |
| Ampere 86 (A10, A40, ...)           | ghcr.io/huggingface/text-embeddings-inference:86-1.8                    |
| Ada Lovelace (RTX 4000 series, ...) | ghcr.io/huggingface/text-embeddings-inference:89-1.8                    |
| Hopper (H100)                       | ghcr.io/huggingface/text-embeddings-inference:hopper-1.8 (experimental) |

Turing uses packed FlashAttention v1 with float16. Models requiring sliding-window
attention or ALiBi require FlashAttention v2 on Ampere or newer GPUs.

### API documentation

You can consult the OpenAPI documentation of the `text-embeddings-inference` REST API using the `/docs` route.
The Swagger UI is also available
at: [https://huggingface.github.io/text-embeddings-inference](https://huggingface.github.io/text-embeddings-inference).

### Using a private or gated model

You have the option to utilize the `HF_TOKEN` environment variable for configuring the token employed by
`text-embeddings-inference`. This allows you to gain access to protected resources.

For example:

1. Go to https://huggingface.co/settings/tokens
2. Copy your CLI READ token
3. Export `HF_TOKEN=<your CLI READ token>`

or with Docker:

```shell
model=<your private model>
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
token=<your CLI READ token>

docker run --gpus all -e HF_TOKEN=$token -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.8 --model-id $model
```

### Air gapped deployment

To deploy Text Embeddings Inference in an air-gapped environment, first download the weights and then mount them inside
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
docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.8 --model-id /data/Qwen3-Embedding-0.6B
```

### Using Re-rankers models

`text-embeddings-inference` v0.4.0 added support for CamemBERT, RoBERTa, XLM-RoBERTa, and GTE Sequence Classification models.
Re-rankers models are Sequence Classification cross-encoders models with a single class that scores the similarity
between a query and a text.

See [this blogpost](https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83) by
the LlamaIndex team to understand how you can use re-rankers models in your RAG pipeline to improve
downstream performance.

```shell
model=BAAI/bge-reranker-large
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.8 --model-id $model
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

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.8 --model-id $model
```

Once you have deployed the model you can use the `predict` endpoint to get the emotions most associated with an input:

```bash
curl 127.0.0.1:8080/predict \
    -X POST \
    -d '{"inputs":"I like you."}' \
    -H 'Content-Type: application/json'
```

### Laya typed decisions (Jev API)

Serve `convaiinnovations/laya-typed-decisions` with the Candle HTTP build:

```shell
text-embeddings-router --model-id convaiinnovations/laya-typed-decisions
```

For CPU inference, add `--dtype float32`.

The existing API server exposes `POST /v1/systemone`. Questions use TEI's shared
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
Laya requires the Candle backend and the HTTP API.

Unknown request/question fields are rejected. Execution extensions such as
`think`, `mode`, `depends_on`, `ask_if`, and `alone` are not implemented.
Temperatures are clamped to [0.5, 5], matching upstream Laya. The published
checkpoint's `choice:11+` temperature is outside this range; confidence for that
bucket is not verified as calibrated. ModernBERT uses approximate (tanh) GELU;
the custom head retains ReLU and exact scorer/action GELU.

### Qwen3-VL image embeddings

With a Candle CUDA/FlashAttention build, serve `Qwen/Qwen3-VL-Embedding-2B`:

```shell
text-embeddings-router --model-id Qwen/Qwen3-VL-Embedding-2B \
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
text-embeddings-router --model-id michaelfeil/rune-26b-a4b \
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

### OneJev text decisions

Qwen3.5-based OneJev checkpoints use the same `/v1/systemone` request and response
format. Start with `--model-id OmniJev/OneJev-0.8B --decision-protocol onejev
--dtype bfloat16`. The checkpoint's native chat template renders the trained
`qev-labels-v2` prompt with thinking disabled. RadixMLP remains enabled.
Choice, noul, and score are supported; images, audio, and video are rejected.
Prompts exceeding the token limit return 422 without truncation. Limits: 255
choice options, 10 score levels, and 512 total options per request.

### Using SPLADE pooling

You can choose to activate SPLADE pooling for Bert and Distilbert MaskedLM architectures:

```shell
model=naver/efficient-splade-VI-BT-large-query
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.8 --model-id $model --pooling splade
```

Once you have deployed the model you can use the `/embed_sparse` endpoint to get the sparse embedding:

```bash
curl 127.0.0.1:8080/embed_sparse \
    -X POST \
    -d '{"inputs":"I like you."}' \
    -H 'Content-Type: application/json'
```

### Distributed Tracing

`text-embeddings-inference` is instrumented with distributed tracing using OpenTelemetry. You can use this feature
by setting the address to an OTLP collector with the `--otlp-endpoint` argument.

### gRPC

`text-embeddings-inference` offers a gRPC API as an alternative to the default HTTP API for high performance
deployments. The API protobuf definition can be
found [here](https://github.com/huggingface/text-embeddings-inference/blob/main/proto/tei.proto).

You can use the gRPC API by adding the `-grpc` tag to any TEI Docker image. For example:

```shell
model=Qwen/Qwen3-Embedding-0.6B
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.8-grpc --model-id $model
```

```shell
grpcurl -d '{"inputs": "What is Deep Learning"}' -plaintext 0.0.0.0:8080 tei.v1.Embed/Embed
```

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


You can also opt to install `text-embeddings-inference` locally.

First [install Rust](https://rustup.rs/):

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then run:

```shell
# On x86 with Candle
cargo install --path router -F candle
# On x86 with Intel backend
cargo install --path router -F mkl
# On M1 or M2
cargo install --path router -F metal
```

You can now launch Text Embeddings Inference on CPU with:

```shell
model=Qwen/Qwen3-Embedding-0.6B

text-embeddings-router --model-id $model --port 8080
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

You can now launch Text Embeddings Inference on GPU with:

```shell
model=Qwen/Qwen3-Embedding-0.6B

text-embeddings-router --model-id $model --port 8080
```

## Docker build

You can build the CPU container with:

```shell
docker build .
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

docker build . -f Dockerfile-cuda --build-arg CUDA_COMPUTE_CAP=$runtime_compute_cap
```

### Apple M1/M2 arm64 architectures

#### DISCLAIMER

As explained here [MPS-Ready, ARM64 Docker Image](https://github.com/pytorch/pytorch/issues/81224), Metal / MPS is not
supported via Docker. As such inference will be CPU bound and most likely pretty slow when using this docker image on an
M1/M2 ARM CPU.

```
docker build . -f Dockerfile --platform=linux/arm64
```

## Examples

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
on first use; restart the process to change it. FP8 is independently opt-in. The earlier
`TEI_ATTENTION_BACKEND` and `TEI_PERF_FA4` experimental controls are no longer used.

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

For text decisions, launch `perplexity-ai/pplx-decider-v1-27b` with
`--decision-protocol pplx --dtype bfloat16` and use `/v1/systemone`.
The Qwen3.5 Candle backbone loads the checkpoint's `readout.safetensors` and
applies its saved calibration temperature. Choice, noul and score questions use
the upstream prompt and answer formulas, with up to 255 options per question.
Each question has a separate branch in the shared batch; causal RadixMLP remains
available. Prompts exceeding 8192 tokens (or a smaller `max_len`) return 422.
Image/video inputs are not supported by this protocol yet.

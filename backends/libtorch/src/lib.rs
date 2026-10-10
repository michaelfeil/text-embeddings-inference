//! Experimental native packed inference. Python and exported TorchScript models are not required.
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::{
    ffi::{c_char, c_void, CStr, CString},
    fs,
    path::Path,
    ptr::NonNull,
};
use text_embeddings_backend_core::{
    Backend, BackendError, Batch, DecisionInput, DecisionOutput, Embedding, Embeddings, ModelType,
    Pool, Predictions, TokenPredictions,
};

#[repr(C)]
struct NativeConfig {
    hidden: i64,
    heads: i64,
    layers: i64,
    intermediate: i64,
    vocab: i64,
    positions: i64,
    types: i64,
    epsilon: f64,
    activation: i32,
}
#[repr(C)]
struct NativeImage {
    pixels: *const f32,
    rows: i64,
    patch_dim: i64,
    grid: [i64; 3],
    merge_size: i64,
    token_start: i64,
    token_count: i64,
    sequence_start: i64,
}
#[repr(C)]
struct NativeAudio {
    values: *const f32,
    mask: *const u8,
    frames: i64,
    feature_size: i64,
    token_start: i64,
    token_count: i64,
}
#[repr(C)]
struct NativeDecisionField {
    kind: i64,
    question_start: i64,
    question_end: i64,
    options: *const i64,
    option_count: usize,
}
#[repr(C)]
struct NativeDecision {
    kind: i32,
    question_type: i64,
    markers: *const i64,
    marker_count: usize,
    token_ids: *const i64,
    token_count: usize,
    fields: *const NativeDecisionField,
    field_count: usize,
}
struct OwnedDecision {
    kind: i32,
    question_type: i64,
    markers: Vec<i64>,
    token_ids: Vec<i64>,
    fields: Vec<NativeDecisionField>,
    // Own the backing storage referenced by each native field descriptor.
    _spans: Vec<Vec<i64>>,
}
impl OwnedDecision {
    fn new(input: &DecisionInput) -> Result<Self, BackendError> {
        let mut result = Self {
            kind: 3,
            question_type: 0,
            markers: vec![],
            token_ids: vec![],
            fields: vec![],
            _spans: vec![],
        };
        match input {
            DecisionInput::Laya {
                question_type,
                markers,
            } => {
                result.kind = 0;
                result.question_type = i64::try_from(*question_type).map_err(inference)?;
                result.markers = markers
                    .iter()
                    .map(|&n| i64::try_from(n).map_err(inference))
                    .collect::<Result<_, _>>()?;
            }
            DecisionInput::OptionTokens { token_ids } => {
                result.kind = 1;
                result.token_ids = token_ids.iter().map(|&n| i64::from(n)).collect();
            }
            DecisionInput::Clef { fields } => {
                result.kind = 2;
                for field in fields {
                    let spans = field
                        .options
                        .iter()
                        .flat_map(|&(a, b)| [a, b])
                        .map(|n| i64::try_from(n).map_err(inference))
                        .collect::<Result<Vec<_>, _>>()?;
                    result.fields.push(NativeDecisionField {
                        kind: i64::try_from(field.kind).map_err(inference)?,
                        question_start: i64::try_from(field.question.0).map_err(inference)?,
                        question_end: i64::try_from(field.question.1).map_err(inference)?,
                        options: spans.as_ptr(),
                        option_count: field.options.len(),
                    });
                    result._spans.push(spans);
                }
            }
            DecisionInput::Warmup => {}
        }
        Ok(result)
    }
    fn native(&self) -> NativeDecision {
        NativeDecision {
            kind: self.kind,
            question_type: self.question_type,
            markers: self.markers.as_ptr(),
            marker_count: self.markers.len(),
            token_ids: self.token_ids.as_ptr(),
            token_count: self.token_ids.len(),
            fields: self.fields.as_ptr(),
            field_count: self.fields.len(),
        }
    }
}
extern "C" {
    fn tei_error() -> *const c_char;
    fn tei_device_count(device: *const c_char) -> i64;
    fn tei_create(config: *const NativeConfig, device: *const c_char, dtype: i32) -> *mut c_void;
    fn tei_weight(
        handle: *mut c_void,
        name: *const c_char,
        data: *const u8,
        shape: *const i64,
        rank: usize,
        dtype: i32,
    ) -> i32;
    fn tei_option(handle: *mut c_void, key: *const c_char, value: *const c_char) -> i32;
    fn tei_output_width(handle: *mut c_void) -> i64;
    fn tei_pooled_width(handle: *mut c_void) -> i64;
    fn tei_classification_width(handle: *mut c_void) -> i64;
    fn tei_graph_count(handle: *mut c_void) -> i64;
    fn tei_ready(handle: *mut c_void) -> i32;
    fn tei_decision_counts(
        handle: *mut c_void,
        requests: *const NativeDecision,
        count: usize,
        counts: *mut i64,
    ) -> i32;
    fn tei_decide(
        handle: *mut c_void,
        ids: *const i64,
        types: *const i64,
        positions: *const i64,
        cumulative: *const i32,
        batch: i64,
        max_sequence: i64,
        requests: *const NativeDecision,
        logits: *mut f32,
        capacity: usize,
        actions: *mut f32,
        media_positions: *const i64,
        images: *const NativeImage,
        image_count: usize,
        audios: *const NativeAudio,
        audio_count: usize,
    ) -> i32;
    fn tei_forward(
        handle: *mut c_void,
        ids: *const i64,
        types: *const i64,
        positions: *const i64,
        cumulative: *const i32,
        batch: i64,
        max_sequence: i64,
        pool: i32,
        pooled: *const i64,
        pooled_count: usize,
        raw: *const i64,
        raw_count: usize,
        output: *mut f32,
        capacity: usize,
        media_positions: *const i64,
        images: *const NativeImage,
        image_count: usize,
        audios: *const NativeAudio,
        audio_count: usize,
    ) -> i32;
    fn tei_destroy(handle: *mut c_void);
}

fn native_error() -> String {
    // SAFETY: the bridge owns a thread-local, NUL-terminated string valid until the next call.
    unsafe { CStr::from_ptr(tei_error()).to_string_lossy().into_owned() }
}
fn start(error: impl ToString) -> BackendError {
    BackendError::Start(error.to_string())
}
fn inference(error: impl ToString) -> BackendError {
    BackendError::Inference(error.to_string())
}

pub fn device_count(device: &str) -> Result<usize, BackendError> {
    let device = CString::new(device).map_err(start)?;
    // SAFETY: the C string is valid for the duration of the synchronous call.
    let count = unsafe { tei_device_count(device.as_ptr()) };
    usize::try_from(count).map_err(|_| start(native_error()))
}

#[derive(Deserialize)]
struct Config {
    model_type: String,
    hidden_size: i64,
    num_attention_heads: i64,
    num_hidden_layers: i64,
    intermediate_size: i64,
    vocab_size: i64,
    max_position_embeddings: i64,
    type_vocab_size: i64,
    layer_norm_eps: f64,
    hidden_act: String,
    #[serde(default)]
    is_decoder: bool,
    #[serde(default)]
    add_cross_attention: bool,
    #[serde(default)]
    position_embedding_type: Option<String>,
}

pub struct LibtorchBackend {
    handle: NonNull<c_void>,
    config: Config,
    pool: Pool,
    output_width: usize,
    pooled_width: usize,
    classification_width: usize,
    decision: bool,
}
// SAFETY: the handle owns its tensors and is used by TEI's single backend worker.
// It can move between threads; it is deliberately not Sync. InferenceMode is set per native call.
unsafe impl Send for LibtorchBackend {}

impl Drop for LibtorchBackend {
    fn drop(&mut self) {
        // SAFETY: this handle is uniquely owned and destroyed exactly once.
        unsafe { tei_destroy(self.handle.as_ptr()) };
    }
}

impl LibtorchBackend {
    /// Number of captured packed shapes; zero also covers eager fallback.
    pub fn cuda_graph_count(&self) -> usize {
        // SAFETY: this uniquely owned handle stays live throughout the synchronous query.
        unsafe { tei_graph_count(self.handle.as_ptr()) as usize }
    }
    pub fn new(
        path: &Path,
        dtype: &str,
        model_type: ModelType,
        device: &str,
    ) -> Result<Self, BackendError> {
        Self::new_with_dense_paths(path, dtype, model_type, device, None)
    }
    pub fn new_with_dense_paths(
        path: &Path,
        dtype: &str,
        model_type: ModelType,
        device: &str,
        dense_paths: Option<Vec<String>>,
    ) -> Result<Self, BackendError> {
        if device != "cpu" && device != "auto" && !device.starts_with("cuda:") {
            return Err(start("LibTorch 2.14.1 varlen attention requires CUDA; only CPU reference and CUDA are supported"));
        }
        if device.starts_with("cuda:") && !matches!(dtype, "float16" | "bfloat16") {
            return Err(start("CUDA varlen attention requires float16 or bfloat16"));
        }
        let pool = match &model_type {
            ModelType::Embedding(pool) => pool.clone(),
            ModelType::Classifier => Pool::Cls,
            ModelType::Decision => Pool::LastToken,
        };
        let decision = model_type == ModelType::Decision;
        let read_config = |name: &str| -> Result<serde_json::Value, BackendError> {
            serde_json::from_slice(&fs::read(path.join(name)).map_err(start)?).map_err(start)
        };
        let laya = decision && path.join("rl_agent_config.json").exists();
        let mut options = read_config(if laya {
            "encoder/config.json"
        } else {
            "config.json"
        })?;
        if laya {
            if options["model_type"] != "modernbert" {
                return Err(start("Laya requires a ModernBERT encoder"));
            }
            for (kind, field) in [
                ("full_attention", "global_rope_theta"),
                ("sliding_attention", "local_rope_theta"),
            ] {
                if options.get(field).is_none() {
                    options[field] = options["rope_parameters"][kind]["rope_theta"].clone();
                }
            }
        }
        let decision_kind = if decision {
            let (kind, config) = if laya {
                ("laya", read_config("rl_agent_config.json")?)
            } else if path.join("decision_config.json").exists() {
                ("pplx", read_config("decision_config.json")?)
            } else if path.join("joint_head_config.json").exists() {
                ("clef", read_config("joint_head_config.json")?)
            } else {
                ("option_tokens", serde_json::json!({}))
            };
            if kind == "pplx" {
                options["_decision_attention_mode"] = config
                    .get("attention_mode")
                    .cloned()
                    .unwrap_or_else(|| "causal".into());
            }
            options["_decision"] = serde_json::json!({"kind":kind,"config":config});
            Some(kind)
        } else {
            None
        };
        let family = options["model_type"]
            .as_str()
            .ok_or_else(|| start("Missing model_type"))?
            .to_owned();
        let normalized_family = match family.as_str() {
            "llama_bidirec" => "llama",
            "ministral3" => "mistral",
            "new" => "gte",
            "gemma4_unified" => "gemma4",
            other => other,
        }
        .to_owned();
        options["model_type"] = normalized_family.clone().into();
        if decision
            && !laya
            && !matches!(
                normalized_family.as_str(),
                "gemma4"
                    | "gemma4_text"
                    | "qwen3_5"
                    | "qwen3_5_text"
                    | "qwen3_5_moe"
                    | "qwen3_5_moe_text"
            )
        {
            return Err(start("Typed decisions require Laya, Gemma4, or Qwen3.5"));
        }
        if matches!(decision_kind, Some("pplx" | "clef"))
            && !normalized_family.starts_with("qwen3_5")
        {
            return Err(start("Pplx and Clef heads require Qwen3.5"));
        }
        options["_cuda_graphs"] = std::env::var("TEI_TORCH_CUDA_GRAPHS")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .into();
        options["_media_cuda_graphs"] = std::env::var("TEI_TORCH_MEDIA_CUDA_GRAPHS")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .into();
        let graph_max_tokens = std::env::var("TEI_TORCH_CUDA_GRAPH_MAX_TOKENS")
            .map(|value| value.parse::<i64>().map_err(start))
            .unwrap_or(Ok(4096))?;
        if graph_max_tokens <= 0 {
            return Err(start("CUDA graph maximum tokens must be positive"));
        }
        options["_cuda_graph_max_tokens"] = graph_max_tokens.into();
        options["_cudnn_varlen"] = std::env::var("TEI_TORCH_CUDNN_VARLEN")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .into();
        if family == "llama_bidirec" {
            options["use_bidirectional_attention"] = true.into();
        }
        let text = options.get("text_config").unwrap_or(&options);
        let integer = |key: &str, alias: &str, fallback: i64| {
            text.get(key)
                .or_else(|| text.get(alias))
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(fallback)
        };
        let config = Config {
            model_type: normalized_family,
            hidden_size: integer("hidden_size", "dim", 0),
            num_attention_heads: integer("num_attention_heads", "n_heads", 0),
            num_hidden_layers: integer("num_hidden_layers", "n_layers", 0),
            intermediate_size: integer("intermediate_size", "hidden_dim", 0),
            vocab_size: integer("vocab_size", "vocab_size", 0),
            max_position_embeddings: integer(
                "max_position_embeddings",
                "max_position_embeddings",
                i64::MAX,
            ),
            type_vocab_size: integer("type_vocab_size", "type_vocab_size", 1),
            layer_norm_eps: text
                .get("layer_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1e-5),
            hidden_act: text
                .get("hidden_act")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("gelu")
                .into(),
            is_decoder: text
                .get("is_decoder")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            add_cross_attention: text
                .get("add_cross_attention")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            position_embedding_type: text
                .get("position_embedding_type")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
        };
        if config.model_type == "bert"
            && ![
                options["auto_map"]["AutoConfig"].as_str(),
                options["_name_or_path"].as_str(),
            ]
            .into_iter()
            .flatten()
            .any(|name| {
                name.contains("jina-bert-implementation")
                    || name.contains("jina-bert-v2-qk-post-norm")
            })
            && (config.is_decoder
                || config.add_cross_attention
                || config
                    .position_embedding_type
                    .as_deref()
                    .is_some_and(|p| p != "absolute"))
        {
            return Err(start(
                "BERT requires an encoder with absolute position embeddings",
            ));
        }
        if config.hidden_size <= 0
            || config.vocab_size <= 0
            || config.num_attention_heads <= 0
            || config.num_hidden_layers <= 0
            || config.hidden_size % config.num_attention_heads != 0
        {
            return Err(start("Invalid model dimensions"));
        }
        let mut module_dense_paths = Vec::new();
        if path.join("modules.json").exists() {
            let modules: serde_json::Value =
                serde_json::from_slice(&fs::read(path.join("modules.json")).map_err(start)?)
                    .map_err(start)?;
            for module in modules
                .as_array()
                .ok_or_else(|| start("modules.json must contain an array"))?
            {
                let kind = module["type"]
                    .as_str()
                    .ok_or_else(|| start("Missing module type"))?;
                if !kind.starts_with("sentence_transformers.")
                    || !matches!(
                        kind.rsplit('.').next(),
                        Some("Transformer" | "Pooling" | "Normalize" | "Dense")
                    )
                {
                    return Err(start(format!(
                        "LibTorch does not yet support Sentence Transformers module {kind}"
                    )));
                }
                if kind.rsplit('.').next() == Some("Dense") {
                    module_dense_paths.push(
                        module["path"]
                            .as_str()
                            .ok_or_else(|| start("Dense module is missing its path"))?
                            .to_owned(),
                    );
                }
            }
        }
        let dense_paths = dense_paths.unwrap_or(module_dense_paths);
        if !dense_paths.is_empty() && !matches!(model_type, ModelType::Embedding(_)) {
            return Err(start(
                "Sentence Transformers Dense modules require an embedding model",
            ));
        }
        options["_dense_count"] = serde_json::json!(dense_paths.len());
        options["_pooling"] = serde_json::json!(if pool == Pool::Splade {
            "splade"
        } else {
            "embedding"
        });
        for (i, dense_path) in dense_paths.iter().enumerate() {
            let relative = Path::new(dense_path);
            if relative.as_os_str().is_empty()
                || relative
                    .components()
                    .any(|part| !matches!(part, std::path::Component::Normal(_)))
            {
                return Err(start(
                    "Dense module path must stay inside the model directory",
                ));
            }
            let dense_config: serde_json::Value = serde_json::from_slice(
                &fs::read(path.join(relative).join("config.json")).map_err(start)?,
            )
            .map_err(start)?;
            if !dense_config.is_object() {
                return Err(start("Dense config must be an object"));
            }
            if dense_config["bias"].as_bool().is_none() {
                return Err(start("Dense config bias must be a boolean"));
            }
            options["_dense"][i.to_string()] = dense_config;
        }
        let activation = match config.hidden_act.as_str() {
            "gelu" => 0,
            "gelu_new" | "gelu_pytorch_tanh" => 1,
            "relu" => 2,
            "silu" => 3,
            _ => 0,
        };
        let native = NativeConfig {
            hidden: config.hidden_size,
            heads: config.num_attention_heads,
            layers: config.num_hidden_layers,
            intermediate: config.intermediate_size,
            vocab: config.vocab_size,
            positions: config.max_position_embeddings,
            types: config.type_vocab_size,
            epsilon: config.layer_norm_eps,
            activation,
        };
        let dtype = match dtype {
            "float32" | "auto" => 0,
            "float16" => 1,
            "bfloat16" => 2,
            _ => return Err(start("Unsupported LibTorch dtype")),
        };
        let device = CString::new(device).map_err(start)?;
        // SAFETY: config and device stay valid during creation, and the bridge copies them.
        let handle = NonNull::new(unsafe { tei_create(&native, device.as_ptr(), dtype) })
            .ok_or_else(|| start(native_error()))?;
        let mut model = Self {
            handle,
            config,
            pool,
            output_width: 0,
            pooled_width: 0,
            classification_width: 0,
            decision,
        };
        fn flatten(value: &serde_json::Value, key: &str, output: &mut Vec<(String, String)>) {
            match value {
                serde_json::Value::Object(map) => {
                    for (name, value) in map {
                        let key = if key.is_empty() {
                            name.clone()
                        } else {
                            format!("{key}.{name}")
                        };
                        flatten(value, &key, output);
                    }
                }
                serde_json::Value::Array(items) => {
                    output.push((key.into(), value.to_string()));
                    for (index, value) in items.iter().enumerate() {
                        flatten(value, &format!("{key}.{index}"), output);
                    }
                }
                serde_json::Value::Null => {}
                serde_json::Value::String(value) => output.push((key.into(), value.clone())),
                value => output.push((key.into(), value.to_string())),
            }
        }
        let mut flattened = Vec::new();
        flatten(&options, "", &mut flattened);
        for (key, value) in flattened {
            let key = CString::new(key).map_err(start)?;
            let value = CString::new(value).map_err(start)?;
            // SAFETY: the bridge copies both NUL-terminated strings before returning.
            if unsafe { tei_option(handle.as_ptr(), key.as_ptr(), value.as_ptr()) } != 0 {
                return Err(start(native_error()));
            }
        }
        let index = path.join("model.safetensors.index.json");
        let files = if index.exists() {
            let index: serde_json::Value =
                serde_json::from_slice(&fs::read(index).map_err(start)?).map_err(start)?;
            let map = index["weight_map"]
                .as_object()
                .ok_or_else(|| start("Invalid safetensors index"))?;
            let mut files = std::collections::BTreeSet::new();
            for file in map.values() {
                let file = file
                    .as_str()
                    .ok_or_else(|| start("Invalid shard filename"))?;
                // Restrict shard references to files inside the model directory.
                if Path::new(file).components().count() != 1 || !file.ends_with(".safetensors") {
                    return Err(start("Invalid shard filename"));
                }
                files.insert(file.to_owned());
            }
            files.into_iter().collect::<Vec<_>>()
        } else {
            vec!["model.safetensors".to_owned()]
        };
        let mut files = files
            .into_iter()
            .map(|file| (file, String::new()))
            .collect::<Vec<_>>();
        if let Some(kind) = decision_kind {
            if kind == "pplx" {
                files.push(("readout.safetensors".into(), "__tei_decision.".into()));
            }
            if kind == "clef" {
                files.push(("joint_head.safetensors".into(), "__tei_decision.".into()));
            }
        }
        for (i, dense_path) in dense_paths.iter().enumerate() {
            files.push((
                format!("{dense_path}/model.safetensors"),
                format!("_dense.{i}."),
            ));
        }
        for (file, prefix) in files {
            let data = fs::read(path.join(file)).map_err(start)?;
            let tensors = SafeTensors::deserialize(&data).map_err(start)?;
            for (name, tensor) in tensors.tensors() {
                // Integer buffers are derived from packed positions rather than uploaded as weights.
                if name.ends_with("position_ids") || name.ends_with("token_type_ids") {
                    continue;
                }
                let dtype = match tensor.dtype() {
                    Dtype::F32 => 0,
                    Dtype::F16 => 1,
                    Dtype::BF16 => 2,
                    other => return Err(start(format!("Unsupported weight dtype {other:?}"))),
                };
                let shape = tensor
                    .shape()
                    .iter()
                    .map(|&n| i64::try_from(n).map_err(start))
                    .collect::<Result<Vec<_>, _>>()?;
                let name = CString::new(format!("{prefix}{name}")).map_err(start)?;
                // SAFETY: safetensors validates byte lengths. Native code copies the data before returning.
                if unsafe {
                    tei_weight(
                        handle.as_ptr(),
                        name.as_ptr(),
                        tensor.data().as_ptr(),
                        shape.as_ptr(),
                        shape.len(),
                        dtype,
                    )
                } != 0
                {
                    return Err(start(native_error()));
                }
            }
        }
        // SAFETY: the live handle is owned by model; this checks every required weight and its shape.
        if unsafe { tei_ready(handle.as_ptr()) } != 0 {
            return Err(start(native_error()));
        }
        // SAFETY: ready() has created and validated the concrete model.
        model.output_width = usize::try_from(unsafe { tei_output_width(handle.as_ptr()) })
            .map_err(|_| start(native_error()))?;
        // SAFETY: ready() has validated the projection chain and its output width.
        model.pooled_width = usize::try_from(unsafe { tei_pooled_width(handle.as_ptr()) })
            .map_err(|_| start(native_error()))?;
        // SAFETY: ready() has validated the concrete model and owns its weights.
        model.classification_width =
            usize::try_from(unsafe { tei_classification_width(handle.as_ptr()) })
                .map_err(|_| start(native_error()))?;
        if model_type == ModelType::Classifier && model.classification_width == 0 {
            return Err(start("Checkpoint has no supported classification head"));
        }
        Ok(model)
    }
}

impl LibtorchBackend {
    fn run(&self, batch: Batch, pool: i32, hidden: usize) -> Result<Embeddings, BackendError> {
        self.run_inner(batch, pool, hidden, None)
            .map(|(embeddings, _)| embeddings)
    }
    fn run_inner(
        &self,
        batch: Batch,
        pool: i32,
        hidden: usize,
        requests: Option<&[DecisionInput]>,
    ) -> Result<(Embeddings, Vec<DecisionOutput>), BackendError> {
        let ends = &batch.cumulative_seq_lengths;
        if ends.len() < 2
            || ends[0] != 0
            || ends.last().copied() != Some(batch.input_ids.len() as u32)
            || ends.windows(2).any(|w| w[0] >= w[1])
            || batch.token_type_ids.len() != batch.input_ids.len()
            || batch.position_ids.len() != batch.input_ids.len()
        {
            return Err(inference("Invalid packed batch"));
        }
        if batch.compact_input_ids.is_some()
            || batch.compact_position_ids.is_some()
            || batch.scatter_unfold.is_some()
            || batch.fold_gather.is_some()
        {
            return Err(inference("LibTorch does not yet support radix folding"));
        }
        let b = ends.len() - 1;
        let lengths: Vec<i64> = ends.windows(2).map(|w| (w[1] - w[0]) as i64).collect();
        let seq = *lengths.iter().max().unwrap() as usize;
        if !batch.multimodal.is_empty() && batch.multimodal.len() != b {
            return Err(inference("Invalid multimodal batch length"));
        }
        let mut images = Vec::new();
        let mut audios = Vec::new();
        let mut media_positions = Vec::new();
        if batch.multimodal.iter().any(Option::is_some) {
            media_positions = (0..3)
                .flat_map(|_| batch.position_ids.iter().map(|&p| i64::from(p)))
                .collect();
            for (row, media) in batch.multimodal.iter().enumerate() {
                let Some(media) = media else { continue };
                let begin = ends[row] as usize;
                let length = lengths[row] as usize;
                for axis in 0..3 {
                    if media.position_ids[axis].len() != length {
                        return Err(inference("Invalid multimodal position length"));
                    }
                    for (index, &value) in media.position_ids[axis].iter().enumerate() {
                        media_positions[axis * batch.input_ids.len() + begin + index] =
                            i64::from(value);
                    }
                }
                for (start, image) in &media.images {
                    let patches = image
                        .grid_thw
                        .iter()
                        .try_fold(1usize, |n, &d| n.checked_mul(d))
                        .ok_or_else(|| inference("Image grid overflow"))?;
                    let merge = image
                        .merge_size
                        .checked_mul(image.merge_size)
                        .filter(|&n| n > 0)
                        .ok_or_else(|| inference("Invalid image merge size"))?;
                    if image.patch_dim == 0
                        || patches == 0
                        || patches % merge != 0
                        || patches.checked_mul(image.patch_dim) != Some(image.pixels.len())
                        || start
                            .checked_add(patches / merge)
                            .is_none_or(|end| end > length)
                    {
                        return Err(inference("Invalid image layout or token span"));
                    }
                    images.push(NativeImage {
                        pixels: image.pixels.as_ptr(),
                        rows: patches as i64,
                        patch_dim: image.patch_dim as i64,
                        grid: image.grid_thw.map(|n| n as i64),
                        merge_size: image.merge_size as i64,
                        token_start: (begin + start) as i64,
                        token_count: (patches / merge) as i64,
                        sequence_start: begin as i64,
                    });
                }
                for (start, audio) in &media.audios {
                    if audio.feature_size == 0
                        || audio.mask.is_empty()
                        || audio.mask.len().checked_mul(audio.feature_size)
                            != Some(audio.values.len())
                        || start
                            .checked_add(audio.token_count())
                            .is_none_or(|end| end > length)
                    {
                        return Err(inference("Invalid audio layout or token span"));
                    }
                    audios.push(NativeAudio {
                        values: audio.values.as_ptr(),
                        mask: audio.mask.as_ptr(),
                        frames: audio.mask.len() as i64,
                        feature_size: audio.feature_size as i64,
                        token_start: (begin + start) as i64,
                        token_count: audio.token_count() as i64,
                    });
                }
            }
        }
        // Preserve TEI's packed layout: allocations scale with actual tokens, never B * max_length.
        let ids: Vec<i64> = batch.input_ids.iter().map(|&id| id as i64).collect();
        let types: Vec<i64> = batch.token_type_ids.iter().map(|&id| id as i64).collect();
        let positions: Vec<i64> = batch.position_ids.iter().map(|&id| id as i64).collect();
        let cumulative = ends
            .iter()
            .map(|&end| i32::try_from(end).map_err(inference))
            .collect::<Result<Vec<_>, _>>()?;
        if ids.iter().any(|&id| id >= self.config.vocab_size)
            || (self.config.type_vocab_size > 0
                && types.iter().any(|&id| id >= self.config.type_vocab_size))
            || positions
                .iter()
                .any(|&id| id >= self.config.max_position_embeddings)
        {
            return Err(inference(
                "Token/type/position ID exceeds configured limits",
            ));
        }
        if let Some(requests) = requests {
            if !self.decision || requests.len() != b {
                return Err(inference(
                    "Decision metadata count does not match a typed decision model/batch",
                ));
            }
            let owned = requests
                .iter()
                .map(OwnedDecision::new)
                .collect::<Result<Vec<_>, _>>()?;
            let native = owned.iter().map(OwnedDecision::native).collect::<Vec<_>>();
            let mut counts = vec![0i64; b];
            // SAFETY: descriptors and their backing span/token buffers remain owned above.
            if unsafe {
                tei_decision_counts(
                    self.handle.as_ptr(),
                    native.as_ptr(),
                    native.len(),
                    counts.as_mut_ptr(),
                )
            } != 0
            {
                return Err(inference(native_error()));
            }
            let counts = counts
                .into_iter()
                .map(|count| usize::try_from(count).map_err(inference))
                .collect::<Result<Vec<_>, _>>()?;
            let total = counts.iter().try_fold(0usize, |n, &count| {
                n.checked_add(count)
                    .ok_or_else(|| inference("Decision output size overflow"))
            })?;
            let mut logits = vec![0f32; total];
            let mut actions = vec![0f32; b];
            // SAFETY: native output counts bound capacity, validated packed/media buffers stay live,
            // and all exceptions are caught by the bridge before returning across the C ABI.
            if unsafe {
                tei_decide(
                    self.handle.as_ptr(),
                    ids.as_ptr(),
                    types.as_ptr(),
                    positions.as_ptr(),
                    cumulative.as_ptr(),
                    b as i64,
                    seq as i64,
                    native.as_ptr(),
                    logits.as_mut_ptr(),
                    logits.len(),
                    actions.as_mut_ptr(),
                    if media_positions.is_empty() {
                        std::ptr::null()
                    } else {
                        media_positions.as_ptr()
                    },
                    images.as_ptr(),
                    images.len(),
                    audios.as_ptr(),
                    audios.len(),
                )
            } != 0
            {
                return Err(inference(native_error()));
            }
            let mut offset = 0;
            let decisions = counts
                .into_iter()
                .zip(actions)
                .map(|(count, action_probability)| {
                    let result = DecisionOutput {
                        logits: logits[offset..offset + count].to_vec(),
                        action_probability,
                    };
                    offset += count;
                    result
                })
                .collect();
            return Ok((Embeddings::default(), decisions));
        }
        let pooled: Vec<i64> = batch.pooled_indices.iter().map(|&n| n as i64).collect();
        let raw: Vec<i64> = batch.raw_indices.iter().map(|&n| n as i64).collect();
        if pooled.iter().chain(&raw).any(|&n| n as usize >= b) {
            return Err(inference("Invalid output index"));
        }
        if pooled
            .iter()
            .chain(&raw)
            .collect::<std::collections::HashSet<_>>()
            .len()
            != pooled.len() + raw.len()
        {
            return Err(inference("Duplicate output indices"));
        }
        let raw_width = if pool >= 4 { hidden } else { self.output_width };
        let raw_rows = raw
            .iter()
            .map(|&i| lengths[i as usize] as usize)
            .sum::<usize>();
        let size = pooled
            .len()
            .checked_mul(hidden)
            .and_then(|n| {
                raw_rows
                    .checked_mul(raw_width)
                    .and_then(|r| n.checked_add(r))
            })
            .ok_or_else(|| inference("Output size overflow"))?;
        let mut output = vec![0f32; size];
        // SAFETY: validated dimensions and indices bound every native read/write. All buffers stay
        // live until this synchronous call returns; exceptions cannot cross the C ABI.
        if unsafe {
            tei_forward(
                self.handle.as_ptr(),
                ids.as_ptr(),
                types.as_ptr(),
                positions.as_ptr(),
                cumulative.as_ptr(),
                b as i64,
                seq as i64,
                pool,
                pooled.as_ptr(),
                pooled.len(),
                raw.as_ptr(),
                raw.len(),
                output.as_mut_ptr(),
                output.len(),
                if media_positions.is_empty() {
                    std::ptr::null()
                } else {
                    media_positions.as_ptr()
                },
                images.as_ptr(),
                images.len(),
                audios.as_ptr(),
                audios.len(),
            )
        } != 0
        {
            return Err(inference(native_error()));
        }
        let mut result = Embeddings::default();
        let mut offset = 0;
        for index in pooled {
            result.insert(
                index as usize,
                Embedding::Pooled(output[offset..offset + hidden].to_vec()),
            );
            offset += hidden;
        }
        for index in raw {
            let size = lengths[index as usize] as usize * raw_width;
            let values = output[offset..offset + size]
                .chunks_exact(raw_width)
                .map(<[f32]>::to_vec)
                .collect();
            result.insert(index as usize, Embedding::All(values));
            offset += size;
        }
        Ok((result, vec![]))
    }
}

impl Backend for LibtorchBackend {
    fn decide(
        &self,
        mut batch: Batch,
        inputs: Vec<DecisionInput>,
    ) -> Result<Vec<DecisionOutput>, BackendError> {
        batch.pooled_indices.clear();
        batch.raw_indices.clear();
        self.run_inner(batch, 2, self.output_width, Some(&inputs))
            .map(|(_, decisions)| decisions)
    }
    fn health(&self) -> Result<(), BackendError> {
        Ok(())
    }
    fn embed(&self, batch: Batch) -> Result<Embeddings, BackendError> {
        let (pool, width) = match self.pool {
            Pool::Cls => (0, self.pooled_width),
            Pool::Mean => (1, self.pooled_width),
            Pool::LastToken => (2, self.pooled_width),
            Pool::Splade => (3, self.pooled_width),
        };
        self.run(batch, pool, width)
    }
    fn predict(&self, mut batch: Batch) -> Result<Predictions, BackendError> {
        if self.classification_width == 0 {
            return Err(inference("Checkpoint has no classification head"));
        }
        batch.pooled_indices =
            (0..batch.cumulative_seq_lengths.len().saturating_sub(1) as u32).collect();
        batch.raw_indices.clear();
        let values = self.run(batch, 4, self.classification_width)?;
        let mut result = Predictions::default();
        for (index, value) in values {
            if let Embedding::Pooled(value) = value {
                result.insert(index, value);
            }
        }
        Ok(result)
    }
    fn predict_tokens(&self, mut batch: Batch) -> Result<TokenPredictions, BackendError> {
        if self.classification_width == 0 {
            return Err(inference("Checkpoint has no classification head"));
        }
        batch.raw_indices =
            (0..batch.cumulative_seq_lengths.len().saturating_sub(1) as u32).collect();
        batch.pooled_indices.clear();
        let values = self.run(batch, 5, self.classification_width)?;
        let mut result = TokenPredictions::default();
        for (index, value) in values {
            if let Embedding::All(value) = value {
                result.insert(index, value);
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, TensorView};
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[cfg(not(feature = "benchmark-cuda"))]
    use text_embeddings_backend_candle::CandleBackend;

    struct Fixture(std::path::PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn fixture(prefix: &str, sharded: bool) -> Fixture {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "tei-libtorch-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        let config = serde_json::json!({"model_type":"bert", "hidden_size":16, "num_attention_heads":2,
            "num_hidden_layers":2, "intermediate_size":24, "vocab_size":16, "max_position_embeddings":16,
            "type_vocab_size":2, "id2label":{"0":"a","1":"b","2":"c"}, "layer_norm_eps":1e-5, "hidden_act":"gelu_pytorch_tanh", "pad_token_id":0,
            "hidden_dropout_prob":0.0, "attention_probs_dropout_prob":0.0, "initializer_range":0.02});
        fs::write(
            path.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let mut weights = Vec::new();
        let mut add = |name: String, shape: Vec<usize>| {
            let salt = weights.len();
            let values = (0..shape.iter().product())
                .map(|i| {
                    let value = (((i * 17 + salt * 13) % 101) as f32 - 50.) / 200.;
                    if name.ends_with("LayerNorm.weight") {
                        1.0 + value
                    } else {
                        value
                    }
                })
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            weights.push((format!("{prefix}{name}"), shape, values));
        };
        add("embeddings.word_embeddings.weight".into(), vec![16, 16]);
        add("embeddings.position_embeddings.weight".into(), vec![16, 16]);
        add(
            "embeddings.token_type_embeddings.weight".into(),
            vec![2, 16],
        );
        add("embeddings.LayerNorm.weight".into(), vec![16]);
        add("embeddings.LayerNorm.bias".into(), vec![16]);
        for i in 0..2 {
            let p = format!("encoder.layer.{i}.");
            for name in [
                "attention.self.query",
                "attention.self.key",
                "attention.self.value",
                "attention.output.dense",
            ] {
                add(format!("{p}{name}.weight"), vec![16, 16]);
                add(format!("{p}{name}.bias"), vec![16]);
            }
            for name in ["attention.output.LayerNorm", "output.LayerNorm"] {
                add(format!("{p}{name}.weight"), vec![16]);
                add(format!("{p}{name}.bias"), vec![16]);
            }
            add(format!("{p}intermediate.dense.weight"), vec![24, 16]);
            add(format!("{p}intermediate.dense.bias"), vec![24]);
            add(format!("{p}output.dense.weight"), vec![16, 24]);
            add(format!("{p}output.dense.bias"), vec![16]);
        }
        if prefix.is_empty() {
            add("classifier.weight".into(), vec![3, 16]);
            add("classifier.bias".into(), vec![3]);
            add(
                "cls.predictions.transform.dense.weight".into(),
                vec![16, 16],
            );
            add("cls.predictions.transform.dense.bias".into(), vec![16]);
            add(
                "cls.predictions.transform.LayerNorm.weight".into(),
                vec![16],
            );
            add("cls.predictions.transform.LayerNorm.bias".into(), vec![16]);
            add("cls.predictions.decoder.weight".into(), vec![16, 16]);
            add("cls.predictions.bias".into(), vec![16]);
        }
        let views = weights
            .iter()
            .map(|(name, shape, data)| {
                (
                    name.clone(),
                    TensorView::new(Dtype::F32, shape.clone(), data).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        if sharded {
            let mut map = serde_json::Map::new();
            for (i, slice) in views.chunks(views.len().div_ceil(2)).enumerate() {
                let file = format!("model-{i}.safetensors");
                fs::write(
                    path.join(&file),
                    serialize(slice.iter().map(|(n, t)| (n.clone(), t.clone())), None).unwrap(),
                )
                .unwrap();
                for (name, _) in slice {
                    map.insert(name.clone(), serde_json::json!(file));
                }
            }
            fs::write(
                path.join("model.safetensors.index.json"),
                serde_json::to_vec(&serde_json::json!({"weight_map":map})).unwrap(),
            )
            .unwrap();
        } else {
            fs::write(
                path.join("model.safetensors"),
                serialize(views, None).unwrap(),
            )
            .unwrap();
        }
        Fixture(path)
    }
    fn batch() -> Batch {
        Batch {
            input_ids: vec![1, 5, 7, 2, 4],
            token_type_ids: vec![0, 0, 1, 1, 1],
            position_ids: vec![0, 1, 2, 0, 1],
            cumulative_seq_lengths: vec![0, 3, 5],
            max_length: 3,
            pooled_indices: vec![1],
            raw_indices: vec![0],
            multimodal: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        }
    }
    fn values(value: &Embedding) -> Vec<f32> {
        match value {
            Embedding::Pooled(v) => v.clone(),
            Embedding::All(v) => v.iter().flatten().copied().collect(),
        }
    }
    fn dense_fixture() -> (Fixture, Vec<String>) {
        let fixture = fixture("", false);
        let paths = vec!["2_Dense".to_owned(), "3_Dense".to_owned()];
        fs::write(fixture.0.join("modules.json"), serde_json::to_vec(&serde_json::json!([
            {"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"},
            {"idx":1,"name":"1","path":"1_Pooling","type":"sentence_transformers.models.Pooling"},
            {"idx":2,"name":"2","path":"2_Dense","type":"sentence_transformers.models.Dense"},
            {"idx":3,"name":"3","path":"3_Dense","type":"sentence_transformers.models.Dense"}
        ])).unwrap()).unwrap();
        for (index, (input, output)) in [(16, 7), (7, 3)].into_iter().enumerate() {
            let dir = fixture.0.join(&paths[index]);
            fs::create_dir(&dir).unwrap();
            fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({
                "in_features":input,"out_features":output,"bias":index == 0,
                "activation_function":if index == 0 { "torch.nn.modules.activation.Tanh" } else { "torch.nn.modules.linear.Identity" }
            })).unwrap()).unwrap();
            let weight = (0..input * output)
                .flat_map(|i| (((i % 11) as f32 - 5.) * 0.04).to_le_bytes())
                .collect::<Vec<_>>();
            let bias = (0..output)
                .flat_map(|i| (i as f32 * 0.01).to_le_bytes())
                .collect::<Vec<_>>();
            let mut tensors = vec![(
                "linear.weight",
                TensorView::new(Dtype::F32, vec![output, input], &weight).unwrap(),
            )];
            if index == 0 {
                tensors.push((
                    "linear.bias",
                    TensorView::new(Dtype::F32, vec![output], &bias).unwrap(),
                ));
            }
            fs::write(
                dir.join("model.safetensors"),
                serialize(tensors, None).unwrap(),
            )
            .unwrap();
        }
        (fixture, paths)
    }
    #[cfg(not(feature = "benchmark-cuda"))]
    #[test]
    fn dense_chain_matches_candle_and_preserves_raw_width() {
        let (fixture, paths) = dense_fixture();
        for pool in [Pool::Cls, Pool::Mean, Pool::LastToken] {
            let model_type = ModelType::Embedding(pool);
            let native =
                LibtorchBackend::new(&fixture.0, "float32", model_type.clone(), "cpu").unwrap();
            let candle = CandleBackend::new(
                &fixture.0,
                "float32".into(),
                model_type,
                Some(paths.clone()),
                0,
            )
            .unwrap();
            let expected = candle.embed(batch()).unwrap();
            let actual = native.embed(batch()).unwrap();
            for (row, embedding) in &actual {
                match embedding {
                    Embedding::Pooled(v) => assert_eq!(v.len(), 3),
                    Embedding::All(v) => assert!(v.iter().all(|token| token.len() == 16)),
                }
                let got = values(embedding);
                let want = values(&expected[row]);
                assert_eq!(got.len(), want.len());
                assert!(got.iter().zip(want).all(|(a, b)| (a - b).abs() < 0.0001));
            }
        }
        fs::write(
            fixture.0.join("2_Dense/config.json"),
            br#"{"in_features":15,"out_features":7,"bias":true}"#,
        )
        .unwrap();
        assert!(LibtorchBackend::new(
            &fixture.0,
            "float32",
            ModelType::Embedding(Pool::Cls),
            "cpu"
        )
        .is_err());
    }
    #[test]
    #[ignore = "Requires an NVIDIA GPU and CUDA LibTorch 2.14.1"]
    fn cuda_dense_chain_matches_cpu_with_mixed_widths() {
        let (fixture, _) = dense_fixture();
        for pool in [Pool::Cls, Pool::Mean, Pool::LastToken] {
            let model_type = ModelType::Embedding(pool);
            let cpu =
                LibtorchBackend::new(&fixture.0, "float32", model_type.clone(), "cpu").unwrap();
            let gpu = LibtorchBackend::new(&fixture.0, "float16", model_type, "cuda:0").unwrap();
            let expected = cpu.embed(batch()).unwrap();
            let actual = gpu.embed(batch()).unwrap();
            for (row, embedding) in actual {
                let got = values(&embedding);
                let want = values(&expected[&row]);
                assert_eq!(got.len(), want.len());
                assert!(got.iter().zip(want).all(|(a, b)| (a - b).abs() < 0.02));
                match embedding {
                    Embedding::Pooled(v) => assert_eq!(v.len(), 3),
                    Embedding::All(v) => assert!(v.iter().all(|token| token.len() == 16)),
                }
            }
        }
    }
    #[cfg(not(feature = "benchmark-cuda"))]
    #[test]
    fn native_classifier_and_splade_match_candle() {
        let fixture = fixture("", false);
        let torch =
            LibtorchBackend::new(&fixture.0, "float32", ModelType::Classifier, "cpu").unwrap();
        let candle =
            CandleBackend::new(&fixture.0, "float32".into(), ModelType::Classifier, None, 0)
                .unwrap();
        let mut classifier_batch = batch();
        classifier_batch.pooled_indices = vec![0, 1];
        classifier_batch.raw_indices.clear();
        let expected = candle.predict(classifier_batch.clone()).unwrap();
        let actual = torch.predict(classifier_batch).unwrap();
        for (row, values) in expected {
            for (a, b) in actual[&row].iter().zip(values) {
                assert!((a - b).abs() < 0.0001);
            }
        }
        let mut token_batch = batch();
        token_batch.pooled_indices.clear();
        token_batch.raw_indices = vec![0, 1];
        let expected = candle.predict_tokens(token_batch.clone()).unwrap();
        let actual = torch.predict_tokens(token_batch).unwrap();
        for (row, values) in expected {
            for (a, b) in actual[&row].iter().flatten().zip(values.iter().flatten()) {
                assert!((a - b).abs() < 0.0001);
            }
        }
        let torch = LibtorchBackend::new(
            &fixture.0,
            "float32",
            ModelType::Embedding(Pool::Splade),
            "cpu",
        )
        .unwrap();
        let candle = CandleBackend::new(
            &fixture.0,
            "float32".into(),
            ModelType::Embedding(Pool::Splade),
            None,
            0,
        )
        .unwrap();
        let actual = torch.embed(batch()).unwrap();
        let expected = candle.embed(batch()).unwrap();
        for (row, value) in expected {
            let actual = values(&actual[&row]);
            let expected = values(&value);
            assert_eq!(actual.len(), expected.len());
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() < 0.0001);
            }
        }
    }
    #[cfg(not(feature = "benchmark-cuda"))]
    #[test]
    fn bert_relu_and_silu_match_candle_for_embeddings_and_heads() {
        for activation in ["relu", "silu"] {
            let fixture = fixture("", false);
            let mut config: serde_json::Value =
                serde_json::from_slice(&fs::read(fixture.0.join("config.json")).unwrap()).unwrap();
            config["hidden_act"] = activation.into();
            fs::write(
                fixture.0.join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();
            for pool in [Pool::Cls, Pool::Mean, Pool::LastToken, Pool::Splade] {
                let model_type = ModelType::Embedding(pool);
                let torch =
                    LibtorchBackend::new(&fixture.0, "float32", model_type.clone(), "cpu").unwrap();
                let candle =
                    CandleBackend::new(&fixture.0, "float32".into(), model_type, None, 0).unwrap();
                let actual = torch.embed(batch()).unwrap();
                for (row, expected) in candle.embed(batch()).unwrap() {
                    let got = values(&actual[&row]);
                    let want = values(&expected);
                    assert_eq!(got.len(), want.len());
                    assert!(
                        got.iter().zip(want).all(|(a, b)| (a - b).abs() < 0.0001),
                        "{activation} embedding row {row}"
                    );
                }
            }
            let torch =
                LibtorchBackend::new(&fixture.0, "float32", ModelType::Classifier, "cpu").unwrap();
            let candle =
                CandleBackend::new(&fixture.0, "float32".into(), ModelType::Classifier, None, 0)
                    .unwrap();
            let mut input = batch();
            input.pooled_indices = vec![0, 1];
            input.raw_indices.clear();
            let actual = torch.predict(input.clone()).unwrap();
            for (row, expected) in candle.predict(input.clone()).unwrap() {
                assert!(
                    actual[&row]
                        .iter()
                        .zip(expected)
                        .all(|(a, b)| (a - b).abs() < 0.0001),
                    "{activation} classifier row {row}"
                );
            }
            input.pooled_indices.clear();
            input.raw_indices = vec![0, 1];
            let actual = torch.predict_tokens(input.clone()).unwrap();
            for (row, expected) in candle.predict_tokens(input).unwrap() {
                assert!(
                    actual[&row]
                        .iter()
                        .flatten()
                        .zip(expected.iter().flatten())
                        .all(|(a, b)| (a - b).abs() < 0.0001),
                    "{activation} token classifier row {row}"
                );
            }
            config["hidden_act"] = "unsupported".into();
            fs::write(
                fixture.0.join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();
            assert!(LibtorchBackend::new(
                &fixture.0,
                "float32",
                ModelType::Embedding(Pool::Mean),
                "cpu"
            )
            .is_err());
        }
    }
    // Candle selects CUDA when compiled with it; CPU parity runs in the default feature build.
    #[cfg(not(feature = "benchmark-cuda"))]
    #[test]
    fn roberta_aliases_match_candle_with_offset_positions_and_heads() {
        for (family, prefix) in [
            ("roberta", "roberta."),
            ("xlm-roberta", "xlm-roberta."),
            ("camembert", "camembert."),
        ] {
            let fixture = fixture(prefix, false);
            let mut config: serde_json::Value =
                serde_json::from_slice(&fs::read(fixture.0.join("config.json")).unwrap()).unwrap();
            config["model_type"] = family.into();
            config["pad_token_id"] = 1.into();
            fs::write(
                fixture.0.join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();
            let bytes = fs::read(fixture.0.join("model.safetensors")).unwrap();
            let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            let mut weights = tensors
                .tensors()
                .into_iter()
                .map(|(name, tensor)| (name, tensor.shape().to_vec(), tensor.data().to_vec()))
                .collect::<Vec<_>>();
            for (name, shape) in [
                ("classifier.dense.weight", vec![16, 16]),
                ("classifier.dense.bias", vec![16]),
                ("classifier.out_proj.weight", vec![3, 16]),
                ("classifier.out_proj.bias", vec![3]),
                ("lm_head.dense.weight", vec![16, 16]),
                ("lm_head.dense.bias", vec![16]),
                ("lm_head.layer_norm.weight", vec![16]),
                ("lm_head.layer_norm.bias", vec![16]),
                ("lm_head.decoder.weight", vec![16, 16]),
                ("lm_head.bias", vec![16]),
            ] {
                let data = (0..shape.iter().product())
                    .map(|i| {
                        let value = ((i * 13 % 79) as f32 - 39.) / 180.;
                        if name.ends_with("layer_norm.weight") {
                            1. + value
                        } else {
                            value
                        }
                    })
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>();
                weights.push((name.to_owned(), shape, data));
            }
            let views = weights.iter().map(|(name, shape, data)| {
                (
                    name.clone(),
                    TensorView::new(Dtype::F32, shape.clone(), data).unwrap(),
                )
            });
            fs::write(
                fixture.0.join("model.safetensors"),
                serialize(views, None).unwrap(),
            )
            .unwrap();
            let mut input = batch();
            for position in &mut input.position_ids {
                *position += 2;
            }
            for pool in [Pool::Cls, Pool::Mean, Pool::LastToken, Pool::Splade] {
                let kind = ModelType::Embedding(pool);
                let native =
                    LibtorchBackend::new(&fixture.0, "float32", kind.clone(), "cpu").unwrap();
                let candle =
                    CandleBackend::new(&fixture.0, "float32".into(), kind, None, 0).unwrap();
                let actual = native.embed(input.clone()).unwrap();
                let expected = candle.embed(input.clone()).unwrap();
                for (row, embedding) in actual {
                    let got = values(&embedding);
                    let want = values(&expected[&row]);
                    assert_eq!(got.len(), want.len());
                    assert!(
                        got.iter().zip(want).all(|(a, b)| (a - b).abs() < 0.0002),
                        "{family} pooling mismatch"
                    );
                }
            }
            let native =
                LibtorchBackend::new(&fixture.0, "float32", ModelType::Classifier, "cpu").unwrap();
            let candle =
                CandleBackend::new(&fixture.0, "float32".into(), ModelType::Classifier, None, 0)
                    .unwrap();
            input.pooled_indices = vec![0, 1];
            input.raw_indices.clear();
            let actual = native.predict(input.clone()).unwrap();
            let expected = candle.predict(input.clone()).unwrap();
            let embedding = LibtorchBackend::new(
                &fixture.0,
                "float32",
                ModelType::Embedding(Pool::Cls),
                "cpu",
            )
            .unwrap()
            .embed(input.clone())
            .unwrap();
            let parameters = weights
                .iter()
                .map(|(name, _, data)| {
                    (
                        name.as_str(),
                        data.chunks_exact(4)
                            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<std::collections::HashMap<_, _>>();
            for (&row, got) in &actual {
                let hidden = values(&embedding[&row]);
                let mid = (0..16)
                    .map(|i| {
                        (parameters["classifier.dense.bias"][i]
                            + hidden
                                .iter()
                                .enumerate()
                                .map(|(j, x)| x * parameters["classifier.dense.weight"][i * 16 + j])
                                .sum::<f32>())
                        .tanh()
                    })
                    .collect::<Vec<_>>();
                let reference = (0..3)
                    .map(|i| {
                        parameters["classifier.out_proj.bias"][i]
                            + mid
                                .iter()
                                .enumerate()
                                .map(|(j, x)| {
                                    x * parameters["classifier.out_proj.weight"][i * 16 + j]
                                })
                                .sum::<f32>()
                    })
                    .collect::<Vec<_>>();
                assert!(got.iter().zip(&reference).all(|(a,b)| (a-b).abs()<0.0002), "Native head disagrees with independent scalar reference: {got:?} vs {reference:?}");
            }
            for (row, got) in actual {
                assert!(
                    got.iter()
                        .zip(&expected[&row])
                        .all(|(a, b)| (a - b).abs() < 0.0002),
                    "{family} classifier mismatch for row {row}: {got:?} vs {:?}",
                    expected[&row]
                );
            }
            input.pooled_indices.clear();
            input.raw_indices = vec![0, 1];
            let actual = native.predict_tokens(input.clone()).unwrap();
            let expected = candle.predict_tokens(input).unwrap();
            for (row, got) in actual {
                assert!(
                    got.iter()
                        .flatten()
                        .zip(expected[&row].iter().flatten())
                        .all(|(a, b)| (a - b).abs() < 0.0002),
                    "{family} token classifier mismatch"
                );
            }
        }
    }
    #[cfg(not(feature = "benchmark-cuda"))]
    #[test]
    fn native_bert_matches_candle_for_ragged_pooled_and_raw_outputs() {
        for (prefix, sharded) in [("", false), ("bert.", true)] {
            let fixture = fixture(prefix, sharded);
            for pool in [Pool::Cls, Pool::Mean, Pool::LastToken] {
                let model_type = ModelType::Embedding(pool);
                let torch =
                    LibtorchBackend::new(&fixture.0, "float32", model_type.clone(), "cpu").unwrap();
                let candle =
                    CandleBackend::new(&fixture.0, "float32".into(), model_type, None, 0).unwrap();
                let actual = torch.embed(batch()).unwrap();
                let expected = candle.embed(batch()).unwrap();
                for index in [0, 1] {
                    let actual = values(&actual[&index]);
                    let expected = values(&expected[&index]);
                    assert_eq!(actual.len(), expected.len());
                    for (a, e) in actual.iter().zip(&expected) {
                        assert!((a - e).abs() < 2e-4, "index {index}: {a} != {e}");
                    }
                }
                // Packed attention must preserve sequence boundaries.
                let mut short = batch();
                short.input_ids = vec![2, 4];
                short.token_type_ids = vec![1, 1];
                short.position_ids = vec![0, 1];
                short.cumulative_seq_lengths = vec![0, 2];
                short.pooled_indices = vec![0];
                short.raw_indices.clear();
                short.max_length = 2;
                let single = torch.embed(short).unwrap();
                for (a, e) in values(&single[&0]).iter().zip(values(&actual[&1])) {
                    assert!((a - e).abs() < 2e-5);
                }
            }
        }
    }
    #[test]
    #[ignore = "Requires an NVIDIA GPU and CUDA LibTorch 2.14.1"]
    fn cuda_varlen_matches_cpu_for_ragged_outputs() {
        let fixture = fixture("", false);
        let mut config: serde_json::Value =
            serde_json::from_slice(&fs::read(fixture.0.join("config.json")).unwrap()).unwrap();
        for activation in ["gelu_pytorch_tanh", "relu", "silu"] {
            config["hidden_act"] = activation.into();
            fs::write(
                fixture.0.join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();
            for pool in [Pool::Cls, Pool::Mean, Pool::LastToken, Pool::Splade] {
                let model_type = ModelType::Embedding(pool);
                let cpu =
                    LibtorchBackend::new(&fixture.0, "float32", model_type.clone(), "cpu").unwrap();
                let gpu =
                    LibtorchBackend::new(&fixture.0, "float16", model_type, "cuda:0").unwrap();
                let expected = cpu.embed(batch()).unwrap();
                let actual = std::thread::spawn(move || gpu.embed(batch()).unwrap())
                    .join()
                    .unwrap();
                for index in [0, 1] {
                    for (a, e) in values(&actual[&index])
                        .iter()
                        .zip(values(&expected[&index]))
                    {
                        assert!((a - e).abs() < 1e-2, "CUDA varlen {activation}: {a} != {e}");
                    }
                }
            }
        }
    }

    #[test]
    fn malformed_batches_and_native_weight_errors_are_recoverable() {
        let fixture = fixture("", false);
        let torch = LibtorchBackend::new(
            &fixture.0,
            "float32",
            ModelType::Embedding(Pool::Mean),
            "cpu",
        )
        .unwrap();
        let mut bad = batch();
        bad.cumulative_seq_lengths = vec![];
        assert!(torch.embed(bad).is_err());
        let mut bad = batch();
        bad.input_ids[0] = 16;
        assert!(torch.embed(bad).is_err());
        let mut bad = batch();
        bad.pooled_indices = vec![2];
        assert!(torch.embed(bad).is_err());
        assert!(torch.embed(batch()).is_ok());
        fs::write(
            fixture.0.join("model.safetensors"),
            serialize(Vec::<(String, TensorView<'_>)>::new(), None).unwrap(),
        )
        .unwrap();
        let err = LibtorchBackend::new(
            &fixture.0,
            "float32",
            ModelType::Embedding(Pool::Mean),
            "cpu",
        )
        .err()
        .unwrap();
        assert!(matches!(err, BackendError::Start(_)));
    }
}
